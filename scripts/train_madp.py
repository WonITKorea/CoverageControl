import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import coverage_control as cc
from coverage_control import IOUtils
from coverage_control.nn import LPAC

# ==============================================================================
# 1. DATASET
# ==============================================================================
class CoverageDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}. Run generate_cvt_dataset_4ch.py first.")

    def len(self):
        return len(self.files)

    def get(self, idx):
        return torch.load(self.files[idx], map_location='cpu', weights_only=False)

# ==============================================================================
# 2. MODEL: CONDITIONAL UNET 1D (MLP) - RESTORED
# ==============================================================================
class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, num_train_timesteps=100):
        super().__init__()
        self.time_embed = nn.Embedding(num_train_timesteps, diffusion_step_embed_dim)

        self.diffusion_step_encoder = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )

        # 4-Layer MLP Architecture (Proven to work)
        self.model = nn.Sequential(
            nn.Linear(input_dim + global_cond_dim + diffusion_step_embed_dim, 512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.Mish(),
            nn.Linear(512, input_dim)
        )

    def forward(self, sample, timestep, global_cond):
        t_emb = self.diffusion_step_encoder(self.time_embed(timestep))
        x = torch.cat([sample, global_cond, t_emb], dim=-1)
        return self.model(x)

# ==============================================================================
# 3. UTILS: NORMALIZATION
# ==============================================================================
def compute_stats(loader):
    print("Computing dataset statistics...")
    all_actions = []
    max_samples = 10000 
    count = 0
    for batch in tqdm(loader, desc="Scanning Dataset"):
        all_actions.append(batch.y)
        count += batch.y.shape[0]
        if count >= max_samples: break
    all_actions = torch.cat(all_actions, dim=0)
    stats = {
        "action_mean": all_actions.mean(dim=0),
        "action_std": all_actions.std(dim=0) + 1e-6,
        "pos_mean": torch.tensor([512.0, 512.0]),
        "pos_std": torch.tensor([256.0, 256.0])
    }
    print(f"Stats Computed -> Action Mean: {stats['action_mean'].numpy()}")
    return stats

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Optimizations ---
    NUM_EPOCHS = 1000  # Train longer!
    BATCH_SIZE = 256  # Keep stable gradients
    LR = 1e-4
    
    config_path = "params/learning_params.toml"
    if not os.path.exists(config_path): config_path = "../../params/learning_params.toml"
    try: config = IOUtils.load_toml(config_path)
    except: config = {"CNNBackBone": {"InputDim": 4, "OutputDim": 32, "ImageSize": 32}}

    print("Loading LPAC (CNN Encoder)...")
    lpac = LPAC(config).to(device).train()
    
    # Restore MLP
    net = ConditionalUnet1D(input_dim=2, global_cond_dim=32, num_train_timesteps=100).to(device)
    
    data_dir = "lpac/data/data/train"
    dataset = CoverageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(list(net.parameters()) + list(lpac.parameters()), lr=LR)
    
    lr_scheduler = get_scheduler("cosine", optimizer=opt, num_warmup_steps=500, num_training_steps=NUM_EPOCHS * len(loader))
    
    # Keep the Improved Scheduler!
    sched = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

    stats = compute_stats(loader)
    action_mean = stats['action_mean'].to(device)
    action_std = stats['action_std'].to(device)

    print("Starting Training (MLP)...")
    save_dir = "checkpoints/madp"  # Back to original folder
    os.makedirs(save_dir, exist_ok=True)
    torch.save(stats, os.path.join(save_dir, "normalization_stats.pt"))

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        net.train()
        lpac.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            raw_action = batch.y
            norm_action = (raw_action - action_mean) / action_std
            
            map_data = batch.map if hasattr(batch, 'map') else batch.x_map
            map_data = map_data.float()
            if map_data.dim() == 5:
                B, N, C, H, W = map_data.shape
                map_input = map_data.view(B*N, C, H, W)
            else: map_input = map_data
            
            if map_input.shape[1] < 4:
                pad = torch.zeros(map_input.shape[0], 4 - map_input.shape[1], 32, 32, device=device)
                map_input = torch.cat([map_input, pad], dim=1)
            else: map_input = map_input[:, :4, :, :]

            map_feat = lpac.cnn_backbone(map_input)
            noise = torch.randn_like(norm_action)
            t = torch.randint(0, 100, (norm_action.shape[0],), device=device).long()
            noisy_act = sched.add_noise(norm_action, noise, t)
            
            noise_pred = net(noisy_act, t, map_feat)
            loss = F.mse_loss(noise_pred, noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        if (epoch+1) % 50 == 0:
            torch.save({
                'model': net.state_dict(),
                'lpac': lpac.state_dict(),
                'stats': stats,
                'epoch': epoch+1,
                'loss': avg_loss
            }, f"{save_dir}/epoch_{epoch+1}.pth")
            print(f"Saved epoch {epoch+1}, Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()
