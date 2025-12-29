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
# 2. MODEL
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
        # Concatenate: Action(2) + Map(32) + Time(256) = 290
        x = torch.cat([sample, global_cond, t_emb], dim=-1)
        return self.model(x)

# ==============================================================================
# 3. UTILS: NORMALIZATION
# ==============================================================================
def compute_stats(loader):
    """Computes Mean and Std for Actions AND Positions."""
    print("Computing dataset statistics (this may take a moment)...")
    all_actions = []
    all_positions = []
    
    # We only need a subset to estimate stats if dataset is huge
    max_samples = 5000
    count = 0
    
    for batch in tqdm(loader, desc="Scanning Dataset"):
        all_actions.append(batch.y)
        pos = batch.pos if hasattr(batch, 'pos') else batch.robot_pos
        all_positions.append(pos)
        count += batch.y.shape[0]
        if count >= max_samples: break
            
    all_actions = torch.cat(all_actions, dim=0)
    all_positions = torch.cat(all_positions, dim=0)
    
    stats = {
        "action_mean": all_actions.mean(dim=0),
        "action_std": all_actions.std(dim=0) + 1e-6,
        "pos_mean": all_positions.mean(dim=0),
        "pos_std": all_positions.std(dim=0) + 1e-6,
    }
    
    print(f"Stats Computed:")
    print(f" Action Mean: {stats['action_mean'].numpy()}")
    print(f" Pos Mean: {stats['pos_mean'].numpy()}")
    return stats

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Hyperparameters ---
    NUM_EPOCHS = 1000  # Defined EARLY
    BATCH_SIZE = 64    # Larger batch size helps stability
    LR = 1e-4

    # --- Config & Models ---
    config_path = "params/learning_params.toml"
    if not os.path.exists(config_path):
        config_path = "../../params/learning_params.toml"
    
    try:
        config = IOUtils.load_toml(config_path)
    except Exception as e:
        print(f"Warning: Could not load config. Using default LPAC config.")
        config = {"CNNBackBone": {"InputDim": 4, "OutputDim": 32, "ImageSize": 32}}

    # 1. Models
    print("Loading LPAC (Pre-trained Encoder)...")
    lpac = LPAC(config).to(device).train() # Ensure it is in TRAIN mode
    
    num_timesteps = 100
    # Condition: 32 (CNN only)
    net = ConditionalUnet1D(input_dim=2, global_cond_dim=32, num_train_timesteps=num_timesteps).to(device)
    
    # 2. Data
    data_dir = "lpac/data/data/train"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return
        
    print(f"Loading data from {data_dir}...")
    dataset = CoverageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 3. Optimizer & Scheduler
    # OPTIMIZE BOTH NETWORKS
    opt = torch.optim.AdamW(list(net.parameters()) + list(lpac.parameters()), lr=LR)
    
    # Scheduler needs NUM_EPOCHS and loader length
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=opt,
        num_warmup_steps=500,
        num_training_steps=NUM_EPOCHS * len(loader)
    )
    
    sched = DDPMScheduler(num_train_timesteps=num_timesteps)

    # 4. COMPUTE NORMALIZATION STATS
    stats = compute_stats(loader)
    
    # Move stats to device
    action_mean = stats['action_mean'].to(device)
    action_std = stats['action_std'].to(device)
    # pos stats not strictly needed for training input anymore, but good to keep
    
    # 5. Training
    print("Starting Training...")
    save_dir = "checkpoints/madp"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(stats, os.path.join(save_dir, "normalization_stats.pt"))

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        net.train()
        lpac.train() # Ensure CNN is training
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for batch in pbar:
            batch = batch.to(device)
            
            # --- Normalize Inputs ---
            raw_action = batch.y
            norm_action = (raw_action - action_mean) / action_std
            
            # Map Processing
            map_data = batch.map if hasattr(batch, 'map') else batch.x_map
            map_data = map_data.float()
            
            # Fix Shapes [B, 4, 32, 32]
            if map_data.dim() == 5: # [B, 1, 4, 32, 32] -> [B, 4, 32, 32]
                B, N, C, H, W = map_data.shape
                map_input = map_data.view(B*N, C, H, W)
            else:
                map_input = map_data
                
            # Verify Channel Count (Pad if needed)
            if map_input.shape[1] != 4:
                if map_input.shape[1] < 4:
                    pad = torch.zeros(map_input.shape[0], 4 - map_input.shape[1], 32, 32, device=device)
                    map_input = torch.cat([map_input, pad], dim=1)
                else:
                    map_input = map_input[:, :4, :, :]

            # --- Forward ---
            # NO torch.no_grad() here! We are training the CNN.
            map_feat = lpac.cnn_backbone(map_input)
            
            # Condition = Just Map Features (Shift Invariant)
            cond = map_feat # Shape: 32
            
            # --- Diffusion Step ---
            noise = torch.randn_like(norm_action)
            t = torch.randint(0, num_timesteps, (norm_action.shape[0],), device=device).long()
            
            noisy_act = sched.add_noise(norm_action, noise, t)
            noise_pred = net(noisy_act, t, cond)
            
            loss = F.mse_loss(noise_pred, noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step() # Step Scheduler
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(loader)
        
        if (epoch+1) % 50 == 0:
            torch.save({
                'model': net.state_dict(),
                'lpac': lpac.state_dict(), # Save CNN too!
                'stats': stats,
                'epoch': epoch+1,
                'loss': avg_loss
            }, f"{save_dir}/epoch_{epoch+1}.pth")
            
            print(f"Saved epoch {epoch+1}, Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()
