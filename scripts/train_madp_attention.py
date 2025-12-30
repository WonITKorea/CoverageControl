import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
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
# 2. MODEL: SPATIAL TRANSFORMER DIFFUSION
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-Attention
        # x: [Batch, SeqLen=1, Hidden] -> We are processing single agents in this implementation
        
        # 1. Attention Block
        resid = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x) 
        x = x + resid
        
        # 2. MLP Block
        resid = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + resid
        return x

class TransformerDiffusionModel(nn.Module):
    def __init__(self, input_dim=2, global_cond_dim=32, hidden_size=256, num_layers=6, num_heads=8, num_timesteps=100):
        super().__init__()
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Input Embedding (Action + Condition)
        self.input_proj = nn.Linear(input_dim + global_cond_dim, hidden_size)

        # Transformer Backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])

        # Output Head
        self.final_norm = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, input_dim)

    def forward(self, sample, timestep, global_cond):
        """
        sample: [B, 2] or [B, N, 2] (Noisy Action)
        timestep: [B]
        global_cond: [B, 32] (Map Features)
        """
        # 1. Embed Time
        t_emb = self.time_mlp(timestep) # [B, Hidden]
        
        # 2. Embed Input
        if sample.dim() == 3:
            # Sequence case: [B, N, 2]
            B, N, _ = sample.shape
            # Expand global_cond: [B, 32] -> [B, N, 32]
            global_cond_expanded = global_cond.unsqueeze(1).expand(-1, N, -1)
            x = torch.cat([sample, global_cond_expanded], dim=-1)
            x = self.input_proj(x) # [B, N, Hidden]
            x = x + t_emb.unsqueeze(1)
        else:
            # Single item case: [B, 2]
            x = torch.cat([sample, global_cond], dim=-1)
            x = self.input_proj(x) # [B, Hidden]
            x = x + t_emb
            x = x.unsqueeze(1) # [B, 1, Hidden]

        # 5. Pass through Transformer
        for block in self.blocks:
            x = block(x)

        # 6. Output
        x = self.final_norm(x)
        x = self.action_head(x)
        
        if sample.dim() == 3:
            return x
        else:
            return x.squeeze(1) # [B, 2]


# ==============================================================================
# 3. UTILS: NORMALIZATION
# ==============================================================================
def compute_stats(loader):
    print("Computing dataset statistics...")
    all_actions = []
    # We use a larger subset for better stats
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
        "pos_mean": torch.tensor([512.0, 512.0]), # Fallback if not used
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

    # --- Hyperparameters ---
    NUM_EPOCHS = 500  # Train longer for Transformer
    BATCH_SIZE = 256  # Larger batch size for stability
    LR = 2e-4         # Slightly higher LR for Transformer
    EMA_DECAY = 0.9999
    
    # --- Config & Models ---
    config_path = "params/learning_params.toml"
    if not os.path.exists(config_path):
        config_path = "../../params/learning_params.toml"

    try:
        config = IOUtils.load_toml(config_path)
    except Exception:
        print("Warning: Could not load config. Using default LPAC config.")
        config = {"CNNBackBone": {"InputDim": 4, "OutputDim": 32, "ImageSize": 32}}

    # 1. Models
    print("Loading LPAC (CNN Encoder)...")
    lpac = LPAC(config).to(device).train()
    
    # Transformer Diffusion Policy
    # Hidden Size 256, 4 Layers, 4 Heads
    net = TransformerDiffusionModel(
        input_dim=2, 
        global_cond_dim=32, 
        hidden_size=256, 
        num_layers=4, 
        num_heads=4,
        num_timesteps=100
    ).to(device)
    
    # 2. Data
    data_dir = "lpac/data/data/train"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found. Run generation script first.")
        return

    print(f"Loading data from {data_dir}...")
    dataset = CoverageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 3. Optimizer & Scheduler
    opt = torch.optim.AdamW(list(net.parameters()) + list(lpac.parameters()), lr=LR)
    
    # EMA Model
    ema_net = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=opt,
        num_warmup_steps=1000,
        num_training_steps=NUM_EPOCHS * len(loader)
    )

    # Improved Noise Scheduler (Cosine is better for images/spatial)
    sched = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

    # 4. Stats
    stats = compute_stats(loader)
    action_mean = stats['action_mean'].to(device)
    action_std = stats['action_std'].to(device)

    # 5. Training
    print("Starting Training...")
    save_dir = "checkpoints/madp_transformer"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(stats, os.path.join(save_dir, "normalization_stats.pt"))

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        net.train()
        lpac.train()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for batch in pbar:
            batch = batch.to(device)

            # --- Normalize Actions ---
            raw_action = batch.y
            norm_action = (raw_action - action_mean) / action_std

            # --- Prepare Maps ---
            map_data = batch.map if hasattr(batch, 'map') else batch.x_map
            map_data = map_data.float()

            # Fix Dimensions [B, 4, 32, 32]
            if map_data.dim() == 5:
                B, N, C, H, W = map_data.shape
                map_input = map_data.view(B*N, C, H, W)
            else:
                map_input = map_data

            # Padding Check
            if map_input.shape[1] < 4:
                pad = torch.zeros(map_input.shape[0], 4 - map_input.shape[1], 32, 32, device=device)
                map_input = torch.cat([map_input, pad], dim=1)
            else:
                map_input = map_input[:, :4, :, :]

            # --- Forward ---
            # 1. Encode Map
            map_feat = lpac.cnn_backbone(map_input) # [B, 32]
            
            # 2. Add Noise
            noise = torch.randn_like(norm_action)
            t = torch.randint(0, 100, (norm_action.shape[0],), device=device).long()
            noisy_act = sched.add_noise(norm_action, noise, t)
            
            # 3. Predict Noise (Transformer)
            noise_pred = net(noisy_act, t, map_feat)
            
            # 4. Loss
            loss = F.mse_loss(noise_pred, noise)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) # Gradient Clipping
            opt.step()
            ema_net.update_parameters(net)
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        
        # Save Checkpoint
        if (epoch+1) % 50 == 0:
            torch.save({
                'model': net.state_dict(),
                'ema_model': ema_net.module.state_dict(),
                'lpac': lpac.state_dict(),
                'stats': stats,
                'epoch': epoch+1,
                'loss': avg_loss
            }, f"{save_dir}/epoch_{epoch+1}.pth")  # Corrected syntax here
            print(f"Saved epoch {epoch+1}, Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()
