import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

# --- Imports from your existing library ---
from coverage_control import IOUtils
from coverage_control.nn import LPAC

# =========================
# DATASET DEFINITION (Added here since it's not in coverage_control.data)
# =========================
class CoverageDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {data_dir}")

    def len(self):
        return len(self.files)

    def get(self, idx):
        return torch.load(self.files[idx], map_location="cpu", weights_only=False)

# -----------------------------------------------------------------------------
# 1. SCIE-Worthy Architecture: Factorized Spatiotemporal U-Net (1D CNN)
# -----------------------------------------------------------------------------

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

class ResidualTemporalBlock(nn.Module):
    """
    A 1D Convolutional Residual Block that fuses temporal features 
    with global spatial conditioning.
    """
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.act = nn.Mish()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        # Condition projection layer
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2) 
        
        # Residual connection adapter if channels change
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        # x: (B, C, T)
        # cond: (B, cond_dim)
        
        # 1. First Conv Block
        resid = self.residual_conv(x)
        h = self.conv1(x)
        h = self.gn1(h)
        
        # 2. Inject Conditioning (Scale and Shift)
        # Project condition to (B, C*2) -> split to scale, shift
        style = self.cond_proj(cond).unsqueeze(-1) # (B, 2*C, 1)
        gamma, beta = style.chunk(2, dim=1)
        
        # Apply FiLM (Feature-wise Linear Modulation) or simple add
        h = h * (1 + gamma) + beta
        h = self.act(h)
        
        # 3. Second Conv Block
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h + resid)
        
        return h

class TrajectoryFlowUnet1D(nn.Module):
    """
    The Temporal 1D U-Net backbone.
    Replaces the MLP to handle trajectory temporal consistency efficiently.
    """
    def __init__(self, input_dim=2, global_cond_dim=32, base_dim=64):
        super().__init__()
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.Mish(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )
        
        # Condition embedding (Map features + Time)
        self.cond_dim = base_dim * 4 + global_cond_dim
        
        # Encoder (Downsampling)
        self.down1 = ResidualTemporalBlock(input_dim, base_dim, self.cond_dim)
        self.down2 = ResidualTemporalBlock(base_dim, base_dim*2, self.cond_dim)
        self.down3 = ResidualTemporalBlock(base_dim*2, base_dim*4, self.cond_dim)
        
        # Bottleneck
        self.mid1 = ResidualTemporalBlock(base_dim*4, base_dim*4, self.cond_dim)
        self.mid2 = ResidualTemporalBlock(base_dim*4, base_dim*4, self.cond_dim)
        
        # Decoder (Upsampling)
        self.up3 = ResidualTemporalBlock(base_dim*4 + base_dim*4, base_dim*2, self.cond_dim)
        self.up2 = ResidualTemporalBlock(base_dim*2 + base_dim*2, base_dim, self.cond_dim)
        self.up1 = ResidualTemporalBlock(base_dim + base_dim, base_dim, self.cond_dim)
        
        # Output
        self.final_conv = nn.Conv1d(base_dim, input_dim, 1)
        
        # Pooling for U-Net
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, global_cond):
        """
        x: (B, T, 2) -> Trajectory
        t: (B,) -> Timesteps (0 to 1 float, or embeddings)
        global_cond: (B, 32) -> Map features
        """
        # 1. Prepare Inputs
        # Permute for Conv1d: (B, T, 2) -> (B, 2, T)
        x = x.transpose(1, 2)
        
        # 2. Prepare Conditioning
        # Time Embedding
        t_emb = self.time_mlp(t) # (B, base*4)
        # Concatenate with Map Features
        cond = torch.cat([t_emb, global_cond], dim=-1) # (B, cond_dim)
        
        # 3. Encoder
        h1 = self.down1(x, cond)        # (B, 64, 16)
        
        h2 = self.down2(self.pool(h1), cond) # (B, 128, 8)
        
        h3 = self.down3(self.pool(h2), cond) # (B, 256, 4)
        
        # 4. Bottleneck
        mid = self.mid1(h3, cond)
        mid = self.mid2(mid, cond)      # (B, 256, 4)
        
        # 5. Decoder
        # Concatenate skip connections
        x_up3 = torch.cat([mid, h3], dim=1) 
        h_up3 = self.up3(x_up3, cond)   # (B, 128, 4)
        
        h_up2 = self.up2(torch.cat([self.upsample(h_up3), h2], dim=1), cond) # (B, 64, 8)
        
        h_up1 = self.up1(torch.cat([self.upsample(h_up2), h1], dim=1), cond) # (B, 64, 16)
        
        # 6. Final Projection
        out = self.final_conv(h_up1)    # (B, 2, 16)
        
        # Permute back: (B, 2, 16) -> (B, 16, 2)
        return out.transpose(1, 2)

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------

def compute_action_stats(loader, action_dim=2):
    print("Computing action normalization stats...")
    actions = []
    seen = 0
    max_samples = 10000
    for batch in tqdm(loader, desc="Scanning dataset"):
        if seen >= max_samples: break
        y = batch.y # (B, T, 2)
        flat = y.reshape(-1, action_dim)
        actions.append(flat)
        seen += y.size(0)
        
    all_actions = torch.cat(actions, dim=0)
    mean = all_actions.mean(dim=0)
    std = all_actions.std(dim=0).clamp(min=1e-6)
    return mean, std

def augment_batch(map_data, action_seq):
    # Same augmentation as your original code
    B = map_data.size(0)
    k = torch.randint(0, 4, (1,)).item()
    
    if k == 0: return map_data, action_seq
    
    # Rotate Map
    map_aug = torch.rot90(map_data, k, [2, 3])
    
    # Rotate Actions
    x, y = action_seq[..., 0], action_seq[..., 1]
    if k == 1: action_aug = torch.stack([-y, x], dim=-1) # 90
    elif k == 2: action_aug = torch.stack([-x, -y], dim=-1) # 180
    elif k == 3: action_aug = torch.stack([y, -x], dim=-1) # 270
    
    return map_aug, action_aug

# -----------------------------------------------------------------------------
# 3. Main Training Loop (Flow Matching)
# -----------------------------------------------------------------------------

def main():
    # Hyperparameters
    PRED_HORIZON = 16
    OUTPUT_DIM = 2
    NUM_EPOCHS = 1000
    BATCH_SIZE = 256
    LR = 1e-4
    EMA_DECAY = 0.999
    DATA_DIR = "lpac/data/data/train" # Adjust to your path
    SAVE_DIR = "checkpoints_flow_unet"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Config & LPAC ---
    try:
        config = IOUtils.load_toml("params/learning_params.toml")
    except:
        print("Config not found, using defaults.")
        config = {}
        
    lpac = LPAC(config).to(device).train()

    # --- Flow Model ---
    model = TrajectoryFlowUnet1D(
        input_dim=OUTPUT_DIM,
        global_cond_dim=32, # Output of LPAC backbone
        base_dim=64
    ).to(device)
    
    # EMA Model for smoother inference
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))

    # --- Data ---
    dataset = CoverageDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # --- Stats ---
    # Check if stats exist, else compute
    stats_path = os.path.join(SAVE_DIR, "normalization_stats.pt")
    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
    else:
        act_mean, act_std = compute_action_stats(loader)
        stats = {'action_mean': act_mean, 'action_std': act_std}
        torch.save(stats, stats_path)
    
    act_mean = stats['action_mean'].to(device)
    act_std = stats['action_std'].to(device)

    # --- Optim ---
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(lpac.parameters()), lr=LR, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print("Training Started (Flow Matching)...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        lpac.train()
        total_loss = 0.0
        pbar = tqdm(loader, leave=False, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            batch = batch.to(device)
            
            # 1. Process Map
            map_data = batch.map if hasattr(batch, 'map') else batch.x_map
            if map_data.dim() == 5: map_data = map_data.view(-1, *map_data.shape[2:]) # Flatten agents
            if map_data.size(1) < 4: # Pad if needed
                pad = torch.zeros(map_data.size(0), 4 - map_data.size(1), map_data.size(2), map_data.size(3), device=device)
                map_data = torch.cat([map_data, pad], dim=1)
                
            # 2. Augment
            map_data, action_seq = augment_batch(map_data, batch.y)
            
            # 3. Normalize Actions (Data x0)
            # action_seq: (B, 16, 2)
            B, T, C = action_seq.shape
            x0 = (action_seq.reshape(-1, C) - act_mean) / act_std
            x0 = x0.reshape(B, T, C) # Back to sequence
            
            # 4. Get Conditioning
            map_feat = lpac.cnn_backbone(map_data) # (B, 32)
            
            # ====================================================
            # FLOW MATCHING TRAINING LOGIC
            # ====================================================
            
            # Sample Noise (x1)
            x1 = torch.randn_like(x0)
            
            # Sample Time t ~ Uniform(0, 1)
            t = torch.rand(x0.shape[0], device=device)
            
            # Interpolate: x_t = (1 - t) * x0 + t * x1
            # Note: Broadcase t for correct shape
            t_reshaped = t.view(-1, 1, 1)
            x_t = (1 - t_reshaped) * x0 + t_reshaped * x1
            
            # Target Velocity: v = x1 - x0
            target_v = x1 - x0
            
            # Predict Velocity
            # Pass t directly (model handles embedding)
            pred_v = model(x_t, t, map_feat)
            
            # Loss: MSE(pred, target)
            loss = F.mse_loss(pred_v, target_v)
            
            # ====================================================
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema_model.update_parameters(model)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        lr_scheduler.step()
        avg_loss = total_loss / len(loader)
        
        if (epoch + 1) % 50 == 0:
            save_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth")
            torch.save({
                'model': model.state_dict(),
                'ema_model': ema_model.module.state_dict(),
                'lpac': lpac.state_dict(),
                'stats': stats,
                'epoch': epoch,
            }, save_path)
            print(f"Saved {save_path} | Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    main()
