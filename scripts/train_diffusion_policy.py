# scripts/train_diffusion_policy.py
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import coverage_control as cc
from coverage_control import IOUtils

try:
    from coverage_control.coverage_env_utils import CNNGNNDataset, LPAC
except ImportError:
    from coverage_control.nn import CNNGNNDataset, LPAC

# --- 1. EMA Helper (Essential for Stable Diffusion Training) ---
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]

# --- 2. Diffusion Scheduler ---
class DDPMScheduler(nn.Module):
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, t):
        if self.sqrt_alphas_cumprod.device != x_start.device:
            dev = x_start.device
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(dev)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(dev)

        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

# --- 3. Network (Bigger Capacity) ---
class NoisePredNet(nn.Module):
    def __init__(self, action_dim, condition_dim, hidden_dim=512):
        super().__init__()
        # Increased depth and width for better learning
        self.net = nn.Sequential(
            nn.Linear(action_dim + condition_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, noisy_action, condition, t):
        t_norm = t.view(-1, 1).float() / 100.0
        x = torch.cat([noisy_action, condition, t_norm], dim=1)
        return self.net(x)

# --- 4. Wrapper ---
class DiffusionPolicy(nn.Module):
    def __init__(self, lpac_model, action_dim=2, hidden_dim=512, num_timesteps=100):
        super().__init__()
        self.lpac_encoder = lpac_model
        self.scheduler = DDPMScheduler(num_timesteps=num_timesteps)
        self.noise_net = NoisePredNet(action_dim, 2, hidden_dim)

    def forward(self, pyg_data):
        with torch.no_grad():
            condition = self.lpac_encoder(pyg_data)
        return condition

# --- 5. Main ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_diffusion_policy.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = IOUtils.load_toml(config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    dataset_path = config["DataDir"] # Make sure this points to your new 'lpac' folder
    data_dir = os.path.join(dataset_path, "data")
    save_dir = config["LPACModel"]["Dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading Data from {data_dir}...")
    # NOTE: Ensure you ran the generation script!
    train_dataset = CNNGNNDataset(data_dir, "train", True, 1024)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # 2. Setup Models
    print("Setting up Diffusion Model...")
    lpac_model = LPAC(config).to(device)
    diffusion_steps = 100
    model = DiffusionPolicy(lpac_model, num_timesteps=diffusion_steps, hidden_dim=512).to(device)
    
    # --- CRITICAL: Normalization Stats ---
    print("Calculating Dataset Statistics...")
    # If dataset is large, we compute on a subset to save time
    all_actions = []
    num_samples = 0
    max_samples = 10000 
    
    for batch in tqdm(train_loader, desc="Scanning"):
        if isinstance(batch, list): y = batch[1]
        else: y = batch.y
        if y.dim() > 2: y = y.view(-1, 2)
        
        all_actions.append(y)
        num_samples += y.shape[0]
        if num_samples > max_samples: break
    
    all_actions = torch.cat(all_actions, dim=0)
    mean = all_actions.mean(dim=0).to(device)
    std = all_actions.std(dim=0).to(device)
    
    # Avoid div by zero
    std = torch.clamp(std, min=1e-5)

    print(f"Stats -> Mean: {mean.cpu().numpy()}, Std: {std.cpu().numpy()}")
    model.lpac_encoder.register_buffer("actions_mean", mean)
    model.lpac_encoder.register_buffer("actions_std", std)
    
    # Initialize EMA
    ema = EMA(model)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # 3. Training Loop (Aggressive)
    num_epochs = 100 # High epoch count for convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"Starting training for {num_epochs} epochs...")
    loss_history = []
    
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch in pbar:
            if isinstance(batch, list):
                batch_data = batch[0].to(device)
                x_start = batch[1].to(device)
            else:
                batch_data = batch.to(device)
                x_start = batch_data.y

            # Flatten
            if x_start.dim() > 2:
                x_start = x_start.view(-1, x_start.shape[-1])
            
            # 1. Normalize Action (Training on Normalized Data is easier!)
            x_norm = (x_start - mean) / std

            # 2. Condition
            cond = model.lpac_encoder(batch_data)
            
            # 3. Diffusion Loss
            B = x_start.shape[0]
            t = torch.randint(0, diffusion_steps, (B,), device=device).long()
            
            x_noisy, noise = model.scheduler.add_noise(x_norm, t)
            noise_pred = model.noise_net(x_noisy, cond, t)
            
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            ema.update(model)
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

        # Save Checkpoint
        if (epoch+1) % 50 == 0:
            # Save EMA weights (better for inference)
            ema.apply_shadow(model)
            torch.save(model.state_dict(), os.path.join(save_dir, f"diffusion_{epoch+1}.pt"))
            ema.restore(model) # Restore for continued training

    # Final Plot
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title("Training Loss")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    print("Training Complete.")

if __name__ == "__main__":
    main()

