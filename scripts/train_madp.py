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
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from coverage_control import IOUtils
from coverage_control.nn import LPAC

# =========================
# CONFIG
# =========================
PRED_HORIZON = 16
ACTION_DIM = 2
OUTPUT_DIM = PRED_HORIZON * ACTION_DIM  # 32

NUM_DIFFUSION_STEPS = 100
NUM_EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-4
CFG_DROPOUT = 0.1  # Probability of dropping the condition (Map)
EMA_DECAY = 0.9999

DATA_DIR = "lpac/data/data/train"
SAVE_DIR = "checkpoints/madp_traj"

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# DATASET
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


# =========================
# MODEL
# =========================
class TrajectoryDiffusionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        num_train_timesteps: int = NUM_DIFFUSION_STEPS,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        
        # Learnable "null" embedding for Classifier-Free Guidance
        self.null_cond = nn.Parameter(torch.randn(1, global_cond_dim))

        self.time_embed = nn.Embedding(num_train_timesteps, diffusion_step_embed_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + global_cond_dim + diffusion_step_embed_dim, 1024),
            nn.Mish(),
            nn.Linear(1024, 1024),
            nn.Mish(),
            nn.Linear(1024, 1024),
            nn.Mish(),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x, t, global_cond):
        # Classifier-Free Guidance: Randomly drop condition during training
        if self.training and self.dropout_prob > 0:
            # Create mask: 1 = keep, 0 = drop
            mask = torch.bernoulli(torch.full((x.shape[0], 1), 1 - self.dropout_prob, device=x.device))
            # Replace dropped samples with null_cond
            global_cond = mask * global_cond + (1 - mask) * self.null_cond
            
        t_emb = self.time_encoder(self.time_embed(t))
        h = torch.cat([x, global_cond, t_emb], dim=-1)
        return self.net(h)


# =========================
# NORMALIZATION STATS
# =========================
@torch.no_grad()
def compute_action_stats(loader):
    print("▶ Computing action normalization stats (trajectory-aware)")

    actions = []
    max_samples = 10000
    seen = 0

    for batch in tqdm(loader, desc="Scanning dataset"):
        assert batch.y.dim() == 3, f"Expected [B, T, 2], got {batch.y.shape}"

        # [B, 16, 2] → [B*16, 2]
        flat = batch.y.reshape(-1, ACTION_DIM)
        actions.append(flat)

        seen += batch.y.size(0)
        if seen >= max_samples:
            break

    actions = torch.cat(actions, dim=0)

    mean = actions.mean(dim=0)               # [2]
    std = actions.std(dim=0).clamp_min(1e-6) # [2]

    print(f"✔ Action mean: {mean.tolist()}")
    print(f"✔ Action std : {std.tolist()}")

    return {
        "action_mean": mean,
        "action_std": std,
    }


# =========================
# AUGMENTATION
# =========================
def augment_batch(map_data, action_seq):
    """
    Randomly rotates the batch by 0, 90, 180, or 270 degrees.
    map_data: [B, C, H, W]
    action_seq: [B, T, 2] (Vectors)
    """
    k = torch.randint(0, 4, (1,)).item()
    
    if k == 0:
        return map_data, action_seq
        
    # Rotate Map (CCW)
    map_aug = torch.rot90(map_data, k, [2, 3])
    
    # Rotate Action Vectors (CCW)
    x, y = action_seq[..., 0], action_seq[..., 1]
    if k == 1:   action_aug = torch.stack([-y, x], dim=-1) # 90 deg
    elif k == 2: action_aug = torch.stack([-x, -y], dim=-1) # 180 deg
    elif k == 3: action_aug = torch.stack([y, -x], dim=-1) # 270 deg
    
    return map_aug, action_aug


# =========================
# TRAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    # --- LPAC ---
    try:
        config = IOUtils.load_toml("params/learning_params.toml")
        print("✔ Loaded config from params/learning_params.toml")
    except Exception:
        print("⚠ Config not found, using default LPAC config.")
        config = {"CNNBackBone": {"InputDim": 4, "OutputDim": 32, "ImageSize": 32}}

    lpac = LPAC(config).to(device).train()

    # --- Diffusion Model ---
    model = TrajectoryDiffusionMLP(
        input_dim=OUTPUT_DIM,
        global_cond_dim=32,
        dropout_prob=CFG_DROPOUT,
    ).to(device)

    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY)
    ).to(device)

    # --- Data ---
    dataset = CoverageDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # --- Optim ---
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(lpac.parameters()),
        lr=LR,
    )

    lr_sched = get_scheduler(
        "cosine",
        optimizer=opt,
        num_warmup_steps=500,
        num_training_steps=NUM_EPOCHS * len(loader),
    )

    noise_sched = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_STEPS,
        beta_schedule="squaredcos_cap_v2",
    )

    # --- Stats ---
    stats = compute_action_stats(loader)

    act_mean = stats["action_mean"].to(device).repeat(PRED_HORIZON)  # [32]
    act_std = stats["action_std"].to(device).repeat(PRED_HORIZON)    # [32]

    torch.save(stats, os.path.join(SAVE_DIR, "normalization_stats.pt"))

    print("▶ Training started")

    # =========================
    # LOOP
    # =========================
    for epoch in range(NUM_EPOCHS):
        model.train()
        lpac.train()

        total_loss = 0.0
        pbar = tqdm(loader, leave=False)

        for batch in pbar:
            batch = batch.to(device)

            # ---------- MAP ----------
            map_data = batch.map if hasattr(batch, "map") else batch.x_map
            map_data = map_data.float()

            if map_data.dim() == 5:
                B, N, C, H, W = map_data.shape
                map_data = map_data.view(B * N, C, H, W)

            if map_data.size(1) < 4:
                pad = torch.zeros(
                    map_data.size(0),
                    4 - map_data.size(1),
                    map_data.size(2),
                    map_data.size(3),
                    device=device,
                )
                map_data = torch.cat([map_data, pad], dim=1)
            else:
                map_data = map_data[:, :4]

            # ---------- AUGMENTATION ----------
            # Rotate map and actions together to increase data diversity
            map_data, raw_action_seq = augment_batch(map_data, batch.y)

            # ---------- ACTION NORM ----------
            assert raw_action_seq.dim() == 3
            # [B, 16, 2] → [B, 32]
            raw_action_flat = raw_action_seq.reshape(raw_action_seq.size(0), -1)
            
            norm_action = (raw_action_flat - act_mean) / act_std

            # ---------- ENCODE MAP ----------
            map_feat = lpac.cnn_backbone(map_data)

            # ---------- DIFFUSION ----------
            noise = torch.randn_like(norm_action)
            t = torch.randint(
                0, NUM_DIFFUSION_STEPS,
                (norm_action.size(0),),
                device=device,
            ).long()

            noisy_action = noise_sched.add_noise(norm_action, noise, t)
            noise_pred = model(noisy_action, t, map_feat)

            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_model.update_parameters(model)
            lr_sched.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}")
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 50 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema_model.module.state_dict(),
                    "lpac": lpac.state_dict(),
                    "stats": stats,
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                },
                f"{SAVE_DIR}/epoch_{epoch+1}.pth",
            )
            print(f"✔ Saved epoch {epoch+1} | loss={avg_loss:.6f}")


if __name__ == "__main__":
    main()
