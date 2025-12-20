import os, glob, math, time, random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------- Key inference ----------
def _pick_key(files: List[str], d: np.lib.npyio.NpzFile, kind: str) -> str:
    """
    kind:
      - "maps": expects ndim in {4,5} (T,N,H,W) or (T,N,C,H,W)
      - "actions": expects shape (T,N,2)
    """
    # user preferred
    prefs = {
        "maps": ["maps", "map", "local_maps", "local", "obs", "states", "state", "x"],
        "actions": ["actions", "action", "u", "ctrl", "controls", "y"],
    }[kind]

    # 1) prefer explicit names
    for k in prefs:
        if k in files:
            return k

    # 2) infer by shape
    for k in files:
        a = d[k]
        if kind == "maps":
            if a.ndim in (4, 5):
                # must start with (T,N,...)
                if a.shape[0] > 0 and a.shape[1] > 0:
                    return k
        else:
            if a.ndim == 3 and a.shape[-1] == 2:
                return k

    raise KeyError(f"Could not infer '{kind}' key from: {files}")

def _load_npz(fp: str) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(fp)
    files = list(d.files)
    if not files:
        raise ValueError(f"{fp}: empty npz")

    mk = _pick_key(files, d, "maps")
    ak = _pick_key(files, d, "actions")

    maps = d[mk]
    acts = d[ak]
    return maps, acts

# ---------- Dataset ----------
class CoverageNPZDataset(Dataset):
    def __init__(
        self,
        npz_glob: str,
        max_files: Optional[int] = None,
        use_channels: Optional[List[int]] = None,
        action_scale: float = 1.0,
        map_norm: str = "per_sample",  # "none" | "per_sample"
        cache_meta_only: bool = True,
    ):
        self.files = sorted(glob.glob(npz_glob))
        if not self.files:
            raise FileNotFoundError(f"No files matched glob: {npz_glob}")
        if max_files:
            self.files = self.files[:max_files]

        self.use_channels = use_channels
        self.action_scale = float(action_scale)
        self.map_norm = map_norm
        self.cache_meta_only = cache_meta_only

        self.index = []  # (file_idx, t, n)
        self._meta = []

        for fi, fp in enumerate(self.files):
            maps, acts = _load_npz(fp)

            if maps.ndim not in (4, 5):
                raise ValueError(f"{fp}: maps ndim must be 4 or 5, got {maps.ndim}")
            if acts.ndim != 3 or acts.shape[-1] != 2:
                raise ValueError(f"{fp}: actions must be (T,N,2), got {acts.shape}")

            T, N = acts.shape[0], acts.shape[1]
            for t in range(T):
                for n in range(N):
                    self.index.append((fi, t, n))

            self._meta.append((fp, maps.shape, acts.shape))

    def __len__(self):
        return len(self.index)

    def _norm_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.map_norm == "none":
            return x
        if self.map_norm == "per_sample":
            out = []
            for c in range(x.shape[0]):
                xc = x[c]
                mn = torch.min(xc)
                mx = torch.max(xc)
                if (mx - mn) < 1e-8:
                    out.append(torch.zeros_like(xc))
                else:
                    out.append((xc - mn) / (mx - mn))
            return torch.stack(out, dim=0)
        raise ValueError(f"Unknown map_norm: {self.map_norm}")

    def __getitem__(self, idx: int):
        fi, t, n = self.index[idx]
        fp = self.files[fi]
        maps, acts = _load_npz(fp)

        if maps.ndim == 4:
            m = maps[t, n][None, ...]  # (1,H,W)
        else:
            m = maps[t, n]  # (C,H,W)
            if self.use_channels is not None:
                m = m[self.use_channels, ...]

        a = acts[t, n]  # (2,)

        m = torch.from_numpy(m).float()
        a = torch.from_numpy(a).float() * self.action_scale

        # optional log1p for channel0 if it's IDF-like
        # m[0] = torch.log1p(torch.clamp(m[0], min=0.0))

        m = self._norm_map(m)
        return m, a

# ---------- Diffusion ----------
@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

class GaussianDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.cfg = cfg
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sc = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        so = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sc * x0 + so * noise, noise

# ---------- Model ----------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class SmallMapEncoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 32, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),

            nn.Conv2d(base, base*2, 3, padding=1),
            nn.GroupNorm(8, base*2),
            nn.SiLU(),
            nn.Conv2d(base*2, base*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base*2),
            nn.SiLU(),

            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.GroupNorm(8, base*4),
            nn.SiLU(),
            nn.Conv2d(base*4, base*4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base*4),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base*4, out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.proj(self.net(x))

class ActionDenoiser(nn.Module):
    def __init__(self, map_in_ch: int, map_feat_dim: int = 256, time_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.map_enc = SmallMapEncoder(map_in_ch, base=32, out_dim=map_feat_dim)
        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, time_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 + map_feat_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, a_t: torch.Tensor, t: torch.Tensor, maps: torch.Tensor):
        mf = self.map_enc(maps)
        tf = self.t_emb(t)
        return self.mlp(torch.cat([a_t, mf, tf], dim=-1))

# ---------- Training ----------
@dataclass
class TrainConfig:
    data_glob: str = "data_cvt/*.npz"
    batch_size: int = 256
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    log_every: int = 200
    save_every_epoch: int = 1
    out_dir: str = "runs/diffusion_policy"
    max_files: Optional[int] = None
    map_norm: str = "per_sample"
    action_scale: float = 1.0
    use_channels: Optional[List[int]] = None

def ddpm_sample(model: ActionDenoiser, diff: GaussianDiffusion, maps: torch.Tensor, steps: int = 200):
    device = maps.device
    B = maps.shape[0]
    a = torch.randn(B, 2, device=device)

    betas = diff.betas
    alphas = diff.alphas
    acp = diff.alphas_cumprod

    for i in reversed(range(steps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        eps = model(a, t, maps)

        a_cum = acp[i]
        a_cum_prev = acp[i-1] if i > 0 else torch.tensor(1.0, device=device)
        beta = betas[i]
        alpha = alphas[i]

        x0 = (a - torch.sqrt(1 - a_cum) * eps) / torch.sqrt(a_cum)

        coef1 = torch.sqrt(a_cum_prev) * beta / (1 - a_cum)
        coef2 = torch.sqrt(alpha) * (1 - a_cum_prev) / (1 - a_cum)
        mean = coef1 * x0 + coef2 * a

        if i > 0:
            noise = torch.randn_like(a)
            var = beta * (1 - a_cum_prev) / (1 - a_cum)
            a = mean + torch.sqrt(var) * noise
        else:
            a = mean
    return a

def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tcfg = TrainConfig()
    dcfg = DiffusionConfig()
    os.makedirs(tcfg.out_dir, exist_ok=True)

    ds = CoverageNPZDataset(
        npz_glob=tcfg.data_glob,
        max_files=tcfg.max_files,
        use_channels=tcfg.use_channels,
        action_scale=tcfg.action_scale,
        map_norm=tcfg.map_norm,
    )

    m0, a0 = ds[0]
    C = m0.shape[0]
    print("dataset size:", len(ds))
    print("map shape:", tuple(m0.shape), "action shape:", tuple(a0.shape))
    print("first files meta sample:", ds._meta[0])

    dl = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True,
                    num_workers=tcfg.num_workers, pin_memory=True, drop_last=True)

    model = ActionDenoiser(map_in_ch=C).to(device)
    diff = GaussianDiffusion(dcfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    step = 0
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        t0 = time.time()
        for maps, a0 in dl:
            maps = maps.to(device, non_blocking=True)
            a0 = a0.to(device, non_blocking=True)

            B = a0.shape[0]
            t = torch.randint(0, dcfg.timesteps, (B,), device=device, dtype=torch.long)
            a_t, noise = diff.q_sample(a0, t)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                eps_pred = model(a_t, t, maps)
                loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % tcfg.log_every == 0:
                dt = time.time() - t0
                print(f"epoch {epoch:03d} step {step:06d} loss {loss.item():.6f} ({dt:.1f}s)")
                t0 = time.time()
            step += 1

        if epoch % tcfg.save_every_epoch == 0:
            path = os.path.join(tcfg.out_dir, f"ckpt_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "diff_cfg": dcfg.__dict__,
                "train_cfg": tcfg.__dict__,
            }, path)
            print("saved:", path)

        model.eval()
        with torch.no_grad():
            maps, _ = next(iter(dl))
            maps = maps.to(device)[:64]
            a_samp = ddpm_sample(model, diff, maps, steps=min(200, dcfg.timesteps))
            print("sample actions mean:", a_samp.mean(0).cpu().numpy(),
                  "std:", a_samp.std(0).cpu().numpy())

    print("done")

if __name__ == "__main__":
    main()
