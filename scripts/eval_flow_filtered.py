import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# --- FilterPy for Nonlinear Filtering ---
try:
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
except ImportError:
    print("Please install filterpy: pip install filterpy")
    sys.exit(1)

# --- Imports from your library ---
from coverage_control import CoverageSystem, IOUtils, WorldIDF, PointVector, Parameters

# Robust Import for CoverageEnvUtils
try:
    from coverage_control.nn import CoverageEnvUtils
except ImportError:
    try:
        from coverage_control.coverage_env_utils import CoverageEnvUtils
    except ImportError:
        import coverage_control.coverage_env_utils as CoverageEnvUtils

# --- Architecture Definition ---
import math

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
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.act = nn.Mish()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2) 
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        resid = self.residual_conv(x)
        h = self.gn1(self.conv1(x))
        style = self.cond_proj(cond).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        h = h * (1 + gamma) + beta
        h = self.act(h)
        h = self.gn2(self.conv2(h))
        return self.act(h + resid)

class TrajectoryFlowUnet1D(nn.Module):
    def __init__(self, input_dim=2, global_cond_dim=32, base_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.Mish(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )
        self.cond_dim = base_dim * 4 + global_cond_dim
        self.down1 = ResidualTemporalBlock(input_dim, base_dim, self.cond_dim)
        self.down2 = ResidualTemporalBlock(base_dim, base_dim*2, self.cond_dim)
        self.down3 = ResidualTemporalBlock(base_dim*2, base_dim*4, self.cond_dim)
        self.mid1 = ResidualTemporalBlock(base_dim*4, base_dim*4, self.cond_dim)
        self.mid2 = ResidualTemporalBlock(base_dim*4, base_dim*4, self.cond_dim)
        self.up3 = ResidualTemporalBlock(base_dim*8, base_dim*2, self.cond_dim)
        self.up2 = ResidualTemporalBlock(base_dim*4, base_dim, self.cond_dim)
        self.up1 = ResidualTemporalBlock(base_dim*2, base_dim, self.cond_dim)
        self.final_conv = nn.Conv1d(base_dim, input_dim, 1)
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, global_cond):
        x = x.transpose(1, 2)
        t_emb = self.time_mlp(t)
        cond = torch.cat([t_emb, global_cond], dim=-1)
        h1 = self.down1(x, cond)
        h2 = self.down2(self.pool(h1), cond)
        h3 = self.down3(self.pool(h2), cond)
        mid = self.mid2(self.mid1(h3, cond), cond)
        h_up3 = self.up3(torch.cat([mid, h3], dim=1), cond)
        h_up2 = self.up2(torch.cat([self.upsample(h_up3), h2], dim=1), cond)
        h_up1 = self.up1(torch.cat([self.upsample(h_up2), h1], dim=1), cond)
        out = self.final_conv(h_up1)
        return out.transpose(1, 2)

# -----------------------------------------------------------------------------
# 1. Nonlinear Filter Implementation (UKF)
# -----------------------------------------------------------------------------
class AgentUKF:
    def __init__(self, dt=0.1):
        self.dim_x = 4
        self.dt = dt
        points = MerweScaledSigmaPoints(self.dim_x, alpha=0.1, beta=2., kappa=-1)
        self.ukf = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=2, dt=dt, fx=self.fx, hx=self.hx, points=points)
        self.ukf.P *= 1.0
        self.ukf.Q = np.eye(4) * 0.01 
        self.ukf.R = np.eye(2) * 0.1 

    def fx(self, x, dt):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F @ x

    def hx(self, x):
        return x[2:] 

    def init_state(self, pos, vel):
        self.ukf.x = np.array([pos[0], pos[1], vel[0], vel[1]])

    def filter(self, diffusion_vel):
        self.ukf.predict()
        self.ukf.update(diffusion_vel)
        return self.ukf.x[2:]

# -----------------------------------------------------------------------------
# 2. Flow Matching Controller with Filter
# -----------------------------------------------------------------------------
class ControllerFlowFiltered:
    def __init__(self, cfg, cc_params, env):
        self.config = cfg
        self.cc_params = cc_params
        self.name = cfg.get("Name", "FlowFiltered")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- Load Model ---
        chk_path = cfg.get("ModelStateDict", "checkpoints_flow_unet/epoch_200.pth")
        if not os.path.exists(chk_path):
             chk_path = "checkpoints_flow_unet/epoch_200.pth"
        
        print(f"Loading Checkpoint: {chk_path}")
        checkpoint = torch.load(chk_path, map_location=self.device)
        
        # Stats
        stats = checkpoint['stats']
        self.act_mean = stats['action_mean'].to(self.device)
        self.act_std = stats['action_std'].to(self.device)
        
        # Perception (LPAC)
        from coverage_control.nn import LPAC
        
        lpac_cfg = {
            "ModelConfig": {
                "UseCommMaps": True
            },
            "CNNBackBone": {
                "InputDim": 4, 
                "NumLayers": 3,
                "LatentSize": 32,
                "KernelSize": 3,
                "ImageSize": 32,
                "OutputDim": 7
            },
            "GNNBackBone": {
                "InputDim": 7,
                "NumHops": 3,
                "NumLayers": 5,
                "LatentSize": 256,
                "OutputDim": 2
            }
        }
        
        self.raw_lpac = LPAC(lpac_cfg).to(self.device)
        self.raw_lpac.load_state_dict(checkpoint['lpac'])
        self.cnn_backbone = self.raw_lpac.cnn_backbone
        
        # Flow Model
        self.model = TrajectoryFlowUnet1D(input_dim=2, global_cond_dim=32, base_dim=64).to(self.device)
        if 'ema_model' in checkpoint:
            self.model.load_state_dict(checkpoint['ema_model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # Configs
        self.pred_horizon = 16
        self.map_size = 32
        
        # --- Initialize Filters ---
        self.filters = None

    def step(self, env):
        with torch.no_grad():
            # 1. Environment Data
            robot_positions = env.GetRobotPositions() 
            
            if hasattr(robot_positions, 'to_numpy'): 
                pos_np = robot_positions.to_numpy()
            elif isinstance(robot_positions, list):
                data = []
                for p in robot_positions:
                    if hasattr(p, 'x'): data.append([p.x, p.y])
                    else: data.append([p[0], p[1]])
                pos_np = np.array(data)
            else:
                pos_np = np.array(robot_positions)

            N = len(pos_np)
            
            # Initialize Filters if needed
            if self.filters is None:
                self.filters = [AgentUKF(dt=0.1) for _ in range(N)]
                for i in range(N):
                    self.filters[i].init_state(pos_np[i], np.zeros(2))

            # 2. Prepare Maps (Using self.cc_params)
            local_maps = CoverageEnvUtils.get_raw_local_maps(env, self.cc_params)
            obst_maps = CoverageEnvUtils.get_raw_obstacle_maps(env, self.cc_params)
            comm_maps = CoverageEnvUtils.get_communication_maps(env, self.cc_params, self.map_size)
            
            # FIX: Ensure 4D shape [Batch, Channels, H, W] for interpolation
            if local_maps.dim() == 3: local_maps = local_maps.unsqueeze(1)
            if obst_maps.dim() == 3: obst_maps = obst_maps.unsqueeze(1)
            if comm_maps.dim() == 3: comm_maps = comm_maps.unsqueeze(1)
            
            # If 5D (Batch, 1, Channels, H, W) -> Squeeze to 4D
            if local_maps.dim() == 5: local_maps = local_maps.squeeze(1)
            if obst_maps.dim() == 5: obst_maps = obst_maps.squeeze(1)
            if comm_maps.dim() == 5: comm_maps = comm_maps.squeeze(1)
            
            # Obstacle map expansion if shared
            if obst_maps.shape[0] == 1: obst_maps = obst_maps.expand(N, -1, -1, -1)

            # Resize
            if local_maps.shape[-1] != 32:
                local_maps = F.interpolate(local_maps, size=32, mode='bilinear')
                obst_maps = F.interpolate(obst_maps, size=32, mode='bilinear')
                comm_maps = F.interpolate(comm_maps, size=32, mode='bilinear')
            
            # Model Input
            full_input = torch.cat([local_maps, obst_maps, comm_maps], dim=1).float().to(self.device)
            map_feat = self.cnn_backbone(full_input)

            # 3. Flow Matching Inference (10 Steps)
            x = torch.randn(N, self.pred_horizon, 2, device=self.device)
            inference_steps = 10
            dt = 1.0 / inference_steps
            
            for i in range(inference_steps):
                t_curr = 1.0 - (i / inference_steps)
                t_tensor = torch.full((N,), t_curr, device=self.device)
                pred_v = self.model(x, t_tensor, map_feat)
                x = x - pred_v * dt
            
            # 4. Un-normalize
            actions_seq = x * self.act_std + self.act_mean
            
            # Take immediate next action
            raw_actions = actions_seq[:, 0, :].cpu().numpy() 
            
            # 5. Apply Nonlinear Filter (Fusion)
            smoothed_actions = []
            for i in range(N):
                filt_vel = self.filters[i].filter(raw_actions[i])
                filt_vel = np.clip(filt_vel, -2.0, 2.0)
                smoothed_actions.append(filt_vel)
            
            smoothed_actions = np.array(smoothed_actions)
            
            # 6. Step Env
            pv_actions = PointVector(smoothed_actions.astype(np.float64))
            if hasattr(env, 'StepActions'):
                env.StepActions(pv_actions)
            else:
                env.Step(pv_actions)
                
            return env.GetObjectiveValue(), False

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_flow_filtered.py <config_file>")
        config_file = "params/eval_single.toml" 
        if not os.path.exists(config_file):
             print(f"File not found: {config_file}")
             sys.exit(1)
    else:
        config_file = sys.argv[1]

    print(f"Loading Config: {config_file}")
    config = IOUtils.load_toml(config_file)
    
    # Setup Params
    cc_params = Parameters(IOUtils.sanitize_path(config["CCParams"]))
    
    # Setup Env
    env = CoverageSystem(cc_params)
    
    # Setup Controller
    ctrl_cfg = {
        "Name": "FlowFiltered",
        "ModelStateDict": "checkpoints_flow_unet/epoch_50.pth"
    }
    
    controller = ControllerFlowFiltered(ctrl_cfg, cc_params, env)
    
    print("Starting Filtered Evaluation...")
    
    # --- Normalization Logic ---
    initial_cost = None
    
    for step in range(600):
        obj, _ = controller.step(env)
        
        # Capture initial cost at step 0
        if step == 0:
            initial_cost = obj
            
        # Normalize: Current / Initial
        norm_cost = obj / initial_cost
        
        if step % 50 == 0:
            print(f"Step {step}: Normalized Cost = {norm_cost:.4f} (Raw: {obj:.2f})")
            
    print("Done!")
