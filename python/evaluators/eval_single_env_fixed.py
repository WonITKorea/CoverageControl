import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
try:
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
except ImportError:
    pass # FilterPy is required for FlowFiltered but optional for others

import coverage_control as cc
from coverage_control import CoverageSystem
from coverage_control import IOUtils
from coverage_control import WorldIDF
from coverage_control import PointVector
from coverage_control.algorithms import ControllerCVT
from coverage_control.algorithms import ControllerNN

# -----------------------------
# Robust Imports
# -----------------------------
try:
    from coverage_control.nn import CoverageEnvUtils
except ImportError:
    try:
        from coverage_control.coverage_env_utils import CoverageEnvUtils
    except ImportError:
        import coverage_control.coverage_env_utils as CoverageEnvUtils

def _add_scripts_to_syspath():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "../../.."))
    scripts_dir = os.path.join(project_root, "scripts")
    if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
        sys.path.append(scripts_dir)

_add_scripts_to_syspath()

# ==============================================================================
# MODEL DEFINITION (Must Match Training)
# ==============================================================================
PRED_HORIZON = 16

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, num_train_timesteps=100, hidden_dim=1024):
        super().__init__()
        self.null_cond = nn.Parameter(torch.randn(1, global_cond_dim))
        self.time_embed = nn.Embedding(num_train_timesteps, diffusion_step_embed_dim)

        self.diffusion_step_encoder = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )

        # Wider MLP for Trajectory
        self.model = nn.Sequential(
            nn.Linear(input_dim + global_cond_dim + diffusion_step_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, sample, timestep, global_cond):
        t_emb = self.diffusion_step_encoder(self.time_embed(timestep))
        x = torch.cat([sample, global_cond, t_emb], dim=-1)
        return self.model(x)

# -----------------------------
# LPAC Wrapper
# -----------------------------
class LPACWrapper(nn.Module):
    def __init__(self, original_lpac):
        super().__init__()
        self.original_lpac = original_lpac # Save reference!
        self.cnn_backbone = original_lpac.cnn_backbone

    def forward(self, x):
        return self.cnn_backbone(x)

# -----------------------------
# Controller Logic
# -----------------------------
# -----------------------------
# Controller Logic (DDPM Version)
# -----------------------------
# -----------------------------
# DDPF Controller (Particle Filter)
# -----------------------------
class ControllerDDPF:
    def __init__(self, cfg, cc_params, env):
        self.config = cfg
        self.name = cfg.get("Name", "DDPF")
        self.cc_params = cc_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_count = 0

        # --- Load Model & Config (Same as before) ---
        learning_params_file = IOUtils.sanitize_path(cfg["LearningParams"])
        self.learning_config = IOUtils.load_toml(learning_params_file)

        try: from coverage_control.nn import LPAC
        except: from coverage_control.coverage_env_utils import LPAC
        
        raw_lpac = LPAC(self.learning_config).to(self.device)
        self.lpac_wrapper = LPACWrapper(raw_lpac).to(self.device)

        model_path = IOUtils.sanitize_path(cfg["ModelStateDict"])
        print(f"Loading Checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # --- Stats ---
        self.actions_mean = torch.zeros(1, 2, device=self.device)
        self.actions_std = torch.ones(1, 2, device=self.device)

        # --- Load State Dict ---
        raw_state_dict = checkpoint['ema_model'] if 'ema_model' in checkpoint else checkpoint['model']
        state_dict = {}
        for k, v in raw_state_dict.items():
            new_k = k
            if k.startswith("module."): new_k = k[7:]
            if new_k.startswith("time_encoder."): new_k = new_k.replace("time_encoder.", "diffusion_step_encoder.")
            if new_k.startswith("net."): new_k = new_k.replace("net.", "model.")
            state_dict[new_k] = v

        if 'model.0.weight' in state_dict:
            w0 = state_dict['model.0.weight']
            hidden_dim = w0.shape[0]
            in_feat_total = w0.shape[1]
            input_dim = state_dict['model.6.weight'].shape[0] if 'model.6.weight' in state_dict else 32
            diffusion_step_embed_dim = 256
            global_cond_dim = in_feat_total - input_dim - diffusion_step_embed_dim
        else:
            input_dim = 32; global_cond_dim = 32; hidden_dim = 1024

        self.pred_horizon = input_dim // 2
        self.train_timesteps = cfg.get("TrainTimesteps", 100)

        # FIXED: Pass arguments by name to avoid order mismatch
        self.model = ConditionalUnet1D(
            input_dim=input_dim, 
            global_cond_dim=global_cond_dim, 
            diffusion_step_embed_dim=256,       # Fixed dim from checkpoint
            num_train_timesteps=self.train_timesteps, 
            hidden_dim=hidden_dim
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        if 'lpac' in checkpoint: self.lpac_wrapper.original_lpac.load_state_dict(checkpoint['lpac'])
        
        # --- DDPF Specific Params ---
        self.num_particles = cfg.get("NumParticles", 16) # K=16 particles per agent
        self.resample_freq = cfg.get("ResampleFreq", 2)  # Resample every 2 steps
        self.softmax_temp = cfg.get("SoftmaxTemp", 1.0)
        
        from diffusers import DDPMScheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=self.train_timesteps, beta_schedule="squaredcos_cap_v2")
        self.scheduler.set_timesteps(self.train_timesteps) # Use full steps for high quality

        if 'stats' in checkpoint:
            stats = checkpoint['stats']
            self.actions_mean.copy_(stats['action_mean'].to(self.device))
            self.actions_std.copy_(stats['action_std'].to(self.device))
            self.act_mean_seq = self.actions_mean.repeat(1, self.pred_horizon)
            self.act_std_seq = self.actions_std.repeat(1, self.pred_horizon)

        self.model.eval()
        self.use_comm_map = cfg.get("UseCommMap", True)
        self.map_size = cfg.get("CNNMapSize", 32)

    def compute_particle_cost(self, pred_x0_flat, N):
        """
        Compute cost for N agents * K particles.
        pred_x0_flat: [N*K, 32]
        """
        # Reshape to [N, K, 16, 2]
        pred_traj = pred_x0_flat.view(N, self.num_particles, self.pred_horizon, 2)
        
        cost = torch.zeros(N, self.num_particles, device=self.device)
        
        # 1. Boundary Cost (Normalized space is approx [-2, 2])
        # Penalize if it goes too far (likely invalid)
        out_of_bounds = (pred_traj.abs() - 2.0).clamp(min=0).sum(dim=(2, 3))
        cost += 100.0 * out_of_bounds

        # 2. Smoothness (Minimize Acceleration/Jerk)
        # diff: [N, K, 15, 2]
        diffs = pred_traj[:, :, 1:, :] - pred_traj[:, :, :-1, :]
        smoothness = diffs.square().sum(dim=(2, 3))
        cost += 1.0 * smoothness
        
        # 3. Magnitude (Encourage movement? Or discourage explosion?)
        # Let's discourage huge jumps
        mag = pred_traj.square().sum(dim=(2, 3))
        cost += 0.01 * mag

        return cost # [N, K]

    def step(self, env):
        self.step_count += 1
        with torch.no_grad():
            # 1. Get Environment Data
            data = CoverageEnvUtils.get_torch_geometric_data(env, self.cc_params, True, self.use_comm_map, self.map_size).to(self.device)
            N = data.pos.shape[0]

            def to_device(x):
                if isinstance(x, np.ndarray): return torch.from_numpy(x).float().to(self.device)
                elif isinstance(x, torch.Tensor): return x.float().to(self.device)
                return x

            # Prepare Maps (Same as before)
 # 1. Retrieve Raw Maps
            local = to_device(CoverageEnvUtils.get_raw_local_maps(env, self.cc_params))
            obst = to_device(CoverageEnvUtils.get_raw_obstacle_maps(env, self.cc_params))
            if self.use_comm_map:
                comm = to_device(CoverageEnvUtils.get_communication_maps(env, self.cc_params, self.map_size))
            else:
                comm = torch.zeros(N, 2, 32, 32, device=self.device)

            # 2. Standardize to 4D: [N, C, H, W]
            # Helper to ensure 4D
            def ensure_4d(t, default_ch=1):
                if t.dim() == 2: return t.unsqueeze(0).unsqueeze(0) # [H, W] -> [1, 1, H, W]
                if t.dim() == 3: return t.unsqueeze(1) # [N, H, W] -> [N, 1, H, W]
                return t # Already 4D [N, C, H, W]

            local = ensure_4d(local, 1)
            obst = ensure_4d(obst, 1)
            comm = ensure_4d(comm, 2)

            # 3. Handle Obstacle Broadcast (Single map for multiple agents)
            if obst.shape[0] == 1 and N > 1:
                obst = obst.expand(N, -1, -1, -1)

            # 4. Resize if needed (Force 32x32)
            if local.shape[-1] != 32: local = F.interpolate(local, size=(32,32), mode='bilinear')
            if obst.shape[-1] != 32: obst = F.interpolate(obst, size=(32,32), mode='bilinear')
            if comm.shape[-1] != 32: comm = F.interpolate(comm, size=(32,32), mode='bilinear')

            # 5. Concatenate
            # local: [N, 1, 32, 32]
            # obst:  [N, 1, 32, 32]
            # comm:  [N, 2, 32, 32]
            # Result: [N, 4, 32, 32]
            full_input = torch.cat([local, obst, comm], dim=1)
            
            map_feat = self.lpac_wrapper.cnn_backbone(full_input)

            # --- DDPF PREPARATION ---
            # Replicate for particles: [N, 32] -> [N*K, 32]
            map_feat_expanded = map_feat.repeat_interleave(self.num_particles, dim=0)
            
            # Init Noise [N*K, 32]
            x = torch.randn(N * self.num_particles, self.model.model[-1].out_features, device=self.device)

            # --- DENOISING LOOP WITH RESAMPLING ---
            for i, t in enumerate(self.scheduler.timesteps):
                t_tensor = torch.full((N * self.num_particles,), t, device=self.device, dtype=torch.long)
                
                # Predict Noise
                noise_pred = self.model(x, t_tensor, map_feat_expanded)
                
                # Step (Get prev_sample AND pred_original_sample)
                step_out = self.scheduler.step(noise_pred, t, x)
                prev_sample = step_out.prev_sample
                pred_x0 = step_out.pred_original_sample
                
                # --- RESAMPLING STEP ---
                if i % self.resample_freq == 0 and i < len(self.scheduler.timesteps) - 1:
                    # Calculate Cost [N, K]
                    costs = self.compute_particle_cost(pred_x0, N)
                    
                    # Convert cost to weights (Softmax)
                    weights = torch.softmax(-costs / self.softmax_temp, dim=1) # [N, K]
                    
                    # Resample indices for each agent
                    # indices: [N, K]
                    indices = torch.multinomial(weights, self.num_particles, replacement=True)
                    
                    # Flatten indices to gather from [N*K, ...]
                    # offset: [0, K, 2K, ...]
                    offset = torch.arange(N, device=self.device).view(-1, 1) * self.num_particles
                    gather_idx = (indices + offset).view(-1) # [N*K]
                    
                    # Update particles
                    x = prev_sample[gather_idx]
                else:
                    x = prev_sample

            # 4. Select Best Particle (Final)
            final_costs = self.compute_particle_cost(x, N) # [N, K]
            best_idx = torch.argmin(final_costs, dim=1) # [N]
            
            # Gather best
            offset = torch.arange(N, device=self.device) * self.num_particles
            best_gather_idx = best_idx + offset
            best_x = x[best_gather_idx] # [N, 32]

            # 5. Denormalize & Execute
            actions_flat = best_x * self.act_std_seq + self.act_mean_seq
            actions_seq = actions_flat.view(N, self.pred_horizon, 2)
            
            step_idx = min(self.config.get("HorizonStep", 0), self.pred_horizon - 1)
            actions_step = actions_seq[:, step_idx, :] 

            # Tuning
            scale = self.config.get("ActionScale", 1.5)
            clamp = self.config.get("MaxAction", 2.0)
            actions_step = torch.clamp(actions_step * scale, -clamp, clamp)

            pv_actions = PointVector(np.asarray(actions_step.cpu().numpy(), dtype=np.float64))
            if hasattr(env, "StepActions"): env.StepActions(pv_actions)
            else: env.Step(pv_actions)

            return env.GetObjectiveValue(), False


# ==============================================================================
# FLOW ARCHITECTURE & FILTER (Add this section)
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

class AgentUKF:
    def __init__(self, dt=0.1):
        self.dim_x = 4
        self.dt = dt
        try:
            from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
            points = MerweScaledSigmaPoints(self.dim_x, alpha=0.1, beta=2., kappa=-1)
            self.ukf = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=2, dt=dt, fx=self.fx, hx=self.hx, points=points)
            self.ukf.P *= 1.0
            self.ukf.Q = np.eye(4) * 0.1 
            self.ukf.R = np.eye(2) * 0.05 
        except ImportError:
            print("FilterPy not installed.")
            
    def fx(self, x, dt):
        F = np.eye(4); F[0, 2] = dt; F[1, 3] = dt
        return F @ x
    def hx(self, x): return x[2:] 
    def init_state(self, pos, vel): self.ukf.x = np.array([pos[0], pos[1], vel[0], vel[1]])
    def filter(self, diffusion_vel):
        self.ukf.predict()
        self.ukf.update(diffusion_vel)
        return self.ukf.x[2:]

class ControllerFlowFiltered:
    def __init__(self, cfg, cc_params, env):
        self.config = cfg
        self.cc_params = cc_params
        self.name = cfg.get("Name", "FlowFiltered")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        chk_path = cfg.get("ModelStateDict")
        print(f"Loading Checkpoint: {chk_path}")
        checkpoint = torch.load(chk_path, map_location=self.device)
        self.act_mean = torch.zeros(1, 2, device=self.device)
        self.act_std = torch.ones(1, 2, device=self.device)
        if 'stats' in checkpoint:
            self.act_mean = checkpoint['stats']['action_mean'].to(self.device)
            self.act_std = checkpoint['stats']['action_std'].to(self.device)
        
        lpac_cfg = {
            "ModelConfig": {"UseCommMaps": True},
            "CNNBackBone": {"InputDim": 4, "OutputDim": 32, "ImageSize": 32, "KernelSize":3, "NumLayers":3, "LatentSize":32},
            "GNNBackBone": {"InputDim": 7, "NumHops": 3, "NumLayers": 5, "LatentSize": 256, "OutputDim": 2}
        }
        
        # Robust LPAC Import inside class to avoid top-level fail
        try: from coverage_control.nn import LPAC
        except: from coverage_control.coverage_env_utils import LPAC
        
        self.raw_lpac = LPAC(lpac_cfg).to(self.device)
        if 'lpac' in checkpoint: self.raw_lpac.load_state_dict(checkpoint['lpac'])
        self.cnn_backbone = self.raw_lpac.cnn_backbone
        
        self.model = TrajectoryFlowUnet1D(input_dim=2, global_cond_dim=32, base_dim=64).to(self.device)
        if 'ema_model' in checkpoint: self.model.load_state_dict(checkpoint['ema_model'])
        else: self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.pred_horizon = 16
        self.map_size = 32
        self.filters = None

    def step(self, env):
        with torch.no_grad():
            robot_positions = env.GetRobotPositions()
            if hasattr(robot_positions, 'to_numpy'): pos_np = robot_positions.to_numpy()
            else:
                 data = []
                 for p in robot_positions:
                    if hasattr(p, 'x'): data.append([p.x, p.y])
                    else: data.append([p[0], p[1]])
                 pos_np = np.array(data)
            N = len(pos_np)
            if self.filters is None:
                self.filters = [AgentUKF(dt=0.1) for _ in range(N)]
                for i in range(N): self.filters[i].init_state(pos_np[i], np.zeros(2))

            local_maps = CoverageEnvUtils.get_raw_local_maps(env, self.cc_params)
            obst_maps = CoverageEnvUtils.get_raw_obstacle_maps(env, self.cc_params)
            comm_maps = CoverageEnvUtils.get_communication_maps(env, self.cc_params, self.map_size)
            if local_maps.dim() == 3: local_maps = local_maps.unsqueeze(1)
            if obst_maps.dim() == 3: obst_maps = obst_maps.unsqueeze(1)
            if comm_maps.dim() == 3: comm_maps = comm_maps.unsqueeze(1)
            if local_maps.dim() == 5: local_maps = local_maps.squeeze(1)
            if obst_maps.dim() == 5: obst_maps = obst_maps.squeeze(1)
            if comm_maps.dim() == 5: comm_maps = comm_maps.squeeze(1)
            if obst_maps.shape[0] == 1: obst_maps = obst_maps.expand(N, -1, -1, -1)
            if local_maps.shape[-1] != 32:
                local_maps = F.interpolate(local_maps, size=32)
                obst_maps = F.interpolate(obst_maps, size=32)
                comm_maps = F.interpolate(comm_maps, size=32)
            
            full_input = torch.cat([local_maps, obst_maps, comm_maps], dim=1).float().to(self.device)
            map_feat = self.cnn_backbone(full_input)

            x = torch.randn(N, self.pred_horizon, 2, device=self.device)
            inference_steps = 10
            dt = 1.0 / inference_steps
            for i in range(inference_steps):
                t_curr = 1.0 - (i / inference_steps)
                t_tensor = torch.full((N,), t_curr, device=self.device)
                pred_v = self.model(x, t_tensor, map_feat)
                x = x - pred_v * dt
            
            actions_seq = x * self.act_std + self.act_mean
            
            # Lookahead 2
            lookahead_step = 2
            if actions_seq.shape[1] > lookahead_step:
                raw_actions = actions_seq[:, lookahead_step, :].cpu().numpy()
            else:
                raw_actions = actions_seq[:, 0, :].cpu().numpy()
            
            smoothed_actions = []
            for i in range(N):
                filt_vel = self.filters[i].filter(raw_actions[i])
                filt_vel = np.clip(filt_vel, -2.0, 2.0)
                smoothed_actions.append(filt_vel)
            
            smoothed_actions = np.array(smoothed_actions)
            pv_actions = PointVector(smoothed_actions.astype(np.float64))
            if hasattr(env, 'StepActions'): env.StepActions(pv_actions)
            else: env.Step(pv_actions)
                
            return env.GetObjectiveValue(), False

# -----------------------------
# Evaluator Class
# -----------------------------
class EvaluatorSingle:
    def __init__(self, in_config):
        self.config = in_config
        self.eval_dir = IOUtils.sanitize_path(self.config["EvalDir"]) + "/"
        self.env_dir = IOUtils.sanitize_path(self.config["EnvironmentDataDir"]) + "/"
        os.makedirs(self.env_dir, exist_ok=True)

        self.controllers_configs = self.config["Controllers"]
        self.num_controllers = len(self.controllers_configs)

        if "EnvironmentConfig" in self.config:
            self.env_config_file = IOUtils.sanitize_path(self.config["EnvironmentConfig"])
        else:
            self.env_config_file = IOUtils.sanitize_path(self.config["CCParams"])

        self.cc_params = cc.Parameters(self.env_config_file)
        self.num_steps = int(self.config["NumSteps"])
        self.plot_map = bool(self.config.get("PlotMap", False))
        self.generate_video = bool(self.config.get("GenerateVideo", False))

        if "FeatureFile" in self.config and "RobotPosFile" in self.config:
            self.feature_file = self.env_dir + self.config["FeatureFile"]
            self.pos_file = self.env_dir + self.config["RobotPosFile"]
            if os.path.isfile(self.feature_file) and os.path.isfile(self.pos_file):
                self.world_idf = WorldIDF(self.cc_params, self.feature_file)
                self.env_main = CoverageSystem(self.cc_params, self.world_idf, self.pos_file)
            else:
                self.env_main = CoverageSystem(self.cc_params)
                self.world_idf = self.env_main.GetWorldIDF()
        else:
            self.env_main = CoverageSystem(self.cc_params)
            self.world_idf = self.env_main.GetWorldIDF()

    def evaluate(self, save=True):
        cost_data = np.zeros((self.num_controllers, self.num_steps), dtype=np.float64)
        robot_init_pos = self.env_main.GetRobotPositions(force_no_noise=True)

        if self.plot_map:
            map_dir = self.eval_dir + "/plots/"
            os.makedirs(map_dir, exist_ok=True)
            self.env_main.PlotInitMap(map_dir, "InitMap")

        for controller_id in range(self.num_controllers):
            env = CoverageSystem(self.cc_params, self.world_idf, robot_init_pos)
            cfg = self.controllers_configs[controller_id]
            controller_name = cfg["Name"]
            ctype = cfg["Type"]
            print(f"--- Evaluating Controller: {controller_name} ({ctype}) ---")

            if ctype == "Learning" or ctype == "NN":
                controller = ControllerNN(cfg, self.cc_params, env)
            elif ctype == "CVT":
                controller = ControllerCVT(cfg, self.cc_params, env)
            elif ctype == "DiffusionPolicy":
                controller = ControllerDDPF(cfg, self.cc_params, env)

            elif ctype == "FlowFiltered": # <--- ADD THIS BLOCK
                controller = ControllerFlowFiltered(cfg, self.cc_params, env)            
            else:
                continue

            if self.generate_video:
                env.RecordPlotData()

            initial_objective_value = env.GetObjectiveValue()
            denom = initial_objective_value if abs(initial_objective_value) > 1e-6 else 1.0
            cost_data[controller_id, 0] = 1.0

            for step in range(1, self.num_steps):
                objective_value, converged = controller.step(env)
                cost_data[controller_id, step] = objective_value / denom
                
                if self.generate_video:
                    env.RecordPlotData()
                
                if step % 50 == 0:
                    print(f"    Step {step}: Normalized Cost = {cost_data[controller_id, step]:.4f}")

            print(f"    Final Cost: {cost_data[controller_id, self.num_steps-1]:.4f}")

            if save:
                c_dir = os.path.join(self.eval_dir, controller_name)
                os.makedirs(c_dir, exist_ok=True)
                np.savetxt(os.path.join(c_dir, "eval.csv"), cost_data[controller_id, :], delimiter=",")
                
                if self.generate_video:
                    try:
                        env.RenderRecordedMap(c_dir, "video.mp4")
                    except:
                        pass
            
            del controller
            del env

        return cost_data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_single_env_fixed.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    config = IOUtils.load_toml(config_file)
    evaluator = EvaluatorSingle(config)
    evaluator.evaluate()
