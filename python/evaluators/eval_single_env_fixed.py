import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    from coverage_control.nn import CoverageEnvUtils, CNNGNNDataset
except ImportError:
    try:
        from coverage_control.coverage_env_utils import CoverageEnvUtils
        from coverage_control.nn import CNNGNNDataset
    except ImportError:
        import coverage_control.coverage_env_utils as CoverageEnvUtils
        CNNGNNDataset = None

def _add_scripts_to_syspath():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "../../.."))
    scripts_dir = os.path.join(project_root, "scripts")
    if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
        sys.path.append(scripts_dir)

_add_scripts_to_syspath()

# ==============================================================================
# MATCHING ARCHITECTURE FROM TRAIN_MADP.PY
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
        x = torch.cat([sample, global_cond, t_emb], dim=-1)
        return self.model(x)

# -----------------------------
# LPAC Wrapper for Inference
# -----------------------------
class LPACWrapper(nn.Module):
    def __init__(self, original_lpac):
        super().__init__()
        self.original_lpac = original_lpac
        self.cnn_backbone = original_lpac.cnn_backbone

    def forward(self, x):
        return self.cnn_backbone(x)

# -----------------------------
# Controller Diffusion Wrapper
# -----------------------------
class ControllerDiffusion:
    def __init__(self, cfg, cc_params, env):
        self.config = cfg
        self.name = cfg["Name"]
        self.cc_params = cc_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        learning_params_file = IOUtils.sanitize_path(cfg["LearningParams"])
        self.learning_config = IOUtils.load_toml(learning_params_file)
        
        try: from coverage_control.nn import LPAC
        except: from coverage_control.coverage_env_utils import LPAC
        
        # Load LPAC Encoder (Frozen)
        raw_lpac = LPAC(self.learning_config).to(self.device)
        self.lpac_wrapper = LPACWrapper(raw_lpac).to(self.device)
        
        # Load Checkpoint
        model_path = IOUtils.sanitize_path(cfg["ModelStateDict"])
        print(f"Loading Checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # STATS BUFFERS
        self.register_stats()

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print("(!) Detected NEW Checkpoint Format (ConditionalUnet1D)")
            state_dict = checkpoint['model']
            
            self.model = ConditionalUnet1D(input_dim=2, global_cond_dim=32, num_train_timesteps=100).to(self.device)
            self.model.load_state_dict(state_dict)
            
            from diffusers import DDIMScheduler
            self.scheduler = DDIMScheduler(num_train_timesteps=100)
            self.scheduler.set_timesteps(20)
            
            if 'stats' in checkpoint:
                stats = checkpoint['stats']
                self.actions_mean.copy_(stats['action_mean'].to(self.device))
                self.actions_std.copy_(stats['action_std'].to(self.device))
                self.pos_mean.copy_(stats['pos_mean'].to(self.device))
                self.pos_std.copy_(stats['pos_std'].to(self.device))
                
                print(f"    Loaded Stats -> Action Mean: {self.actions_mean.mean():.4f}, Pos Mean: {self.pos_mean.mean():.4f}")
        else:
            raise ValueError("Old Checkpoint format not supported with fixed script. Please retrain.")

        self.model.eval()
        self.use_comm_map = cfg.get("UseCommMap", True)
        self.map_size = cfg.get("CNNMapSize", 32)
        
        self.world_size_meters = 1024.0
        self.env_limit = self.world_size_meters / 2.0 
        self.local_map_size_meters = 256.0 
        self.local_map_radius = self.local_map_size_meters / 2.0 
        self.comm_radius = 128.0 

    def register_stats(self):
        self.actions_mean = torch.zeros(1, 2, device=self.device)
        self.actions_std = torch.ones(1, 2, device=self.device)
        self.pos_mean = torch.zeros(1, 2, device=self.device)
        self.pos_std = torch.ones(1, 2, device=self.device)

    def _extract_local_maps(self, global_map, pos_tensor):
        N = pos_tensor.shape[0]
        C, H, W = global_map.shape[1], global_map.shape[2], global_map.shape[3]
        
        centers = pos_tensor / self.env_limit
        scale_x = self.local_map_size_meters / self.world_size_meters
        scale_y = self.local_map_size_meters / self.world_size_meters
        
        theta = torch.zeros(N, 2, 3, device=self.device)
        theta[:, 0, 0] = scale_x
        theta[:, 1, 1] = scale_y 
        theta[:, 0, 2] = centers[:, 0]
        theta[:, 1, 2] = centers[:, 1]
        
        grid = F.affine_grid(theta, torch.Size((N, C, self.map_size, self.map_size)), align_corners=False)
        
        if global_map.shape[0] == 1:
            global_map_expand = global_map.expand(N, -1, -1, -1)
        else:
            global_map_expand = global_map 
            
        local_maps = F.grid_sample(global_map_expand, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return local_maps

    def _inject_neighbor_channels(self, map_tensor, pos_tensor, batch_idx):
        N = pos_tensor.shape[0]
        H, W = self.map_size, self.map_size
        r_meter = self.local_map_radius
        
        map_tensor[:, 2:, :, :] = 0.0
        
        for i in range(N):
            ego_pos = pos_tensor[i]
            diff = pos_tensor - ego_pos
            mask_x = (diff[:, 0].abs() <= r_meter)
            mask_y = (diff[:, 1].abs() <= r_meter)
            mask = mask_x & mask_y
            mask[i] = False
            
            neighbors_idx = torch.where(mask)[0]
            if len(neighbors_idx) == 0: continue

            rel_pos = diff[neighbors_idx]
            
            px = ((rel_pos[:, 0] + r_meter) / (2 * r_meter) * W).long()
            py = ((rel_pos[:, 1] + r_meter) / (2 * r_meter) * H).long()
            
            px = torch.clamp(px, 0, W-1)
            py = torch.clamp(py, 0, H-1)
            
            for k in range(len(neighbors_idx)):
                cx, cy = px[k], py[k]
                if 0 <= cx < W and 0 <= cy < H:
                    map_tensor[i, 2, cy, cx] = rel_pos[k, 0] / r_meter
                    map_tensor[i, 3, cy, cx] = rel_pos[k, 1] / r_meter
                            
        return map_tensor

    def step(self, env):
        with torch.no_grad():
            data = CoverageEnvUtils.get_torch_geometric_data(
                env, self.cc_params, True, self.use_comm_map, self.map_size
            ).to(self.device)
            
            # --- 1. Prepare Maps ---
            density_map = None
            if isinstance(data.x, torch.Tensor): density_map = data.x
            if density_map.dim() < 4: density_map = density_map.view(1, -1, self.map_size, self.map_size)

            try:
                obst_raw = CoverageEnvUtils.get_raw_obstacle_maps(env, self.cc_params)
                if not isinstance(obst_raw, torch.Tensor):
                    obst_raw = torch.from_numpy(obst_raw).float().to(self.device)
                else:
                    obst_raw = obst_raw.to(self.device)
                
                if obst_raw.dim() == 2: obst_raw = obst_raw.unsqueeze(0).unsqueeze(0)
                elif obst_raw.dim() == 3: obst_raw = obst_raw.unsqueeze(1)
                
                if obst_raw.shape[-1] != self.map_size:
                    obst_raw = F.interpolate(obst_raw, size=(self.map_size, self.map_size), mode='bilinear')
                
                global_maps = torch.cat([density_map, obst_raw], dim=1)
            except Exception:
                global_maps = density_map

            N = data.pos.shape[0]
            local_maps_raw = CoverageEnvUtils.get_raw_local_maps(env, self.cc_params)
            local_maps = torch.from_numpy(local_maps_raw).float().to(self.device)
            
            full_input = torch.zeros((N, 4, self.map_size, self.map_size), device=self.device)
            full_input[:, 0:1, :, :] = local_maps[:, 0:1, :, :]
            #full_input = torch.flip(full_input, dims=[2]) 
            if local_maps.shape[1] >= 2:
                full_input[:, 1:2, :, :] = local_maps[:, 1:2, :, :]
                
            full_input = self._inject_neighbor_channels(full_input, data.pos, None)
            
            # --- 2. Normalize Position (Fixing the disconnect) ---
            norm_pos = (data.pos - self.pos_mean) / self.pos_std
            
            # --- 3. Forward ---
            map_feat = self.lpac_wrapper.cnn_backbone(full_input)
            cond = map_feat
            
            batch_size = cond.shape[0]
            x = torch.randn(batch_size, 2, device=self.device)
            
            for t in self.scheduler.timesteps:
                t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                noise_pred = self.model(x, t_tensor, cond)
                if hasattr(noise_pred, "sample"): noise_pred = noise_pred.sample
                step_output = self.scheduler.step(noise_pred, t, x)
                x = step_output.prev_sample
            # --- 4. Denormalize Actions ---
            actions_tensor = x * self.actions_std + self.actions_mean
            # 1. Reduce Speed slightly to prevent overshooting
            actions_tensor = actions_tensor * 0.8
            actions_tensor = torch.clamp(actions_tensor, -2.0, 2.0)

            print(f"DEBUG Action [0]: {actions_tensor[0].cpu().detach().numpy()}")
            pv_actions = PointVector(np.asarray(actions_tensor.cpu().numpy(), dtype=np.float64))
            if hasattr(env, "StepActions"): env.StepActions(pv_actions)
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
                controller = ControllerDiffusion(cfg, self.cc_params, env)
            else:
                print(f"Skipping unknown controller type: {ctype}")
                continue
            
            if self.generate_video: env.RecordPlotData()
            
            initial_objective_value = env.GetObjectiveValue()
            denom = initial_objective_value if abs(initial_objective_value) > 1e-6 else 1.0
            cost_data[controller_id, 0] = 1.0
            
            for step in range(1, self.num_steps):
                objective_value, converged = controller.step(env)
                cost_data[controller_id, step] = objective_value / denom
                if self.generate_video: env.RecordPlotData()
                if step % 50 == 0:
                    print(f"    Step {step}: Normalized Cost = {cost_data[controller_id, step]:.4f}")
            
            print(f"    Final Cost: {cost_data[controller_id, self.num_steps-1]:.4f}")
            
            if save:
                c_dir = os.path.join(self.eval_dir, controller_name)
                os.makedirs(c_dir, exist_ok=True)
                np.savetxt(os.path.join(c_dir, "eval.csv"), cost_data[controller_id, :], delimiter=",")
                if self.generate_video:
                    print("    Generating Video...")
                    try: env.RenderRecordedMap(c_dir, "video.mp4")
                    except Exception as e: print(f"    Video generation failed: {e}")
            del controller
            del env
        return cost_data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python python/evaluators/eval_single_env_fixed.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    config = IOUtils.load_toml(config_file)
    evaluator = EvaluatorSingle(config)
    evaluator.evaluate()

