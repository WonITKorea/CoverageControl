import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coverage_control as cc
from coverage_control import CoverageSystem, IOUtils, WorldIDF, PointVector
from coverage_control.algorithms import ControllerCVT, ControllerNN

try: from coverage_control.nn import CoverageEnvUtils
except: from coverage_control.coverage_env_utils import CoverageEnvUtils

def _add_scripts_to_syspath():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(this_dir, "../../..", "scripts")))
_add_scripts_to_syspath()

# ==============================================================================
# RESTORED MODEL: MLP (ConditionalUnet1D)
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

class LPACWrapper(nn.Module):
    def __init__(self, original_lpac):
        super().__init__()
        self.original_lpac = original_lpac
        self.cnn_backbone = original_lpac.cnn_backbone
    def forward(self, x): return self.cnn_backbone(x)

class ControllerDiffusion:
    def __init__(self, cfg, cc_params, env):
        self.config = cfg
        self.name = cfg["Name"]
        self.cc_params = cc_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_count = 0
        
        learning_params = IOUtils.load_toml(IOUtils.sanitize_path(cfg["LearningParams"]))
        try: from coverage_control.nn import LPAC
        except: from coverage_control.coverage_env_utils import LPAC
        
        raw_lpac = LPAC(learning_params).to(self.device)
        self.lpac_wrapper = LPACWrapper(raw_lpac).to(self.device)

        model_path = IOUtils.sanitize_path(cfg["ModelStateDict"])
        print(f"Loading Checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.actions_mean = torch.zeros(1, 2, device=self.device)
        self.actions_std = torch.ones(1, 2, device=self.device)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print("(!) Detected MLP Checkpoint")
            self.model = ConditionalUnet1D(2, 32, 256, 100).to(self.device)
            self.model.load_state_dict(checkpoint['model'])
            
            if 'lpac' in checkpoint:
                print("    Loading LPAC weights...")
                self.lpac_wrapper.original_lpac.load_state_dict(checkpoint['lpac'])
            
            from diffusers import DDPMScheduler
            self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
            
            if 'stats' in checkpoint:
                self.actions_mean.copy_(checkpoint['stats']['action_mean'].to(self.device))
                self.actions_std.copy_(checkpoint['stats']['action_std'].to(self.device))
        else: raise ValueError("Invalid Checkpoint")

        self.model.eval()
        self.use_comm_map = cfg.get("UseCommMap", True)
        self.map_size = cfg.get("CNNMapSize", 32)

    def step(self, env):
        self.step_count += 1
        with torch.no_grad():
            data = CoverageEnvUtils.get_torch_geometric_data(env, self.cc_params, True, self.use_comm_map, self.map_size).to(self.device)
            N = data.pos.shape[0]
            def to_dev(x):
                if isinstance(x, np.ndarray): return torch.from_numpy(x).float().to(self.device)
                return x.float().to(self.device)

            local = to_dev(CoverageEnvUtils.get_raw_local_maps(env, self.cc_params))
            if local.dim() == 3: local = local.unsqueeze(1)
            
            obst = to_dev(CoverageEnvUtils.get_raw_obstacle_maps(env, self.cc_params))
            if obst.dim() == 2: obst = obst.unsqueeze(0).unsqueeze(0)
            elif obst.dim() == 3: obst = obst.unsqueeze(1)
            if obst.shape[0] == 1 and N > 1: obst = obst.expand(N, -1, -1, -1)

            if self.use_comm_map:
                comm = to_dev(CoverageEnvUtils.get_communication_maps(env, self.cc_params, self.map_size))
                if comm.dim() == 3: comm = comm.unsqueeze(1)
            else: comm = torch.zeros(N, 2, self.map_size, self.map_size, device=self.device)

            # Resize
            if local.shape[-1]!=32: local=F.interpolate(local, size=(32,32), mode='bilinear')
            if obst.shape[-1]!=32: obst=F.interpolate(obst, size=(32,32), mode='bilinear')
            if comm.shape[-1]!=32: comm=F.interpolate(comm, size=(32,32), mode='bilinear')

            full_input = torch.cat([local[:,0:1], obst[:,0:1], comm[:,0:2]], dim=1)
            cond = self.lpac_wrapper.cnn_backbone(full_input)

            x = torch.randn(N, 2, device=self.device)
            for t in self.scheduler.timesteps:
                t_tensor = torch.full((N,), t, device=self.device, dtype=torch.long)
                noise_pred = self.model(x, t_tensor, cond)
                x = self.scheduler.step(noise_pred, t, x).prev_sample

            actions = x * self.actions_std + self.actions_mean
            actions = actions * 1.2 # Speed boost
            actions = torch.clamp(actions, -2.0, 2.0)

            if self.step_count % 50 == 0:
                print(f"DEBUG Act[0]: {actions[0].cpu().numpy()}")

            pv = PointVector(np.asarray(actions.cpu().numpy(), dtype=np.float64))
            if hasattr(env, "StepActions"): env.StepActions(pv)
            else: env.Step(pv)

            return env.GetObjectiveValue(), False

class EvaluatorSingle:
    def __init__(self, in_config):
        self.config = in_config
        self.eval_dir = IOUtils.sanitize_path(self.config["EvalDir"]) + "/"
        self.env_dir = IOUtils.sanitize_path(self.config["EnvironmentDataDir"]) + "/"
        os.makedirs(self.env_dir, exist_ok=True)
        self.controllers_configs = self.config["Controllers"]
        self.num_controllers = len(self.controllers_configs)
        if "EnvironmentConfig" in self.config: self.env_config_file = IOUtils.sanitize_path(self.config["EnvironmentConfig"])
        else: self.env_config_file = IOUtils.sanitize_path(self.config["CCParams"])
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

            if ctype == "Learning" or ctype == "NN": controller = ControllerNN(cfg, self.cc_params, env)
            elif ctype == "CVT": controller = ControllerCVT(cfg, self.cc_params, env)
            elif ctype == "DiffusionPolicy": controller = ControllerDiffusion(cfg, self.cc_params, env)
            else: continue

            if self.generate_video: env.RecordPlotData()
            initial_objective_value = env.GetObjectiveValue()
            denom = initial_objective_value if abs(initial_objective_value) > 1e-6 else 1.0
            cost_data[controller_id, 0] = 1.0

            for step in range(1, self.num_steps):
                objective_value, converged = controller.step(env)
                cost_data[controller_id, step] = objective_value / denom
                if self.generate_video: env.RecordPlotData()
                if step % 50 == 0: print(f"    Step {step}: Normalized Cost = {cost_data[controller_id, step]:.4f}")
            print(f"    Final Cost: {cost_data[controller_id, self.num_steps-1]:.4f}")

            if save:
                c_dir = os.path.join(self.eval_dir, controller_name)
                os.makedirs(c_dir, exist_ok=True)
                np.savetxt(os.path.join(c_dir, "eval.csv"), cost_data[controller_id, :], delimiter=",")
                if self.generate_video:
                    try: env.RenderRecordedMap(c_dir, "video.mp4")
                    except: pass
            del controller
            del env
        return cost_data

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    config_file = sys.argv[1]
    config = IOUtils.load_toml(config_file)
    evaluator = EvaluatorSingle(config)
    evaluator.evaluate()
