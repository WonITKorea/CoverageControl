import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Imports from your library
from coveragecontrol import CoverageSystem, IOUtils, WorldIDF, PointVector
from coveragecontrol.algorithms import ControllerNN
import coveragecontrol.coverageenvutils as CoverageEnvUtils

# Import architecture from training script (assuming they are in same folder)
# If you run this as script, ensure train_flow can be imported
from train_flow import TrajectoryFlowUnet1D

class ControllerFlowMatching:
    def __init__(self, cfg, cc_params, env):
        self.config = cfg
        self.name = cfg.get("Name", "FlowController")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.step_count = 0
        
        # Load Checkpoint
        model_path = IOUtils.sanitize_path(cfg["ModelStateDict"])
        print(f"Loading Flow Model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load Stats
        stats = checkpoint['stats']
        self.act_mean = stats['action_mean'].to(self.device)
        self.act_std = stats['action_std'].to(self.device)
        
        # Initialize LPAC Wrapper (Perception)
        # We need to recreate the LPAC structure to load weights
        from coveragecontrol.nn import LPAC
        raw_lpac = LPAC(cfg).to(self.device)
        raw_lpac.load_state_dict(checkpoint['lpac'])
        self.cnn_backbone = raw_lpac.cnn_backbone
        
        # Initialize Flow Model
        # Ensure dims match training
        self.pred_horizon = 16 
        self.model = TrajectoryFlowUnet1D(input_dim=2, global_cond_dim=32, base_dim=64).to(self.device)
        
        # Load Weights (Prefer EMA)
        if 'ema_model' in checkpoint:
            print("Loading EMA Weights...")
            self.model.load_state_dict(checkpoint['ema_model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
            
        self.model.eval()
        
        # Eval Configs
        self.use_comm_map = cfg.get("UseCommMap", True)
        self.map_size = cfg.get("CNNMapSize", 32)
        self.horizon_step = cfg.get("HorizonStep", 0) # Receding horizon index

    def step(self, env):
        self.step_count += 1
        
        with torch.no_grad():
            # 1. Prepare Data (Same as original)
            N = len(env.GetRobotPositions())
            
            # Helper to get maps (Reusing your utility logic)
            local_maps = CoverageEnvUtils.get_raw_local_maps(env, env.params).unsqueeze(1)
            obst_maps = CoverageEnvUtils.get_raw_obstacle_maps(env, env.params).unsqueeze(1)
            if obst_maps.shape[0] == 1: obst_maps = obst_maps.expand(N, -1, -1, -1)
            
            if self.use_comm_map:
                comm_maps = CoverageEnvUtils.get_communication_maps(env, env.params, self.map_size).unsqueeze(1)
            else:
                comm_maps = torch.zeros(N, 2, self.map_size, self.map_size)

            # Resize if needed
            if local_maps.shape[-1] != 32:
                local_maps = F.interpolate(local_maps, size=32, mode='bilinear')
                obst_maps = F.interpolate(obst_maps, size=32, mode='bilinear')
                comm_maps = F.interpolate(comm_maps, size=32, mode='bilinear')
                
            # Move to device
            local_maps = local_maps.float().to(self.device)
            obst_maps = obst_maps.float().to(self.device)
            comm_maps = comm_maps.float().to(self.device)
            
            # Concatenate
            full_input = torch.cat([local_maps, obst_maps, comm_maps], dim=1) # (N, 4, 32, 32)
            
            # 2. Get Conditioning
            map_feat = self.cnn_backbone(full_input) # (N, 32)
            
            # ====================================================
            # FLOW MATCHING INFERENCE (Euler ODE Solver)
            # ====================================================
            
            # Start from Pure Noise (t=1.0)
            # Shape: (N, 16, 2)
            x = torch.randn(N, self.pred_horizon, 2, device=self.device)
            
            # ODE Solver Steps (10 is usually sufficient for Flow Matching)
            inference_steps = 10
            dt = 1.0 / inference_steps
            
            for i in range(inference_steps):
                # Current time t (starts at 1.0, goes down to 0.0)
                # We solve BACKWARDS from Noise(1) to Data(0)
                t_curr = 1.0 - (i / inference_steps)
                
                # Create time tensor
                t_tensor = torch.full((N,), t_curr, device=self.device)
                
                # Predict Velocity v = x1 - x0
                pred_v = self.model(x, t_tensor, map_feat)
                
                # Euler Step: x_{t-dt} = x_t - v * dt
                x = x - pred_v * dt
                
            # ====================================================
            
            # 3. Un-normalize
            # x is now the predicted normalized action sequence
            actions_seq = x * self.act_std + self.act_mean
            
            # 4. Receding Horizon Control
            # Take the action at index `horizon_step` (usually 0)
            step_idx = min(self.horizon_step, self.pred_horizon - 1)
            actions = actions_seq[:, step_idx, :] # (N, 2)
            
            # Clamp and Convert
            max_action = 2.0 # From your config defaults
            actions = torch.clamp(actions, -max_action, max_action)
            
            pv_actions = PointVector(actions.cpu().numpy().astype(np.float64))
            
            if hasattr(env, 'StepActions'):
                env.StepActions(pv_actions)
            else:
                env.Step(pv_actions)
                
            return env.GetObjectiveValue(), False

# -----------------------------------------------------------------------------
# Evaluation Runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_flow.py <config_file>")
        sys.exit(1)
        
    config_file = sys.argv[1]
    config = IOUtils.load_toml(config_file)
    
    # Setup Single Evaluator Logic (simplified from your original)
    from coveragecontrol import Parameters
    cc_params = Parameters(IOUtils.sanitize_path(config["CCParams"]))
    
    # Assuming standard WorldIDF and System setup
    world_idf = WorldIDF(cc_params)
    env = CoverageSystem(cc_params, world_idf)
    
    # Init Controller
    # Mocking a config entry for the controller
    ctrl_cfg = {
        "Name": "FlowMatch",
        "ModelStateDict": "checkpoints_flow_unet/epoch_1000.pth", # Adjust
        "CCParams": config["CCParams"]
    }
    
    controller = ControllerFlowMatching(ctrl_cfg, cc_params, env)
    
    print("Starting Eval...")
    cost_data = []
    
    for step in range(config.get("NumSteps", 600)):
        obj, _ = controller.step(env)
        cost_data.append(obj)
        if step % 50 == 0:
            print(f"Step {step}: {obj:.4f}")
            
    print(f"Final Cost: {cost_data[-1]:.4f}")
