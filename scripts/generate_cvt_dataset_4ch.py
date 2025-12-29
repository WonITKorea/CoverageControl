import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gc
import coverage_control as cc
from coverage_control import IOUtils, CoverageSystem
from coverage_control.algorithms import ControllerCVT

# Import Utils
try:
    from coverage_control.nn import CoverageEnvUtils
    from torch_geometric.data import Data
except ImportError:
    from coverage_control.coverage_env_utils import CoverageEnvUtils
    from torch_geometric.data import Data

# --- Configuration ---
OUTPUT_DIR = "lpac/data"
NUM_TRAIN_SAMPLES = 50000
NUM_TEST_SAMPLES = 2000
MAP_SIZE = 32
USE_COMM_MAP = True
EPISODE_LEN = 100
MAX_VELOCITY = 1.0  # Clamp velocity magnitude

def to_tensor(x):
    """Converts numpy array to float tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()

def point_vector_to_tensor(pv, num_agents):
    """
    Converts C++ PointVector OR Numpy Array to torch.Tensor [N, 2].
    Handles:
      - Object with .x, .y
      - Array/List with [0], [1]
      - Full Numpy Array [N, 2]
    """
    # Shortcut: If it's already a clean numpy array [N, 2]
    if isinstance(pv, np.ndarray) and pv.ndim == 2 and pv.shape[1] == 2:
        return torch.from_numpy(pv).float()
        
    data = []
    # Safeguard against list length vs requested agents
    limit = min(num_agents, len(pv)) if hasattr(pv, '__len__') else num_agents
    
    for i in range(limit):
        p = pv[i]
        try:
            # Case 1: Object with .x, .y (C++ Binding)
            if hasattr(p, 'x') and hasattr(p, 'y'):
                data.append([float(p.x), float(p.y)])
            # Case 2: Numpy array or List [x, y]
            else:
                data.append([float(p[0]), float(p[1])])
        except Exception:
            # If extraction fails, stop or skip
            break
            
    if len(data) == 0:
        return torch.zeros(num_agents, 2)
        
    return torch.tensor(data).float()

def resize_map(tensor, size):
    """
    Resizes a map tensor to [B, C, size, size].
    Expected input: [N, C, H, W] or [N, H, W]
    """
    if tensor.dim() == 3:  # [N, H, W] -> [N, 1, H, W]
        tensor = tensor.unsqueeze(1)
    
    if tensor.shape[-1] != size:
        tensor = F.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)
    
    return tensor

def get_num_agents(params):
    """Robustly retrieve number of agents from params."""
    if hasattr(params, 'num_agents'): return params.num_agents
    if hasattr(params, 'n_agents'): return params.n_agents
    if hasattr(params, 'N'): return params.N
    # Fallback: Check if we can infer from environment or default
    print("Warning: Could not find 'num_agents' in params. Defaulting to 32.")
    return 32

def generate_subset(subset_name, num_samples, params):
    subset_dir = os.path.join(OUTPUT_DIR, "data", subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    
    # Robustly get num agents
    NUM_AGENTS = get_num_agents(params)
    print(f"Generating {num_samples} samples for '{subset_name}' (Agents: {NUM_AGENTS})...")
    
    # We use ClairvoyantCVT (Perfect Info) as the Expert
    ctrl_cfg = {"Name": "Expert", "Algorithm": "ClairvoyantCVT", "Type": "CVT"}
    
    count = 0
    pbar = tqdm(total=num_samples)
    
    while count < num_samples:
        env = CoverageSystem(params)
        controller = ControllerCVT(ctrl_cfg, params, env)
        
        for step in range(EPISODE_LEN):
            if count >= num_samples: break
            
            try:
                # 1. Compute Expert Action (Target Positions)
                controller.alg.ComputeActions()
                targets_pv = controller.alg.GetActions() # PointVector or Array
                
                # 2. Get Current Positions
                pos_pv = env.GetRobotPositions() # PointVector or Array
                
                # 3. Convert C++ Objects to Tensors [N, 2]
                raw_velocity = point_vector_to_tensor(targets_pv, NUM_AGENTS) 
                current_pos = point_vector_to_tensor(pos_pv, NUM_AGENTS)
                targets = raw_velocity
                if count % 100 == 0:
                    print(f"DEBUG RANGE: Pos [{current_pos.min():.1f}, {current_pos.max():.1f}] | Target [{targets.min():.1f}, {targets.max():.1f}]")
                # Verify shapes matches expected agents
                if targets.shape[0] != NUM_AGENTS:
                    NUM_AGENTS = targets.shape[0] # Adjust dynamically if needed
                 
                # 4. Calculate Velocity = Target - Current_Pos
                velocity = raw_velocity
                
                # 5. Clamp Velocity (Normalize speed to simulate max_step)
                vel_norm = torch.norm(velocity, dim=1, keepdim=True)
                scale = torch.clamp(MAX_VELOCITY / (vel_norm + 1e-6), max=1.0)
                velocity = velocity * scale
                
                # 6. Extract & Resize Maps
                # Local View [N, 1, 32, 32]
                local = to_tensor(CoverageEnvUtils.get_raw_local_maps(env, params))
                local = resize_map(local, MAP_SIZE)
                
                # Obstacle Map [N, 1, 32, 32]
                obst = to_tensor(CoverageEnvUtils.get_raw_obstacle_maps(env, params))
                obst = resize_map(obst, MAP_SIZE)
                
                # Comm Map [N, 2, 32, 32]
                comm = to_tensor(CoverageEnvUtils.get_communication_maps(env, params, MAP_SIZE))
                comm = resize_map(comm, MAP_SIZE)
                
                # Concatenate: [N, 4, 32, 32]
                full_map = torch.cat([local, obst, comm], dim=1)
                
                # 7. SAVE INDIVIDUAL SAMPLES
                # Iterate over each robot and save as a separate training example
                for i in range(NUM_AGENTS):
                    if count >= num_samples: break
                    
                    data = Data()
                    data.map = full_map[i].unsqueeze(0)    # [1, 4, 32, 32]
                    data.y = velocity[i].unsqueeze(0)      # [1, 2] (Velocity)
                    data.pos = current_pos[i].unsqueeze(0) # [1, 2]
                    
                    # Skip NaN
                    if torch.isnan(data.y).any(): continue
                    
                    torch.save(data, os.path.join(subset_dir, f"data_{count}.pt"))
                    count += 1
                    pbar.update(1)
                
            except Exception as e:
                print(f"Error extracting sample: {e}")
                # Optional: print full trace if needed
                # import traceback; traceback.print_exc()
                break
            
            # Step Environment
            if hasattr(env, "StepActions"): 
                env.StepActions(targets_pv)
            else: 
                env.Step(targets_pv)
        
        del controller
        del env
        if count % 500 == 0: gc.collect()
            
    pbar.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_cvt_dataset_4ch.py <params_file>")
        sys.exit(1)
        
    params_file = sys.argv[1]
    params = cc.Parameters(params_file)
    
    # Generate Data
    generate_subset("train", NUM_TRAIN_SAMPLES, params)
    generate_subset("test", NUM_TEST_SAMPLES, params)

if __name__ == "__main__":
    main()

