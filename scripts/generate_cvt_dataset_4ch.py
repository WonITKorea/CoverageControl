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
NUM_TRAIN_SAMPLES = 100000  # Large dataset
NUM_TEST_SAMPLES = 2000
MAP_SIZE = 32
USE_COMM_MAP = True
EPISODE_LEN = 100
MAX_VELOCITY = 1.0
PRED_HORIZON = 16 # <--- NEW: Trajectory Horizon

def to_tensor(x):
    if isinstance(x, np.ndarray): return torch.from_numpy(x).float()
    return x.float()

def point_vector_to_tensor(pv, num_agents):
    if isinstance(pv, np.ndarray) and pv.ndim == 2: return torch.from_numpy(pv).float()
    data = []
    limit = min(num_agents, len(pv)) if hasattr(pv, '__len__') else num_agents
    for i in range(limit):
        p = pv[i]
        try:
            if hasattr(p, 'x'): data.append([float(p.x), float(p.y)])
            else: data.append([float(p[0]), float(p[1])])
        except: break
    if len(data) == 0: return torch.zeros(num_agents, 2)
    return torch.tensor(data).float()

def resize_map(tensor, size):
    if tensor.dim() == 3: tensor = tensor.unsqueeze(1)
    if tensor.shape[-1] != size:
        tensor = F.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)
    return tensor

def get_num_agents(params):
    if hasattr(params, 'num_agents'): return params.num_agents
    if hasattr(params, 'n_agents'): return params.n_agents
    if hasattr(params, 'N'): return params.N
    return 32

def generate_subset(subset_name, num_samples, params):
    subset_dir = os.path.join(OUTPUT_DIR, "data", subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    NUM_AGENTS = get_num_agents(params)
    print(f"Generating {num_samples} samples (Horizon={PRED_HORIZON})...")

    ctrl_cfg = {"Name": "Expert", "Algorithm": "ClairvoyantCVT", "Type": "CVT"}
    count = 0
    pbar = tqdm(total=num_samples)

    while count < num_samples:
        env = CoverageSystem(params)
        controller = ControllerCVT(ctrl_cfg, params, env)
        
        # We need to run the episode and buffer states to create trajectories
        episode_obs = []     # List of maps
        episode_acts = []    # List of velocities
        episode_pos = []     # List of positions

        # Run full episode first
        for step in range(EPISODE_LEN + PRED_HORIZON): # Run a bit longer for padding
            controller.alg.ComputeActions()
            targets_pv = controller.alg.GetActions()
            pos_pv = env.GetRobotPositions()
            
            raw_vel = point_vector_to_tensor(targets_pv, NUM_AGENTS)
            curr_pos = point_vector_to_tensor(pos_pv, NUM_AGENTS)
            
            # Normalize Velocity
            vel_norm = torch.norm(raw_vel, dim=1, keepdim=True)
            scale = torch.clamp(MAX_VELOCITY / (vel_norm + 1e-6), max=1.0)
            velocity = raw_vel * scale

            # Capture Maps (Only needed up to EPISODE_LEN)
            if step < EPISODE_LEN:
                local = to_tensor(CoverageEnvUtils.get_raw_local_maps(env, params))
                obst = to_tensor(CoverageEnvUtils.get_raw_obstacle_maps(env, params))
                comm = to_tensor(CoverageEnvUtils.get_communication_maps(env, params, MAP_SIZE))
                
                local = resize_map(local, MAP_SIZE)
                obst = resize_map(obst, MAP_SIZE)
                comm = resize_map(comm, MAP_SIZE)
                full_map = torch.cat([local, obst, comm], dim=1) # [N, 4, 32, 32]
                
                episode_obs.append(full_map)
                episode_pos.append(curr_pos)
            
            episode_acts.append(velocity) # Keep collecting actions for horizon

            # Step Env
            if hasattr(env, "StepActions"): env.StepActions(targets_pv)
            else: env.Step(targets_pv)

        # Create Samples with Horizon
        # For each time step t, target is actions[t : t+16]
        for t in range(len(episode_obs)):
            if count >= num_samples: break
            
            # Get Horizon
            # If we are near end of episode, pad with last action
            actions_chunk = []
            for k in range(PRED_HORIZON):
                idx = t + k
                if idx < len(episode_acts):
                    actions_chunk.append(episode_acts[idx])
                else:
                    actions_chunk.append(episode_acts[-1]) # Pad with last
            
            # Stack: [Horizon, N, 2] -> [N, Horizon, 2]
            action_seq = torch.stack(actions_chunk, dim=0).transpose(0, 1) 
            
            # Save N samples (one per agent)
            current_map = episode_obs[t]
            current_p = episode_pos[t]
            
            for i in range(NUM_AGENTS):
                if count >= num_samples: break
                
                data = Data()
                data.map = current_map[i].unsqueeze(0)  # [1, 4, 32, 32]
                data.y = action_seq[i].unsqueeze(0)     # [1, 16, 2] <--- TRAJECTORY!
                data.pos = current_p[i].unsqueeze(0)
                
                if torch.isnan(data.y).any(): continue
                torch.save(data, os.path.join(subset_dir, f"data_{count}.pt"))
                
                count += 1
                pbar.update(1)

        del controller
        del env
        if count % 2000 == 0: gc.collect()
    
    pbar.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_cvt_dataset_4ch.py <params_file>")
        sys.exit(1)
    params = cc.Parameters(sys.argv[1])
    generate_subset("train", NUM_TRAIN_SAMPLES, params)
    generate_subset("test", NUM_TEST_SAMPLES, params)

if __name__ == "__main__":
    main()
