import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import coverage_control as cc
import coverage_control.nn as cc_nn
from coverage_control import IOUtils, PointVector
from coverage_control.nn import CoverageEnvUtils

# --- Configuration ---
script_dir = os.path.dirname(os.path.realpath(__file__))
# Adjust paths if your folder structure is different
params_file = os.path.join(script_dir, "data/lpac/coverage_control_params.toml")
model_file = os.path.join(script_dir, "data/lpac/models/model_k3_1024_state_dict.pt") 
learning_config_file = os.path.join(script_dir, "data/lpac/models/learning_params.toml")
robot_pos_file = os.path.join(script_dir, "data/nn/robots_positions")
features_file = os.path.join(script_dir, "data/nn/features")
# ---------------------

def simulate_and_plot():
    # 1. Setup Environment
    print("Setting up environment...")
    params = cc.Parameters(params_file)
    world_idf = cc.WorldIDF(params, features_file)
    env = cc.CoverageSystem(params, world_idf, robot_pos_file)
    params.pNumRobots = env.GetNumRobots()
    
    # 2. Load Model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_config = IOUtils.load_toml(learning_config_file)
    lpac_model = cc_nn.LPAC(learning_config).to(device)
    
    # Load state dict (handling potential key mismatches if saved differently)
    state_dict = torch.load(model_file, map_location=device)
    lpac_model.load_state_dict(state_dict)
    
    # Register buffers if missing (for older checkpoints)
    if not hasattr(lpac_model, "actions_mean"):
        # Create dummy buffers if not in checkpoint (will be overwritten if in state_dict)
        lpac_model.register_buffer("actions_mean", torch.zeros(1, 2).to(device))
        lpac_model.register_buffer("actions_std", torch.ones(1, 2).to(device))
        
    lpac_model.eval()
    
    use_comm_maps = learning_config['ModelConfig']['UseCommMaps']
    map_size = learning_config['CNNBackBone']['ImageSize']

    # 3. Simulation Loop
    num_steps = 100  # Adjust as needed
    trajectories = [] # Store positions: list of (N, 2) arrays
    obj_values = []
    
    print(f"Running simulation for {num_steps} steps...")
    
    with torch.no_grad():
        for step in range(num_steps):
            # Get data from environment
            pyg_data = CoverageEnvUtils.get_torch_geometric_data(
                env, params, True, use_comm_maps, map_size
            ).to(device)
            
            # --- CAPTURE POSITIONS ---
            # Try to get positions from pyg_data.x (node features)
            # Usually x is [num_nodes, features], where features[0:2] are x, y
            current_pos = pyg_data.x[:, :2].cpu().numpy()
            trajectories.append(current_pos)
            # -------------------------
            
            # Forward pass
            actions = lpac_model(pyg_data)
            
            # Unnormalize actions
            actions = actions * lpac_model.actions_std + lpac_model.actions_mean
            
            # Step environment
            point_vector_actions = PointVector(actions.cpu().numpy())
            env.StepActions(point_vector_actions)
            
            # Record objective
            obj_values.append(env.GetObjectiveValue())

    # 4. Visualization
    print("Generating plots...")
    trajectories = np.array(trajectories) # Shape: (Steps, Robots, 2)
    
    # Get the IDF map
    world_map = env.GetWorldMap()
    
    # Plot 1: Map + Trajectories
    plt.figure(figsize=(10, 8))
    plt.imshow(world_map, origin='lower', cmap='viridis') # Transpose if needed: .T
    
    # Iterate over robots to plot paths
    num_robots = trajectories.shape[1]
    for i in range(num_robots):
        # Plot full path
        plt.plot(trajectories[:, i, 0], trajectories[:, i, 1], 
                 '-', linewidth=1, alpha=0.7)
        # Plot start point
        plt.plot(trajectories[0, i, 0], trajectories[0, i, 1], 'g.', markersize=5)
        # Plot end point
        plt.plot(trajectories[-1, i, 0], trajectories[-1, i, 1], 'r*', markersize=8)
        
    plt.colorbar(label='IDF Value')
    plt.title(f"Simulation Trajectories ({num_robots} robots)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()

    # Plot 2: Objective Value
    plt.figure(figsize=(8, 4))
    plt.plot(obj_values)
    plt.title("Coverage Cost over Time")
    plt.xlabel("Step")
    plt.ylabel("Objective Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate_and_plot()
