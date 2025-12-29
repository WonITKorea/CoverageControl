import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import coverage_control as cc
import coverage_control.nn as cc_nn
from coverage_control import IOUtils, PointVector
from coverage_control.nn import CoverageEnvUtils

# --- PATH CONFIGURATION ---
# Based on your tree: ~/CoverageControl_ws/src/CoverageControl/python/tests/run_sample.py
current_dir = os.path.dirname(os.path.realpath(__file__))
ws_root = os.path.abspath(os.path.join(current_dir, "../../../")) # Points to ~/CoverageControl_ws/src

# Define paths to your sample data
# You listed `data/lpac/models` and `data/features`, so we use those.
data_root = os.path.join(ws_root, "data")
params_file = os.path.join(data_root, "params/coverage_control_params.toml")
learning_config_file = os.path.join(data_root, "params/learning_params.toml")
features_file = os.path.join(data_root, "features") 
robot_pos_file = os.path.join(data_root, "robots_positions")

# We need a trained model. Your tree shows `data/lpac/models`. 
# Check if a .pt file exists inside there. If not, the script will fail.
# Let's assume there is one or we skip loading if just testing map.
model_dir = os.path.join(data_root, "lpac/models")
# Find the first .pt file in the model dir
try:
    model_file = [f for f in os.listdir(model_dir) if f.endswith(".pt")][0]
    model_path = os.path.join(model_dir, model_file)
    print(f"Found model: {model_path}")
except (FileNotFoundError, IndexError):
    print(f"WARNING: No .pt model found in {model_dir}. Simulation will fail without a model.")
    model_path = None
# --------------------------

def main():
    if not model_path:
        return

    print("1. Setting up Environment...")
    # Initialize parameters and world
    params = cc.Parameters(params_file)
    world_idf = cc.WorldIDF(params, features_file)
    env = cc.CoverageSystem(params, world_idf, robot_pos_file)
    
    print("2. Loading Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_config = IOUtils.load_toml(learning_config_file)
    
    # Force the model to use the correct map size from config
    # (Sometimes config needs manual override if it doesn't match data)
    
    lpac_model = cc_nn.LPAC(learning_config).to(device)
    lpac_model.load_state_dict(torch.load(model_path, map_location=device))
    lpac_model.eval()

    # Create dummy buffers if missing (common in this codebase)
    if not hasattr(lpac_model, "actions_mean"):
        lpac_model.register_buffer("actions_mean", torch.zeros(1, 2).to(device))
        lpac_model.register_buffer("actions_std", torch.ones(1, 2).to(device))

    print("3. Running Simulation Loop...")
    num_steps = 50
    trajectories = []
    costs = []
    
    use_comm_maps = learning_config['ModelConfig']['UseCommMaps']
    map_size = learning_config['CNNBackBone']['ImageSize']

    with torch.no_grad():
        for step in range(num_steps):
            # Get graph data
            data = CoverageEnvUtils.get_torch_geometric_data(
                env, params, True, use_comm_maps, map_size
            ).to(device)
            
            # Store current positions (from graph features)
            # data.x is usually [N, features], first 2 are (x,y)
            pos = data.x[:, :2].cpu().numpy()
            trajectories.append(pos)
            
            # Predict actions
            actions = lpac_model(data)
            
            # Un-normalize
            actions = actions * lpac_model.actions_std + lpac_model.actions_mean
            
            # Step environment
            env.StepActions(PointVector(actions.cpu().numpy()))
            
            # Track cost
            costs.append(env.GetObjectiveValue())

    print("4. Plotting...")
    trajectories = np.array(trajectories) # (Steps, Robots, 2)
    world_map = env.GetWorldMap()

    # Plot Map + Trajectories
    plt.figure(figsize=(8, 8))
    plt.imshow(world_map, origin='lower', cmap='viridis')
    for i in range(trajectories.shape[1]):
        plt.plot(trajectories[:, i, 0], trajectories[:, i, 1], '-w', alpha=0.5)
        plt.plot(trajectories[-1, i, 0], trajectories[-1, i, 1], '*r') # End
    plt.title("Simulation with Sample Data")
    plt.show()

    # Plot Cost
    plt.figure()
    plt.plot(costs)
    plt.title("Coverage Cost")
    plt.show()

if __name__ == "__main__":
    main()
