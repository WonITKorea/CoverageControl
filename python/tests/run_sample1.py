import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import coverage_control as cc
import coverage_control.nn as cc_nn
from coverage_control import IOUtils, PointVector

# Import fix
try:
    from coverage_control.coverage_env_utils import CoverageEnvUtils
except ImportError:
    from coverage_control.nn import CoverageEnvUtils

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.realpath(__file__))
ws_root = os.path.abspath(os.path.join(current_dir, "../../../")) 
data_root = os.path.join(ws_root, "data")

params_file = os.path.join(data_root, "params/coverage_control_params.toml")
learning_config_file = os.path.join(data_root, "params/learning_params.toml")
features_file = os.path.join(data_root, "features") 
robot_pos_file = os.path.join(data_root, "robots_positions")
model_dir = os.path.join(data_root, "lpac/models")

try:
    model_file = [f for f in os.listdir(model_dir) if f.endswith(".pt")][0]
    model_path = os.path.join(model_dir, model_file)
except:
    print("Error: No model found.")
    sys.exit(1)

def main():
    print(f"Model: {model_path}")

    # 1. Setup
    params = cc.Parameters(params_file)
    params.pNumRobots = 15 # Force match sample data
    
    world_idf = cc.WorldIDF(params, features_file)
    env = cc.CoverageSystem(params, world_idf, robot_pos_file)
    
    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_config = IOUtils.load_toml(learning_config_file)
    lpac_model = cc_nn.LPAC(learning_config).to(device)
    lpac_model.load_state_dict(torch.load(model_path, map_location=device))
    lpac_model.eval()

    if not hasattr(lpac_model, "actions_mean"):
        lpac_model.register_buffer("actions_mean", torch.zeros(1, 2).to(device))
        lpac_model.register_buffer("actions_std", torch.ones(1, 2).to(device))

    # 3. Simulate
    print("Running Simulation...")
    num_steps = 100
    trajectories = []
    costs = []
    
    use_comm_maps = learning_config['ModelConfig']['UseCommMaps']
    map_size = learning_config['CNNBackBone']['ImageSize']

    with torch.no_grad():
        for step in range(num_steps):
            data = CoverageEnvUtils.get_torch_geometric_data(
                env, params, True, use_comm_maps, map_size
            ).to(device)
            
            # Extract Positions
            if hasattr(data, 'pos'): pos = data.pos.cpu().numpy()
            elif hasattr(data, 'robot_positions'): pos = data.robot_positions.cpu().numpy()
            else: pos = data.x[:, :2].cpu().numpy()
            
            trajectories.append(pos)
            
            actions = lpac_model(data)
            actions = actions * lpac_model.actions_std + lpac_model.actions_mean
            env.StepActions(PointVector(actions.cpu().numpy()))
            costs.append(env.GetObjectiveValue())

    # 4. Plot
     # ... inside plotting section ...
    print("4. Plotting...")
    trajectories = np.array(trajectories)
    X, Y = trajectories[:, :, 0], trajectories[:, :, 1]
    world_map = env.GetWorldMap()
    map_h, map_w = world_map.shape

    # --- FIX: Center Robots on Map ---
    # Calculate where the robots start
    start_x_avg = np.mean(X[0])
    start_y_avg = np.mean(Y[0])
    
    # If they are far outside the map (e.g., > 1024), shift them to the center (512)
    if start_x_avg > map_w or start_y_avg > map_h:
        print(f"Robots detected outside map (Avg Pos: {start_x_avg:.1f}, {start_y_avg:.1f}). Shifting to center...")
        shift_x = (map_w / 2.0) - start_x_avg
        shift_y = (map_h / 2.0) - start_y_avg
        X = X + shift_x
        Y = Y + shift_y
    # ---------------------------------

    # Plot 1: Map
    plt.figure(figsize=(10, 10))
    plt.imshow(world_map, origin='lower', cmap='viridis', extent=[0, map_w, 0, map_h])
    
    for i in range(trajectories.shape[1]):
        plt.plot(X[:, i], Y[:, i], '-w', alpha=0.7)
        plt.plot(X[0, i], Y[0, i], '.r', markersize=8)
        plt.plot(X[-1, i], Y[-1, i], '*y', markersize=10)

    # Automatically zoom out if robots are outside map
    plt.xlim(min(0, np.min(X)-50), max(map_w, np.max(X)+50))
    plt.ylim(min(0, np.min(Y)-50), max(map_h, np.max(Y)+50))
    
    plt.title(f"Simulation: {trajectories.shape[1]} Robots")
    plt.savefig("simulation_map_final.png")
    print("Saved simulation_map_final.png")
    plt.close()

    # Plot 2: Cost
    plt.figure(figsize=(8, 4))
    plt.plot(costs)
    plt.title("Coverage Cost")
    plt.grid(True)
    plt.savefig("coverage_cost_final.png")
    print("Saved coverage_cost_final.png")
    plt.close()

if __name__ == "__main__":
    main()

