import os
import numpy as np
from coverage_control.algorithms.controllers import ControllerCVT, CoverageEnvUtils
from coverage_control import _core

OUT_DIR = "data_cvt_4ch"
EPISODES = 50
HORIZON = 200
MAP_SIZE = 256

def load_params(toml_path: str):
    P = _core.Parameters
    for name in ["from_toml","from_file","load","read","load_from_file"]:
        if hasattr(P, name):
            try:
                return getattr(P, name)(toml_path)
            except Exception:
                pass
    p = P()
    for name in ["load","read","load_from_file","loadFromFile","readFromFile","from_toml"]:
        if hasattr(p, name):
            try:
                getattr(p, name)(toml_path)
                return p
            except Exception:
                pass
    return p

def step_env(env, actions):
    # actions는 core 타입 그대로 넣는 게 가장 안전
    for name in ["StepActions", "StepAction", "StepControl"]:
        if hasattr(env, name):
            return getattr(env, name)(actions)
    raise RuntimeError("No StepActions/StepAction/StepControl found.")

def get_obs_4ch(env, params):
    # (N,H,W)
    local = CoverageEnvUtils.get_raw_local_maps(env, params)
    obst  = CoverageEnvUtils.get_raw_obstacle_maps(env, params)
    # (N,2,H,W)
    comm  = CoverageEnvUtils.get_communication_maps(env, params, MAP_SIZE)

    # torch -> numpy
    local_np = local.cpu().numpy().astype(np.float32)   # (N,H,W)
    obst_np  = obst.cpu().numpy().astype(np.float32)    # (N,H,W)
    comm_np  = comm.cpu().numpy().astype(np.float32)    # (N,2,H,W)

    # stack -> (N,4,H,W)
    obs4 = np.concatenate(
        [
            local_np[:, None, :, :],      # (N,1,H,W)
            obst_np[:, None, :, :],       # (N,1,H,W)
            comm_np                       # (N,2,H,W)
        ],
        axis=1
    )
    return obs4

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    params = load_params("params/coverage_control_params.toml")

    # N은 환경에서 받아도 되지만, 지금은 32로 고정되어 있어 보임
    # 첫 env로 확인
    env0 = _core.CoverageSystem(params)
    N = int(env0.GetNumRobots())
    print("N robots:", N)

    for ep in range(EPISODES):
        env = _core.CoverageSystem(params)
        ctrl = ControllerCVT({"Name":"ClairvoyantCVT","Algorithm":"ClairvoyantCVT"}, params, env)

        obs_buf  = np.empty((HORIZON, N, 4, MAP_SIZE, MAP_SIZE), dtype=np.float32)
        act_buf  = np.empty((HORIZON, N, 2), dtype=np.float32)
        cost_buf = np.empty((HORIZON,), dtype=np.float32)

        for t in range(HORIZON):
            # 4채널 관측
            obs_buf[t] = get_obs_4ch(env, params)

            # expert actions
            ctrl.alg.ComputeActions()
            actions = ctrl.alg.GetActions()
            act_t = CoverageEnvUtils.to_tensor(actions)  # (N,2)
            act_buf[t] = act_t.cpu().numpy().astype(np.float32)

            # cost
            cost_buf[t] = float(env.GetObjectiveValue())

            # apply
            step_env(env, actions)

            if t % 50 == 0:
                print(f"ep={ep:03d} t={t:03d} cost={cost_buf[t]:.6f}", flush=True)

        out_path = os.path.join(OUT_DIR, f"cvt4_ep{ep:04d}.npz")
        np.savez_compressed(out_path, obs=obs_buf, actions=act_buf, cost=cost_buf,
                            meta=np.array([MAP_SIZE, N], dtype=np.int32))
        print("saved:", out_path, flush=True)

    print("done", flush=True)

if __name__ == "__main__":
    main()
