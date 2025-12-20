import os
import time
import tempfile
import numpy as np
import torch

from coverage_control.algorithms.controllers import ControllerCVT, CoverageEnvUtils
from coverage_control import _core

# -----------------------------
# Atomic NPZ save
# -----------------------------
def atomic_save_npz(path: str, **arrays):
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=out_dir)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

# -----------------------------
# Utils
# -----------------------------
def ensure_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def np_actions_to_pointvector(actions_np: np.ndarray):
    """
    (N,2) numpy -> _core.PointVector<_core.Point2>
    """
    pv = _core.PointVector()
    for i in range(actions_np.shape[0]):
        pv.append(
            _core.Point2(
                float(actions_np[i, 0]),
                float(actions_np[i, 1]),
            )
        )
    return pv

def step_env(env, actions_np):
    pv = np_actions_to_pointvector(actions_np)
    return env.StepActions(pv)

def load_params(toml_path="params/coverage_control_params.toml"):
    P = _core.Parameters
    for name in ["from_toml", "from_file", "load", "read", "load_from_file"]:
        if hasattr(P, name):
            try:
                return getattr(P, name)(toml_path)
            except Exception:
                pass
    p = P()
    for name in ["load", "read", "load_from_file", "from_toml"]:
        if hasattr(p, name):
            try:
                getattr(p, name)(toml_path)
                return p
            except Exception:
                pass
    return p

# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR = "data_cvt"
    EPISODES = 50
    STEPS = 200
    MAP_SIZE = 256
    SAVE_4CH = True

    os.makedirs(OUT_DIR, exist_ok=True)

    params = load_params()
    env = _core.CoverageSystem(params)

    ctrl = ControllerCVT(
        {"Name": "ClairvoyantCVT", "Algorithm": "ClairvoyantCVT"},
        params,
        env,
    )

    for ep in range(EPISODES):
        maps_buf, acts_buf, costs_buf = [], [], []
        t0 = time.time()

        for t in range(STEPS):
            local = ensure_numpy(CoverageEnvUtils.get_raw_local_maps(env, params))
            obst  = ensure_numpy(CoverageEnvUtils.get_raw_obstacle_maps(env, params))
            comm  = ensure_numpy(CoverageEnvUtils.get_communication_maps(env, params, MAP_SIZE))

            if SAVE_4CH:
                m = np.stack([local, obst, comm[:, 0], comm[:, 1]], axis=1)
            else:
                m = local

            ctrl.alg.ComputeActions()
            actions = ensure_numpy(ctrl.alg.GetActions()).astype(np.float32)

            cost = float(env.GetObjectiveValue())

            maps_buf.append(m)
            acts_buf.append(actions)
            costs_buf.append(cost)

            step_env(env, actions)

            if t % 50 == 0:
                print(
                    f"ep={ep:03d} t={t:03d} cost={cost:.3e} "
                    f"maps={m.shape} act={actions.shape}",
                    flush=True,
                )

        maps_np = np.stack(maps_buf, axis=0)
        acts_np = np.stack(acts_buf, axis=0)
        costs_np = np.asarray(costs_buf, dtype=np.float32)

        out_path = os.path.join(OUT_DIR, f"cvt_ep{ep+1:04d}.npz")
        atomic_save_npz(out_path, maps=maps_np, actions=acts_np, costs=costs_np)

        print(
            f"saved: {out_path} "
            f"samples={maps_np.shape[0]*maps_np.shape[1]} "
            f"time={time.time()-t0:.1f}s",
            flush=True,
        )

        # verify
        d = np.load(out_path)
        _ = d["maps"].shape
        _ = d["actions"].shape

    print("DONE.")

if __name__ == "__main__":
    main()
