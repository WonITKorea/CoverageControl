import time
import numpy as np
import matplotlib.pyplot as plt

from coverage_control.algorithms.controllers import ControllerCVT, CoverageEnvUtils
from coverage_control import _core

MAP_SIZE = 256
ROBOT_IDX = 0
FPS = 10
STEPS = 1000

def load_params():
    P = _core.Parameters
    for name in ["from_toml", "from_file", "load", "read", "load_from_file"]:
        if hasattr(P, name):
            try:
                return getattr(P, name)("params/coverage_control_params.toml")
            except Exception:
                pass
    p = P()
    for name in ["load", "read", "load_from_file", "loadFromFile", "readFromFile", "from_toml"]:
        if hasattr(p, name):
            try:
                getattr(p, name)("params/coverage_control_params.toml")
                return p
            except Exception:
                pass
    return p

def step_env(env, actions):
    for name in ["StepActions", "StepAction", "StepControl"]:
        if hasattr(env, name):
            return getattr(env, name)(actions)
    raise RuntimeError("No StepActions/StepAction/StepControl found.")

def norm01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def sym_scale_from_maxabs(x: np.ndarray, eps: float = 1e-6) -> float:
    """comm 같이 sparse한 signed 맵은 percentile이 0이 될 수 있음 → max(|x|)로 스케일 잡는 게 안전"""
    x = x.astype(np.float32, copy=False)
    return max(float(np.max(np.abs(x))), eps)

def main():
    params = load_params()
    env = _core.CoverageSystem(params)
    ctrl = ControllerCVT({"Name":"ClairvoyantCVT","Algorithm":"ClairvoyantCVT"}, params, env)

    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    axs = axs.flatten()

    im_local = axs[0].imshow(np.zeros((MAP_SIZE, MAP_SIZE), np.float32),
                             vmin=0.0, vmax=1.0, interpolation="nearest")
    axs[0].set_title("local(IDF) [log+norm]")
    axs[0].axis("off")

    im_obs = axs[1].imshow(np.zeros((MAP_SIZE, MAP_SIZE), np.float32),
                           vmin=0.0, vmax=1.0, interpolation="nearest")
    axs[1].set_title("obstacles [norm]")
    axs[1].axis("off")

    im_cx = axs[2].imshow(np.zeros((MAP_SIZE, MAP_SIZE), np.float32),
                          vmin=-1.0, vmax=1.0, interpolation="nearest", cmap="coolwarm")
    axs[2].set_title("comm_x [signed]")
    axs[2].axis("off")

    im_cy = axs[3].imshow(np.zeros((MAP_SIZE, MAP_SIZE), np.float32),
                          vmin=-1.0, vmax=1.0, interpolation="nearest", cmap="coolwarm")
    axs[3].set_title("comm_y [signed]")
    axs[3].axis("off")

    suptxt = fig.suptitle("")

    for t in range(STEPS):
        local = CoverageEnvUtils.get_raw_local_maps(env, params).cpu().numpy()
        obst  = CoverageEnvUtils.get_raw_obstacle_maps(env, params).cpu().numpy()
        comm  = CoverageEnvUtils.get_communication_maps(env, params, MAP_SIZE).cpu().numpy()

        loc = np.log1p(local[ROBOT_IDX])
        obs = obst[ROBOT_IDX]
        cx  = comm[ROBOT_IDX, 0]
        cy  = comm[ROBOT_IDX, 1]

        # local/obstacles: 0~1 정규화
        loc_v = norm01(loc)
        obs_v = norm01(obs)

        # comm: sparse 대비를 위해 maxabs 기반으로 ±스케일
        c = max(sym_scale_from_maxabs(cx), sym_scale_from_maxabs(cy))
        cx_v = np.clip(cx / c, -1.0, 1.0)
        cy_v = np.clip(cy / c, -1.0, 1.0)

        im_local.set_data(loc_v)
        im_obs.set_data(obs_v)
        im_cx.set_data(cx_v)
        im_cy.set_data(cy_v)

        # autoscale 방지
        im_local.set_clim(0.0, 1.0)
        im_obs.set_clim(0.0, 1.0)
        im_cx.set_clim(-1.0, 1.0)
        im_cy.set_clim(-1.0, 1.0)

        if t % 50 == 0:
            cost = float(env.GetObjectiveValue())
            print(
                f"t={t:04d} cost={cost:.3e} | "
                f"local[min,max,std]=({loc.min():.3e},{loc.max():.3e},{loc.std():.3e}) "
                f"obs[min,max,std]=({obs.min():.3e},{obs.max():.3e},{obs.std():.3e}) "
                f"cx[min,max,std]=({cx.min():.3e},{cx.max():.3e},{cx.std():.3e}) "
                f"cy[min,max,std]=({cy.min():.3e},{cy.max():.3e},{cy.std():.3e}) "
                f"c={c:.3e}",
                flush=True
            )
            suptxt.set_text(f"t={t:04d}  robot={ROBOT_IDX}  cost={cost:.3e}")
        else:
            suptxt.set_text(f"t={t:04d}  robot={ROBOT_IDX}")

        fig.canvas.draw()
        fig.canvas.flush_events()

        ctrl.alg.ComputeActions()
        actions = ctrl.alg.GetActions()
        step_env(env, actions)

        time.sleep(1.0 / FPS)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
