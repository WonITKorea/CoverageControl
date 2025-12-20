import glob
from coverage_control.algorithms.controllers import ControllerCVT, CoverageEnvUtils
from coverage_control import _core

def pick_toml():
    return "params/coverage_control_params.toml"

def load_params(toml_path: str):
    P = _core.Parameters
    for name in ["from_toml", "from_file", "load", "read", "load_from_file"]:
        if hasattr(P, name):
            try:
                return getattr(P, name)(toml_path)
            except Exception:
                pass
    p = P()
    for name in ["load", "read", "load_from_file", "loadFromFile", "readFromFile", "from_toml"]:
        if hasattr(p, name):
            try:
                getattr(p, name)(toml_path)
                return p
            except Exception:
                pass
    return p

def main():
    params = load_params(pick_toml())
    env = _core.CoverageSystem(params)

    ctrl = ControllerCVT({"Name":"ClairvoyantCVT","Algorithm":"ClairvoyantCVT"}, params, env)

    u = ctrl.step(env)
    print("type(u):", type(u))
    print("len(u):", len(u))
    for i, item in enumerate(u):
        print(f"\n[{i}] type={type(item)}")
        # pybind vector류는 repr이 도움됨
        try:
            print("repr:", repr(item)[:300])
        except Exception:
            pass
        # tensor 변환 시도
        try:
            import torch
            t = CoverageEnvUtils.to_tensor(item) if hasattr(CoverageEnvUtils, "to_tensor") else None
            if t is not None:
                print("as tensor shape:", tuple(t.shape), "dtype:", t.dtype)
        except Exception as e:
            print("to_tensor failed:", e)

if __name__ == "__main__":
    main()
