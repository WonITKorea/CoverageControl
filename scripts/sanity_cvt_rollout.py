import glob
import coverage_control
from coverage_control.algorithms.controllers import ControllerCVT, CoverageEnvUtils
from coverage_control import _core

def pick_toml():
    # 메인으로 보이는 파일 우선
    if glob.glob("params/coverage_control_params.toml"):
        return "params/coverage_control_params.toml"
    # fallback: 아무 toml이나
    cands = sorted(glob.glob("params/*.toml"))
    return cands[0] if cands else None

def load_params(toml_path: str):
    P = _core.Parameters

    # classmethod / staticmethod 형태 우선
    for name in ["from_toml", "from_file", "load", "read", "load_from_file"]:
        if hasattr(P, name):
            fn = getattr(P, name)
            try:
                return fn(toml_path)
            except Exception:
                pass

    # instance 메서드 형태
    p = P()
    for name in ["load", "read", "load_from_file", "loadFromFile", "readFromFile", "from_toml"]:
        if hasattr(p, name):
            fn = getattr(p, name)
            try:
                fn(toml_path)
                return p
            except Exception:
                pass

    # 마지막: 그냥 생성만 반환
    return p

def main():
    toml_path = pick_toml()
    print("coverage_control:", coverage_control.__file__)
    print("Using TOML:", toml_path)

    params = load_params(toml_path)
    env = _core.CoverageSystem(params)  # 이건 이미 성공했었음

    # CVT expert 컨트롤러 구성
    cvt_config = {"Name": "ClairvoyantCVT", "Algorithm": "ClairvoyantCVT"}
    ctrl = ControllerCVT(cvt_config, params, env)

    # 여기서 핵심: ControllerCVT.step(env) 형태
    # step이 없거나 다른 이름이면 아래 hint로 잡아낼 것
    if not hasattr(ctrl, "step"):
        print("[error] ControllerCVT has no .step. Available:", [x for x in dir(ctrl) if "step" in x.lower() or "act" in x.lower() or "comp" in x.lower()])
        return

    # env 쪽에서 action 적용 메서드가 뭔지 모르니 후보로 찾자
    env_step_candidates = [x for x in dir(env) if any(k in x.lower() for k in ["step","update","apply","move","integrate","advance"])]
    cost_candidates = [x for x in dir(env) if "cost" in x.lower()]

    print("env step candidates:", env_step_candidates[:30])
    print("env cost candidates:", cost_candidates[:30])

    def try_env_step(u):
        # 흔한 이름 순서대로 시도
        for name in ["step","Step","update","Update","apply_control","ApplyControl","advance","Advance"]:
            if hasattr(env, name):
                fn = getattr(env, name)
                if callable(fn):
                    try:
                        return fn(u)
                    except TypeError:
                        # 시그니처가 (u, params) 같은 변종일 수도
                        try:
                            return fn(u, params)
                        except Exception:
                            pass
                    except Exception:
                        pass
        return None

    def try_cost():
        for name in ["get_cost","cost","coverage_cost","getCoverageCost","get_coverage_cost","evaluate_cost"]:
            if hasattr(env, name):
                try:
                    return float(getattr(env, name)())
                except Exception:
                    pass
        return None

    print("\n=== sanity rollout (ClairvoyantCVT) ===")
    for t in range(30):
        # action 계산: ctrl.step(env) 가 action을 반환하는 형태
        u = ctrl.step(env)

        out = try_env_step(u)  # env에 action 적용(있으면)
        cost = try_cost()

        if t % 5 == 0:
            msg = f"t={t:03d}"
            if cost is not None:
                msg += f" cost={cost:.6f}"
            msg += f" u_type={type(u)}"
            print(msg)

        # 관측맵 확인 (논문 연결 포인트)
        if t == 0:
            try:
                raw = CoverageEnvUtils.get_raw_local_maps(env, params)
                print("raw_local_maps shape:", tuple(raw.shape))
            except Exception as e:
                print("[warn] get_raw_local_maps failed:", e)

    print("\nOK: rollout loop finished")

if __name__ == "__main__":
    main()
