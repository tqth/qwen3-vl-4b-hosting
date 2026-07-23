"""
Launcher dung chung cho tat ca MLLM servers.

Cach dung:
    python start_server.py --model qwen3vl        # mac dinh port 8000
    python start_server.py --model llava
    python start_server.py --model qwen3vl --port 8001

Chay 4-bit (tiet kiem VRAM, chia se GPU voi tien trinh khac):
    LOAD_IN_4BIT=1 python start_server.py --model qwen3vl

Doi model_id khac trong cung ho Qwen3-VL (vd ban 8B):
    MODEL_ID=Qwen/Qwen3-VL-8B-Instruct python start_server.py --model qwen3vl

Them model moi:
    1. Tao file servers/<ten>_server.py
    2. Them entry vao MODEL_REGISTRY ben duoi
"""
import subprocess, time, requests, os, argparse

# -- Registry: them model moi vao day ------------------------------------
MODEL_REGISTRY = {
    "qwen3vl": {
        "script": "servers/qwen3vl_server.py",
        "pip":    ["transformers>=4.57.0", "accelerate"],
        "log":    "/kaggle/working/qwen3vl_server.log",
    },
    "llava": {
        "script": "servers/llava_server.py",
        "pip":    [],
        "log":    "/kaggle/working/llava_server.log",
    },
    "qwen3": {
        "script": "servers/qwen3_server.py",
        "pip": [
            "transformers>=4.57.0",
            "accelerate",
        ],
        "log": "/kaggle/working/qwen3_server.log",
    },
    # Vi du them InternVL sau nay:
    # "internvl": {
    #     "script": "servers/internvl_server.py",
    #     "pip":    ["timm"],
    #     "log":    "/kaggle/working/internvl_server.log",
    # },
}
# --------------------------------------------------------------------------

HF_CACHE_DIR = "/kaggle/working/hf_cache"
REPO_DIR     = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys(),
                        help="Ten model muon host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port cho Flask server (mac dinh: 8000)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Thoi gian cho toi da (giay) de server ready")
    return parser.parse_args()


def install_deps(packages: list):
    if not packages:
        return
    pkgs = " ".join(f'"{p}"' for p in packages)
    print(f"Cai dependencies: {pkgs}")
    os.system(f"pip install -q {pkgs}")


def start_server(script_path: str, port: int, log_path: str) -> subprocess.Popen:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    env = {
        **os.environ,
        "HF_HOME": HF_CACHE_DIR,
        "TRANSFORMERS_CACHE": HF_CACHE_DIR,
        "SERVER_PORT": str(port),
    }
    proc = subprocess.Popen(
        ["python", script_path],
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def wait_until_ready(port: int, timeout: int, interval: int = 15):
    elapsed = 0
    while elapsed < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval
        print(f"   ...{elapsed}s", end="\r")
    return False


def main():
    args = parse_args()
    cfg  = MODEL_REGISTRY[args.model]
    script_path = os.path.join(REPO_DIR, cfg["script"])

    if not os.path.exists(script_path):
        print(f"[ERROR] Khong tim thay script: {script_path}")
        return

    print(f"\n{'='*50}")
    print(f"  Model  : {args.model}")
    print(f"  Port   : {args.port}")
    print(f"  Script : {script_path}")
    print(f"  Log    : {cfg['log']}")
    print(f"{'='*50}\n")

    install_deps(cfg["pip"])

    proc = start_server(script_path, args.port, cfg["log"])
    print(f"Server dang khoi dong (PID={proc.pid})...")

    if wait_until_ready(args.port, args.timeout):
        print(f"\n✓ {args.model} server san sang tai http://localhost:{args.port}")
    else:
        print(f"\n✗ Timeout sau {args.timeout}s! Log cuoi:")
        os.system(f"tail -30 {cfg['log']}")


if __name__ == "__main__":
    main()
