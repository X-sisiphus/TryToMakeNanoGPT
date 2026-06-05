from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="out/ablation")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--use-mps", action="store_true")
    return parser.parse_args()
args = parse_args()

baseArgs = [
    sys.executable,
    str(ROOT / "train.py"),
    "--max-iters", str(args.max_iters),
    "--eval-interval", str(args.eval_interval),
    "--eval-iters", str(args.eval_iters),
    "--batch-size", "16",
    "--block-size", "64",
    "--n-embd", "128",
    "--n-layer", "2",
    "--num-heads", "4",
    "--dropout", "0.0",
]

experiments = [
    (
        "baseline",
        ["--norm", "layernorm", "--ffn", "gelu", "--no-rope", "--num-kv-heads", "4", "--no-flash"],
    ),
    (
        "rmsnorm",
        ["--norm", "rmsnorm", "--ffn", "gelu", "--no-rope", "--num-kv-heads", "4", "--no-flash"],
    ),
    (
        "swiglu",
        ["--norm", "rmsnorm", "--ffn", "swiglu", "--no-rope", "--num-kv-heads", "4", "--no-flash"],
    ),
    (
        "rope",
        ["--norm", "rmsnorm", "--ffn", "swiglu", "--use-rope", "--num-kv-heads", "4", "--no-flash"],
    ),
    (
        "gqa",
        ["--norm", "rmsnorm", "--ffn", "swiglu", "--use-rope", "--num-kv-heads", "2", "--no-flash"],
    ),
    (
        "sdpa",
        ["--norm", "rmsnorm", "--ffn", "swiglu", "--use-rope", "--num-kv-heads", "2", "--use-flash"],
    ),
]

for name, extraArgs in experiments:
    runDir = f"{args.out_dir}/{name}"
    cmd = baseArgs + extraArgs + ["--out-dir", runDir]
    print(f"\n=== running {name} ===", flush=True)
    print(" ".join(cmd), flush=True)

    env = None
    if args.use_mps:
        import os
        env = os.environ.copy()
        env["USE_MPS"] = "1"

    subprocess.run(cmd, check=True, env=env)
