from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="out/ablation")
    return parser.parse_args()
args = parse_args()

experiments = []

for name in sorted(os.listdir(args.root)):
    runDir = os.path.join(args.root, name)
    logPath = os.path.join(runDir, "log.csv")
    if os.path.isdir(runDir) and os.path.exists(logPath):
        experiments.append((name, runDir, logPath))

if not experiments:
    raise SystemExit(f"没有在 {args.root} 下找到包含 log.csv 的实验目录")

for name, runDir, logPath in experiments:
    outPath = os.path.join(runDir, "loss.png")
    cmd = [
        sys.executable,
        str(ROOT / "tools/plots/plot_log.py"),
        "--log", logPath,
        "--out", outPath,
    ]

    print(f"plotting {name} -> {outPath}", flush=True)
    subprocess.run(cmd, check=True)
