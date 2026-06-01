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

def run(cmd):
    print("\n" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

args = parse_args()

ablationCmd = [
    sys.executable,
    "run_ablation.py",
    "--out-dir", args.out_dir,
    "--max-iters", str(args.max_iters),
    "--eval-interval", str(args.eval_interval),
    "--eval-iters", str(args.eval_iters),
]

if args.use_mps:
    ablationCmd.append("--use-mps")

run(ablationCmd)

summaryPath = f"{args.out_dir}/summary.csv"

run([
    sys.executable,
    "summarize_ablation.py",
    "--root", args.out_dir,
    "--out", summaryPath,
])

run([
    sys.executable,
    "plot_ablation.py",
    "--root", args.out_dir,
])

run([
    sys.executable,
    "plot_ablation_summary.py",
    "--summary", summaryPath,
])