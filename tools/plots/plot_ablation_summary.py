from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import os

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


args = parse_args()
outPath = args.out or os.path.join(os.path.dirname(args.summary), "summary.png")

experiments = []
bestValLosses = []
tokensPerSec = []

with open(args.summary, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        experiments.append(row["experiment"])
        bestValLosses.append(float(row["best_val_loss"]))
        tokensPerSec.append(float(row["final_tokens_per_sec"]))

if not experiments:
    raise SystemExit(f"{args.summary} 中没有可绘制的实验结果")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(experiments, bestValLosses)
axes[0].set_title("Best validation loss")
axes[0].set_ylabel("loss")
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(experiments, tokensPerSec)
axes[1].set_title("Final tokens/sec")
axes[1].set_ylabel("tokens/sec")
axes[1].tick_params(axis="x", rotation=30)

fig.tight_layout()
plt.savefig(outPath, dpi=200)
print(f"saved figure to {outPath}")
