from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="out/ablation")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()
args = parse_args()

outPath = args.out or os.path.join(args.root, "summary.csv")

experiments = []

for name in sorted(os.listdir(args.root)):
    runDir = os.path.join(args.root, name)
    logPath = os.path.join(runDir, "log.csv")
    if os.path.isdir(runDir) and os.path.exists(logPath):
        experiments.append((name, logPath))

def read_log(logPath):
    rows = []
    with open(logPath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

summaryRows = []

for name, logPath in experiments:
    rows = read_log(logPath)
    if not rows:
        continue

    final = rows[-1]
    best = min(rows, key=lambda r: float(r["val_loss"]))

    summaryRows.append({
        "experiment": name,
        "final_step": int(final["step"]),
        "final_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "best_val_loss": float(best["val_loss"]),
        "best_step": int(best["step"]),
        "final_lr": float(final["lr"]),
        "final_tokens_per_sec": float(final.get("tokens_per_sec", 0.0)),
    })

fieldnames = [
    "experiment",
    "final_step",
    "final_train_loss",
    "final_val_loss",
    "best_val_loss",
    "best_step",
    "final_lr",
    "final_tokens_per_sec",
]

with open(outPath, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summaryRows)

print(f"saved summary to {outPath}")

for row in summaryRows:
    print(
        f"{row['experiment']:>10s} | "
        f"final val {row['final_val_loss']:.4f} | "
        f"best val {row['best_val_loss']:.4f} | "
        f"tok/s {row['final_tokens_per_sec']:.0f}"
    )
