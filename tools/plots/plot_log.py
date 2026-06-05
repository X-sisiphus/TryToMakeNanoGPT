from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()
args = parse_args()
steps = []
trainLosses = []
valLosses = []
lrs = []

with open(args.log, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        trainLosses.append(float(row["train_loss"]))
        valLosses.append(float(row["val_loss"]))
        if "lr" in row:
            lrs.append(float(row["lr"]))

plt.figure(figsize=(8, 5))
plt.plot(steps, trainLosses, label="train loss")
plt.plot(steps, valLosses, label="val loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

if args.out is not None:
    plt.savefig(args.out, dpi=200)
    print(f"saved figure to {args.out}")
else:
    plt.show()