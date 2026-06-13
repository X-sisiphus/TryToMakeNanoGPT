from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import os
from collections import defaultdict

import torch
import tiktoken

from dpo_data import encode_dpo_example, load_dpo_jsonl, pad_dpo_batch
from model import BigramLanguageModel, GPTConfig
from train_dpo import sequence_logps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dpo-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--encoding", type=str, default="gpt2")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_model(checkpointPath, blockSize, device):
    checkpoint = torch.load(
        checkpointPath,
        map_location="cpu",
        weights_only=False,
    )
    config = GPTConfig(**checkpoint["config"])
    config.blockSize = blockSize

    model = BigramLanguageModel(
        config.vocabSize,
        config.blockSize,
        config=config,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def load_encoded_examples(path, enc, blockSize):
    examples = load_dpo_jsonl(path)
    encoded = []

    for example in examples:
        item = encode_dpo_example(example, enc)
        item["preference_type"] = example.get("preference_type", "unknown")

        if (
            item["chosen_tokens"] <= blockSize
            and item["rejected_tokens"] <= blockSize
        ):
            encoded.append(item)

    return encoded


def batch_items(items, batchSize):
    for start in range(0, len(items), batchSize):
        yield items[start:start + batchSize]


@torch.no_grad()
def evaluate_batch(model, items, device):
    batch = pad_dpo_batch(items)
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    chosenLogps = sequence_logps(
        model,
        batch["chosen_input_ids"],
        batch["chosen_answer_mask"],
    )
    rejectedLogps = sequence_logps(
        model,
        batch["rejected_input_ids"],
        batch["rejected_answer_mask"],
    )

    rows = []

    for item, chosenLogp, rejectedLogp in zip(items, chosenLogps, rejectedLogps):
        margin = float(chosenLogp.item() - rejectedLogp.item())
        rows.append(
            {
                "preference_type": item["preference_type"],
                "chosen_logp": float(chosenLogp.item()),
                "rejected_logp": float(rejectedLogp.item()),
                "margin": margin,
                "correct": margin > 0,
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            }
        )

    return rows


def summarize(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[row["preference_type"]].append(row)

    summary = {}
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    avgMargin = sum(row["margin"] for row in rows) / total

    summary["all"] = {
        "total": total,
        "accuracy": correct / total,
        "avg_margin": avgMargin,
    }

    for preferenceType, groupRows in groups.items():
        groupTotal = len(groupRows)
        groupCorrect = sum(row["correct"] for row in groupRows)
        groupAvgMargin = sum(row["margin"] for row in groupRows) / groupTotal

        summary[preferenceType] = {
            "total": groupTotal,
            "accuracy": groupCorrect / groupTotal,
            "avg_margin": groupAvgMargin,
        }

    return summary


def write_outputs(args, rows, summary):
    os.makedirs(args.out_dir, exist_ok=True)

    csvPath = os.path.join(args.out_dir, "preference_results.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    fieldnames = [
        "preference_type",
        "chosen_logp",
        "rejected_logp",
        "margin",
        "correct",
        "prompt",
        "chosen",
        "rejected",
    ]

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# DPO Preference Evaluation\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n")
        f.write(f"DPO data: `{args.dpo_path}`\n\n")

        f.write("## Summary\n\n")
        for preferenceType, item in sorted(summary.items()):
            f.write(
                f"- {preferenceType}: "
                f"{item['accuracy']:.2%}, "
                f"avg margin {item['avg_margin']:.4f}, "
                f"total {item['total']}\n"
            )

        f.write("\n## Wrong Examples\n\n")
        wrongRows = [
            row for row in rows
            if not row["correct"]
        ]

        for row in wrongRows[:10]:
            f.write("### Example\n\n")
            f.write(f"Type: `{row['preference_type']}`\n\n")
            f.write("Prompt:\n\n```text\n")
            f.write(row["prompt"])
            f.write("\n```\n\n")
            f.write("Chosen:\n\n```text\n")
            f.write(row["chosen"])
            f.write("\n```\n\n")
            f.write("Rejected:\n\n```text\n")
            f.write(row["rejected"])
            f.write("\n```\n\n")
            f.write(f"Margin: `{row['margin']:.4f}`\n\n")

    print(f"saved csv to {csvPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()

    useMps = os.environ.get("USE_MPS") == "1"
    device = "mps" if torch.backends.mps.is_available() and useMps else "cpu"
    print(f"using device: {device}", flush=True)

    enc = tiktoken.get_encoding(args.encoding)
    model = load_model(args.checkpoint, args.block_size, device)
    examples = load_encoded_examples(args.dpo_path, enc, args.block_size)

    rows = []
    for items in batch_items(examples, args.batch_size):
        rows.extend(evaluate_batch(model, items, device))

    summary = summarize(rows)
    write_outputs(args, rows, summary)


if __name__ == "__main__":
    main()
