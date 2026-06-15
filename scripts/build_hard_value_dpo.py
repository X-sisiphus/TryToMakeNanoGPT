import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sft_data import load_sft_jsonl


FIELDS = ["station", "signal", "value", "unit"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def parse_fields(text):
    fields = {}

    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key in FIELDS:
            fields[key] = value

    return fields


def render_fields(fields):
    return (
        f"station: {fields['station']}\n"
        f"signal: {fields['signal']}\n"
        f"value: {fields['value']}\n"
        f"unit: {fields['unit']}"
    )


def values_in_input(text):
    return re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])", text)


def build_hard_rejected(rng, example):
    fields = parse_fields(example["output"])

    if any(field not in fields for field in FIELDS):
        raise ValueError(f"输出字段不完整: {example['output']}")

    candidates = [
        value
        for value in values_in_input(example["input"])
        if value != fields["value"]
    ]

    if len(candidates) == 0:
        return None

    rejectedFields = dict(fields)
    rejectedFields["value"] = rng.choice(candidates)
    return render_fields(rejectedFields)


def build_dpo_examples(examples, seed):
    rng = random.Random(seed)
    dpoExamples = []

    for example in examples:
        rejected = build_hard_rejected(rng, example)

        if rejected is None:
            continue

        dpoExamples.append(
            {
                "task": example.get("task", "field_extraction"),
                "preference_type": "hard_wrong_value_from_input",
                "instruction": example["instruction"],
                "input": example["input"],
                "chosen": example["output"],
                "rejected": rejected,
            }
        )

    return dpoExamples


def save_jsonl(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    examples = load_sft_jsonl(args.sft_path)
    dpoExamples = build_dpo_examples(examples, args.seed)
    save_jsonl(dpoExamples, args.out)

    counts = Counter(example["preference_type"] for example in dpoExamples)

    print(f"loaded {len(examples)} sft examples")
    print(f"saved {len(dpoExamples)} dpo examples to {args.out}")
    print(f"preference types: {dict(counts)}")
    print("preview:")

    for example in dpoExamples[:5]:
        print("-" * 80)
        print(example["input"])
        print("chosen:")
        print(example["chosen"])
        print("rejected:")
        print(example["rejected"])


if __name__ == "__main__":
    main()
