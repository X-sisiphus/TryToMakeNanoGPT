import argparse
import json
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sft/astro_sft_small.jsonl")
    parser.add_argument("--out", type=str, default="data/sft/astro_sft_field.jsonl")
    parser.add_argument("--task", type=str, default="field_extraction")
    return parser.parse_args()


def load_jsonl(path):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    return examples


def filter_by_task(examples, task):
    return [
        example
        for example in examples
        if example.get("task") == task
    ]


def save_jsonl(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    examples = load_jsonl(args.input)
    filtered = filter_by_task(examples, args.task)

    if len(filtered) == 0:
        raise ValueError(f"没有找到 task={args.task} 的样本")

    save_jsonl(filtered, args.out)

    counts = Counter(example.get("task", "unknown") for example in filtered)

    print(f"loaded examples: {len(examples)}")
    print(f"saved examples: {len(filtered)}")
    print(f"output: {args.out}")
    print(f"tasks: {dict(counts)}")


if __name__ == "__main__":
    main()
