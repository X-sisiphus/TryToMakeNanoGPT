import argparse
import json
import os
import random
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def load_jsonl(path):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    return examples


def save_jsonl(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    examples = []
    sourceCounts = {}

    for path in args.inputs:
        loaded = load_jsonl(path)
        sourceCounts[path] = len(loaded)
        examples.extend(loaded)

    if args.shuffle:
        rng.shuffle(examples)

    save_jsonl(examples, args.out)

    taskCounts = Counter(example.get("task", "unknown") for example in examples)
    print(f"saved {len(examples)} examples to {args.out}")
    print(f"sources: {sourceCounts}")
    print(f"tasks: {dict(taskCounts)}")


if __name__ == "__main__":
    main()
