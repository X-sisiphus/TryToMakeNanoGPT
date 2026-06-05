import argparse
import json
import os
import random
from collections import Counter


VALUES = [
    -12.0,
    -10.5,
    -8.5,
    -6.2,
    -4.4,
    -3.2,
    -2.5,
    -0.8,
    0.4,
    0.8,
    1.2,
    1.5,
    2.2,
    2.4,
    3.6,
    4.7,
    5.6,
    5.8,
    6.8,
    8.0,
    9.1,
    11.3,
    12.0,
    12.5,
    18.5,
    24.0,
    25.0,
    33.5,
    38.5,
    52.2,
    71.4,
]

TEMPLATES = [
    "value={value}",
    "value = {value}",
    "measurement value={value}",
    "reported value is {value}",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/value_copy_500.jsonl")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def build_example(value, template):
    inputText = template.format(value=value)
    outputText = f"value: {value}"

    return {
        "task": "value_copy",
        "instruction": "Extract the value from the text.",
        "input": inputText,
        "output": outputText,
    }


def build_examples(numExamples, seed):
    rng = random.Random(seed)
    examples = []

    for _ in range(numExamples):
        value = rng.choice(VALUES)
        template = rng.choice(TEMPLATES)
        examples.append(build_example(value, template))

    return examples


def save_jsonl(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    examples = build_examples(args.num_examples, args.seed)
    save_jsonl(examples, args.out)

    counts = Counter(example["task"] for example in examples)

    print(f"saved {len(examples)} examples to {args.out}")
    print(f"task counts: {dict(counts)}")
    print("preview:")
    for example in examples[:3]:
        print("-" * 80)
        print(example["input"])
        print(example["output"])


if __name__ == "__main__":
    main()
