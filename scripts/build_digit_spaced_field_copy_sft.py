import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_field_copy_sft import COPY_TEMPLATES, SIGNALS, STATIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/astro_sft_field_copy_spaced_500.jsonl")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def space_value(value):
    return " ".join(list(str(value)))


def build_example(station, signal, value, unit, template):
    spacedValue = space_value(value)

    inputText = template.format(
        station=station,
        signal=signal,
        value=spacedValue,
        unit=unit,
    )

    outputText = (
        f"station: {station}\n"
        f"signal: {signal}\n"
        f"value: {spacedValue}\n"
        f"unit: {unit}"
    )

    return {
        "task": "field_extraction",
        "instruction": "Extract the station, signal, value, and unit from the text.",
        "input": inputText,
        "output": outputText,
    }


def build_examples(numExamples, seed):
    rng = random.Random(seed)
    examples = []
    seen = set()

    maxCombinations = (
        len(STATIONS)
        * len(COPY_TEMPLATES)
        * sum(len(values) for _, _, values in SIGNALS)
    )

    if numExamples > maxCombinations:
        raise ValueError(
            f"num_examples={numExamples} 超过可生成的不重复组合数 {maxCombinations}"
        )

    while len(examples) < numExamples:
        station = rng.choice(STATIONS)
        signal, unit, values = rng.choice(SIGNALS)
        value = rng.choice(values)
        template = rng.choice(COPY_TEMPLATES)

        key = (station, signal, value, unit, template)
        if key in seen:
            continue

        seen.add(key)
        examples.append(
            build_example(
                station=station,
                signal=signal,
                value=value,
                unit=unit,
                template=template,
            )
        )

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
