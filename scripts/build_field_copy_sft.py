import argparse
import json
import os
import random
from collections import Counter


STATIONS = [
    "BJFS",
    "WETTZELL",
    "KOKEE",
    "NYALES20",
    "HOBART12",
    "ONSA",
    "TSKB",
    "GOLD",
    "MATE",
    "YEBES40M",
]

SIGNALS = [
    ("vertical velocity", "mm/yr", [-8.5, -3.2, 0.8, 2.4, 5.6]),
    ("east displacement", "mm", [-12.0, -6.2, 1.5, 4.7, 9.1]),
    ("north displacement", "mm", [-10.5, -4.4, 2.2, 6.8, 11.3]),
    ("clock bias", "ns", [-2.5, -0.8, 0.4, 1.2, 3.6]),
    ("zenith wet delay", "mm", [12.5, 24.0, 38.5, 52.2, 71.4]),
    ("tropospheric delay", "ps", [8.0, 12.0, 18.5, 25.0, 33.5]),
    ("seasonal amplitude", "mm", [1.2, 2.4, 3.6, 5.8, 8.1]),
]

COPY_TEMPLATES = [
    "station={station}; signal={signal}; value={value}; unit={unit}",
    "signal={signal}; station={station}; unit={unit}; value={value}",
    "value={value}; unit={unit}; station={station}; signal={signal}",
    "unit={unit}; value={value}; signal={signal}; station={station}",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/astro_sft_field_copy_500.jsonl")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def build_example(station, signal, value, unit, template):
    inputText = template.format(
        station=station,
        signal=signal,
        value=value,
        unit=unit,
    )

    outputText = (
        f"station: {station}\n"
        f"signal: {signal}\n"
        f"value: {value}\n"
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
                station,
                signal,
                value,
                unit,
                template,
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
