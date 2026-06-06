import argparse
import json
import os
import random
from collections import Counter


STATIONS = [
    "BJFS",
    "KOKEE",
    "ONSA",
    "TSKB",
    "MATE",
]

SIGNALS = [
    ("vertical velocity", "mm/yr", ["-3.2", "0.8", "2.4", "5.6"]),
    ("clock bias", "ns", ["-0.8", "0.4", "1.2", "3.6"]),
    ("east displacement", "mm", ["-6.2", "1.5", "4.7", "9.1"]),
]

TEMPLATES = [
    "station={station}; signal={signal}; value={value}; unit={unit}",
    "value={value}; unit={unit}; station={station}; signal={signal}",
    "signal={signal}; station={station}; unit={unit}; value={value}",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/small_field_copy_spaced_100.jsonl")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--digit-spaced", action="store_true")
    return parser.parse_args()


def maybe_space_value(value, digitSpaced):
    if digitSpaced:
        return " ".join(list(str(value)))

    return value


def build_example(station, signal, value, unit, template, digitSpaced):
    valueText = maybe_space_value(value, digitSpaced)

    inputText = template.format(
        station=station,
        signal=signal,
        value=valueText,
        unit=unit,
    )

    outputText = (
        f"station: {station}\n"
        f"signal: {signal}\n"
        f"value: {valueText}\n"
        f"unit: {unit}"
    )

    return {
        "task": "field_extraction",
        "instruction": "Extract the station, signal, value, and unit from the text.",
        "input": inputText,
        "output": outputText,
    }


def build_examples(numExamples, seed, digitSpaced):
    rng = random.Random(seed)
    examples = []
    seen = set()

    maxCombinations = (
        len(STATIONS)
        * len(TEMPLATES)
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
        template = rng.choice(TEMPLATES)

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
                digitSpaced=digitSpaced,
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

    examples = build_examples(
        numExamples=args.num_examples,
        seed=args.seed,
        digitSpaced=args.digit_spaced,
    )

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
