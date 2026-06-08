import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_field_sft import SIGNALS, STATIONS


TEMPLATES = [
    (
        "During epoch {epoch}, station {station} reported {signal} of {value} {unit}; "
        "the formal uncertainty was {uncertainty} {unit}."
    ),
    (
        "The solution used {samples} samples. For {station}, the target {signal} is "
        "{value} {unit}, while the residual RMS is {rms} {unit}."
    ),
    (
        "At {station}, {signal} equals {value} {unit}. The previous solution listed "
        "{previous} {unit}, and the quality flag is {flag}."
    ),
    (
        "Report {report_id}: {station} has {signal} = {value} {unit}; the network "
        "average for this product is {network_avg} {unit}."
    ),
    (
        "For station {station}, ignore the reference value {reference} {unit}; "
        "the extracted {signal} should be {value} {unit}."
    ),
    (
        "The {signal} at {station} is not {distractor} {unit}; the accepted estimate "
        "is {value} {unit} after {iterations} iterations."
    ),
    (
        "{station} was processed with window length {window} days. The measured "
        "{signal} is {value} {unit}, with threshold {threshold} {unit}."
    ),
    (
        "A preliminary value of {preliminary} {unit} was rejected for {station}; "
        "the final {signal} is {value} {unit}."
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/field_distractor_500.jsonl")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def format_value(value):
    return str(value)


def random_decimal(rng, values):
    value = rng.choice(values)
    return format_value(value)


def build_example(rng, station, signal, value, unit, values, template):
    targetValue = format_value(value)

    fields = {
        "station": station,
        "signal": signal,
        "value": targetValue,
        "unit": unit,
        "epoch": rng.choice(["2018.0", "2019.5", "2020.0", "2021.5", "2022.0"]),
        "samples": rng.choice(["120", "240", "360", "480"]),
        "uncertainty": rng.choice(["0.1", "0.2", "0.5", "1.0"]),
        "rms": rng.choice(["0.3", "0.6", "1.2", "2.4"]),
        "previous": random_decimal(rng, values),
        "flag": rng.choice(["0", "1", "2", "3"]),
        "report_id": rng.choice(["101", "204", "305", "409"]),
        "network_avg": random_decimal(rng, values),
        "reference": random_decimal(rng, values),
        "distractor": random_decimal(rng, values),
        "iterations": rng.choice(["3", "5", "8", "13"]),
        "window": rng.choice(["7", "14", "30", "90"]),
        "threshold": rng.choice(["0.5", "1.0", "2.0", "5.0"]),
        "preliminary": random_decimal(rng, values),
    }

    inputText = template.format(**fields)

    outputText = (
        f"station: {station}\n"
        f"signal: {signal}\n"
        f"value: {targetValue}\n"
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
                rng=rng,
                station=station,
                signal=signal,
                value=value,
                unit=unit,
                values=values,
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

    examples = build_examples(
        numExamples=args.num_examples,
        seed=args.seed,
    )
    save_jsonl(examples, args.out)

    counts = Counter(example["task"] for example in examples)

    print(f"saved {len(examples)} examples to {args.out}")
    print(f"task counts: {dict(counts)}")
    print(f"templates: {len(TEMPLATES)}")
    print("preview:")
    for example in examples[:3]:
        print("-" * 80)
        print(example["input"])
        print(example["output"])


if __name__ == "__main__":
    main()
