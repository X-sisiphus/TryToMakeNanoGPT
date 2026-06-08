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


TEMPLATES_BY_TYPE = {
    "previous": [
        (
            "At {station}, {signal} equals {value} {unit}. The previous solution "
            "listed {previous} {unit}."
        ),
        (
            "The current {signal} for {station} is {value} {unit}; last week it was "
            "{previous} {unit}."
        ),
    ],
    "negative": [
        (
            "The {signal} at {station} is not {distractor} {unit}; the accepted "
            "estimate is {value} {unit}."
        ),
        (
            "For {station}, reject {distractor} {unit} and use {value} {unit} as "
            "the {signal}."
        ),
    ],
    "network": [
        (
            "Report {report_id}: {station} has {signal} = {value} {unit}; the "
            "network average is {network_avg} {unit}."
        ),
        (
            "The network mean is {network_avg} {unit}, but station {station} has "
            "{signal} of {value} {unit}."
        ),
    ],
    "uncertainty": [
        (
            "During epoch {epoch}, station {station} reported {signal} of {value} "
            "{unit}; the formal uncertainty was {uncertainty} {unit}."
        ),
        (
            "{station} has {signal} = {value} {unit} with uncertainty "
            "{uncertainty} {unit}."
        ),
    ],
    "metadata": [
        (
            "{station} was processed with window length {window} days. The measured "
            "{signal} is {value} {unit}."
        ),
        (
            "The solution used {samples} samples; for {station}, {signal} is "
            "{value} {unit}."
        ),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distractor-type",
        choices=sorted(TEMPLATES_BY_TYPE),
        required=True,
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def random_decimal(rng, values, targetValue=None):
    candidates = [value for value in values if value != targetValue]
    if len(candidates) == 0:
        candidates = list(values)

    return str(rng.choice(candidates))


def build_example(rng, distractorType, station, signal, value, unit, values, template):
    fields = {
        "station": station,
        "signal": signal,
        "value": str(value),
        "unit": unit,
        "previous": random_decimal(rng, values, value),
        "distractor": random_decimal(rng, values, value),
        "network_avg": random_decimal(rng, values, value),
        "report_id": rng.choice(["101", "204", "305", "409"]),
        "epoch": rng.choice(["2018.0", "2019.5", "2020.0", "2021.5", "2022.0"]),
        "uncertainty": rng.choice(["0.1", "0.2", "0.5", "1.0"]),
        "window": rng.choice(["7", "14", "30", "90"]),
        "samples": rng.choice(["120", "240", "360", "480"]),
    }

    inputText = template.format(**fields)
    outputText = (
        f"station: {station}\n"
        f"signal: {signal}\n"
        f"value: {value}\n"
        f"unit: {unit}"
    )

    return {
        "task": "field_extraction",
        "distractor_type": distractorType,
        "instruction": "Extract the station, signal, value, and unit from the text.",
        "input": inputText,
        "output": outputText,
    }


def build_examples(distractorType, numExamples, seed):
    rng = random.Random(seed)
    templates = TEMPLATES_BY_TYPE[distractorType]
    examples = []
    seen = set()

    maxCombinations = (
        len(STATIONS)
        * len(templates)
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
        template = rng.choice(templates)

        key = (station, signal, value, unit, template)
        if key in seen:
            continue

        seen.add(key)
        examples.append(
            build_example(
                rng=rng,
                distractorType=distractorType,
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
        distractorType=args.distractor_type,
        numExamples=args.num_examples,
        seed=args.seed,
    )
    save_jsonl(examples, args.out)

    counts = Counter(example["distractor_type"] for example in examples)

    print(f"saved {len(examples)} examples to {args.out}")
    print(f"distractor counts: {dict(counts)}")
    print("preview:")
    for example in examples[:3]:
        print("-" * 80)
        print(example["input"])
        print(example["output"])


if __name__ == "__main__":
    main()
