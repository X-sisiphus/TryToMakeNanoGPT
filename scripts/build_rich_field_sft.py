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
    "Station {station} shows a {signal} of {value} {unit} from space geodetic observations.",
    "The {signal} measured at station {station} is {value} {unit}.",
    "For {station}, the estimated {signal} equals {value} {unit}.",
    "Space geodetic processing reports {station} with {signal} = {value} {unit}.",
    "{station} has a reported {signal} of {value} {unit} in the latest solution.",
    "In the latest solution, {station}'s {signal} is {value} {unit}.",
    "A {value} {unit} {signal} was estimated for station {station}.",
    "For station {station}, analysts report {value} {unit} as the {signal}.",
    "The solution lists {station}: {signal}, {value} {unit}.",
    "{station} -- {signal}: {value} {unit}.",
    "At {station}, the processing chain found {signal} to be {value} {unit}.",
    "The reported value for {signal} at {station} is {value} {unit}.",
    "Using space geodetic data, {station} was assigned a {signal} of {value} {unit}.",
    "The station {station} record contains {signal} = {value} {unit}.",
    "{signal} for {station} was solved as {value} {unit}.",
    "A current estimate gives {station} a {signal} value of {value} {unit}.",
    "Processing output: station {station}, {signal} {value} {unit}.",
    "{station}'s latest {signal} estimate is {value} {unit}.",
    "The {signal} entry for station {station} reads {value} {unit}.",
    "In the geodetic report, {station} is associated with {value} {unit} of {signal}.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/astro_sft_field_rich_1000.jsonl")
    parser.add_argument("--num-examples", type=int, default=1000)
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
