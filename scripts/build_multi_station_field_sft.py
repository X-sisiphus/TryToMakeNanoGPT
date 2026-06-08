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
        "Target station {station} reports {signal} of {value} {unit}. "
        "Reference station {other_station} reports {other_signal} of {other_value} {other_unit}."
    ),
    (
        "Compare {station} and {other_station}: {station} has {signal} = {value} {unit}, "
        "while {other_station} has {other_signal} = {other_value} {other_unit}."
    ),
    (
        "Extract the measurement for {station}. In the same solution, {other_station} has "
        "{other_signal} of {other_value} {other_unit}, and {station} has {signal} of {value} {unit}."
    ),
    (
        "For {other_station}, the {other_signal} is {other_value} {other_unit}. "
        "For the requested station {station}, the {signal} is {value} {unit}."
    ),
    (
        "The report lists two stations: {other_station} with {other_signal} {other_value} {other_unit}; "
        "{station} with {signal} {value} {unit}. Extract {station}."
    ),
    (
        "{station}: {signal}, {value} {unit}. {other_station}: {other_signal}, "
        "{other_value} {other_unit}. The requested record is {station}."
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/field_multi_station_500.jsonl")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def choose_other_station(rng, station):
    choices = [item for item in STATIONS if item != station]
    return rng.choice(choices)


def choose_signal(rng):
    signal, unit, values = rng.choice(SIGNALS)
    value = rng.choice(values)
    return signal, unit, str(value)


def build_example(rng, station, signal, value, unit, template):
    otherStation = choose_other_station(rng, station)
    otherSignal, otherUnit, otherValue = choose_signal(rng)

    inputText = template.format(
        station=station,
        signal=signal,
        value=value,
        unit=unit,
        other_station=otherStation,
        other_signal=otherSignal,
        other_value=otherValue,
        other_unit=otherUnit,
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
        signal, unit, value = choose_signal(rng)
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
