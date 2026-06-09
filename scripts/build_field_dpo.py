import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sft_data import load_sft_jsonl
from scripts.build_field_sft import SIGNALS, STATIONS


FIELDS = ["station", "signal", "value", "unit"]
SIGNAL_TO_UNIT_VALUES = {
    signal: (unit, values)
    for signal, unit, values in SIGNALS
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def parse_fields(text):
    fields = {}

    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key in FIELDS:
            fields[key] = value

    return fields


def render_fields(fields):
    return (
        f"station: {fields['station']}\n"
        f"signal: {fields['signal']}\n"
        f"value: {fields['value']}\n"
        f"unit: {fields['unit']}"
    )


def values_in_input(inputText):
    return re.findall(r"[-+]?\d+(?:\.\d+)?", inputText)


def choose_different(rng, values, current):
    choices = [
        str(value)
        for value in values
        if str(value) != str(current)
    ]

    if len(choices) == 0:
        return str(current)

    return rng.choice(choices)


def corrupt_station(rng, fields):
    corrupted = dict(fields)
    choices = [
        station
        for station in STATIONS
        if station != fields["station"]
    ]
    corrupted["station"] = rng.choice(choices)
    return corrupted, "wrong_station"


def corrupt_signal(rng, fields):
    corrupted = dict(fields)
    choices = [
        signal
        for signal in SIGNAL_TO_UNIT_VALUES
        if signal != fields["signal"]
    ]
    signal = rng.choice(choices)
    unit, values = SIGNAL_TO_UNIT_VALUES[signal]
    corrupted["signal"] = signal
    corrupted["unit"] = unit
    corrupted["value"] = str(rng.choice(values))
    return corrupted, "wrong_signal_group"


def corrupt_value_from_input(rng, fields, inputText):
    corrupted = dict(fields)
    candidates = [
        value
        for value in values_in_input(inputText)
        if value != fields["value"]
    ]

    if not candidates:
        return corrupt_value_same_signal(rng, fields)

    corrupted["value"] = rng.choice(candidates)
    return corrupted, "wrong_value_from_input"


def corrupt_value_same_signal(rng, fields):
    corrupted = dict(fields)
    _, values = SIGNAL_TO_UNIT_VALUES[fields["signal"]]
    corrupted["value"] = choose_different(rng, values, fields["value"])
    return corrupted, "wrong_value_same_signal"


def build_rejected(rng, example):
    fields = parse_fields(example["output"])
    if any(field not in fields for field in FIELDS):
        raise ValueError(f"输出字段不完整: {example['output']}")

    corruptors = [
        corrupt_station,
        corrupt_signal,
        corrupt_value_same_signal,
    ]

    if len(values_in_input(example["input"])) >= 2:
        corruptors.append(corrupt_value_from_input)

    corruptor = rng.choice(corruptors)
    if corruptor is corrupt_value_from_input:
        rejectedFields, rejectType = corruptor(rng, fields, example["input"])
    else:
        rejectedFields, rejectType = corruptor(rng, fields)

    return render_fields(rejectedFields), rejectType


def build_dpo_examples(examples, seed):
    rng = random.Random(seed)
    dpoExamples = []

    for example in examples:
        rejected, rejectType = build_rejected(rng, example)

        dpoExamples.append(
            {
                "task": example.get("task", "field_extraction"),
                "preference_type": rejectType,
                "instruction": example["instruction"],
                "input": example["input"],
                "chosen": example["output"],
                "rejected": rejected,
            }
        )

    return dpoExamples


def save_jsonl(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    examples = load_sft_jsonl(args.sft_path)
    dpoExamples = build_dpo_examples(examples, args.seed)
    save_jsonl(dpoExamples, args.out)

    counts = Counter(example["preference_type"] for example in dpoExamples)
    print(f"saved {len(dpoExamples)} examples to {args.out}")
    print(f"preference types: {dict(sorted(counts.items()))}")
    print("preview:")

    for example in dpoExamples[:3]:
        print("-" * 80)
        print(example["input"])
        print("chosen:")
        print(example["chosen"])
        print("rejected:")
        print(example["rejected"])
        print(f"type: {example['preference_type']}")


if __name__ == "__main__":
    main()
