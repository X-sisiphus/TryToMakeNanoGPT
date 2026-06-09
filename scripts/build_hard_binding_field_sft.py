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


BINDING_TYPES = [
    "same_signal",
    "same_unit",
    "near_value",
    "target_first",
    "target_second",
]

TEMPLATES = [
    (
        "{first_station}: {first_signal}, {first_value} {first_unit}. "
        "{second_station}: {second_signal}, {second_value} {second_unit}. "
        "Extract {target_station}."
    ),
    (
        "Two records are listed. For {first_station}, {first_signal} is "
        "{first_value} {first_unit}. For {second_station}, {second_signal} is "
        "{second_value} {second_unit}. Return the record for {target_station}."
    ),
    (
        "Do not mix the stations: {first_station} has {first_signal} = "
        "{first_value} {first_unit}, while {second_station} has {second_signal} = "
        "{second_value} {second_unit}. The requested station is {target_station}."
    ),
    (
        "In the station table, {first_station} reports {first_signal} of "
        "{first_value} {first_unit}; {second_station} reports {second_signal} of "
        "{second_value} {second_unit}. Extract only {target_station}."
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/field_hard_binding_800.jsonl")
    parser.add_argument("--num-examples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def choose_other_station(rng, station):
    choices = [item for item in STATIONS if item != station]
    return rng.choice(choices)


def choose_signal(rng):
    signal, unit, values = rng.choice(SIGNALS)
    value = rng.choice(values)
    return {
        "signal": signal,
        "unit": unit,
        "value": str(value),
        "values": values,
    }


def choose_same_signal_record(rng, targetRecord):
    candidates = [
        value
        for value in targetRecord["values"]
        if str(value) != targetRecord["value"]
    ]
    if not candidates:
        candidates = targetRecord["values"]

    return {
        "signal": targetRecord["signal"],
        "unit": targetRecord["unit"],
        "value": str(rng.choice(candidates)),
        "values": targetRecord["values"],
    }


def choose_same_unit_record(rng, targetRecord):
    candidates = [
        (signal, unit, values)
        for signal, unit, values in SIGNALS
        if unit == targetRecord["unit"] and signal != targetRecord["signal"]
    ]

    if not candidates:
        return choose_same_signal_record(rng, targetRecord)

    signal, unit, values = rng.choice(candidates)
    return {
        "signal": signal,
        "unit": unit,
        "value": str(rng.choice(values)),
        "values": values,
    }


def choose_near_value_record(rng, targetRecord):
    targetValue = float(targetRecord["value"])
    candidates = []

    for signal, unit, values in SIGNALS:
        if unit != targetRecord["unit"]:
            continue

        for value in values:
            if str(value) == targetRecord["value"] and signal == targetRecord["signal"]:
                continue
            candidates.append((abs(float(value) - targetValue), signal, unit, value, values))

    if not candidates:
        return choose_same_unit_record(rng, targetRecord)

    candidates.sort(key=lambda item: item[0])
    _, signal, unit, value, values = rng.choice(candidates[: min(4, len(candidates))])
    return {
        "signal": signal,
        "unit": unit,
        "value": str(value),
        "values": values,
    }


def choose_other_record(rng, bindingType, targetRecord):
    if bindingType == "same_signal":
        return choose_same_signal_record(rng, targetRecord)

    if bindingType == "same_unit":
        return choose_same_unit_record(rng, targetRecord)

    if bindingType == "near_value":
        return choose_near_value_record(rng, targetRecord)

    return choose_signal(rng)


def render_example(
    station,
    otherStation,
    targetRecord,
    otherRecord,
    targetFirst,
    template,
    bindingType,
):
    targetFields = {
        "station": station,
        "signal": targetRecord["signal"],
        "value": targetRecord["value"],
        "unit": targetRecord["unit"],
    }
    otherFields = {
        "station": otherStation,
        "signal": otherRecord["signal"],
        "value": otherRecord["value"],
        "unit": otherRecord["unit"],
    }

    first = targetFields if targetFirst else otherFields
    second = otherFields if targetFirst else targetFields

    inputText = template.format(
        first_station=first["station"],
        first_signal=first["signal"],
        first_value=first["value"],
        first_unit=first["unit"],
        second_station=second["station"],
        second_signal=second["signal"],
        second_value=second["value"],
        second_unit=second["unit"],
        target_station=station,
    )

    outputText = (
        f"station: {station}\n"
        f"signal: {targetRecord['signal']}\n"
        f"value: {targetRecord['value']}\n"
        f"unit: {targetRecord['unit']}"
    )

    return {
        "task": "field_extraction",
        "binding_type": bindingType,
        "instruction": "Extract the station, signal, value, and unit from the text.",
        "input": inputText,
        "output": outputText,
    }


def build_examples(numExamples, seed):
    rng = random.Random(seed)
    examples = []
    seen = set()

    while len(examples) < numExamples:
        station = rng.choice(STATIONS)
        otherStation = choose_other_station(rng, station)
        targetRecord = choose_signal(rng)
        bindingType = rng.choice(BINDING_TYPES)

        if bindingType == "target_first":
            targetFirst = True
        elif bindingType == "target_second":
            targetFirst = False
        else:
            targetFirst = rng.choice([True, False])

        otherRecord = choose_other_record(rng, bindingType, targetRecord)
        template = rng.choice(TEMPLATES)

        key = (
            station,
            otherStation,
            targetRecord["signal"],
            targetRecord["value"],
            targetRecord["unit"],
            otherRecord["signal"],
            otherRecord["value"],
            otherRecord["unit"],
            targetFirst,
            template,
        )
        if key in seen:
            continue

        seen.add(key)
        examples.append(
            render_example(
                station=station,
                otherStation=otherStation,
                targetRecord=targetRecord,
                otherRecord=otherRecord,
                targetFirst=targetFirst,
                template=template,
                bindingType=bindingType,
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

    counts = Counter(example["binding_type"] for example in examples)

    print(f"saved {len(examples)} examples to {args.out}")
    print(f"binding counts: {dict(counts)}")
    print("preview:")
    for example in examples[:5]:
        print("-" * 80)
        print(f"binding_type: {example['binding_type']}")
        print(example["input"])
        print(example["output"])


if __name__ == "__main__":
    main()
