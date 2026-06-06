import argparse
import json
import os
import random
from collections import Counter


BASE_STATIONS = [
    "BJFS",
    "KOKEE",
    "ONSA",
    "TSKB",
    "MATE",
]

FULL_STATIONS = [
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

BASE_SIGNALS = [
    ("vertical velocity", "mm/yr", ["-3.2", "0.8", "2.4", "5.6"]),
    ("clock bias", "ns", ["-0.8", "0.4", "1.2", "3.6"]),
    ("east displacement", "mm", ["-6.2", "1.5", "4.7", "9.1"]),
]

FULL_SIGNALS = [
    ("vertical velocity", "mm/yr", ["-8.5", "-3.2", "0.8", "2.4", "5.6"]),
    ("east displacement", "mm", ["-12.0", "-6.2", "1.5", "4.7", "9.1"]),
    ("north displacement", "mm", ["-10.5", "-4.4", "2.2", "6.8", "11.3"]),
    ("clock bias", "ns", ["-2.5", "-0.8", "0.4", "1.2", "3.6"]),
    ("zenith wet delay", "mm", ["12.5", "24.0", "38.5", "52.2", "71.4"]),
    ("tropospheric delay", "ps", ["8.0", "12.0", "18.5", "25.0", "33.5"]),
    ("seasonal amplitude", "mm", ["1.2", "2.4", "3.6", "5.8", "8.1"]),
]

VALUE_STRESS_SIGNALS = [
    (
        "vertical velocity",
        "mm/yr",
        ["-9.6", "-8.5", "-6.4", "-3.2", "-1.6", "0.8", "2.4", "4.8", "5.6", "7.2"],
    ),
    (
        "clock bias",
        "ns",
        ["-2.5", "-1.6", "-0.8", "-0.4", "0.4", "0.8", "1.2", "2.4", "3.6", "4.8"],
    ),
    (
        "east displacement",
        "mm",
        ["-12.0", "-9.6", "-6.2", "-3.1", "1.5", "3.0", "4.7", "6.2", "9.1", "12.4"],
    ),
]

TEMPLATES = [
    "station={station}; signal={signal}; value={value}; unit={unit}",
    "value={value}; unit={unit}; station={station}; signal={signal}",
    "signal={signal}; station={station}; unit={unit}; value={value}",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["station", "signal", "value"],
        required=True,
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num-examples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--digit-spaced", action="store_true")
    return parser.parse_args()


def select_space(mode):
    if mode == "station":
        return FULL_STATIONS, BASE_SIGNALS

    if mode == "signal":
        return BASE_STATIONS, FULL_SIGNALS

    if mode == "value":
        return BASE_STATIONS, VALUE_STRESS_SIGNALS

    raise ValueError(f"unknown mode: {mode}")


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


def build_examples(mode, numExamples, seed, digitSpaced):
    rng = random.Random(seed)
    stations, signals = select_space(mode)
    examples = []
    seen = set()

    maxCombinations = (
        len(stations)
        * len(TEMPLATES)
        * sum(len(values) for _, _, values in signals)
    )

    if numExamples > maxCombinations:
        raise ValueError(
            f"num_examples={numExamples} 超过可生成的不重复组合数 {maxCombinations}"
        )

    while len(examples) < numExamples:
        station = rng.choice(stations)
        signal, unit, values = rng.choice(signals)
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
        mode=args.mode,
        numExamples=args.num_examples,
        seed=args.seed,
        digitSpaced=args.digit_spaced,
    )

    save_jsonl(examples, args.out)

    counts = Counter(example["task"] for example in examples)
    stations = Counter(example["output"].splitlines()[0] for example in examples)
    signals = Counter(example["output"].splitlines()[1] for example in examples)

    print(f"saved {len(examples)} examples to {args.out}")
    print(f"mode: {args.mode}")
    print(f"task counts: {dict(counts)}")
    print(f"unique stations: {len(stations)}")
    print(f"unique signals: {len(signals)}")
    print("preview:")
    for example in examples[:3]:
        print("-" * 80)
        print(example["input"])
        print(example["output"])


if __name__ == "__main__":
    main()
