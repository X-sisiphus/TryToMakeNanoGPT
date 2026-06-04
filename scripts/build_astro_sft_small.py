import argparse
import json
import os
from collections import Counter


TASK_TARGETS = {
    "concept_explanation": 40,
    "field_extraction": 40,
    "format_conversion": 40,
    "qa": 40,
    "summary": 40,
}

CONCEPTS = [
    (
        "GNSS time series",
        "a record of station coordinate changes over time, often used to estimate trends, seasonal motion, offsets, and noise",
    ),
    (
        "terrestrial reference frame",
        "a time-dependent coordinate system that keeps station positions and velocities comparable across space and time",
    ),
    (
        "VLBI baseline delay",
        "a measured signal delay between radio telescopes observing distant compact radio sources",
    ),
    (
        "satellite clock bias",
        "the difference between a satellite clock and a reference time scale used in precise positioning",
    ),
    (
        "ionospheric delay",
        "a signal delay caused by charged particles in the ionosphere, especially important for radio navigation signals",
    ),
    (
        "tropospheric delay",
        "a signal delay caused by the neutral atmosphere and often modeled with zenith wet delay parameters",
    ),
    (
        "Earth orientation parameters",
        "parameters describing how Earth rotates relative to celestial and terrestrial reference frames",
    ),
    (
        "station velocity",
        "the long-term motion rate of a geodetic station in east, north, or up components",
    ),
    (
        "seasonal deformation",
        "periodic station motion caused by effects such as hydrology, atmosphere, temperature, and loading",
    ),
    (
        "sea level trend",
        "a long-term change in sea level that can be interpreted together with vertical land motion",
    ),
]

CONCEPT_INSTRUCTIONS = [
    "Explain the geodesy or astronomy concept in one paragraph.",
    "Describe the meaning of this space geodesy term.",
    "Give a concise explanation of this technical term.",
    "Explain why this concept matters for space-time intelligent analysis.",
]

STATIONS = [
    "BJFS",
    "WETTZELL",
    "NYALES20",
    "HOBART12",
    "KOKEE",
    "MATERA",
    "ONSA",
    "TSKB",
]

SIGNALS = [
    ("vertical velocity", 2.4, "mm/yr"),
    ("east displacement", -6.2, "mm"),
    ("north displacement", 4.7, "mm"),
    ("clock bias", 12.0, "ns"),
    ("zenith wet delay", 38.5, "mm"),
    ("baseline residual", 11.0, "ps"),
    ("sea level trend", 3.1, "mm/yr"),
    ("seasonal amplitude", 5.8, "mm"),
]

OBSERVABLES = [
    ("VLBI baseline residual", "delay", 12, "ps", "X"),
    ("GNSS station velocity", "velocity", 2.4, "mm/yr", "L1/L2"),
    ("SLR range residual", "range_residual", 18, "mm", "laser"),
    ("satellite clock product", "clock_bias", 9, "ns", "GNSS"),
    ("tropospheric estimate", "zenith_wet_delay", 42, "mm", "GNSS"),
]

QA_PAIRS = [
    (
        "Why can GNSS time series show seasonal signals?",
        "Because station coordinates can be affected by hydrology, atmosphere, thermal loading, and other periodic geophysical processes.",
    ),
    (
        "What does a terrestrial reference frame provide?",
        "It provides a consistent coordinate basis for measuring station positions, velocities, and Earth system changes.",
    ),
    (
        "Why is VLBI useful for Earth orientation?",
        "VLBI observes distant radio sources and helps estimate Earth rotation and orientation relative to a celestial frame.",
    ),
    (
        "Why can clock bias affect precise positioning?",
        "A timing error changes the inferred signal travel time, which directly affects the estimated receiver position.",
    ),
    (
        "Why is the ionosphere important in GNSS processing?",
        "The ionosphere delays radio signals, so unmodeled ionospheric effects can bias positioning and time transfer.",
    ),
    (
        "What is the role of tropospheric delay estimation?",
        "It reduces atmospheric path delay errors and can also provide information related to water vapor.",
    ),
    (
        "Why should station offsets be modeled?",
        "Offsets create discontinuities in coordinate time series and can bias velocity estimates if they are ignored.",
    ),
    (
        "Why compare GNSS and InSAR for deformation monitoring?",
        "GNSS gives precise station time series, while InSAR gives dense spatial coverage, so the two observations are complementary.",
    ),
    (
        "When is SFT more appropriate than continued pretraining?",
        "SFT is more appropriate when the goal is instruction following, structured output, extraction, or a specific answer style.",
    ),
    (
        "Why does a tiny SFT validation set give unstable conclusions?",
        "Because a few validation examples can be dominated by sampling noise and may not represent the broader task distribution.",
    ),
]

QA_INSTRUCTIONS = [
    "Answer the question briefly.",
    "Give a concise technical answer.",
    "Answer in one or two sentences.",
    "Provide a short explanation for the question.",
]

SUMMARY_TEXTS = [
    (
        "GNSS observations can be used to estimate station velocity, seasonal deformation, and offsets caused by earthquakes or equipment changes.",
        "GNSS time series support velocity, seasonal deformation, and offset analysis.",
    ),
    (
        "VLBI measures delays between radio telescopes observing distant quasars, supporting Earth orientation and reference frame estimation.",
        "VLBI delay measurements help estimate Earth orientation and reference frames.",
    ),
    (
        "Tropospheric delay modeling reduces atmospheric errors in GNSS positioning and can also provide information about water vapor.",
        "Tropospheric delay modeling improves positioning and can inform atmospheric analysis.",
    ),
    (
        "A terrestrial reference frame connects station coordinates, velocities, Earth orientation, satellite orbits, and geophysical corrections.",
        "A terrestrial reference frame provides a consistent basis for geodetic measurements.",
    ),
    (
        "InSAR provides dense spatial deformation maps, while GNSS provides precise point-based coordinate time series.",
        "InSAR and GNSS provide complementary deformation observations.",
    ),
    (
        "Satellite clock products estimate timing corrections that are required for high-accuracy GNSS positioning and time transfer.",
        "Satellite clock products support precise positioning and time transfer.",
    ),
    (
        "Sea level analysis should consider vertical land motion, because land movement can change relative sea level trends at tide gauges.",
        "Vertical land motion is important for interpreting relative sea level trends.",
    ),
    (
        "Earth orientation parameters describe polar motion, universal time variations, and nutation terms used in precise geodesy.",
        "Earth orientation parameters describe changes in Earth's rotation state.",
    ),
    (
        "Station metadata records antenna changes, receiver changes, monument events, and other information needed to interpret discontinuities.",
        "Station metadata helps explain discontinuities in coordinate time series.",
    ),
    (
        "Space-time intelligent analysis links geodetic observations with temporal patterns, spatial context, and domain-specific physical meaning.",
        "Space-time intelligent analysis combines observations, time patterns, spatial context, and physical interpretation.",
    ),
]

SUMMARY_INSTRUCTIONS = [
    "Summarize the observation in one sentence.",
    "Write a concise summary of the paragraph.",
    "Compress the paragraph into a short technical summary.",
    "Summarize the key geodetic meaning.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/astro_sft_small.jsonl")
    return parser.parse_args()


def add_example(examples, task, instruction, input_text, output_text):
    examples.append(
        {
            "task": task,
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        }
    )


def build_concept_examples(examples):
    for concept, meaning in CONCEPTS:
        for instruction in CONCEPT_INSTRUCTIONS:
            output_text = (
                f"{concept} is {meaning}. "
                "It helps connect space geodetic observations with time-dependent Earth system change, "
                "reference frame maintenance, or precise positioning analysis."
            )
            add_example(
                examples,
                "concept_explanation",
                instruction,
                concept,
                output_text,
            )


def build_field_extraction_examples(examples):
    for station in STATIONS:
        for signal, value, unit in SIGNALS:
            input_text = (
                f"Station {station} shows a {signal} of {value:.1f} {unit} "
                "from space geodetic observations."
            )
            output_text = (
                f"station: {station}\n"
                f"signal: {signal}\n"
                f"value: {value:.1f}\n"
                f"unit: {unit}"
            )
            add_example(
                examples,
                "field_extraction",
                "Extract the station, signal, value, and unit from the text.",
                input_text,
                output_text,
            )


def build_format_conversion_examples(examples):
    for station in STATIONS:
        for observable, key, value, unit, band in OBSERVABLES:
            input_text = (
                f"{observable}: station={station}, {key}={value} {unit}, band={band}."
            )
            output = {
                "station": station,
                "observable": observable,
                key: value,
                "unit": unit,
                "band": band,
            }
            add_example(
                examples,
                "format_conversion",
                "Convert the observation into a compact JSON object.",
                input_text,
                json.dumps(output, ensure_ascii=False),
            )


def build_qa_examples(examples):
    for question, answer in QA_PAIRS:
        for instruction in QA_INSTRUCTIONS:
            add_example(
                examples,
                "qa",
                instruction,
                question,
                answer,
            )


def build_summary_examples(examples):
    for input_text, output_text in SUMMARY_TEXTS:
        for instruction in SUMMARY_INSTRUCTIONS:
            add_example(
                examples,
                "summary",
                instruction,
                input_text,
                output_text,
            )


def trim_by_task(examples):
    counts = Counter()
    trimmed = []

    for example in examples:
        task = example["task"]
        if counts[task] >= TASK_TARGETS[task]:
            continue

        trimmed.append(example)
        counts[task] += 1

    missing = {
        task: target - counts[task]
        for task, target in TASK_TARGETS.items()
        if counts[task] < target
    }
    if missing:
        raise ValueError(f"not enough examples for task targets: {missing}")

    return trimmed


def main():
    args = parse_args()
    examples = []

    build_concept_examples(examples)
    build_field_extraction_examples(examples)
    build_format_conversion_examples(examples)
    build_qa_examples(examples)
    build_summary_examples(examples)

    examples = trim_by_task(examples)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    counts = Counter(example["task"] for example in examples)

    print(f"saved {len(examples)} examples to {args.out}")
    for task, count in sorted(counts.items()):
        print(f"{task}: {count}")


if __name__ == "__main__":
    main()
