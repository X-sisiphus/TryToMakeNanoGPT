import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/sft/astro_sft_tiny.jsonl")
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


def build_examples():
    examples = []

    concepts = {
        "terrestrial reference frame": "A terrestrial reference frame is a time-dependent coordinate system used to describe station positions, velocities, Earth orientation, and related geodetic quantities.",
        "GNSS time series": "A GNSS time series records how a station position changes over time, usually in east, north, and up components.",
        "Earth orientation parameters": "Earth orientation parameters describe how Earth rotates relative to celestial and terrestrial reference frames.",
        "space geodesy": "Space geodesy uses space-based and ground-based observations to measure Earth's shape, rotation, gravity-related motion, and surface deformation.",
        "InSAR deformation monitoring": "InSAR deformation monitoring uses radar phase differences to estimate surface displacement over broad areas.",
        "satellite clock product": "A satellite clock product estimates timing corrections for navigation satellites, which is essential for precise positioning.",
        "station offset": "A station offset is a discontinuity in a coordinate time series caused by an earthquake, equipment change, monument change, or processing change.",
        "vertical land motion": "Vertical land motion is upward or downward movement of the land surface, often measured with GNSS and used in sea-level and hazard studies.",
    }

    for term, explanation in concepts.items():
        add_example(
            examples,
            "concept_explanation",
            "Explain the geodesy or astronomy concept in one paragraph.",
            term,
            explanation,
        )

    summaries = [
        (
            "GNSS stations provide continuous observations of station position. The east, north, and up components can contain a linear trend, seasonal signals, offsets, and noise. These records are useful for studying tectonic motion and local deformation.",
            "GNSS station time series describe position changes over time. They can reveal trends, seasonal motion, offsets, and deformation signals.",
        ),
        (
            "InSAR provides spatially dense deformation measurements, while GNSS provides precise point-based time series. Combining the two data sources can improve deformation monitoring because they have complementary spatial and temporal properties.",
            "InSAR and GNSS are complementary deformation sensors. InSAR gives dense spatial coverage, while GNSS anchors time-dependent motion at precise stations.",
        ),
        (
            "A terrestrial reference frame connects station coordinates, velocities, Earth orientation, satellite orbits, and geophysical corrections. It allows measurements collected at different places and times to be compared consistently.",
            "A terrestrial reference frame provides a consistent coordinate basis for comparing geodetic measurements across space and time.",
        ),
        (
            "Precise satellite orbit and clock products are required for high-accuracy GNSS positioning. Errors in satellite position or time transfer directly affect receiver coordinate estimates.",
            "Accurate orbit and clock products are necessary for precise GNSS positioning because satellite errors propagate into station coordinates.",
        ),
    ]

    for paragraph, summary in summaries:
        add_example(
            examples,
            "summary",
            "Summarize the paragraph in two concise sentences.",
            paragraph,
            summary,
        )

    station_records = [
        ("BJFS", "east", "2.30 mm/yr", "2022-09-18", "antenna change"),
        ("SHAO", "north", "-1.20 mm/yr", "none", "stable monument"),
        ("WUH2", "up", "4.85 mm/yr", "2021-05-22", "possible hydrologic loading"),
        ("LHAZ", "east", "8.10 mm/yr", "2015-04-25", "earthquake offset"),
        ("KUNM", "north", "3.65 mm/yr", "none", "seasonal signal"),
        ("URUM", "up", "-2.40 mm/yr", "2020-01-10", "receiver change"),
        ("TWTF", "east", "1.05 mm/yr", "none", "low residual noise"),
        ("YELL", "north", "-0.75 mm/yr", "2019-08-03", "metadata discontinuity"),
    ]

    for station, component, velocity, offset_date, note in station_records:
        input_text = (
            f"station: {station}\n"
            f"component: {component}\n"
            f"velocity: {velocity}\n"
            f"offset_date: {offset_date}\n"
            f"note: {note}"
        )
        output = {
            "station": station,
            "component": component,
            "velocity": velocity,
            "offset_date": offset_date,
            "note": note,
        }
        add_example(
            examples,
            "field_extraction",
            "Extract the GNSS station metadata as compact JSON.",
            input_text,
            json.dumps(output, ensure_ascii=False),
        )

    coordinates = [
        ("12h30m00s", "+45d00m00s", "compact radio source"),
        ("05h12m33s", "-12d44m18s", "quasar candidate"),
        ("18h05m10s", "+22d31m09s", "calibration source"),
        ("23h59m01s", "-01d12m30s", "deep sky target"),
        ("01h08m45s", "+08d20m15s", "VLBI source"),
        ("09h41m27s", "-30d10m05s", "reference source"),
    ]

    for ra, dec, source_type in coordinates:
        input_text = f"RA={ra}, Dec={dec}, source_type={source_type}"
        output_text = (
            f"The source has right ascension {ra}, declination {dec}, "
            f"and is labeled as a {source_type}."
        )
        add_example(
            examples,
            "format_conversion",
            "Convert the coordinate record into a readable sentence.",
            input_text,
            output_text,
        )

    qa_pairs = [
        (
            "Why can a small domain corpus change generation vocabulary without creating real expertise?",
            "Because continued pretraining shifts token probabilities toward the domain. The model may emit domain words more often, but that does not guarantee reliable reasoning or factual understanding.",
        ),
        (
            "Why did lower temperature fail to fix broken station metadata generation?",
            "Lower temperature amplifies the highest-probability tokens. If the model has learned fragments but not structure, lower temperature can make repetition worse.",
        ),
        (
            "When should SFT be preferred over continued pretraining?",
            "SFT is preferred when the desired behavior is instruction following, structured output, extraction, or a specific response format.",
        ),
        (
            "Why is GPT-2 BPE inefficient for many Chinese astronomy terms?",
            "GPT-2 BPE was trained primarily on English-like text, so Chinese characters and domain terms are often split into many tokens.",
        ),
    ]

    for question, answer in qa_pairs:
        add_example(
            examples,
            "qa",
            "Answer the question clearly and concisely.",
            question,
            answer,
        )

    return examples


def main():
    args = parse_args()
    examples = build_examples()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    task_counts = {}
    for example in examples:
        task_counts[example["task"]] = task_counts.get(example["task"], 0) + 1

    print(f"saved {len(examples)} examples to {args.out}")
    for task, count in sorted(task_counts.items()):
        print(f"{task}: {count}")


if __name__ == "__main__":
    main()

