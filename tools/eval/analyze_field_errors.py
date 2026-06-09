from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import os
from collections import Counter, defaultdict

from scripts.build_field_sft import SIGNALS, STATIONS


FIELDS = ["station", "signal", "value", "unit"]
SIGNAL_NAMES = [signal for signal, _, _ in SIGNALS]
UNITS = sorted({unit for _, unit, _ in SIGNALS}, key=len, reverse=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-accuracy", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-examples-per-type", type=int, default=5)
    return parser.parse_args()


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_value(value):
    return str(value).strip().lower()


def value_in_text(value, text):
    value = clean_value(value)
    text = clean_value(text)
    return value != "" and value in text


def get_present_stations(text):
    return [
        station.lower()
        for station in STATIONS
        if station.lower() in clean_value(text)
    ]


def get_present_signals(text):
    return [
        signal.lower()
        for signal in SIGNAL_NAMES
        if signal.lower() in clean_value(text)
    ]


def get_present_units(text):
    return [
        unit.lower()
        for unit in UNITS
        if unit.lower() in clean_value(text)
    ]


def get_field(row, prefix, field):
    return clean_value(row.get(f"{prefix}_{field}", ""))


def is_correct(row, field):
    return clean_value(row.get(f"{field}_correct", "")) == "true"


def classify_row(row):
    if clean_value(row.get("all_fields_correct", "")) == "true":
        return "all_correct"

    inputText = row["input"]
    presentStations = get_present_stations(inputText)
    presentSignals = get_present_signals(inputText)
    presentUnits = get_present_units(inputText)

    targetStation = get_field(row, "target", "station")
    targetSignal = get_field(row, "target", "signal")
    targetValue = get_field(row, "target", "value")
    targetUnit = get_field(row, "target", "unit")

    predictionStation = get_field(row, "prediction", "station")
    predictionSignal = get_field(row, "prediction", "signal")
    predictionValue = get_field(row, "prediction", "value")
    predictionUnit = get_field(row, "prediction", "unit")

    missingFields = [
        field
        for field in FIELDS
        if get_field(row, "prediction", field) == ""
    ]
    if missingFields:
        return "missing_field"

    otherStationSelected = (
        predictionStation in presentStations
        and predictionStation != targetStation
    )
    if otherStationSelected:
        return "wrong_station_or_record"

    if is_correct(row, "station") and not all(
        is_correct(row, field)
        for field in ["signal", "value", "unit"]
    ):
        valueFromContext = (
            predictionValue != ""
            and predictionValue != targetValue
            and value_in_text(predictionValue, inputText)
        )
        signalFromContext = (
            predictionSignal in presentSignals
            and predictionSignal != targetSignal
        )
        unitFromContext = (
            predictionUnit in presentUnits
            and predictionUnit != targetUnit
        )

        if valueFromContext or signalFromContext or unitFromContext:
            return "target_station_wrong_measurement"

        return "target_station_field_noise"

    if is_correct(row, "value") and not is_correct(row, "station"):
        return "right_value_wrong_station"

    return "other_field_error"


def add_analysis(row):
    analyzed = dict(row)
    analyzed["error_type"] = classify_row(row)

    wrongFields = [
        field
        for field in FIELDS
        if not is_correct(row, field)
    ]
    analyzed["wrong_fields"] = ",".join(wrongFields)
    analyzed["present_stations"] = ",".join(get_present_stations(row["input"]))
    analyzed["present_signals"] = ",".join(get_present_signals(row["input"]))
    return analyzed


def percent(numerator, denominator):
    if denominator == 0:
        return "0.00%"
    return f"{numerator / denominator:.2%}"


def write_markdown(path, rows, grouped, maxExamplesPerType):
    total = len(rows)
    wrongRows = [row for row in rows if row["error_type"] != "all_correct"]
    counter = Counter(row["error_type"] for row in rows)
    wrongFieldCounter = Counter()

    for row in wrongRows:
        for field in row["wrong_fields"].split(","):
            if field:
                wrongFieldCounter[field] += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Field Error Analysis\n\n")
        f.write("## Summary\n\n")
        f.write(f"- total: {total}\n")
        f.write(f"- all correct: {counter['all_correct']} ({percent(counter['all_correct'], total)})\n")
        f.write(f"- wrong: {len(wrongRows)} ({percent(len(wrongRows), total)})\n\n")

        f.write("## Error Types\n\n")
        for errorType, count in counter.most_common():
            if errorType == "all_correct":
                continue
            f.write(f"- {errorType}: {count} ({percent(count, total)})\n")

        f.write("\n## Wrong Fields\n\n")
        for field, count in wrongFieldCounter.most_common():
            f.write(f"- {field}: {count} ({percent(count, len(wrongRows))} of wrong rows)\n")

        f.write("\n## Type Meanings\n\n")
        f.write("- wrong_station_or_record: prediction selected another station or record from the input.\n")
        f.write("- target_station_wrong_measurement: station is correct, but signal/value/unit appears copied from another measurement in the input.\n")
        f.write("- target_station_field_noise: station is correct, but the wrong field does not look like a direct copy from context.\n")
        f.write("- right_value_wrong_station: value is correct, but station is wrong.\n")
        f.write("- missing_field: prediction omitted at least one field line.\n")
        f.write("- other_field_error: remaining mixed errors.\n\n")

        f.write("## Examples\n\n")
        for errorType, examples in grouped.items():
            if errorType == "all_correct":
                continue

            f.write(f"### {errorType}\n\n")
            for row in examples[:maxExamplesPerType]:
                f.write("Input:\n\n")
                f.write("```text\n")
                f.write(row["input"])
                f.write("\n```\n\n")

                f.write("Target:\n\n")
                f.write("```text\n")
                f.write(row["target"])
                f.write("\n```\n\n")

                f.write("Prediction:\n\n")
                f.write("```text\n")
                f.write(row["prediction"])
                f.write("\n```\n\n")

                f.write(f"Wrong fields: `{row['wrong_fields']}`\n\n")


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = [add_analysis(row) for row in load_rows(args.field_accuracy)]

    os.makedirs(args.out_dir, exist_ok=True)
    csvPath = os.path.join(args.out_dir, "field_error_analysis.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["error_type"]].append(row)

    write_csv(csvPath, rows)
    write_markdown(reportPath, rows, grouped, args.max_examples_per_type)

    print(f"saved csv to {csvPath}")
    print(f"saved report to {reportPath}")


if __name__ == "__main__":
    main()
