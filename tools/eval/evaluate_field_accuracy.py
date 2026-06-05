from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import os


FIELDS = ["station", "signal", "value", "unit"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="out/sft_quality_field_500_val/results.csv")
    parser.add_argument("--out-dir", type=str, default="out/field_accuracy_field_500")
    return parser.parse_args()


def parse_fields(text):
    fields = {}

    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip().lower()

        if key in FIELDS:
            fields[key] = value

    return fields


def load_rows(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


def evaluate_row(row):
    targetFields = parse_fields(row["target"])
    predictionFields = parse_fields(row["prediction"])

    result = {
        "input": row["input"],
        "target": row["target"],
        "prediction": row["prediction"],
    }

    correctCount = 0

    for field in FIELDS:
        targetValue = targetFields.get(field, "")
        predictionValue = predictionFields.get(field, "")

        correct = targetValue == predictionValue and targetValue != ""

        result[f"target_{field}"] = targetValue
        result[f"prediction_{field}"] = predictionValue
        result[f"{field}_correct"] = correct

        if correct:
            correctCount += 1

    result["field_correct_count"] = correctCount
    result["all_fields_correct"] = correctCount == len(FIELDS)

    return result


def summarize(results):
    total = len(results)
    if total == 0:
        raise ValueError("没有可评测的结果")

    summary = {
        "total": total,
        "all_fields_accuracy": sum(row["all_fields_correct"] for row in results) / total,
    }

    for field in FIELDS:
        key = f"{field}_correct"
        summary[f"{field}_accuracy"] = sum(row[key] for row in results) / total

    summary["avg_field_correct_count"] = (
        sum(row["field_correct_count"] for row in results) / total
    )

    return summary


def write_outputs(args, results, summary):
    os.makedirs(args.out_dir, exist_ok=True)

    csvPath = os.path.join(args.out_dir, "field_accuracy.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    fieldnames = [
        "input",
        "target",
        "prediction",
        "target_station",
        "prediction_station",
        "station_correct",
        "target_signal",
        "prediction_signal",
        "signal_correct",
        "target_value",
        "prediction_value",
        "value_correct",
        "target_unit",
        "prediction_unit",
        "unit_correct",
        "field_correct_count",
        "all_fields_correct",
    ]

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# Field Accuracy Evaluation\n\n")
        f.write(f"Results: `{args.results}`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- total: {summary['total']}\n")
        f.write(f"- station accuracy: {summary['station_accuracy']:.2%}\n")
        f.write(f"- signal accuracy: {summary['signal_accuracy']:.2%}\n")
        f.write(f"- value accuracy: {summary['value_accuracy']:.2%}\n")
        f.write(f"- unit accuracy: {summary['unit_accuracy']:.2%}\n")
        f.write(f"- all fields accuracy: {summary['all_fields_accuracy']:.2%}\n")
        f.write(f"- avg correct fields: {summary['avg_field_correct_count']:.2f}/4\n\n")

        f.write("## Wrong Examples\n\n")
        wrongRows = [
            row for row in results
            if not row["all_fields_correct"]
        ]

        for row in wrongRows[:10]:
            f.write("### Example\n\n")
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

            f.write(
                f"- station: {row['station_correct']}\n"
                f"- signal: {row['signal_correct']}\n"
                f"- value: {row['value_correct']}\n"
                f"- unit: {row['unit_correct']}\n\n"
            )

        f.write("## Files\n\n")
        f.write(f"- `{csvPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved csv to {csvPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()

    rows = load_rows(args.results)
    results = [
        evaluate_row(row)
        for row in rows
    ]

    summary = summarize(results)

    write_outputs(args, results, summary)


if __name__ == "__main__":
    main()
