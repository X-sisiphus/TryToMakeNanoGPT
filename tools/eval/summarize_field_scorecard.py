import argparse
import csv
import os
import re


METRICS = [
    "station",
    "signal",
    "value",
    "unit",
    "all_fields",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=str, default="out/field_scorecard.csv")
    parser.add_argument("--out-md", type=str, default="out/field_scorecard.md")
    parser.add_argument("items", nargs="+")
    return parser.parse_args()


def parse_item(item):
    if "=" not in item:
        raise ValueError(f"item 必须使用 name=report.md 格式: {item}")

    name, path = item.split("=", 1)
    return name, path


def parse_accuracy(text, metric):
    if metric == "all_fields":
        pattern = r"- all fields accuracy: ([0-9.]+)%"
    else:
        pattern = rf"- {metric} accuracy: ([0-9.]+)%"

    match = re.search(pattern, text)
    if match is None:
        return ""

    return float(match.group(1))


def load_report(name, path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    row = {
        "name": name,
        "report": path,
    }
    for metric in METRICS:
        row[metric] = parse_accuracy(text, metric)

    return row


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["name", *METRICS, "report"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value == "":
        return ""
    return f"{value:.2f}%"


def write_markdown(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Field Scorecard\n\n")
        f.write("| checkpoint | station | signal | value | unit | all fields |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")

        for row in rows:
            f.write(
                f"| {row['name']} | "
                f"{fmt(row['station'])} | "
                f"{fmt(row['signal'])} | "
                f"{fmt(row['value'])} | "
                f"{fmt(row['unit'])} | "
                f"{fmt(row['all_fields'])} |\n"
            )


def main():
    args = parse_args()
    rows = [
        load_report(name, path)
        for name, path in (parse_item(item) for item in args.items)
    ]

    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows)

    print(f"saved csv to {args.out_csv}")
    print(f"saved markdown to {args.out_md}")


if __name__ == "__main__":
    main()
