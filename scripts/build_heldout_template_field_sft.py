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
from scripts.build_rich_field_sft import TEMPLATES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", type=str, default="data/sft/field_rich_train_templates_1000.jsonl")
    parser.add_argument("--heldout-out", type=str, default="data/sft/field_rich_heldout_templates_200.jsonl")
    parser.add_argument("--train-examples", type=int, default=1000)
    parser.add_argument("--heldout-examples", type=int, default=200)
    parser.add_argument("--heldout-templates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def build_example(station, signal, value, unit, template, templateId):
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
        "template_id": templateId,
        "instruction": "Extract the station, signal, value, and unit from the text.",
        "input": inputText,
        "output": outputText,
    }


def build_examples(numExamples, seed, templateItems):
    rng = random.Random(seed)
    examples = []
    seen = set()

    maxCombinations = (
        len(STATIONS)
        * len(templateItems)
        * sum(len(values) for _, _, values in SIGNALS)
    )

    if numExamples > maxCombinations:
        raise ValueError(
            f"num_examples={numExamples} 超过可生成的不重复组合数 {maxCombinations}"
        )

    while len(examples) < numExamples:
        templateId, template = rng.choice(templateItems)
        station = rng.choice(STATIONS)
        signal, unit, values = rng.choice(SIGNALS)
        value = rng.choice(values)

        key = (templateId, station, signal, value, unit)
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
                templateId=templateId,
            )
        )

    return examples


def save_jsonl(examples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def print_summary(name, examples, path):
    templateCounts = Counter(example["template_id"] for example in examples)
    taskCounts = Counter(example["task"] for example in examples)

    print(f"{name}: saved {len(examples)} examples to {path}")
    print(f"{name}: task counts: {dict(taskCounts)}")
    print(f"{name}: templates: {sorted(templateCounts)}")
    print(f"{name}: template counts: {dict(sorted(templateCounts.items()))}")
    print(f"{name}: preview:")
    for example in examples[:3]:
        print("-" * 80)
        print(example["input"])
        print(example["output"])


def main():
    args = parse_args()

    if args.heldout_templates <= 0 or args.heldout_templates >= len(TEMPLATES):
        raise ValueError("--heldout-templates 必须大于 0 且小于模板总数")

    indexedTemplates = list(enumerate(TEMPLATES))
    trainTemplates = indexedTemplates[:-args.heldout_templates]
    heldoutTemplates = indexedTemplates[-args.heldout_templates:]

    trainExamples = build_examples(
        numExamples=args.train_examples,
        seed=args.seed,
        templateItems=trainTemplates,
    )
    heldoutExamples = build_examples(
        numExamples=args.heldout_examples,
        seed=args.seed + 1,
        templateItems=heldoutTemplates,
    )

    save_jsonl(trainExamples, args.train_out)
    save_jsonl(heldoutExamples, args.heldout_out)

    print(f"total templates: {len(TEMPLATES)}")
    print(f"train template ids: {[idx for idx, _ in trainTemplates]}")
    print(f"heldout template ids: {[idx for idx, _ in heldoutTemplates]}")
    print_summary("train", trainExamples, args.train_out)
    print_summary("heldout", heldoutExamples, args.heldout_out)


if __name__ == "__main__":
    main()
