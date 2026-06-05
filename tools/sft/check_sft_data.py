from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import Counter


REQUIRED_FIELDS = ["instruction", "input", "output"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/sft/astro_sft_tiny.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    taskCounts = Counter()
    inputLengths = []
    outputLengths = []
    examples = []

    with open(args.path, "r", encoding="utf-8") as f:
        for lineNo, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {lineNo} 行不是合法 JSON: {exc}") from exc

            for field in REQUIRED_FIELDS:
                if field not in item:
                    raise ValueError(f"第 {lineNo} 行缺少字段: {field}")
                if not isinstance(item[field], str):
                    raise ValueError(f"第 {lineNo} 行字段 {field} 必须是字符串")

            if item["instruction"].strip() == "":
                raise ValueError(f"第 {lineNo} 行 instruction 不能为空")
            if item["output"].strip() == "":
                raise ValueError(f"第 {lineNo} 行 output 不能为空")

            task = item.get("task", "unknown")
            taskCounts[task] += 1
            inputLengths.append(len(item["input"]))
            outputLengths.append(len(item["output"]))
            if len(examples) < 3:
                examples.append(item)

    total = sum(taskCounts.values())
    if total == 0:
        raise ValueError("SFT 数据为空")

    avgInputLen = sum(inputLengths) / len(inputLengths)
    avgOutputLen = sum(outputLengths) / len(outputLengths)

    print(f"path: {args.path}")
    print(f"examples: {total}")
    print(f"avg input chars: {avgInputLen:.1f}")
    print(f"avg output chars: {avgOutputLen:.1f}")
    print("task counts:")
    for task, count in sorted(taskCounts.items()):
        print(f"  {task}: {count}")

    print("\npreview:")
    for item in examples:
        print("-" * 80)
        print(f"task: {item.get('task', 'unknown')}")
        print(f"instruction: {item['instruction']}")
        print(f"input: {item['input'][:160]}")
        print(f"output: {item['output'][:160]}")

    print("\nall SFT data checks passed")


if __name__ == "__main__":
    main()

