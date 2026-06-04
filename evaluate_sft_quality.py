import argparse
import csv
import os
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher

import torch
import tiktoken

from model import BigramLanguageModel, GPTConfig
from sft_data import format_sft_example, load_sft_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sft-path", type=str, default="data/sft/astro_sft_small.jsonl")
    parser.add_argument("--out-dir", type=str, default="out/sft_quality_eval")
    parser.add_argument("--split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-mode", choices=["stratified", "shuffle", "sequential"], default="stratified")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def load_model(checkpointPath, device):
    checkpoint = torch.load(checkpointPath, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])

    model = BigramLanguageModel(config.vocabSize, config.blockSize, config=config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    vocabInfo = checkpoint["vocab"]
    if vocabInfo.get("type", "char") != "tokenizer":
        raise ValueError("evaluate_sft_quality.py 目前只支持 tokenizer checkpoint。")

    enc = tiktoken.get_encoding(vocabInfo["meta"]["encoding"])
    return model, enc


def split_examples(examples, trainRatio, splitMode, seed):
    rng = random.Random(seed)

    if splitMode == "sequential":
        splitIndex = int(len(examples) * trainRatio)
        return examples[:splitIndex], examples[splitIndex:]

    if splitMode == "shuffle":
        shuffled = list(examples)
        rng.shuffle(shuffled)
        splitIndex = int(len(shuffled) * trainRatio)
        return shuffled[:splitIndex], shuffled[splitIndex:]

    groups = defaultdict(list)
    for example in examples:
        groups[example.get("task", "unknown")].append(example)

    trainExamples = []
    valExamples = []
    for _, groupExamples in sorted(groups.items()):
        groupExamples = list(groupExamples)
        rng.shuffle(groupExamples)

        splitIndex = int(len(groupExamples) * trainRatio)
        if len(groupExamples) > 1:
            splitIndex = min(max(splitIndex, 1), len(groupExamples) - 1)

        trainExamples.extend(groupExamples[:splitIndex])
        valExamples.extend(groupExamples[splitIndex:])

    rng.shuffle(trainExamples)
    rng.shuffle(valExamples)
    return trainExamples, valExamples


def select_split(examples, split, trainRatio, splitMode, seed):
    trainExamples, valExamples = split_examples(
        examples,
        trainRatio,
        splitMode,
        seed,
    )

    if split == "train":
        return trainExamples

    if split == "val":
        return valExamples

    return examples


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def word_tokens(text):
    return re.findall(r"[a-zA-Z0-9_.:/+-]+", normalize_text(text))


def token_f1(prediction, target):
    predictionTokens = word_tokens(prediction)
    targetTokens = word_tokens(target)

    if len(predictionTokens) == 0 and len(targetTokens) == 0:
        return 1.0

    if len(predictionTokens) == 0 or len(targetTokens) == 0:
        return 0.0

    predictionCounts = defaultdict(int)
    for token in predictionTokens:
        predictionCounts[token] += 1

    overlap = 0
    for token in targetTokens:
        if predictionCounts[token] > 0:
            overlap += 1
            predictionCounts[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(predictionTokens)
    recall = overlap / len(targetTokens)
    return 2 * precision * recall / (precision + recall)


def target_recall(prediction, target):
    predictionTokens = set(word_tokens(prediction))
    targetTokens = set(word_tokens(target))

    if len(targetTokens) == 0:
        return 1.0

    return len(predictionTokens & targetTokens) / len(targetTokens)


def repeated_bigram_ratio(tokenIds):
    if len(tokenIds) < 2:
        return 0.0

    bigrams = [
        (tokenIds[i], tokenIds[i + 1])
        for i in range(len(tokenIds) - 1)
    ]
    return 1.0 - len(set(bigrams)) / len(bigrams)


def apply_repetition_penalty(logits, generatedIds, repetitionPenalty):
    if repetitionPenalty <= 1.0 or len(generatedIds) == 0:
        return logits

    seenTokens = torch.unique(torch.tensor(generatedIds, device=logits.device))
    seenLogits = logits[seenTokens]
    logits[seenTokens] = torch.where(
        seenLogits < 0,
        seenLogits * repetitionPenalty,
        seenLogits / repetitionPenalty,
    )
    return logits


@torch.no_grad()
def generate_completion(
    model,
    enc,
    prompt,
    maxNewTokens,
    temperature,
    topK,
    repetitionPenalty,
    device,
):
    promptIds = enc.encode(prompt)
    generatedIds = []
    idx = torch.tensor([promptIds], dtype=torch.long, device=device)
    eosId = enc.eot_token
    hitEos = False
    eosStep = None

    for step in range(maxNewTokens):
        idxCond = idx[:, -model.config.blockSize:]
        logits, _ = model(idxCond)
        logits = logits[0, -1, :].clone()
        logits = apply_repetition_penalty(logits, generatedIds, repetitionPenalty)

        if temperature == 0.0:
            nextId = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if topK is not None:
                values, _ = torch.topk(logits, min(topK, logits.size(-1)))
                logits[logits < values[-1]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nextId = int(torch.multinomial(probs, num_samples=1).item())

        if nextId == eosId:
            hitEos = True
            eosStep = step + 1
            break

        generatedIds.append(nextId)
        nextTensor = torch.tensor([[nextId]], dtype=torch.long, device=device)
        idx = torch.cat([idx, nextTensor], dim=1)

    return {
        "text": enc.decode(generatedIds),
        "token_ids": generatedIds,
        "hit_eos": hitEos,
        "eos_step": eosStep,
    }


def evaluate_example(model, enc, example, args, device):
    prompt, target = format_sft_example(example)
    result = generate_completion(
        model,
        enc,
        prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.repetition_penalty,
        device,
    )
    prediction = result["text"]

    return {
        "task": example.get("task", "unknown"),
        "input": example.get("input", ""),
        "target": target,
        "prediction": prediction,
        "hit_eos": result["hit_eos"],
        "eos_step": result["eos_step"],
        "completion_tokens": len(result["token_ids"]),
        "exact_match": normalize_text(prediction) == normalize_text(target),
        "token_f1": token_f1(prediction, target),
        "target_recall": target_recall(prediction, target),
        "char_similarity": SequenceMatcher(
            None,
            normalize_text(prediction),
            normalize_text(target),
        ).ratio(),
        "repeat_bigram_ratio": repeated_bigram_ratio(result["token_ids"]),
    }


def mean(items, key):
    if len(items) == 0:
        return 0.0
    return sum(float(item[key]) for item in items) / len(items)


def write_outputs(args, rows):
    os.makedirs(args.out_dir, exist_ok=True)
    csvPath = os.path.join(args.out_dir, "results.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    fieldnames = [
        "task",
        "input",
        "target",
        "prediction",
        "hit_eos",
        "eos_step",
        "completion_tokens",
        "exact_match",
        "token_f1",
        "target_recall",
        "char_similarity",
        "repeat_bigram_ratio",
    ]

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# SFT Quality Evaluation\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n\n")
        f.write(f"SFT data: `{args.sft_path}`\n\n")
        f.write(f"Split: `{args.split}`\n\n")
        f.write(f"Split mode: `{args.split_mode}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- examples: {len(rows)}\n")
        f.write(f"- eos rate: {mean(rows, 'hit_eos'):.2%}\n")
        f.write(f"- exact match: {mean(rows, 'exact_match'):.2%}\n")
        f.write(f"- avg token F1: {mean(rows, 'token_f1'):.3f}\n")
        f.write(f"- avg target recall: {mean(rows, 'target_recall'):.3f}\n")
        f.write(f"- avg char similarity: {mean(rows, 'char_similarity'):.3f}\n")
        f.write(f"- avg completion tokens: {mean(rows, 'completion_tokens'):.1f}\n")
        f.write(f"- avg repeated bigram ratio: {mean(rows, 'repeat_bigram_ratio'):.3f}\n\n")

        f.write("## By Task\n\n")
        f.write("| task | n | eos rate | token F1 | target recall | char similarity | exact match |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for task, items in sorted(groups.items()):
            f.write(
                f"| {task} | {len(items)} | "
                f"{mean(items, 'hit_eos'):.2%} | "
                f"{mean(items, 'token_f1'):.3f} | "
                f"{mean(items, 'target_recall'):.3f} | "
                f"{mean(items, 'char_similarity'):.3f} | "
                f"{mean(items, 'exact_match'):.2%} |\n"
            )

        f.write("\n## Lowest F1 Examples\n\n")
        for row in sorted(rows, key=lambda item: item["token_f1"])[:5]:
            f.write(f"### {row['task']} | {row['input'][:80]}\n\n")
            f.write("Target:\n\n")
            f.write("```text\n")
            f.write(row["target"])
            f.write("\n```\n\n")
            f.write("Prediction:\n\n")
            f.write("```text\n")
            f.write(row["prediction"])
            f.write("\n```\n\n")
            f.write(
                f"- token F1: {row['token_f1']:.3f}\n"
                f"- target recall: {row['target_recall']:.3f}\n"
                f"- hit eos: {row['hit_eos']}\n\n"
            )

        f.write("## Files\n\n")
        f.write(f"- `{csvPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved csv to {csvPath}", flush=True)
    print(f"saved report to {reportPath}", flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cpu"
    model, enc = load_model(args.checkpoint, device)

    examples = select_split(
        load_sft_jsonl(args.sft_path),
        args.split,
        args.train_ratio,
        args.split_mode,
        args.seed,
    )
    if args.max_examples is not None:
        examples = examples[:args.max_examples]

    rows = [
        evaluate_example(model, enc, example, args, device)
        for example in examples
    ]

    write_outputs(args, rows)


if __name__ == "__main__":
    main()
