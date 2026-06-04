import argparse
import csv
import os
from collections import Counter

import torch
import tiktoken

from model import BigramLanguageModel, GPTConfig
from sft_data import format_sft_example, load_sft_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sft-path", type=str, default="data/sft/astro_sft_small.jsonl")
    parser.add_argument("--out-dir", type=str, default="out/sft_next_token_diagnostics")
    parser.add_argument("--max-per-task", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def load_model(checkpointPath, device):
    checkpoint = torch.load(checkpointPath, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = BigramLanguageModel(config.vocabSize, config.blockSize, config=config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    vocabInfo = checkpoint["vocab"]
    vocabType = vocabInfo.get("type", "char")
    if vocabType != "tokenizer":
        raise ValueError("diagnose_next_token.py 目前只支持 tokenizer checkpoint。")

    enc = tiktoken.get_encoding(vocabInfo["meta"]["encoding"])
    return model, enc


def select_examples(examples, maxPerTask):
    counts = Counter()
    selected = []

    for example in examples:
        task = example.get("task", "unknown")
        if counts[task] >= maxPerTask:
            continue

        selected.append(example)
        counts[task] += 1

    return selected


def escape_token(text):
    return text.encode("unicode_escape").decode("ascii")


def next_token_distribution(model, enc, text, topK, device):
    ids = enc.encode(text)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]

    eosId = enc.eot_token
    eosProb = probs[eosId].item()
    eosRank = int((probs > probs[eosId]).sum().item()) + 1
    topProbs, topIds = torch.topk(probs, topK)

    topTokens = []
    for rank, (tokenId, prob) in enumerate(zip(topIds.tolist(), topProbs.tolist()), start=1):
        tokenText = enc.decode([tokenId])
        topTokens.append(
            {
                "rank": rank,
                "token_id": tokenId,
                "token_text": tokenText,
                "token_preview": escape_token(tokenText),
                "prob": prob,
                "is_eos": tokenId == eosId,
            }
        )

    return {
        "token_count": len(ids),
        "eos_id": eosId,
        "eos_rank": eosRank,
        "eos_prob": eosProb,
        "top_tokens": topTokens,
    }


def write_outputs(args, rows, topRows):
    os.makedirs(args.out_dir, exist_ok=True)
    summaryPath = os.path.join(args.out_dir, "summary.csv")
    topTokensPath = os.path.join(args.out_dir, "top_tokens.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(summaryPath, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "task",
            "position",
            "input_preview",
            "token_count",
            "eos_id",
            "eos_rank",
            "eos_prob",
            "top1_id",
            "top1_preview",
            "top1_prob",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(topTokensPath, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "task",
            "position",
            "input_preview",
            "rank",
            "token_id",
            "token_preview",
            "prob",
            "is_eos",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topRows)

    answerEndRows = [row for row in rows if row["position"] == "answer_end"]
    promptRows = [row for row in rows if row["position"] == "prompt_start"]

    avgAnswerEndEosRank = sum(row["eos_rank"] for row in answerEndRows) / len(answerEndRows)
    avgAnswerEndEosProb = sum(row["eos_prob"] for row in answerEndRows) / len(answerEndRows)
    avgPromptEosRank = sum(row["eos_rank"] for row in promptRows) / len(promptRows)
    avgPromptEosProb = sum(row["eos_prob"] for row in promptRows) / len(promptRows)

    worstAnswerEnd = sorted(answerEndRows, key=lambda row: row["eos_rank"], reverse=True)[:6]

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# SFT Next-token Diagnostics\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n\n")
        f.write(f"SFT data: `{args.sft_path}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- examples inspected: {len(promptRows)}\n")
        f.write(f"- avg prompt-start EOS rank: {avgPromptEosRank:.1f}\n")
        f.write(f"- avg prompt-start EOS prob: {avgPromptEosProb:.6f}\n")
        f.write(f"- avg answer-end EOS rank: {avgAnswerEndEosRank:.1f}\n")
        f.write(f"- avg answer-end EOS prob: {avgAnswerEndEosProb:.6f}\n\n")

        f.write("Interpretation:\n\n")
        f.write("- At `prompt_start`, EOS should usually be low because the model should begin answering.\n")
        f.write("- At `answer_end`, EOS should be high if the model has learned when to stop.\n\n")

        f.write("## Worst Answer-end EOS Ranks\n\n")
        for row in worstAnswerEnd:
            f.write(
                f"### {row['task']} | {row['input_preview']}\n\n"
                f"- eos rank: {row['eos_rank']}\n"
                f"- eos prob: {row['eos_prob']:.8f}\n"
                f"- top1: `{row['top1_preview']}` prob={row['top1_prob']:.8f}\n\n"
            )

        f.write("## Files\n\n")
        f.write(f"- `{summaryPath}`\n")
        f.write(f"- `{topTokensPath}`\n")

    print(f"saved summary to {summaryPath}")
    print(f"saved top tokens to {topTokensPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    useMps = os.environ.get("USE_MPS") == "1"
    device = "mps" if torch.backends.mps.is_available() and useMps else "cpu"
    print(f"using device: {device}", flush=True)

    model, enc = load_model(args.checkpoint, device)
    examples = select_examples(load_sft_jsonl(args.sft_path), args.max_per_task)

    rows = []
    topRows = []

    for example in examples:
        task = example.get("task", "unknown")
        prompt, answer = format_sft_example(example)
        inputPreview = example["input"].replace("\n", " ")[:80]

        positions = [
            ("prompt_start", prompt),
            ("answer_end", prompt + answer),
        ]

        for position, text in positions:
            result = next_token_distribution(model, enc, text, args.top_k, device)
            top1 = result["top_tokens"][0]
            row = {
                "task": task,
                "position": position,
                "input_preview": inputPreview,
                "token_count": result["token_count"],
                "eos_id": result["eos_id"],
                "eos_rank": result["eos_rank"],
                "eos_prob": result["eos_prob"],
                "top1_id": top1["token_id"],
                "top1_preview": top1["token_preview"],
                "top1_prob": top1["prob"],
            }
            rows.append(row)

            for topToken in result["top_tokens"]:
                topRows.append(
                    {
                        "task": task,
                        "position": position,
                        "input_preview": inputPreview,
                        "rank": topToken["rank"],
                        "token_id": topToken["token_id"],
                        "token_preview": topToken["token_preview"],
                        "prob": topToken["prob"],
                        "is_eos": topToken["is_eos"],
                    }
                )

    write_outputs(args, rows, topRows)


if __name__ == "__main__":
    main()
