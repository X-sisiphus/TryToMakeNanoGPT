import argparse
import csv
import json
import os
from collections import Counter

import torch
import tiktoken

from model import BigramLanguageModel, GPTConfig
from sft_data import END_TOKEN, format_sft_example, load_sft_jsonl


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
    endSequenceIds = enc.encode("\n" + END_TOKEN)
    endFirstId = endSequenceIds[0]
    eosProb = probs[eosId].item()
    eosRank = int((probs > probs[eosId]).sum().item()) + 1
    endFirstProb = probs[endFirstId].item()
    endFirstRank = int((probs > probs[endFirstId]).sum().item()) + 1
    topProbs, topIds = torch.topk(probs, topK)

    endSequenceRanks = []
    endSequenceProbs = []
    prefixIds = list(ids)
    for tokenId in endSequenceIds:
        prefix = torch.tensor([prefixIds], dtype=torch.long, device=device)
        with torch.no_grad():
            stepLogits, _ = model(prefix)
            stepProbs = torch.softmax(stepLogits[:, -1, :], dim=-1)[0]

        endSequenceRanks.append(int((stepProbs > stepProbs[tokenId]).sum().item()) + 1)
        endSequenceProbs.append(stepProbs[tokenId].item())
        prefixIds.append(tokenId)

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
                "is_end_first": tokenId == endFirstId,
            }
        )

    return {
        "token_count": len(ids),
        "eos_id": eosId,
        "eos_rank": eosRank,
        "eos_prob": eosProb,
        "end_token_text": END_TOKEN,
        "end_first_id": endFirstId,
        "end_first_rank": endFirstRank,
        "end_first_prob": endFirstProb,
        "end_sequence_ids": endSequenceIds,
        "end_sequence_previews": [escape_token(enc.decode([tokenId])) for tokenId in endSequenceIds],
        "end_sequence_ranks": endSequenceRanks,
        "end_sequence_probs": endSequenceProbs,
        "end_sequence_max_rank": max(endSequenceRanks),
        "end_sequence_last_rank": endSequenceRanks[-1],
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
            "end_token_text",
            "end_first_id",
            "end_first_rank",
            "end_first_prob",
            "end_sequence_ids",
            "end_sequence_previews",
            "end_sequence_ranks",
            "end_sequence_probs",
            "end_sequence_max_rank",
            "end_sequence_last_rank",
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
            "is_end_first",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topRows)

    answerEndRows = [row for row in rows if row["position"] == "answer_end"]
    promptRows = [row for row in rows if row["position"] == "prompt_start"]

    avgAnswerEndEosRank = sum(row["eos_rank"] for row in answerEndRows) / len(answerEndRows)
    avgAnswerEndEosProb = sum(row["eos_prob"] for row in answerEndRows) / len(answerEndRows)
    avgAnswerEndEndFirstRank = sum(row["end_first_rank"] for row in answerEndRows) / len(answerEndRows)
    avgAnswerEndEndFirstProb = sum(row["end_first_prob"] for row in answerEndRows) / len(answerEndRows)
    avgAnswerEndEndSeqMaxRank = sum(row["end_sequence_max_rank"] for row in answerEndRows) / len(answerEndRows)
    avgAnswerEndEndSeqLastRank = sum(row["end_sequence_last_rank"] for row in answerEndRows) / len(answerEndRows)
    avgPromptEosRank = sum(row["eos_rank"] for row in promptRows) / len(promptRows)
    avgPromptEosProb = sum(row["eos_prob"] for row in promptRows) / len(promptRows)
    avgPromptEndFirstRank = sum(row["end_first_rank"] for row in promptRows) / len(promptRows)
    avgPromptEndFirstProb = sum(row["end_first_prob"] for row in promptRows) / len(promptRows)
    avgPromptEndSeqMaxRank = sum(row["end_sequence_max_rank"] for row in promptRows) / len(promptRows)
    avgPromptEndSeqLastRank = sum(row["end_sequence_last_rank"] for row in promptRows) / len(promptRows)

    worstAnswerEnd = sorted(answerEndRows, key=lambda row: row["eos_rank"], reverse=True)[:6]
    worstAnswerEndEndToken = sorted(answerEndRows, key=lambda row: row["end_first_rank"], reverse=True)[:6]

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# SFT Next-token Diagnostics\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n\n")
        f.write(f"SFT data: `{args.sft_path}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- examples inspected: {len(promptRows)}\n")
        f.write(f"- avg prompt-start EOS rank: {avgPromptEosRank:.1f}\n")
        f.write(f"- avg prompt-start EOS prob: {avgPromptEosProb:.6f}\n")
        f.write(f"- avg prompt-start END-first rank: {avgPromptEndFirstRank:.1f}\n")
        f.write(f"- avg prompt-start END-first prob: {avgPromptEndFirstProb:.6f}\n")
        f.write(f"- avg prompt-start END-sequence max rank: {avgPromptEndSeqMaxRank:.1f}\n")
        f.write(f"- avg prompt-start END-sequence last-token rank: {avgPromptEndSeqLastRank:.1f}\n")
        f.write(f"- avg answer-end EOS rank: {avgAnswerEndEosRank:.1f}\n")
        f.write(f"- avg answer-end EOS prob: {avgAnswerEndEosProb:.6f}\n")
        f.write(f"- avg answer-end END-first rank: {avgAnswerEndEndFirstRank:.1f}\n")
        f.write(f"- avg answer-end END-first prob: {avgAnswerEndEndFirstProb:.6f}\n\n")
        f.write(f"- avg answer-end END-sequence max rank: {avgAnswerEndEndSeqMaxRank:.1f}\n")
        f.write(f"- avg answer-end END-sequence last-token rank: {avgAnswerEndEndSeqLastRank:.1f}\n\n")

        f.write("Interpretation:\n\n")
        f.write("- At `prompt_start`, EOS should usually be low because the model should begin answering.\n")
        f.write("- At `answer_end`, END-first should be high if the model has learned the visible `<END>` stop marker.\n")
        f.write("- EOS is still reported for comparison with earlier experiments.\n\n")

        f.write("## Worst Answer-end EOS Ranks\n\n")
        for row in worstAnswerEnd:
            f.write(
                f"### {row['task']} | {row['input_preview']}\n\n"
                f"- eos rank: {row['eos_rank']}\n"
                f"- eos prob: {row['eos_prob']:.8f}\n"
                f"- END-first rank: {row['end_first_rank']}\n"
                f"- END-first prob: {row['end_first_prob']:.8f}\n"
                f"- END-sequence ranks: `{row['end_sequence_ranks']}`\n"
                f"- END-sequence token previews: `{row['end_sequence_previews']}`\n"
                f"- top1: `{row['top1_preview']}` prob={row['top1_prob']:.8f}\n\n"
            )

        f.write("## Worst Answer-end END-first Ranks\n\n")
        for row in worstAnswerEndEndToken:
            f.write(
                f"### {row['task']} | {row['input_preview']}\n\n"
                f"- END-first rank: {row['end_first_rank']}\n"
                f"- END-first prob: {row['end_first_prob']:.8f}\n"
                f"- END-sequence ranks: `{row['end_sequence_ranks']}`\n"
                f"- END-sequence token previews: `{row['end_sequence_previews']}`\n"
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
                "end_token_text": result["end_token_text"],
                "end_first_id": result["end_first_id"],
                "end_first_rank": result["end_first_rank"],
                "end_first_prob": result["end_first_prob"],
                "end_sequence_ids": json.dumps(result["end_sequence_ids"], ensure_ascii=False),
                "end_sequence_previews": json.dumps(result["end_sequence_previews"], ensure_ascii=False),
                "end_sequence_ranks": json.dumps(result["end_sequence_ranks"], ensure_ascii=False),
                "end_sequence_probs": json.dumps(result["end_sequence_probs"], ensure_ascii=False),
                "end_sequence_max_rank": result["end_sequence_max_rank"],
                "end_sequence_last_rank": result["end_sequence_last_rank"],
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
                        "is_end_first": topToken["is_end_first"],
                    }
                )

    write_outputs(args, rows, topRows)


if __name__ == "__main__":
    main()
