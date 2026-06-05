from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import os

import torch
import tiktoken

from model import BigramLanguageModel, GPTConfig


PROMPTS = [
    (
        "Explain GNSS",
        "Instruction:\n"
        "Explain the geodesy or astronomy concept in one paragraph.\n\n"
        "Input:\n"
        "GNSS time series\n\n"
        "Answer:\n"
    ),
    (
        "Extract Fields",
        "Instruction:\n"
        "Extract the station, signal, value, and unit from the text.\n\n"
        "Input:\n"
        "Station BJFS shows a vertical velocity of 2.4 mm/yr from space geodetic observations.\n\n"
        "Answer:\n"
    ),
    (
        "Convert Format",
        "Instruction:\n"
        "Convert the observation into a compact JSON object.\n\n"
        "Input:\n"
        "VLBI baseline residual: station=WETTZELL, delay=12 ps, band=X.\n\n"
        "Answer:\n"
    ),
]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_top_k_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item == "":
            continue
        if item.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(int(item))
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out/sft_generation_diagnostics")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperatures", type=str, default="0.5,0.7,1.0")
    parser.add_argument("--top-ks", type=str, default="20,40")
    parser.add_argument("--repetition-penalties", type=str, default="1.0")
    parser.add_argument("--stop-text", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=2)
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
    vocabType = vocabInfo.get("type", "char")
    if vocabType != "tokenizer":
        raise ValueError("diagnose_sft_generation.py 目前只支持 tokenizer checkpoint。")

    enc = tiktoken.get_encoding(vocabInfo["meta"]["encoding"])
    return model, enc


def max_token_run(tokenIds):
    if not tokenIds:
        return 0

    best = 1
    current = 1

    for prev, item in zip(tokenIds, tokenIds[1:]):
        if item == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return best


def repeated_ngram_ratio(tokenIds, n):
    if len(tokenIds) < n:
        return 0.0

    ngrams = [
        tuple(tokenIds[i:i + n])
        for i in range(len(tokenIds) - n + 1)
    ]

    if not ngrams:
        return 0.0

    repeated = len(ngrams) - len(set(ngrams))
    return repeated / len(ngrams)


def diagnose_sample(model, enc, prompt, temperature, topK, repetitionPenalty, stopText, maxNewTokens, device):
    promptIds = enc.encode(prompt)
    context = torch.tensor([promptIds], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            context,
            maxNewTokens,
            temperature=temperature,
            topK=topK,
            repetitionPenalty=repetitionPenalty,
            repetitionStart=len(promptIds),
        )

    generatedIds = generated[0].tolist()
    newIds = generatedIds[len(promptIds):]
    eosId = enc.eot_token

    hitEos = eosId in newIds
    eosStep = None
    completionIds = newIds
    hitStopText = False
    stopTextCharPos = None

    if hitEos:
        eosIndex = newIds.index(eosId)
        eosStep = eosIndex + 1
        completionIds = newIds[:eosIndex]

    completionText = enc.decode(completionIds)
    if stopText is not None and stopText in completionText:
        hitStopText = True
        stopTextCharPos = completionText.find(stopText)
        completionText = completionText[:stopTextCharPos]
        completionIds = enc.encode(completionText)

    uniqueRatio = len(set(completionIds)) / len(completionIds) if completionIds else 0.0

    return {
        "hit_eos": hitEos,
        "eos_step": eosStep,
        "hit_stop_text": hitStopText,
        "stop_text_char_pos": stopTextCharPos,
        "generated_tokens": len(newIds),
        "completion_tokens": len(completionIds),
        "unique_token_ratio": uniqueRatio,
        "max_token_run": max_token_run(completionIds),
        "repeat_bigram_ratio": repeated_ngram_ratio(completionIds, 2),
        "repeat_trigram_ratio": repeated_ngram_ratio(completionIds, 3),
        "completion": completionText,
    }


def summarize(rows):
    total = len(rows)
    eosCount = sum(1 for row in rows if row["hit_eos"])
    stopTextCount = sum(1 for row in rows if row["hit_stop_text"])
    emptyCount = sum(1 for row in rows if row["completion_tokens"] == 0)
    avgLen = sum(row["completion_tokens"] for row in rows) / total
    avgUnique = sum(row["unique_token_ratio"] for row in rows) / total
    avgBigramRepeat = sum(row["repeat_bigram_ratio"] for row in rows) / total
    maxRun = max(row["max_token_run"] for row in rows)

    return {
        "total": total,
        "eos_count": eosCount,
        "eos_rate": eosCount / total,
        "stop_text_count": stopTextCount,
        "stop_text_rate": stopTextCount / total,
        "empty_count": emptyCount,
        "empty_rate": emptyCount / total,
        "avg_completion_tokens": avgLen,
        "avg_unique_token_ratio": avgUnique,
        "avg_repeat_bigram_ratio": avgBigramRepeat,
        "max_token_run": maxRun,
    }


def summarize_by_setting(rows):
    groups = {}

    for row in rows:
        key = (row["temperature"], row["top_k"], row["repetition_penalty"])
        groups.setdefault(key, []).append(row)

    summaryRows = []
    for (temperature, topK, repetitionPenalty), items in sorted(groups.items()):
        total = len(items)
        eosCount = sum(1 for row in items if row["hit_eos"])
        stopTextCount = sum(1 for row in items if row["hit_stop_text"])
        emptyCount = sum(1 for row in items if row["completion_tokens"] == 0)
        avgLen = sum(row["completion_tokens"] for row in items) / total
        avgBigramRepeat = sum(row["repeat_bigram_ratio"] for row in items) / total
        maxRun = max(row["max_token_run"] for row in items)

        summaryRows.append(
            {
                "temperature": temperature,
                "top_k": topK,
                "repetition_penalty": repetitionPenalty,
                "total": total,
                "eos_rate": eosCount / total,
                "stop_text_rate": stopTextCount / total,
                "empty_rate": emptyCount / total,
                "avg_completion_tokens": avgLen,
                "avg_repeat_bigram_ratio": avgBigramRepeat,
                "max_token_run": maxRun,
            }
        )

    return summaryRows


def escape_preview(text, limit=220):
    escaped = text.encode("unicode_escape").decode("ascii")
    if len(escaped) > limit:
        return escaped[:limit] + "..."
    return escaped


def write_outputs(args, rows, summary):
    os.makedirs(args.out_dir, exist_ok=True)
    csvPath = os.path.join(args.out_dir, "diagnostics.csv")
    reportPath = os.path.join(args.out_dir, "report.md")
    settingSummaryRows = summarize_by_setting(rows)

    fieldnames = [
        "prompt_name",
        "temperature",
        "top_k",
        "repetition_penalty",
        "sample_id",
        "hit_eos",
        "eos_step",
        "hit_stop_text",
        "stop_text_char_pos",
        "generated_tokens",
        "completion_tokens",
        "unique_token_ratio",
        "max_token_run",
        "repeat_bigram_ratio",
        "repeat_trigram_ratio",
        "escaped_completion_preview",
        "completion",
    ]

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "escaped_completion_preview": escape_preview(row["completion"]),
                }
            )

    worstRows = sorted(
        rows,
        key=lambda row: (
            row["repeat_bigram_ratio"],
            row["max_token_run"],
            row["completion_tokens"],
        ),
        reverse=True,
    )[:6]

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# SFT Generation Diagnostics\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n\n")
        f.write(f"Max new tokens: `{args.max_new_tokens}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- total samples: {summary['total']}\n")
        f.write(f"- eos rate: {summary['eos_rate']:.2%} ({summary['eos_count']}/{summary['total']})\n")
        if args.stop_text is not None:
            f.write(f"- stop-text rate: {summary['stop_text_rate']:.2%} ({summary['stop_text_count']}/{summary['total']})\n")
        f.write(f"- empty completion rate: {summary['empty_rate']:.2%} ({summary['empty_count']}/{summary['total']})\n")
        f.write(f"- avg completion tokens: {summary['avg_completion_tokens']:.1f}\n")
        f.write(f"- avg unique token ratio: {summary['avg_unique_token_ratio']:.3f}\n")
        f.write(f"- avg repeated bigram ratio: {summary['avg_repeat_bigram_ratio']:.3f}\n")
        f.write(f"- max repeated token run: {summary['max_token_run']}\n\n")

        f.write("## By Sampling Setting\n\n")
        f.write("| temperature | top_k | repetition penalty | eos rate | stop-text rate | empty rate | avg length | avg repeated bigram | max token run |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in settingSummaryRows:
            f.write(
                f"| {row['temperature']} | {row['top_k']} | "
                f"{row['repetition_penalty']} | "
                f"{row['eos_rate']:.2%} | {row['stop_text_rate']:.2%} | "
                f"{row['empty_rate']:.2%} | "
                f"{row['avg_completion_tokens']:.1f} | "
                f"{row['avg_repeat_bigram_ratio']:.3f} | "
                f"{row['max_token_run']} |\n"
            )
        f.write("\n")

        f.write("## Worst Repetition Samples\n\n")
        for row in worstRows:
            f.write(
                f"### {row['prompt_name']} | temp={row['temperature']} | "
                f"top_k={row['top_k']} | penalty={row['repetition_penalty']} | "
                f"sample={row['sample_id']}\n\n"
            )
            f.write(
                f"- hit_eos: {row['hit_eos']}\n"
                f"- eos_step: {row['eos_step']}\n"
                f"- hit_stop_text: {row['hit_stop_text']}\n"
                f"- stop_text_char_pos: {row['stop_text_char_pos']}\n"
                f"- completion_tokens: {row['completion_tokens']}\n"
                f"- repeat_bigram_ratio: {row['repeat_bigram_ratio']:.3f}\n"
                f"- max_token_run: {row['max_token_run']}\n\n"
            )
            f.write("Escaped preview:\n\n")
            f.write("```text\n")
            f.write(escape_preview(row["completion"]))
            f.write("\n```\n\n")
            f.write("```text\n")
            f.write(row["completion"])
            f.write("\n```\n\n")

    print(f"saved csv to {csvPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    temperatures = parse_float_list(args.temperatures)
    topKs = parse_top_k_list(args.top_ks)
    repetitionPenalties = parse_float_list(args.repetition_penalties)

    useMps = os.environ.get("USE_MPS") == "1"
    device = "mps" if torch.backends.mps.is_available() and useMps else "cpu"
    print(f"using device: {device}", flush=True)

    model, enc = load_model(args.checkpoint, device)
    rows = []

    for promptName, prompt in PROMPTS:
        for temperature in temperatures:
            for topK in topKs:
                for repetitionPenalty in repetitionPenalties:
                    for sampleId in range(args.num_samples):
                        torch.manual_seed(args.seed + len(rows))
                        print(
                            f"diagnosing: {promptName}, temp={temperature}, top_k={topK}, "
                            f"penalty={repetitionPenalty}, sample={sampleId}",
                            flush=True,
                        )
                        metrics = diagnose_sample(
                            model,
                            enc,
                            prompt,
                            temperature,
                            topK,
                            repetitionPenalty,
                            args.stop_text,
                            args.max_new_tokens,
                            device,
                        )
                        rows.append(
                            {
                                "prompt_name": promptName,
                                "temperature": temperature,
                                "top_k": topK if topK is not None else "none",
                                "repetition_penalty": repetitionPenalty,
                                "sample_id": sampleId,
                                **metrics,
                            }
                        )

    summary = summarize(rows)
    write_outputs(args, rows, summary)


if __name__ == "__main__":
    main()
