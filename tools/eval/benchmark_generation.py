from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import os
import time

import torch
import tiktoken

from model import BigramLanguageModel, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Instruction:\nExtract the station, signal, value, and unit from the text.\n\nInput:\nONSA reports vertical velocity of 2.4 mm/yr.\n\nAnswer:\n")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="out/generation_benchmark")
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true")
    return parser.parse_args()


def load_model(checkpointPath, device):
    checkpoint = torch.load(
        checkpointPath,
        map_location=device,
        weights_only=False,
    )
    config = GPTConfig(**checkpoint["config"])

    model = BigramLanguageModel(
        config.vocabSize,
        config.blockSize,
        config=config,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    vocabInfo = checkpoint["vocab"]
    if vocabInfo.get("type", "char") != "tokenizer":
        raise ValueError("benchmark_generation.py 目前只支持 tokenizer checkpoint。")

    enc = tiktoken.get_encoding(vocabInfo["meta"]["encoding"])
    return model, enc


def sync_if_needed(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_once(model, context, args, device, eosTokenId):
    sync_if_needed(device)
    start = time.perf_counter()
    generated = model.generate(
        context,
        args.max_new_tokens,
        temperature=args.temperature,
        topK=args.top_k,
        repetitionPenalty=args.repetition_penalty,
        repetitionStart=context.shape[1],
        eosTokenId=eosTokenId,
        useKvCache=args.use_kv_cache,
    )
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    newTokens = generated.shape[1] - context.shape[1]
    return elapsed, newTokens, generated


def summarize(rows):
    latencies = [row["latency_sec"] for row in rows]
    tokensPerSec = [row["tokens_per_sec"] for row in rows]
    return {
        "runs": len(rows),
        "avg_latency_sec": sum(latencies) / len(latencies),
        "min_latency_sec": min(latencies),
        "max_latency_sec": max(latencies),
        "avg_tokens_per_sec": sum(tokensPerSec) / len(tokensPerSec),
        "min_tokens_per_sec": min(tokensPerSec),
        "max_tokens_per_sec": max(tokensPerSec),
    }


def write_outputs(args, rows, summary, generatedText, device, promptTokens):
    os.makedirs(args.out_dir, exist_ok=True)
    csvPath = os.path.join(args.out_dir, "benchmark.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "latency_sec",
                "new_tokens",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# Generation Benchmark\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n")
        f.write(f"Device: `{device}`\n")
        f.write(f"Prompt tokens: `{promptTokens}`\n")
        f.write(f"Max new tokens: `{args.max_new_tokens}`\n")
        f.write(f"Use KV cache: `{args.use_kv_cache}`\n")
        f.write(f"Runs: `{summary['runs']}`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- avg latency: {summary['avg_latency_sec']:.4f} sec\n")
        f.write(f"- min latency: {summary['min_latency_sec']:.4f} sec\n")
        f.write(f"- max latency: {summary['max_latency_sec']:.4f} sec\n")
        f.write(f"- avg tokens/s: {summary['avg_tokens_per_sec']:.2f}\n")
        f.write(f"- min tokens/s: {summary['min_tokens_per_sec']:.2f}\n")
        f.write(f"- max tokens/s: {summary['max_tokens_per_sec']:.2f}\n\n")

        f.write("## Sample Output\n\n")
        f.write("```text\n")
        f.write(generatedText)
        f.write("\n```\n\n")

        f.write("## Files\n\n")
        f.write(f"- `{csvPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved csv to {csvPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()

    useMps = os.environ.get("USE_MPS") == "1"
    device = "mps" if torch.backends.mps.is_available() and useMps else "cpu"
    print(f"using device: {device}", flush=True)

    model, enc = load_model(args.checkpoint, device)
    promptIds = enc.encode(args.prompt)
    context = torch.tensor(
        [promptIds],
        dtype=torch.long,
        device=device,
    )

    if context.shape[1] > model.config.blockSize:
        raise ValueError(
            f"prompt tokens={context.shape[1]} 超过 blockSize={model.config.blockSize}"
        )
    if (
        args.use_kv_cache
        and not model.config.useRoPE
        and context.shape[1] + args.max_new_tokens > model.config.blockSize
    ):
        raise ValueError(
            "非 RoPE 模型使用 use-kv-cache 时，prompt tokens + max-new-tokens "
            f"不超过 blockSize={model.config.blockSize}"
        )

    eosTokenId = enc.eot_token if args.stop_at_eos else None

    for _ in range(args.warmup_runs):
        run_once(model, context, args, device, eosTokenId)

    rows = []
    generated = None

    for runIdx in range(args.num_runs):
        elapsed, newTokens, generated = run_once(
            model,
            context,
            args,
            device,
            eosTokenId,
        )
        tokensPerSec = newTokens / elapsed if elapsed > 0 else 0.0
        rows.append(
            {
                "run": runIdx,
                "latency_sec": elapsed,
                "new_tokens": newTokens,
                "tokens_per_sec": tokensPerSec,
            }
        )
        print(
            f"run {runIdx}: {elapsed:.4f}s, {tokensPerSec:.2f} tok/s",
            flush=True,
        )

    summary = summarize(rows)
    generatedText = enc.decode(generated[0].tolist())
    write_outputs(
        args,
        rows,
        summary,
        generatedText,
        device,
        len(promptIds),
    )


if __name__ == "__main__":
    main()
