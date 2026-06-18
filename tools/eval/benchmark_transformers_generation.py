from pathlib import Path
import argparse
import csv
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = """Instruction:
Extract the station, signal, value, and unit from the text.

Input:
ONSA reports vertical velocity of 2.4 mm/yr.

Answer:
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--use-mps", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out/transformers_benchmark")
    return parser.parse_args()


def sync_if_needed(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_once(model, tokenizer, args, device):
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputIds = inputs["input_ids"].to(device)
    attentionMask = inputs.get("attention_mask")
    if attentionMask is not None:
        attentionMask = attentionMask.to(device)

    generationKwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generationKwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_k": args.top_k,
            }
        )
    else:
        generationKwargs["do_sample"] = False

    sync_if_needed(device)
    start = time.perf_counter()
    generated = model.generate(
        input_ids=inputIds,
        attention_mask=attentionMask,
        **generationKwargs,
    )
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    newTokens = generated.shape[1] - inputIds.shape[1]
    text = tokenizer.decode(generated[0], skip_special_tokens=False)
    return elapsed, inputIds.shape[1], newTokens, text


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


def write_outputs(args, rows, summary, sampleText, device, promptTokens):
    os.makedirs(args.out_dir, exist_ok=True)
    csvPath = os.path.join(args.out_dir, "benchmark.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "latency_sec",
                "prompt_tokens",
                "new_tokens",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# Transformers Generation Benchmark\n\n")
        f.write(f"Model: `{args.model_name}`\n")
        f.write(f"Device: `{device}`\n")
        f.write(f"Prompt tokens: `{promptTokens}`\n")
        f.write(f"Max new tokens: `{args.max_new_tokens}`\n")
        f.write(f"Runs: `{summary['runs']}`\n")
        f.write(f"Do sample: `{args.do_sample}`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- avg latency: {summary['avg_latency_sec']:.4f} sec\n")
        f.write(f"- min latency: {summary['min_latency_sec']:.4f} sec\n")
        f.write(f"- max latency: {summary['max_latency_sec']:.4f} sec\n")
        f.write(f"- avg tokens/s: {summary['avg_tokens_per_sec']:.2f}\n")
        f.write(f"- min tokens/s: {summary['min_tokens_per_sec']:.2f}\n")
        f.write(f"- max tokens/s: {summary['max_tokens_per_sec']:.2f}\n\n")

        f.write("## Sample Output\n\n")
        f.write("```text\n")
        f.write(sampleText)
        f.write("\n```\n\n")

        f.write("## Files\n\n")
        f.write(f"- `{csvPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved csv to {csvPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() and args.use_mps else "cpu"
    print(f"using device: {device}", flush=True)
    print(f"loading model: {args.model_name}", flush=True)

    model, tokenizer = load_model(args, device)

    for _ in range(args.warmup_runs):
        run_once(model, tokenizer, args, device)

    rows = []
    sampleText = ""
    promptTokens = 0

    for runIdx in range(args.num_runs):
        elapsed, promptTokens, newTokens, sampleText = run_once(
            model,
            tokenizer,
            args,
            device,
        )
        tokensPerSec = newTokens / elapsed if elapsed > 0 else 0.0
        rows.append(
            {
                "run": runIdx,
                "latency_sec": elapsed,
                "prompt_tokens": promptTokens,
                "new_tokens": newTokens,
                "tokens_per_sec": tokensPerSec,
            }
        )
        print(
            f"run {runIdx}: {elapsed:.4f}s, {tokensPerSec:.2f} tok/s",
            flush=True,
        )

    summary = summarize(rows)
    write_outputs(args, rows, summary, sampleText, device, promptTokens)


if __name__ == "__main__":
    main()
