from pathlib import Path
import argparse
import csv
import os
import time

import torch
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


DEFAULT_PROMPT = """Instruction:
Extract the station, signal, value, and unit from the text.

Input:
ONSA reports vertical velocity of 2.4 mm/yr.

Answer:
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--random-gpt2", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--n-positions", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=112)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
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
    if args.random_gpt2:
        config = GPT2Config(
            vocab_size=args.vocab_size,
            n_positions=args.n_positions,
            n_ctx=args.n_positions,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            bos_token_id=50256,
            eos_token_id=50256,
            pad_token_id=50256,
        )
        model = GPT2LMHeadModel(config)
        model.to(device)
        model.eval()
        enc = tiktoken.get_encoding("gpt2")
        return model, enc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_once(model, tokenizer, args, device):
    if args.random_gpt2:
        promptIds = tokenizer.encode(args.prompt)
        inputIds = torch.tensor([promptIds], dtype=torch.long, device=device)
        attentionMask = torch.ones_like(inputIds)
        padTokenId = 50256
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt")
        inputIds = inputs["input_ids"].to(device)
        attentionMask = inputs.get("attention_mask")
        if attentionMask is not None:
            attentionMask = attentionMask.to(device)
        padTokenId = tokenizer.eos_token_id

    generationKwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": padTokenId,
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
    if args.random_gpt2:
        text = tokenizer.decode(generated[0].tolist())
    else:
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
        modelName = "random-gpt2" if args.random_gpt2 else args.model_name
        f.write(f"Model: `{modelName}`\n")
        f.write(f"Device: `{device}`\n")
        f.write(f"Parameters: `{summary['num_params']}`\n")
        if args.random_gpt2:
            f.write(
                "Config: "
                f"`n_embd={args.n_embd}, n_layer={args.n_layer}, "
                f"n_head={args.n_head}, n_positions={args.n_positions}`\n"
            )
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
    if args.random_gpt2:
        print(
            "building random GPT-2: "
            f"n_embd={args.n_embd}, n_layer={args.n_layer}, "
            f"n_head={args.n_head}, n_positions={args.n_positions}",
            flush=True,
        )
    else:
        print(f"loading model: {args.model_name}", flush=True)

    model, tokenizer = load_model(args, device)
    numParams = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {numParams / 1e6:.2f}M", flush=True)

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
    summary["num_params"] = numParams
    write_outputs(args, rows, summary, sampleText, device, promptTokens)


if __name__ == "__main__":
    main()
