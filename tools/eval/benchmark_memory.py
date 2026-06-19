from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import os
import resource
import subprocess
import time

import torch
import tiktoken

from model import BigramLanguageModel, GPTConfig


DEFAULT_PROMPT = (
    "Instruction:\n"
    "Extract the station, signal, value, and unit from the text.\n\n"
    "Input:\n"
    "ONSA reports vertical velocity of 2.4 mm/yr.\n\n"
    "Answer:\n"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true")
    parser.add_argument("--use-mps", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out/memory_benchmark")
    return parser.parse_args()


def current_rss_mb():
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        capture_output=True,
        text=True,
    )
    rssKb = int(result.stdout.strip())
    return rssKb / 1024.0


def peak_rss_mb():
    maxRss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maxRss / (1024.0 * 1024.0)
    return maxRss / 1024.0


def device_memory_mb(device):
    if device == "cuda" and torch.cuda.is_available():
        return {
            "device_allocated_mb": torch.cuda.memory_allocated() / (1024.0 * 1024.0),
            "device_reserved_mb": torch.cuda.memory_reserved() / (1024.0 * 1024.0),
        }

    if device == "mps" and hasattr(torch, "mps"):
        allocated = None
        driver = None
        if hasattr(torch.mps, "current_allocated_memory"):
            allocated = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
        if hasattr(torch.mps, "driver_allocated_memory"):
            driver = torch.mps.driver_allocated_memory() / (1024.0 * 1024.0)
        return {
            "device_allocated_mb": allocated,
            "device_reserved_mb": driver,
        }

    return {
        "device_allocated_mb": None,
        "device_reserved_mb": None,
    }


def snapshot(stage, device):
    deviceStats = device_memory_mb(device)
    return {
        "stage": stage,
        "rss_mb": current_rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
        "device_allocated_mb": deviceStats["device_allocated_mb"],
        "device_reserved_mb": deviceStats["device_reserved_mb"],
    }


def format_optional(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def sync_if_needed(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


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
        raise ValueError("benchmark_memory.py 目前只支持 tokenizer checkpoint。")
    enc = tiktoken.get_encoding(vocabInfo["meta"]["encoding"])
    return model, enc


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


def write_outputs(args, device, snapshots, rows, summary):
    os.makedirs(args.out_dir, exist_ok=True)
    csvPath = os.path.join(args.out_dir, "memory.csv")
    runsPath = os.path.join(args.out_dir, "runs.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "rss_mb",
                "peak_rss_mb",
                "device_allocated_mb",
                "device_reserved_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(snapshots)

    with open(runsPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "latency_sec",
                "new_tokens",
                "tokens_per_sec",
                "rss_mb",
                "peak_rss_mb",
                "device_allocated_mb",
                "device_reserved_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# Memory Benchmark\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n")
        f.write(f"Device: `{device}`\n")
        f.write(f"Use KV cache: `{args.use_kv_cache}`\n")
        f.write(f"Max new tokens: `{args.max_new_tokens}`\n")
        f.write(f"Runs: `{args.num_runs}`\n\n")

        f.write("## Memory Snapshots\n\n")
        f.write("| stage | rss MB | peak rss MB | device allocated MB | device reserved MB |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for item in snapshots:
            f.write(
                f"| {item['stage']} "
                f"| {item['rss_mb']:.2f} "
                f"| {item['peak_rss_mb']:.2f} "
                f"| {format_optional(item['device_allocated_mb'])} "
                f"| {format_optional(item['device_reserved_mb'])} |\n"
            )

        f.write("\n## Summary\n\n")
        f.write(f"- rss before load: {summary['rss_before_load_mb']:.2f} MB\n")
        f.write(f"- rss after load: {summary['rss_after_load_mb']:.2f} MB\n")
        f.write(f"- model load rss delta: {summary['model_load_delta_mb']:.2f} MB\n")
        f.write(f"- peak rss after generation: {summary['peak_after_generation_mb']:.2f} MB\n")
        f.write(f"- avg latency: {summary['avg_latency_sec']:.4f} sec\n")
        f.write(f"- avg tokens/s: {summary['avg_tokens_per_sec']:.2f}\n\n")

        f.write("## Files\n\n")
        f.write(f"- `{csvPath}`\n")
        f.write(f"- `{runsPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved memory snapshots to {csvPath}")
    print(f"saved run details to {runsPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() and args.use_mps else "cpu"
    print(f"using device: {device}", flush=True)

    snapshots = [snapshot("before_load", device)]

    model, enc = load_model(args.checkpoint, device)
    promptIds = enc.encode(args.prompt)
    context = torch.tensor(
        [promptIds],
        dtype=torch.long,
        device=device,
    )
    eosTokenId = enc.eot_token if args.stop_at_eos else None
    snapshots.append(snapshot("after_load", device))

    for _ in range(args.warmup_runs):
        run_once(model, context, args, device, eosTokenId)
    snapshots.append(snapshot("after_warmup", device))

    rows = []
    for runIdx in range(args.num_runs):
        elapsed, newTokens, _ = run_once(model, context, args, device, eosTokenId)
        tokensPerSec = newTokens / elapsed if elapsed > 0 else 0.0
        mem = snapshot(f"after_run_{runIdx}", device)
        rows.append(
            {
                "run": runIdx,
                "latency_sec": elapsed,
                "new_tokens": newTokens,
                "tokens_per_sec": tokensPerSec,
                "rss_mb": mem["rss_mb"],
                "peak_rss_mb": mem["peak_rss_mb"],
                "device_allocated_mb": mem["device_allocated_mb"],
                "device_reserved_mb": mem["device_reserved_mb"],
            }
        )
        snapshots.append(mem)
        print(
            f"run {runIdx}: {elapsed:.4f}s, {tokensPerSec:.2f} tok/s, "
            f"rss {mem['rss_mb']:.2f} MB, peak {mem['peak_rss_mb']:.2f} MB",
            flush=True,
        )

    latencies = [row["latency_sec"] for row in rows]
    tokensPerSecList = [row["tokens_per_sec"] for row in rows]
    summary = {
        "rss_before_load_mb": snapshots[0]["rss_mb"],
        "rss_after_load_mb": snapshots[1]["rss_mb"],
        "model_load_delta_mb": snapshots[1]["rss_mb"] - snapshots[0]["rss_mb"],
        "peak_after_generation_mb": max(item["peak_rss_mb"] for item in snapshots),
        "avg_latency_sec": sum(latencies) / len(latencies),
        "avg_tokens_per_sec": sum(tokensPerSecList) / len(tokensPerSecList),
    }

    write_outputs(args, device, snapshots, rows, summary)


if __name__ == "__main__":
    main()
