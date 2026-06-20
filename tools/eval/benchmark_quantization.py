from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import io
import os
import time

import torch
import torch.nn as nn
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
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--qengine", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="out/quantization_benchmark")
    return parser.parse_args()


def load_model(checkpointPath):
    checkpoint = torch.load(
        checkpointPath,
        map_location="cpu",
        weights_only=False,
    )
    config = GPTConfig(**checkpoint["config"])
    model = BigramLanguageModel(
        config.vocabSize,
        config.blockSize,
        config=config,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    vocabInfo = checkpoint["vocab"]
    if vocabInfo.get("type", "char") != "tokenizer":
        raise ValueError("benchmark_quantization.py 目前只支持 tokenizer checkpoint。")
    enc = tiktoken.get_encoding(vocabInfo["meta"]["encoding"])
    return model, enc


def quantize_model(model):
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )


def setup_quantized_engine(engine):
    supported = [item for item in torch.backends.quantized.supported_engines if item != "none"]
    if engine == "auto":
        if not supported:
            raise RuntimeError("当前 PyTorch 没有可用的 quantized backend。")
        engine = "qnnpack" if "qnnpack" in supported else supported[0]
    if engine not in supported:
        raise RuntimeError(f"不支持的 quantized backend: {engine}; 可选: {supported}")
    torch.backends.quantized.engine = engine
    return engine


def serialized_state_dict_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024.0 * 1024.0)


def parameter_bytes_mb(model):
    totalBytes = 0
    for param in model.parameters():
        totalBytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        totalBytes += buffer.numel() * buffer.element_size()
    return totalBytes / (1024.0 * 1024.0)


def count_modules(model):
    linearCount = 0
    quantizedLinearCount = 0
    for module in model.modules():
        moduleName = module.__class__.__name__.lower()
        modulePackage = module.__class__.__module__.lower()
        if isinstance(module, nn.Linear):
            linearCount += 1
        if "quantized" in modulePackage and "linear" in moduleName:
            quantizedLinearCount += 1
    return {
        "linear_modules": linearCount,
        "quantized_linear_modules": quantizedLinearCount,
    }


@torch.no_grad()
def run_once(model, context, args, eosTokenId, seed):
    torch.manual_seed(seed)
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
    elapsed = time.perf_counter() - start
    newTokens = generated.shape[1] - context.shape[1]
    tokensPerSec = newTokens / elapsed if elapsed > 0 else 0.0
    return elapsed, newTokens, tokensPerSec, generated


def benchmark_model(name, model, context, args, eosTokenId):
    for warmupIdx in range(args.warmup_runs):
        run_once(model, context, args, eosTokenId, args.seed + warmupIdx)

    rows = []
    generated = None
    for runIdx in range(args.num_runs):
        elapsed, newTokens, tokensPerSec, generated = run_once(
            model,
            context,
            args,
            eosTokenId,
            args.seed + 1000 + runIdx,
        )
        rows.append(
            {
                "mode": name,
                "run": runIdx,
                "latency_sec": elapsed,
                "new_tokens": newTokens,
                "tokens_per_sec": tokensPerSec,
            }
        )
        print(
            f"{name} run {runIdx}: {elapsed:.4f}s, {tokensPerSec:.2f} tok/s",
            flush=True,
        )
    return rows, generated


def summarize_rows(rows):
    latencies = [row["latency_sec"] for row in rows]
    tokensPerSec = [row["tokens_per_sec"] for row in rows]
    return {
        "avg_latency_sec": sum(latencies) / len(latencies),
        "min_latency_sec": min(latencies),
        "max_latency_sec": max(latencies),
        "avg_tokens_per_sec": sum(tokensPerSec) / len(tokensPerSec),
        "min_tokens_per_sec": min(tokensPerSec),
        "max_tokens_per_sec": max(tokensPerSec),
    }


def write_outputs(args, rows, summaries, sampleTexts):
    os.makedirs(args.out_dir, exist_ok=True)
    detailsPath = os.path.join(args.out_dir, "details.csv")
    summaryPath = os.path.join(args.out_dir, "summary.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(detailsPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "run",
                "latency_sec",
                "new_tokens",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(summaryPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "state_dict_mb",
                "parameter_buffer_mb",
                "linear_modules",
                "quantized_linear_modules",
                "avg_latency_sec",
                "avg_tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# Quantization Benchmark\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n")
        f.write("Device: `cpu`\n")
        f.write("Quantization: `torch dynamic int8 for nn.Linear`\n")
        f.write(f"Quantized engine: `{args.qengine}`\n")
        f.write(f"Max new tokens: `{args.max_new_tokens}`\n")
        f.write(f"Use KV cache: `{args.use_kv_cache}`\n")
        f.write(f"Runs: `{args.num_runs}`\n\n")

        f.write("## Summary\n\n")
        f.write("| mode | state dict MB | param/buffer MB | Linear | quantized Linear | avg latency | avg tok/s |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries:
            f.write(
                f"| {row['mode']} "
                f"| {row['state_dict_mb']:.2f} "
                f"| {row['parameter_buffer_mb']:.2f} "
                f"| {row['linear_modules']} "
                f"| {row['quantized_linear_modules']} "
                f"| {row['avg_latency_sec']:.4f}s "
                f"| {row['avg_tokens_per_sec']:.2f} |\n"
            )

        f.write("\n## Sample Outputs\n\n")
        for mode, text in sampleTexts.items():
            f.write(f"### {mode}\n\n")
            f.write("```text\n")
            f.write(text)
            f.write("\n```\n\n")

        f.write("## Files\n\n")
        f.write(f"- `{detailsPath}`\n")
        f.write(f"- `{summaryPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved details to {detailsPath}")
    print(f"saved summary to {summaryPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    args.qengine = setup_quantized_engine(args.qengine)
    print(f"quantized engine: {args.qengine}", flush=True)

    fp32Model, enc = load_model(args.checkpoint)
    int8Model = quantize_model(fp32Model)

    promptIds = enc.encode(args.prompt)
    context = torch.tensor([promptIds], dtype=torch.long)
    eosTokenId = enc.eot_token if args.stop_at_eos else None

    models = {
        "fp32": fp32Model,
        "dynamic_int8": int8Model,
    }

    allRows = []
    summaries = []
    sampleTexts = {}
    for mode, model in models.items():
        modelRows, generated = benchmark_model(mode, model, context, args, eosTokenId)
        allRows.extend(modelRows)
        timingSummary = summarize_rows(modelRows)
        moduleSummary = count_modules(model)
        summaries.append(
            {
                "mode": mode,
                "state_dict_mb": serialized_state_dict_mb(model),
                "parameter_buffer_mb": parameter_bytes_mb(model),
                "linear_modules": moduleSummary["linear_modules"],
                "quantized_linear_modules": moduleSummary["quantized_linear_modules"],
                "avg_latency_sec": timingSummary["avg_latency_sec"],
                "avg_tokens_per_sec": timingSummary["avg_tokens_per_sec"],
            }
        )
        sampleTexts[mode] = enc.decode(generated[0].tolist())

    write_outputs(args, allRows, summaries, sampleTexts)


if __name__ == "__main__":
    main()
