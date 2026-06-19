import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[2]


CONFIGS = {
    "fixed_w1": {
        "dynamic_max_concurrent_batches": 1,
        "extra_args": [],
    },
    "fixed_w2": {
        "dynamic_max_concurrent_batches": 2,
        "extra_args": [],
    },
    "adaptive_w2": {
        "dynamic_max_concurrent_batches": 2,
        "extra_args": [
            "--dynamic-adaptive-wait",
            "--dynamic-min-wait-ms", "1",
            "--dynamic-max-wait-ms", "8",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--configs", type=str, default="fixed_w1,fixed_w2,adaptive_w2")
    parser.add_argument("--concurrency", type=str, default="4,8,12")
    parser.add_argument("--requests-per-level", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true")
    parser.add_argument("--vary-prompts", action="store_true")
    parser.add_argument("--dynamic-max-batch-size", type=int, default=8)
    parser.add_argument("--dynamic-wait-ms", type=float, default=5.0)
    parser.add_argument("--use-mps", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out/dynamic_scheduler_benchmark")
    return parser.parse_args()


def parse_configs(text):
    configs = []
    for item in text.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in CONFIGS:
            raise ValueError(f"未知 config: {name}; 可选: {sorted(CONFIGS)}")
        configs.append(name)
    if not configs:
        raise ValueError("至少需要一个 config")
    return configs


def get_json(url, timeout=10):
    with urlrequest.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(baseUrl, timeoutSec=60):
    start = time.perf_counter()
    lastError = None
    while time.perf_counter() - start < timeoutSec:
        try:
            return get_json(baseUrl.rstrip("/") + "/health", timeout=5)
        except URLError as exc:
            lastError = exc
        except TimeoutError as exc:
            lastError = exc
        time.sleep(0.5)
    raise RuntimeError(f"服务启动超时: {lastError}")


def start_server(args, configName, logPath):
    config = CONFIGS[configName]
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "serve" / "serve_fastapi.py"),
        "--checkpoint", args.checkpoint,
        "--host", args.host,
        "--port", str(args.port),
        "--dynamic-max-batch-size", str(args.dynamic_max_batch_size),
        "--dynamic-wait-ms", str(args.dynamic_wait_ms),
        "--dynamic-max-concurrent-batches",
        str(config["dynamic_max_concurrent_batches"]),
        *config["extra_args"],
    ]
    if args.use_mps:
        cmd.append("--use-mps")

    logFile = open(logPath, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=logFile,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, logFile


def stop_server(process, logFile):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    logFile.close()


def run_benchmark(args, configName, runDir):
    baseUrl = f"http://{args.host}:{args.port}"
    benchmarkCmd = [
        sys.executable,
        str(ROOT / "tools" / "eval" / "benchmark_api_concurrency.py"),
        "--url", baseUrl.rstrip("/") + "/generate_dynamic",
        "--concurrency", args.concurrency,
        "--requests-per-level", str(args.requests_per_level),
        "--warmup-runs", str(args.warmup_runs),
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--out-dir", str(runDir),
    ]
    if args.stop_at_eos:
        benchmarkCmd.append("--stop-at-eos")
    if args.use_kv_cache:
        benchmarkCmd.append("--use-kv-cache")
    if args.vary_prompts:
        benchmarkCmd.append("--vary-prompts")

    subprocess.run(benchmarkCmd, cwd=ROOT, check=True)
    stats = get_json(baseUrl.rstrip("/") + "/dynamic_stats")
    statsPath = runDir / "dynamic_stats.json"
    with open(statsPath, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def read_summary_rows(configName, runDir, stats):
    summaryPath = runDir / "summary.csv"
    rows = []
    with open(summaryPath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["config"] = configName
            row["adaptive_wait"] = stats.get("adaptive_wait", False)
            row["max_concurrent_batches"] = stats.get("max_concurrent_batches", "")
            row["avg_flush_wait_ms"] = stats.get("avg_flush_wait_ms", "")
            row["service_avg_batch_size"] = stats.get("avg_batch_size", "")
            row["service_avg_padding_ratio"] = stats.get("avg_padding_ratio", "")
            rows.append(row)
    return rows


def write_aggregate(outDir, rows):
    csvPath = outDir / "scheduler_summary.csv"
    reportPath = outDir / "report.md"
    fieldnames = [
        "config",
        "concurrency",
        "requests",
        "requests_per_sec",
        "output_tokens_per_sec",
        "avg_end_to_end_latency_sec",
        "p95_end_to_end_latency_sec",
        "avg_dynamic_batch_size",
        "avg_queue_wait_ms",
        "avg_batch_latency_ms",
        "avg_padding_ratio",
        "avg_flush_wait_ms",
        "max_concurrent_batches",
        "adaptive_wait",
        "batch_size_histogram",
    ]

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# Dynamic Scheduler Benchmark\n\n")
        f.write("## Summary\n\n")
        f.write("| config | concurrency | req/s | tok/s | avg latency | p95 latency | avg batch | avg wait | avg padding |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['config']} "
                f"| {row['concurrency']} "
                f"| {float(row['requests_per_sec']):.2f} "
                f"| {float(row['output_tokens_per_sec']):.2f} "
                f"| {float(row['avg_end_to_end_latency_sec']):.4f}s "
                f"| {float(row['p95_end_to_end_latency_sec']):.4f}s "
                f"| {float(row['avg_dynamic_batch_size']):.2f} "
                f"| {float(row['avg_queue_wait_ms']):.2f}ms "
                f"| {float(row['avg_padding_ratio']):.2%} |\n"
            )

        f.write("\n## Files\n\n")
        f.write(f"- `{csvPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved scheduler summary to {csvPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    configNames = parse_configs(args.configs)
    outDir = Path(args.out_dir)
    outDir.mkdir(parents=True, exist_ok=True)

    allRows = []
    for configName in configNames:
        runDir = outDir / configName
        runDir.mkdir(parents=True, exist_ok=True)
        logPath = runDir / "server.log"

        print(f"\n=== {configName} ===", flush=True)
        process, logFile = start_server(args, configName, logPath)
        try:
            wait_for_health(f"http://{args.host}:{args.port}")
            stats = run_benchmark(args, configName, runDir)
            allRows.extend(read_summary_rows(configName, runDir, stats))
        finally:
            stop_server(process, logFile)

    write_aggregate(outDir, allRows)


if __name__ == "__main__":
    main()
