import argparse
import csv
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


DEFAULT_PROMPT = """Instruction:
Extract the station, signal, value, and unit from the text.

Input:
ONSA reports vertical velocity of 2.4 mm/yr.

Answer:
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8010/generate")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--concurrency", type=str, default="1,2,4")
    parser.add_argument("--requests-per-level", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out/api_concurrency_benchmark")
    return parser.parse_args()


def parse_concurrency(text):
    levels = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("concurrency 必须大于 0")
        levels.append(value)
    if not levels:
        raise ValueError("至少需要一个 concurrency")
    return levels


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urlrequest.urlopen(req, timeout=120) as response:
            responseBody = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"请求失败，请确认服务已经启动: {exc}") from exc

    endToEndLatency = time.perf_counter() - start
    response = json.loads(responseBody)
    return response, endToEndLatency


def run_level(args, payload, concurrency, totalRequests):
    rows = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(post_json, args.url, payload)
            for _ in range(totalRequests)
        ]

        for requestIdx, future in enumerate(as_completed(futures)):
            response, endToEndLatency = future.result()
            serverLatency = response["latency_sec"]
            rows.append(
                {
                    "concurrency": concurrency,
                    "request": requestIdx,
                    "end_to_end_latency_sec": endToEndLatency,
                    "server_latency_sec": serverLatency,
                    "http_overhead_sec": max(0.0, endToEndLatency - serverLatency),
                    "new_tokens": response["new_tokens"],
                    "tokens_per_sec": response["tokens_per_sec"],
                }
            )

    wallTime = time.perf_counter() - start
    return rows, wallTime


def summarize_level(rows, wallTime):
    endToEndLatencies = [row["end_to_end_latency_sec"] for row in rows]
    serverLatencies = [row["server_latency_sec"] for row in rows]
    overheads = [row["http_overhead_sec"] for row in rows]
    newTokens = [row["new_tokens"] for row in rows]
    totalTokens = sum(newTokens)

    return {
        "concurrency": rows[0]["concurrency"],
        "requests": len(rows),
        "wall_time_sec": wallTime,
        "requests_per_sec": len(rows) / wallTime if wallTime > 0 else 0.0,
        "output_tokens_per_sec": totalTokens / wallTime if wallTime > 0 else 0.0,
        "avg_end_to_end_latency_sec": statistics.mean(endToEndLatencies),
        "p50_end_to_end_latency_sec": percentile(endToEndLatencies, 0.50),
        "p95_end_to_end_latency_sec": percentile(endToEndLatencies, 0.95),
        "max_end_to_end_latency_sec": max(endToEndLatencies),
        "avg_server_latency_sec": statistics.mean(serverLatencies),
        "avg_http_overhead_sec": statistics.mean(overheads),
        "avg_new_tokens": statistics.mean(newTokens),
    }


def write_outputs(args, allRows, summaries):
    os.makedirs(args.out_dir, exist_ok=True)
    detailPath = os.path.join(args.out_dir, "details.csv")
    summaryPath = os.path.join(args.out_dir, "summary.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(detailPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concurrency",
                "request",
                "end_to_end_latency_sec",
                "server_latency_sec",
                "http_overhead_sec",
                "new_tokens",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(allRows)

    with open(summaryPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concurrency",
                "requests",
                "wall_time_sec",
                "requests_per_sec",
                "output_tokens_per_sec",
                "avg_end_to_end_latency_sec",
                "p50_end_to_end_latency_sec",
                "p95_end_to_end_latency_sec",
                "max_end_to_end_latency_sec",
                "avg_server_latency_sec",
                "avg_http_overhead_sec",
                "avg_new_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# API Concurrency Benchmark\n\n")
        f.write(f"URL: `{args.url}`\n")
        f.write(f"Concurrency levels: `{args.concurrency}`\n")
        f.write(f"Requests per level: `{args.requests_per_level}`\n")
        f.write(f"Max new tokens: `{args.max_new_tokens}`\n\n")

        f.write("## Summary\n\n")
        f.write("| concurrency | req/s | output tok/s | avg latency | p95 latency | avg server latency |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries:
            f.write(
                f"| {row['concurrency']} "
                f"| {row['requests_per_sec']:.2f} "
                f"| {row['output_tokens_per_sec']:.2f} "
                f"| {row['avg_end_to_end_latency_sec']:.4f}s "
                f"| {row['p95_end_to_end_latency_sec']:.4f}s "
                f"| {row['avg_server_latency_sec']:.4f}s |\n"
            )

        f.write("\n## Files\n\n")
        f.write(f"- `{detailPath}`\n")
        f.write(f"- `{summaryPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved details to {detailPath}")
    print(f"saved summary to {summaryPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    concurrencyLevels = parse_concurrency(args.concurrency)
    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "stop_at_eos": args.stop_at_eos,
        "use_kv_cache": args.use_kv_cache,
    }

    for _ in range(args.warmup_runs):
        post_json(args.url, payload)

    allRows = []
    summaries = []

    for concurrency in concurrencyLevels:
        rows, wallTime = run_level(
            args,
            payload,
            concurrency,
            args.requests_per_level,
        )
        summary = summarize_level(rows, wallTime)
        allRows.extend(rows)
        summaries.append(summary)

        print(
            f"concurrency {concurrency}: "
            f"{summary['requests_per_sec']:.2f} req/s, "
            f"{summary['output_tokens_per_sec']:.2f} output tok/s, "
            f"avg {summary['avg_end_to_end_latency_sec']:.4f}s, "
            f"p95 {summary['p95_end_to_end_latency_sec']:.4f}s",
            flush=True,
        )

    write_outputs(args, allRows, summaries)


if __name__ == "__main__":
    main()
