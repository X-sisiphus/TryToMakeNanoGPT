import argparse
import csv
import json
import os
import statistics
import time
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
    parser.add_argument("--output-lengths", type=str, default="8,16,32,64")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out/output_length_benchmark")
    return parser.parse_args()


def parse_lengths(text):
    lengths = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("output length 必须大于 0")
        lengths.append(value)
    if not lengths:
        raise ValueError("至少需要一个 output length")
    return lengths


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
    return json.loads(responseBody), endToEndLatency


def summarize(rows):
    endToEndLatencies = [row["end_to_end_latency_sec"] for row in rows]
    serverLatencies = [row["server_latency_sec"] for row in rows]
    tokensPerSec = [row["tokens_per_sec"] for row in rows]
    newTokens = [row["new_tokens"] for row in rows]

    return {
        "target_output_tokens": rows[0]["target_output_tokens"],
        "runs": len(rows),
        "avg_end_to_end_latency_sec": statistics.mean(endToEndLatencies),
        "min_end_to_end_latency_sec": min(endToEndLatencies),
        "max_end_to_end_latency_sec": max(endToEndLatencies),
        "avg_server_latency_sec": statistics.mean(serverLatencies),
        "avg_tokens_per_sec": statistics.mean(tokensPerSec),
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
                "target_output_tokens",
                "run",
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
                "target_output_tokens",
                "runs",
                "avg_end_to_end_latency_sec",
                "min_end_to_end_latency_sec",
                "max_end_to_end_latency_sec",
                "avg_server_latency_sec",
                "avg_tokens_per_sec",
                "avg_new_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# Output Length Benchmark\n\n")
        f.write(f"URL: `{args.url}`\n")
        f.write(f"Output lengths: `{args.output_lengths}`\n")
        f.write(f"Runs per length: `{args.num_runs}`\n")
        f.write(f"Stop at EOS: `{args.stop_at_eos}`\n\n")

        f.write("## Summary\n\n")
        f.write("| target output | avg new tokens | avg latency | avg server latency | avg tok/s |\n")
        f.write("| ---: | ---: | ---: | ---: | ---: |\n")
        for row in summaries:
            f.write(
                f"| {row['target_output_tokens']} "
                f"| {row['avg_new_tokens']:.1f} "
                f"| {row['avg_end_to_end_latency_sec']:.4f}s "
                f"| {row['avg_server_latency_sec']:.4f}s "
                f"| {row['avg_tokens_per_sec']:.2f} |\n"
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
    outputLengths = parse_lengths(args.output_lengths)

    allRows = []
    summaries = []

    for outputTokens in outputLengths:
        payload = {
            "prompt": args.prompt,
            "max_new_tokens": outputTokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "stop_at_eos": args.stop_at_eos,
            "use_kv_cache": args.use_kv_cache,
        }

        for _ in range(args.warmup_runs):
            post_json(args.url, payload)

        rows = []
        for runIdx in range(args.num_runs):
            response, endToEndLatency = post_json(args.url, payload)
            serverLatency = response["latency_sec"]
            rows.append(
                {
                    "target_output_tokens": outputTokens,
                    "run": runIdx,
                    "end_to_end_latency_sec": endToEndLatency,
                    "server_latency_sec": serverLatency,
                    "http_overhead_sec": max(0.0, endToEndLatency - serverLatency),
                    "new_tokens": response["new_tokens"],
                    "tokens_per_sec": response["tokens_per_sec"],
                }
            )

        summary = summarize(rows)
        allRows.extend(rows)
        summaries.append(summary)

        print(
            f"output {outputTokens} tokens: "
            f"avg {summary['avg_end_to_end_latency_sec']:.4f}s, "
            f"{summary['avg_tokens_per_sec']:.2f} tok/s, "
            f"actual {summary['avg_new_tokens']:.1f} tokens",
            flush=True,
        )

    write_outputs(args, allRows, summaries)


if __name__ == "__main__":
    main()
