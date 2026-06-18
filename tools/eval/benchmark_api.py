import argparse
import csv
import json
import os
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
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out/api_benchmark")
    return parser.parse_args()


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
        with urlrequest.urlopen(req, timeout=60) as response:
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
    overheads = [row["http_overhead_sec"] for row in rows]
    tokensPerSec = [row["tokens_per_sec"] for row in rows]

    return {
        "runs": len(rows),
        "avg_end_to_end_latency_sec": sum(endToEndLatencies) / len(endToEndLatencies),
        "min_end_to_end_latency_sec": min(endToEndLatencies),
        "max_end_to_end_latency_sec": max(endToEndLatencies),
        "avg_server_latency_sec": sum(serverLatencies) / len(serverLatencies),
        "avg_http_overhead_sec": sum(overheads) / len(overheads),
        "avg_tokens_per_sec": sum(tokensPerSec) / len(tokensPerSec),
        "min_tokens_per_sec": min(tokensPerSec),
        "max_tokens_per_sec": max(tokensPerSec),
    }


def write_outputs(args, rows, summary, sampleResponse):
    os.makedirs(args.out_dir, exist_ok=True)
    csvPath = os.path.join(args.out_dir, "benchmark.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "end_to_end_latency_sec",
                "server_latency_sec",
                "http_overhead_sec",
                "new_tokens",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# API Benchmark\n\n")
        f.write(f"URL: `{args.url}`\n")
        f.write(f"Runs: `{summary['runs']}`\n")
        f.write(f"Max new tokens: `{args.max_new_tokens}`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- avg end-to-end latency: {summary['avg_end_to_end_latency_sec']:.4f} sec\n")
        f.write(f"- min end-to-end latency: {summary['min_end_to_end_latency_sec']:.4f} sec\n")
        f.write(f"- max end-to-end latency: {summary['max_end_to_end_latency_sec']:.4f} sec\n")
        f.write(f"- avg server generation latency: {summary['avg_server_latency_sec']:.4f} sec\n")
        f.write(f"- avg HTTP overhead: {summary['avg_http_overhead_sec']:.4f} sec\n")
        f.write(f"- avg tokens/s: {summary['avg_tokens_per_sec']:.2f}\n")
        f.write(f"- min tokens/s: {summary['min_tokens_per_sec']:.2f}\n")
        f.write(f"- max tokens/s: {summary['max_tokens_per_sec']:.2f}\n\n")

        f.write("## Sample Response\n\n")
        f.write("```json\n")
        f.write(json.dumps(sampleResponse, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")

        f.write("## Files\n\n")
        f.write(f"- `{csvPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved csv to {csvPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "stop_at_eos": args.stop_at_eos,
    }

    for _ in range(args.warmup_runs):
        post_json(args.url, payload)

    rows = []
    sampleResponse = None

    for runIdx in range(args.num_runs):
        response, endToEndLatency = post_json(args.url, payload)
        serverLatency = response["latency_sec"]
        httpOverhead = max(0.0, endToEndLatency - serverLatency)
        row = {
            "run": runIdx,
            "end_to_end_latency_sec": endToEndLatency,
            "server_latency_sec": serverLatency,
            "http_overhead_sec": httpOverhead,
            "new_tokens": response["new_tokens"],
            "tokens_per_sec": response["tokens_per_sec"],
        }
        rows.append(row)
        sampleResponse = response

        print(
            f"run {runIdx}: end-to-end {endToEndLatency:.4f}s, "
            f"server {serverLatency:.4f}s, overhead {httpOverhead:.4f}s, "
            f"{response['tokens_per_sec']:.2f} tok/s",
            flush=True,
        )

    summary = summarize(rows)
    write_outputs(args, rows, summary, sampleResponse)


if __name__ == "__main__":
    main()
