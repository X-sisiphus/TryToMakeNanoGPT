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
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8010")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true")
    parser.add_argument("--out-dir", type=str, default="out/batch_api_benchmark")
    return parser.parse_args()


def parse_batch_sizes(text):
    batchSizes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("batch size 必须大于 0")
        batchSizes.append(value)
    if not batchSizes:
        raise ValueError("至少需要一个 batch size")
    return batchSizes


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


def make_generate_payload(args):
    return {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "stop_at_eos": args.stop_at_eos,
        "use_kv_cache": args.use_kv_cache,
    }


def make_batch_payload(args, batchSize):
    return {
        "prompts": [args.prompt for _ in range(batchSize)],
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "stop_at_eos": args.stop_at_eos,
        "use_kv_cache": args.use_kv_cache,
    }


def run_sequential(args, batchSize):
    url = args.base_url.rstrip("/") + "/generate"
    payload = make_generate_payload(args)
    start = time.perf_counter()
    totalNewTokens = 0
    for _ in range(batchSize):
        response, _ = post_json(url, payload)
        totalNewTokens += response["new_tokens"]
    latency = time.perf_counter() - start
    return latency, totalNewTokens


def run_batched(args, batchSize):
    url = args.base_url.rstrip("/") + "/generate_batch"
    payload = make_batch_payload(args, batchSize)
    response, latency = post_json(url, payload)
    return latency, response["total_new_tokens"]


def write_outputs(args, rows):
    os.makedirs(args.out_dir, exist_ok=True)
    csvPath = os.path.join(args.out_dir, "benchmark.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "batch_size",
                "run",
                "latency_sec",
                "total_new_tokens",
                "tokens_per_sec",
                "requests_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        key = (row["mode"], row["batch_size"])
        grouped.setdefault(key, []).append(row)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# Batch API Benchmark\n\n")
        f.write(f"Base URL: `{args.base_url}`\n")
        f.write(f"Batch sizes: `{args.batch_sizes}`\n")
        f.write(f"Max new tokens: `{args.max_new_tokens}`\n")
        f.write(f"Use KV cache: `{args.use_kv_cache}`\n\n")

        f.write("## Summary\n\n")
        f.write("| mode | batch | avg latency | avg tok/s | avg req/s |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for key in sorted(grouped):
            groupRows = grouped[key]
            avgLatency = sum(row["latency_sec"] for row in groupRows) / len(groupRows)
            avgTokensPerSec = sum(row["tokens_per_sec"] for row in groupRows) / len(groupRows)
            avgRequestsPerSec = sum(row["requests_per_sec"] for row in groupRows) / len(groupRows)
            mode, batchSize = key
            f.write(
                f"| {mode} | {batchSize} | {avgLatency:.4f}s | "
                f"{avgTokensPerSec:.2f} | {avgRequestsPerSec:.2f} |\n"
            )

        f.write("\n## Files\n\n")
        f.write(f"- `{csvPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved csv to {csvPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    batchSizes = parse_batch_sizes(args.batch_sizes)

    for _ in range(args.warmup_runs):
        run_batched(args, batchSizes[0])

    rows = []
    for batchSize in batchSizes:
        for runIdx in range(args.num_runs):
            for mode, runner in [
                ("sequential", run_sequential),
                ("batched", run_batched),
            ]:
                latency, totalNewTokens = runner(args, batchSize)
                rows.append(
                    {
                        "mode": mode,
                        "batch_size": batchSize,
                        "run": runIdx,
                        "latency_sec": latency,
                        "total_new_tokens": totalNewTokens,
                        "tokens_per_sec": totalNewTokens / latency if latency > 0 else 0.0,
                        "requests_per_sec": batchSize / latency if latency > 0 else 0.0,
                    }
                )
                print(
                    f"{mode} batch {batchSize} run {runIdx}: "
                    f"{latency:.4f}s, "
                    f"{totalNewTokens / latency if latency > 0 else 0.0:.2f} tok/s, "
                    f"{batchSize / latency if latency > 0 else 0.0:.2f} req/s",
                    flush=True,
                )

    write_outputs(args, rows)


if __name__ == "__main__":
    main()
