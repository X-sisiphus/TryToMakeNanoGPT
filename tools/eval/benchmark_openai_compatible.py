import argparse
import csv
import json
import os
import statistics
import time
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


DEFAULT_PROMPT = """Extract the station, signal, value, and unit from the text.

Text: ONSA reports vertical velocity of 2.4 mm/yr.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--system", type=str, default="You are a precise field extraction assistant.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--out-dir", type=str, default="out/openai_compatible_benchmark")
    return parser.parse_args()


def post_json(url, payload, apiKey):
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apiKey}",
    }
    request = urlrequest.Request(url, data=data, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urlrequest.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"请求失败，请确认 OpenAI-compatible 服务已经启动: {exc}") from exc
    latency = time.perf_counter() - start
    return json.loads(body), latency


def build_payload(args):
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }


def extract_output(response):
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "")


def output_tokens(response, fallbackText):
    usage = response.get("usage", {})
    if "completion_tokens" in usage:
        return usage["completion_tokens"]
    return max(1, len(fallbackText.split()))


def summarize(rows):
    latencies = [row["latency_sec"] for row in rows]
    tokensPerSec = [row["tokens_per_sec"] for row in rows]
    return {
        "runs": len(rows),
        "avg_latency_sec": statistics.mean(latencies),
        "p95_latency_sec": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "avg_tokens_per_sec": statistics.mean(tokensPerSec),
        "min_tokens_per_sec": min(tokensPerSec),
        "max_tokens_per_sec": max(tokensPerSec),
    }


def write_outputs(args, rows, summary, sampleResponse, sampleText):
    os.makedirs(args.out_dir, exist_ok=True)
    detailsPath = os.path.join(args.out_dir, "details.csv")
    reportPath = os.path.join(args.out_dir, "report.md")

    with open(detailsPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "latency_sec",
                "output_tokens",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(reportPath, "w", encoding="utf-8") as f:
        f.write("# OpenAI-Compatible Benchmark\n\n")
        f.write(f"Base URL: `{args.base_url}`\n")
        f.write(f"Model: `{args.model}`\n")
        f.write(f"Max tokens: `{args.max_tokens}`\n")
        f.write(f"Runs: `{summary['runs']}`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- avg latency: {summary['avg_latency_sec']:.4f} sec\n")
        f.write(f"- p95 latency: {summary['p95_latency_sec']:.4f} sec\n")
        f.write(f"- avg tokens/s: {summary['avg_tokens_per_sec']:.2f}\n")
        f.write(f"- min tokens/s: {summary['min_tokens_per_sec']:.2f}\n")
        f.write(f"- max tokens/s: {summary['max_tokens_per_sec']:.2f}\n\n")

        f.write("## Sample Output\n\n")
        f.write("```text\n")
        f.write(sampleText)
        f.write("\n```\n\n")

        f.write("## Sample Response\n\n")
        f.write("```json\n")
        f.write(json.dumps(sampleResponse, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")

        f.write("## Files\n\n")
        f.write(f"- `{detailsPath}`\n")
        f.write(f"- `{reportPath}`\n")

    print(f"saved details to {detailsPath}")
    print(f"saved report to {reportPath}")


def main():
    args = parse_args()
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = build_payload(args)

    for _ in range(args.warmup_runs):
        post_json(url, payload, args.api_key)

    rows = []
    sampleResponse = None
    sampleText = ""
    for runIdx in range(args.num_runs):
        response, latency = post_json(url, payload, args.api_key)
        text = extract_output(response)
        tokens = output_tokens(response, text)
        tokensPerSec = tokens / latency if latency > 0 else 0.0
        rows.append(
            {
                "run": runIdx,
                "latency_sec": latency,
                "output_tokens": tokens,
                "tokens_per_sec": tokensPerSec,
            }
        )
        sampleResponse = response
        sampleText = text
        print(
            f"run {runIdx}: {latency:.4f}s, {tokensPerSec:.2f} tok/s",
            flush=True,
        )

    summary = summarize(rows)
    write_outputs(args, rows, summary, sampleResponse, sampleText)


if __name__ == "__main__":
    main()
