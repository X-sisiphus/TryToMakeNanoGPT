from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
from urllib import request as urlrequest


DEFAULT_PROMPT = (
    "Instruction:\n"
    "Extract the station, signal, value, and unit from the text.\n\n"
    "Input:\n"
    "ONSA reports vertical velocity of 2.4 mm/yr.\n\n"
    "Answer:\n"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8010")
    parser.add_argument("--endpoint", choices=["generate", "generate_dynamic"], default="generate")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--stop-at-eos", action="store_true")
    parser.add_argument("--use-kv-cache", action="store_true")
    return parser.parse_args()


def get_json(url):
    with urlrequest.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def print_section(title, value):
    print(f"\n=== {title} ===")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        print(value)


def main():
    args = parse_args()
    baseUrl = args.base_url.rstrip("/")
    health = get_json(f"{baseUrl}/health")
    print_section("health", health)

    payload = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "stop_at_eos": args.stop_at_eos,
        "use_kv_cache": args.use_kv_cache,
    }
    result = post_json(f"{baseUrl}/{args.endpoint}", payload)

    compactResult = {
        "completion_text": result.get("completion_text", ""),
        "prompt_tokens": result.get("prompt_tokens"),
        "new_tokens": result.get("new_tokens"),
        "latency_sec": result.get("latency_sec"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "device": result.get("device"),
        "use_kv_cache": result.get("use_kv_cache"),
    }
    if args.endpoint == "generate_dynamic":
        compactResult.update(
            {
                "dynamic_batch_size": result.get("dynamic_batch_size"),
                "queue_wait_ms": result.get("queue_wait_ms"),
                "batch_latency_ms": result.get("batch_latency_ms"),
            }
        )

    print_section("request", payload)
    print_section("result", compactResult)

    if args.endpoint == "generate_dynamic":
        stats = get_json(f"{baseUrl}/dynamic_stats")
        print_section("dynamic_stats", stats)


if __name__ == "__main__":
    main()
