from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import tiktoken


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", type=str, default="gpt2")
    return parser.parse_args()

examples = [
    "hello world",
    "Transformer is all you need.",
    "你好，世界",
    "天文学和时空智能",
    "GNSS time series and space geodesy",
    "RA=12h30m, Dec=+45deg",
]

args = parse_args()
enc = tiktoken.get_encoding(args.encoding)

for text in examples:
    ids = enc.encode(text)
    pieces = [enc.decode([i]) for i in ids]

    print("=" * 80)
    print(f"text: {text}")
    print(f"chars: {len(text)}")
    print(f"tokens: {len(ids)}")
    print(f"chars/token: {len(text) / len(ids):.2f}")
    print(f"ids: {ids}")
    print("pieces:")

    for tokenId, piece in zip(ids, pieces):
        print(f"  {tokenId}: {repr(piece)}")
