from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse

import tiktoken


VALUES = [
    -12.0,
    -10.5,
    -8.5,
    -6.2,
    -4.4,
    -3.2,
    -2.5,
    -0.8,
    0.4,
    0.8,
    1.2,
    1.5,
    2.2,
    2.4,
    3.6,
    4.7,
    5.6,
    5.8,
    6.8,
    8.0,
    9.1,
    11.3,
    12.0,
    12.5,
    18.5,
    24.0,
    25.0,
    33.5,
    38.5,
    52.2,
    71.4,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", type=str, default="gpt2")
    return parser.parse_args()


def token_pieces(enc, ids):
    return [
        enc.decode([tokenId]).encode("unicode_escape").decode("ascii")
        for tokenId in ids
    ]


def show(enc, text):
    ids = enc.encode(text)
    pieces = token_pieces(enc, ids)

    print(f"text: {text}")
    print(f"ids: {ids}")
    print(f"pieces: {pieces}")
    print(f"num tokens: {len(ids)}")
    print("-" * 80)


def main():
    args = parse_args()
    enc = tiktoken.get_encoding(args.encoding)

    print(f"encoding: {args.encoding}")
    print("=" * 80)

    for value in VALUES:
        show(enc, str(value))

    print("\nwith output prefix")
    print("=" * 80)

    for value in VALUES:
        show(enc, f"value: {value}")

    print("\nwith input form")
    print("=" * 80)

    for value in VALUES:
        show(enc, f"value={value}")


if __name__ == "__main__":
    main()
