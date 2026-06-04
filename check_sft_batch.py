import argparse

import torch
import tiktoken

from sft_data import (
    IGNORE_INDEX,
    encode_sft_example,
    load_sft_jsonl,
    pad_sft_batch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/sft/astro_sft_tiny.jsonl")
    parser.add_argument("--encoding", type=str, default="gpt2")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_sft_jsonl(args.path)
    if len(examples) < args.batch_size:
        raise ValueError(
            f"SFT 样本数不足: 需要 {args.batch_size} 条，当前只有 {len(examples)} 条。"
        )

    enc = tiktoken.get_encoding(args.encoding)
    encoded = [encode_sft_example(example, enc) for example in examples[:args.batch_size]]
    batch = pad_sft_batch(encoded)

    inputIds = batch["input_ids"]
    labels = batch["labels"]
    attentionMask = batch["attention_mask"]

    assert inputIds.shape == labels.shape
    assert inputIds.shape == attentionMask.shape
    assert inputIds.dtype == torch.long
    assert labels.dtype == torch.long
    assert attentionMask.dtype == torch.long
    assert (labels[attentionMask == 0] == IGNORE_INDEX).all()
    assert (attentionMask.sum(dim=1) > 0).all()

    for row, item in enumerate(encoded):
        realLen = len(item["input_ids"])
        assert attentionMask[row, :realLen].eq(1).all()
        assert attentionMask[row, realLen:].eq(0).all()
        assert labels[row, realLen:].eq(IGNORE_INDEX).all()

    print(f"path: {args.path}")
    print(f"encoding: {args.encoding}")
    print(f"batch size: {args.batch_size}")
    print(f"input_ids shape: {tuple(inputIds.shape)}")
    print(f"labels shape: {tuple(labels.shape)}")
    print(f"attention_mask shape: {tuple(attentionMask.shape)}")
    print("padding mask ok")


if __name__ == "__main__":
    main()
