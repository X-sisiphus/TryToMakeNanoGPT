import argparse
import json
import os

import numpy as np
import tiktoken


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input.txt")
    parser.add_argument("--out-dir", type=str, default="data/tiny")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--encoding", type=str, default="gpt2")
    parser.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    return parser.parse_args()

args = parse_args()

def get_numpy_dtype(dtypeName, vocabSize):
    if dtypeName == "uint16":
        if vocabSize > np.iinfo(np.uint16).max:
            raise ValueError(
                f"vocab_size={vocabSize} 超过 uint16 上限，请改用 --dtype uint32。"
            )
        return np.uint16

    if dtypeName == "uint32":
        return np.uint32

    raise ValueError(f"不支持的 dtype: {dtypeName}")

def read_text(inputPath):
    if os.path.isfile(inputPath):
        with open(inputPath, "r", encoding="utf-8") as f:
            return f.read(), [inputPath]

    if os.path.isdir(inputPath):
        inputFiles = []
        for root, _, names in os.walk(inputPath):
            for name in names:
                if name.endswith(".txt"):
                    inputFiles.append(os.path.join(root, name))

        inputFiles = sorted(inputFiles)
        if len(inputFiles) == 0:
            raise ValueError(f"目录中没有找到 .txt 文件: {inputPath}")

        textParts = []
        for path in inputFiles:
            with open(path, "r", encoding="utf-8") as f:
                textParts.append(f.read())

        return "\n".join(textParts), inputFiles

    raise FileNotFoundError(f"找不到输入路径: {inputPath}")

os.makedirs(args.out_dir, exist_ok=True)

text, inputFiles = read_text(args.input)

enc = tiktoken.get_encoding(args.encoding)
ids = enc.encode(text)

npDtype = get_numpy_dtype(args.dtype, enc.n_vocab)

numChars = len(text)
numTokens = len(ids)
if numTokens == 0:
    raise ValueError("输入文本没有产生任何 token，请检查 --input 文件内容。")
charsPerToken = numChars / numTokens

n = int(len(ids) * args.train_ratio)
trainIds = ids[:n]
valIds = ids[n:]

trainPath = os.path.join(args.out_dir, "train.bin")
valPath = os.path.join(args.out_dir, "val.bin")

np.array(trainIds, dtype=npDtype).tofile(trainPath)
np.array(valIds, dtype=npDtype).tofile(valPath)

meta = {
    "input": args.input,
    "num_files": len(inputFiles),
    "files": inputFiles,
    "tokenizer": "tiktoken",
    "encoding": args.encoding,
    "vocab_size": enc.n_vocab,
    "chars": numChars,
    "tokens": numTokens,
    "chars_per_token": charsPerToken,
    "train_ratio": args.train_ratio,
    "train_tokens": len(trainIds),
    "val_tokens": len(valIds),
    "dtype": args.dtype,
}

metaPath = os.path.join(args.out_dir, "meta.json")
with open(metaPath, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"chars: {numChars}")
print(f"tokens: {numTokens}")
print(f"chars/token: {charsPerToken:.2f}")
print(f"input files: {len(inputFiles)}")
print(f"vocab_size: {enc.n_vocab}")
print(f"train tokens: {len(trainIds)}")
print(f"val tokens: {len(valIds)}")
print(f"saved to: {args.out_dir}")
print(f"dtype: {args.dtype}")
