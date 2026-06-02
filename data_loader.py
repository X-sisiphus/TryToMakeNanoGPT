import json
import os

import numpy as np
import torch

def load_token_data(dataDir):
    metaPath = os.path.join(dataDir, "meta.json")
    trainPath = os.path.join(dataDir, "train.bin")
    valPath = os.path.join(dataDir, "val.bin")

    with open(metaPath, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dtypeName = meta.get("dtype", "uint16")
    npDtype = np.dtype(dtypeName)

    trainData = torch.from_numpy(
        np.fromfile(trainPath, dtype=npDtype).astype(np.int64)
    )
    valData = torch.from_numpy(
        np.fromfile(valPath, dtype=npDtype).astype(np.int64)
    )

    vocabularySize = meta["vocab_size"]
    vocabInfo = {
        "type": "tokenizer",
        "meta": meta,
    }

    print(f"loaded token data from {dataDir}", flush=True)
    print(f"vocab size: {vocabularySize}", flush=True)
    print(f"train tokens: {len(trainData)}", flush=True)
    print(f"val tokens: {len(valData)}", flush=True)

    return trainData, valData, vocabularySize, vocabInfo

def load_char_data(inputPath, trainRatio):
    with open(inputPath, "r", encoding="utf-8") as trainTxt:
        text = trainTxt.read()

    chars = sorted(list(set(text)))
    vocabularySize = len(chars)
    stringToInt = {ch: i for i, ch in enumerate(chars)}
    intToString = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stringToInt[c] for c in s]

    data = torch.tensor(
        encode(text),
        dtype=torch.long,
    )

    n = int(trainRatio * len(data))
    trainData = data[:n]
    valData = data[n:]

    vocabInfo = {
        "type": "char",
        "stringToInt": stringToInt,
        "intToString": intToString,
    }

    print(f"loaded char data from {inputPath}", flush=True)
    print(f"vocab size: {vocabularySize}", flush=True)
    print(f"train tokens: {len(trainData)}", flush=True)
    print(f"val tokens: {len(valData)}", flush=True)

    return trainData, valData, vocabularySize, vocabInfo

def load_data(dataDir=None, inputPath="input.txt", trainRatio=0.9):
    if dataDir is not None:
        return load_token_data(dataDir)

    return load_char_data(inputPath, trainRatio)

def get_batch(split, trainData, valData, blockSize, batchSize, device):
    sourceData = trainData if split == "train" else valData

    ix = torch.randint(
        len(sourceData) - blockSize,
        (batchSize,)
    )

    x = torch.stack([
        sourceData[i:i + blockSize]
        for i in ix
    ])

    y = torch.stack([
        sourceData[i + 1:i + blockSize + 1]
        for i in ix
    ])

    x, y = x.to(device), y.to(device)
    return x, y
