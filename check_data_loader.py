import argparse
import torch

from data_loader import load_data, get_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/tiny")
    parser.add_argument("--input", type=str, default="input.txt")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()

def check_dataset(name, trainData, valData, vocabSize, vocabInfo, blockSize, batchSize):
    print(f"\nchecking {name}")

    assert len(trainData) > blockSize + 1
    assert len(valData) > blockSize + 1
    assert vocabSize > 0
    assert "type" in vocabInfo

    x, y = get_batch(
        "train",
        trainData,
        valData,
        blockSize,
        batchSize,
        device="cpu",
    )

    assert x.shape == (batchSize, blockSize)
    assert y.shape == (batchSize, blockSize)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # y 应该是原始序列中 x 的下一个 token
    assert torch.equal(x[:, 1:], y[:, :-1])

    print(f"{name} ok")

def main():
    args = parse_args()

    tokenTrain, tokenVal, tokenVocabSize, tokenVocabInfo = load_data(
        dataDir=args.data_dir,
    )

    check_dataset(
        "tokenizer",
        tokenTrain,
        tokenVal,
        tokenVocabSize,
        tokenVocabInfo,
        args.block_size,
        args.batch_size,
    )

    charTrain, charVal, charVocabSize, charVocabInfo = load_data(
        inputPath=args.input,
    )

    check_dataset(
        "char",
        charTrain,
        charVal,
        charVocabSize,
        charVocabInfo,
        args.block_size,
        args.batch_size,
    )

    print("\nall data loader checks passed")

if __name__ == "__main__":
    main()
