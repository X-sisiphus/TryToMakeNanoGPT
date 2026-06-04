import argparse

import tiktoken

from sft_data import END_TOKEN, EOS_TOKEN, IGNORE_INDEX, encode_sft_example, load_sft_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/sft/astro_sft_tiny.jsonl")
    parser.add_argument("--encoding", type=str, default="gpt2")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_sft_jsonl(args.path)
    if len(examples) == 0:
        raise ValueError("SFT 数据为空")

    enc = tiktoken.get_encoding(args.encoding)
    endIds = enc.encode(END_TOKEN, allowed_special={EOS_TOKEN})
    encoded = [encode_sft_example(example, enc) for example in examples]

    promptTokens = []
    answerTokens = []

    for idx, item in enumerate(encoded):
        inputIds = item["input_ids"]
        labels = item["labels"]
        promptLen = item["prompt_tokens"]
        answerLen = item["answer_tokens"]

        assert len(inputIds) == len(labels)
        assert promptLen > 0
        assert answerLen > 0
        answerStart = promptLen - 1
        answerEnd = answerStart + answerLen

        assert all(label == IGNORE_INDEX for label in labels[:answerStart])
        assert labels[answerStart:answerEnd] == inputIds[promptLen:]
        assert all(label == IGNORE_INDEX for label in labels[answerEnd:])
        assert inputIds[-len(endIds):] == endIds
        assert labels[answerEnd - len(endIds):answerEnd] == endIds

        promptTokens.append(promptLen)
        answerTokens.append(answerLen)

    avgPromptTokens = sum(promptTokens) / len(promptTokens)
    avgAnswerTokens = sum(answerTokens) / len(answerTokens)
    maxTotalTokens = max(p + a for p, a in zip(promptTokens, answerTokens))

    print(f"path: {args.path}")
    print(f"encoding: {args.encoding}")
    print(f"examples: {len(encoded)}")
    print(f"avg prompt tokens: {avgPromptTokens:.1f}")
    print(f"avg answer tokens: {avgAnswerTokens:.1f}")
    print(f"max total tokens: {maxTotalTokens}")
    print(f"end token text: {END_TOKEN}")
    print(f"end token ids: {endIds}")
    print("labels mask ok")

    first = encoded[0]
    print("\nfirst prompt:")
    print(first["prompt"])
    print("first answer:")
    print(first["answer"])


if __name__ == "__main__":
    main()
