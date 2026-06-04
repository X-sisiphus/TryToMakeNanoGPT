import json
import torch

def format_sft_example(example):
    instruction = example["instruction"].strip()
    inputText = example["input"].strip()
    outputText = example["output"].strip()

    prompt = (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Input:\n"
        f"{inputText}\n\n"
        "Answer:\n"
    )

    answer = outputText

    return prompt, answer

IGNORE_INDEX = -100
PAD_TOKEN_ID = 0
EOS_TOKEN = "<|endoftext|>"
END_TOKEN = EOS_TOKEN

def encode_sft_example(example, enc):
    prompt, answer = format_sft_example(example)

    promptIds = enc.encode(prompt)
    answerIds = enc.encode(
        answer + END_TOKEN,
        allowed_special={EOS_TOKEN},
    )

    inputIds = promptIds + answerIds
    labels = [IGNORE_INDEX] * len(inputIds)

    # 模型在每个位置预测“下一个 token”，所以 answer 的第一个 token
    # 要放在 prompt 最后一个位置的 label 上，END 也要由它前一个 token 预测。
    answerStart = len(promptIds) - 1
    labels[answerStart:answerStart + len(answerIds)] = answerIds

    return {
        "input_ids": inputIds,
        "labels": labels,
        "prompt": prompt,
        "answer": answer,
        "prompt_tokens": len(promptIds),
        "answer_tokens": len(answerIds),
    }

def load_sft_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def pad_sft_batch(items, padTokenId=PAD_TOKEN_ID):
    maxLen = max(len(item["input_ids"]) for item in items)

    inputIds = []
    labels = []
    attentionMask = []

    for item in items:
        ids = item["input_ids"]
        itemLabels = item["labels"]
        padLen = maxLen - len(ids)

        inputIds.append(ids + [padTokenId] * padLen)
        labels.append(itemLabels + [IGNORE_INDEX] * padLen)
        attentionMask.append([1] * len(ids) + [0] * padLen)

    return {
        "input_ids": torch.tensor(inputIds, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attentionMask, dtype=torch.long),
    }
