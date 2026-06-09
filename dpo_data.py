import json
import torch

from sft_data import EOS_TOKEN, END_TOKEN, PAD_TOKEN_ID, format_sft_example


def format_dpo_example(example):
    instruction = example["instruction"].strip()
    inputText = example["input"].strip()
    chosen = example["chosen"].strip()
    rejected = example["rejected"].strip()

    prompt = (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Input:\n"
        f"{inputText}\n\n"
        "Answer:\n"
    )

    return prompt, chosen, rejected


def sft_to_chosen_example(example):
    prompt, answer = format_sft_example(example)
    return prompt, answer


def encode_prompt_answer(prompt, answer, enc):
    promptIds = enc.encode(prompt)
    answerIds = enc.encode(
        answer + END_TOKEN,
        allowed_special={EOS_TOKEN},
    )

    inputIds = promptIds + answerIds
    answerMask = [0] * len(promptIds) + [1] * len(answerIds)

    return {
        "input_ids": inputIds,
        "answer_mask": answerMask,
        "prompt_tokens": len(promptIds),
        "answer_tokens": len(answerIds),
    }


def encode_dpo_example(example, enc):
    prompt, chosen, rejected = format_dpo_example(example)

    chosenItem = encode_prompt_answer(prompt, chosen, enc)
    rejectedItem = encode_prompt_answer(prompt, rejected, enc)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_input_ids": chosenItem["input_ids"],
        "chosen_answer_mask": chosenItem["answer_mask"],
        "rejected_input_ids": rejectedItem["input_ids"],
        "rejected_answer_mask": rejectedItem["answer_mask"],
        "chosen_tokens": len(chosenItem["input_ids"]),
        "rejected_tokens": len(rejectedItem["input_ids"]),
    }


def load_dpo_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def pad_sequences(sequences, padTokenId=PAD_TOKEN_ID):
    maxLen = max(len(sequence) for sequence in sequences)
    padded = []
    masks = []

    for sequence in sequences:
        padLen = maxLen - len(sequence)
        padded.append(sequence + [padTokenId] * padLen)
        masks.append([1] * len(sequence) + [0] * padLen)

    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


def pad_dpo_batch(items, padTokenId=PAD_TOKEN_ID):
    chosenInputIds, chosenAttentionMask = pad_sequences(
        [item["chosen_input_ids"] for item in items],
        padTokenId=padTokenId,
    )
    rejectedInputIds, rejectedAttentionMask = pad_sequences(
        [item["rejected_input_ids"] for item in items],
        padTokenId=padTokenId,
    )
    chosenAnswerMask, _ = pad_sequences(
        [item["chosen_answer_mask"] for item in items],
        padTokenId=0,
    )
    rejectedAnswerMask, _ = pad_sequences(
        [item["rejected_answer_mask"] for item in items],
        padTokenId=0,
    )

    return {
        "chosen_input_ids": chosenInputIds,
        "chosen_attention_mask": chosenAttentionMask,
        "chosen_answer_mask": chosenAnswerMask,
        "rejected_input_ids": rejectedInputIds,
        "rejected_attention_mask": rejectedAttentionMask,
        "rejected_answer_mask": rejectedAnswerMask,
    }
