import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict

import torch
import torch.nn.functional as F
import tiktoken

from dpo_data import encode_dpo_example, load_dpo_jsonl, pad_dpo_batch
from model import BigramLanguageModel, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser()

    # DPO 数据参数
    parser.add_argument("--dpo-path", type=str, default="data/dpo/field_dpo.jsonl")
    parser.add_argument("--encoding", type=str, default="gpt2")

    # checkpoint 参数
    parser.add_argument("--init-from", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out/dpo_debug")

    # batch 和训练循环参数
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-mode", choices=["stratified", "shuffle", "sequential"], default="shuffle")
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def split_items(items, trainRatio, splitMode, seed):
    rng = random.Random(seed)

    if splitMode == "sequential":
        splitIndex = int(len(items) * trainRatio)
        return items[:splitIndex], items[splitIndex:]

    if splitMode == "shuffle":
        shuffled = list(items)
        rng.shuffle(shuffled)
        splitIndex = int(len(shuffled) * trainRatio)
        return shuffled[:splitIndex], shuffled[splitIndex:]

    groups = defaultdict(list)
    for item in items:
        groups[item["preference_type"]].append(item)

    trainItems = []
    valItems = []
    for _, groupItems in sorted(groups.items()):
        groupItems = list(groupItems)
        rng.shuffle(groupItems)

        splitIndex = int(len(groupItems) * trainRatio)
        if len(groupItems) > 1:
            splitIndex = min(max(splitIndex, 1), len(groupItems) - 1)

        trainItems.extend(groupItems[:splitIndex])
        valItems.extend(groupItems[splitIndex:])

    rng.shuffle(trainItems)
    rng.shuffle(valItems)
    return trainItems, valItems


def load_model_from_checkpoint(path, blockSize, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    config.blockSize = blockSize

    model = BigramLanguageModel(
        config.vocabSize,
        config.blockSize,
        config=config,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, config, checkpoint


def sequence_logps(model, inputIds, answerMask):
    logits, _ = model(inputIds)

    targetIds = inputIds[:, 1:]
    targetMask = answerMask[:, 1:].float()
    logits = logits[:, :-1, :]

    logProbs = F.log_softmax(logits, dim=-1)
    tokenLogps = torch.gather(
        logProbs,
        dim=-1,
        index=targetIds.unsqueeze(-1),
    ).squeeze(-1)

    return (tokenLogps * targetMask).sum(dim=-1)


def dpo_batch_loss(policyModel, referenceModel, batch, beta):
    chosenInputIds = batch["chosen_input_ids"]
    chosenAnswerMask = batch["chosen_answer_mask"]
    rejectedInputIds = batch["rejected_input_ids"]
    rejectedAnswerMask = batch["rejected_answer_mask"]

    policyChosenLogps = sequence_logps(
        policyModel,
        chosenInputIds,
        chosenAnswerMask,
    )
    policyRejectedLogps = sequence_logps(
        policyModel,
        rejectedInputIds,
        rejectedAnswerMask,
    )

    with torch.no_grad():
        referenceChosenLogps = sequence_logps(
            referenceModel,
            chosenInputIds,
            chosenAnswerMask,
        )
        referenceRejectedLogps = sequence_logps(
            referenceModel,
            rejectedInputIds,
            rejectedAnswerMask,
        )

    policyLogRatio = policyChosenLogps - policyRejectedLogps
    referenceLogRatio = referenceChosenLogps - referenceRejectedLogps
    logits = beta * (policyLogRatio - referenceLogRatio)
    loss = -F.logsigmoid(logits).mean()

    return {
        "loss": loss,
        "preference_accuracy": (policyLogRatio > referenceLogRatio).float().mean(),
        "policy_margin": policyLogRatio.mean(),
        "reference_margin": referenceLogRatio.mean(),
        "chosen_logp": policyChosenLogps.mean(),
        "rejected_logp": policyRejectedLogps.mean(),
    }


def move_batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    useMps = os.environ.get("USE_MPS") == "1"
    device = "mps" if torch.backends.mps.is_available() and useMps else "cpu"

    print(f"seed: {args.seed}", flush=True)
    print(f"using device: {device}", flush=True)
    print(f"dpo path: {args.dpo_path}", flush=True)
    print(f"init from: {args.init_from}", flush=True)
    print(f"out dir: {args.out_dir}", flush=True)
    print(f"beta: {args.beta}", flush=True)

    enc = tiktoken.get_encoding(args.encoding)
    rawExamples = load_dpo_jsonl(args.dpo_path)
    encoded = []

    for example in rawExamples:
        item = encode_dpo_example(example, enc)
        item["preference_type"] = example.get("preference_type", "unknown")
        encoded.append(item)

    encoded = [
        item
        for item in encoded
        if item["chosen_tokens"] <= args.block_size
        and item["rejected_tokens"] <= args.block_size
    ]

    if len(encoded) == 0:
        raise ValueError("没有样本长度小于等于 block_size，请增大 --block-size。")

    trainItems, valItems = split_items(
        encoded,
        args.train_ratio,
        args.split_mode,
        args.seed,
    )

    if len(trainItems) == 0:
        raise ValueError("训练集为空，请检查 DPO 数据或 --train-ratio。")

    if len(valItems) == 0:
        raise ValueError("验证集为空，请降低 --train-ratio 或增加 DPO 数据。")

    print(f"dpo examples: {len(encoded)}", flush=True)
    print(f"train dpo examples: {len(trainItems)}", flush=True)
    print(f"val dpo examples: {len(valItems)}", flush=True)
    print(f"split mode: {args.split_mode}", flush=True)
    print(f"train preference types: {dict(sorted(Counter(item['preference_type'] for item in trainItems).items()))}", flush=True)
    print(f"val preference types: {dict(sorted(Counter(item['preference_type'] for item in valItems).items()))}", flush=True)

    policyModel, config, checkpoint = load_model_from_checkpoint(
        args.init_from,
        args.block_size,
        device,
    )
    referenceModel, _, _ = load_model_from_checkpoint(
        args.init_from,
        args.block_size,
        device,
    )

    referenceModel.eval()
    for param in referenceModel.parameters():
        param.requires_grad = False

    policyModel.train()

    print(config, flush=True)
    print(f"number of parameters: {policyModel.get_num_params() / 1e6:.2f}M", flush=True)

    optimizer = policyModel.configure_optimizers(
        weightDecay=0.0,
        learningRate=args.learning_rate,
    )

    def get_batch(split):
        sourceData = trainItems if split == "train" else valItems
        ix = torch.randint(len(sourceData), (args.batch_size,))
        items = [sourceData[i] for i in ix]
        return move_batch_to_device(pad_dpo_batch(items), device)

    logPath = os.path.join(args.out_dir, "log.csv")
    with open(logPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "train_loss",
            "val_loss",
            "train_pref_acc",
            "val_pref_acc",
            "train_policy_margin",
            "val_policy_margin",
            "train_reference_margin",
            "val_reference_margin",
        ])

    def log_metrics(step, trainMetrics, valMetrics):
        with open(logPath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                trainMetrics["loss"],
                valMetrics["loss"],
                trainMetrics["preference_accuracy"],
                valMetrics["preference_accuracy"],
                trainMetrics["policy_margin"],
                valMetrics["policy_margin"],
                trainMetrics["reference_margin"],
                valMetrics["reference_margin"],
            ])

    @torch.no_grad()
    def estimate_metrics():
        out = {}
        policyModel.eval()

        for split in ["train", "val"]:
            metrics = defaultdict(float)

            for _ in range(args.eval_iters):
                batch = get_batch(split)
                result = dpo_batch_loss(
                    policyModel,
                    referenceModel,
                    batch,
                    args.beta,
                )

                for key, value in result.items():
                    metrics[key] += float(value.item())

            out[split] = {
                key: value / args.eval_iters
                for key, value in metrics.items()
            }

        policyModel.train()
        return out

    def save_checkpoint(step):
        dpoCheckpoint = {
            "model": policyModel.state_dict(),
            "config": asdict(config),
            "args": vars(args),
            "init_from": args.init_from,
            "step": step,
            "vocab": checkpoint["vocab"],
        }

        path = os.path.join(args.out_dir, "ckpt.pt")
        torch.save(dpoCheckpoint, path)
        print(f"saved checkpoint to {path}", flush=True)

    for step in range(args.max_iters):
        if step % args.eval_interval == 0:
            metrics = estimate_metrics()

            print(
                "step "
                f"{step}: train loss {metrics['train']['loss']:.4f}, "
                f"val loss {metrics['val']['loss']:.4f}, "
                f"train pref acc {metrics['train']['preference_accuracy']:.2%}, "
                f"val pref acc {metrics['val']['preference_accuracy']:.2%}, "
                f"train margin {metrics['train']['policy_margin']:.4f}, "
                f"val margin {metrics['val']['policy_margin']:.4f}",
                flush=True,
            )

            log_metrics(step, metrics["train"], metrics["val"])

        batch = get_batch("train")
        result = dpo_batch_loss(
            policyModel,
            referenceModel,
            batch,
            args.beta,
        )

        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        optimizer.step()

    save_checkpoint(args.max_iters - 1)


if __name__ == "__main__":
    main()
