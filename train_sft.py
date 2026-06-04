import argparse
import os
import torch
import tiktoken
from dataclasses import asdict

from model import BigramLanguageModel, GPTConfig
from sft_data import load_sft_jsonl, encode_sft_example, pad_sft_batch


def parse_args():
    parser = argparse.ArgumentParser()

    # SFT 数据参数
    parser.add_argument("--sft-path", type=str, default="data/sft/astro_sft_tiny.jsonl")
    parser.add_argument("--encoding", type=str, default="gpt2")

    # batch 和训练循环参数
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)

    # 模型结构参数
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 实验记录参数
    parser.add_argument("--out-dir", type=str, default="out/sft_debug")
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


args = parse_args()

torch.manual_seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)

useMps = os.environ.get("USE_MPS") == "1"
device = "mps" if torch.backends.mps.is_available() and useMps else "cpu"

print(f"seed: {args.seed}", flush=True)
print(f"using device: {device}", flush=True)
print(f"sft path: {args.sft_path}", flush=True)
print(f"encoding: {args.encoding}", flush=True)
print(f"out dir: {args.out_dir}", flush=True)


# 1. 加载并编码 SFT 数据
examples = load_sft_jsonl(args.sft_path)
enc = tiktoken.get_encoding(args.encoding)

encoded = [
    encode_sft_example(example, enc)
    for example in examples
]

encoded = [
    item for item in encoded
    if len(item["input_ids"]) <= args.block_size
]

if len(encoded) == 0:
    raise ValueError("没有样本长度小于等于 block_size，请增大 --block-size。")

promptLens = [item["prompt_tokens"] for item in encoded]
answerLens = [item["answer_tokens"] for item in encoded]
totalLens = [len(item["input_ids"]) for item in encoded]

print(f"sft examples: {len(encoded)}", flush=True)
print(f"avg prompt tokens: {sum(promptLens) / len(promptLens):.1f}", flush=True)
print(f"avg answer tokens: {sum(answerLens) / len(answerLens):.1f}", flush=True)
print(f"max total tokens: {max(totalLens)}", flush=True)


# 2. 构造 batch
def get_batch():
    ix = torch.randint(
        len(encoded),
        (args.batch_size,)
    )

    items = [
        encoded[i]
        for i in ix
    ]

    batch = pad_sft_batch(items)

    x = batch["input_ids"].to(device)
    y = batch["labels"].to(device)

    return x, y


# 3. 构造模型
config = GPTConfig(
    vocabSize=enc.n_vocab,
    blockSize=args.block_size,
    nEmbd=args.n_embd,
    nLayer=args.n_layer,
    numHeads=args.num_heads,
    numKvHeads=args.num_kv_heads,
    dropout=args.dropout,
    normType="rmsnorm",
    ffnType="swiglu",
    useRoPE=True,
    useFlashAttention=True,
)

print(config, flush=True)

model = BigramLanguageModel(
    config.vocabSize,
    config.blockSize,
    config=config,
)

model.to(device)

numParams = model.get_num_params()
print(f"number of parameters: {numParams / 1e6:.2f}M", flush=True)


# 4. 优化器
optimizer = model.configure_optimizers(
    weightDecay=0.1,
    learningRate=args.learning_rate,
)


# 5. 保存 checkpoint
def save_checkpoint(step):
    checkpoint = {
        "model": model.state_dict(),
        "config": asdict(config),
        "args": vars(args),
        "step": step,
        "vocab": {
            "type": "tokenizer",
            "meta": {
                "tokenizer": "tiktoken",
                "encoding": args.encoding,
                "vocab_size": enc.n_vocab,
            },
        },
    }

    path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save(checkpoint, path)
    print(f"saved checkpoint to {path}", flush=True)


# 6. 训练循环
model.train()

for step in range(args.max_iters):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % args.eval_interval == 0:
        print(
            f"step {step}: loss {loss.item():.4f}",
            flush=True,
        )

save_checkpoint(args.max_iters - 1)