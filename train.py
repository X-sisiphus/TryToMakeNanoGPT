import torch
import os
from model import BigramLanguageModel, GPTConfig
import argparse
import csv
from dataclasses import asdict
import json
import time
from data_loader import load_data, get_batch

#argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # 数据和训练循环参数：控制上下文长度、batch 大小、训练步数和学习率。
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)

    # 模型规模参数：这些参数会改变模型结构，通常会导致 checkpoint 彼此不兼容。
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)

    # 架构消融开关：baseline 可设为 layernorm / gelu / no-rope / no-flash / MHA。
    parser.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="rmsnorm")
    parser.add_argument("--ffn", choices=["gelu", "swiglu"], default="swiglu")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--no-rope", dest="use_rope", action="store_false")
    parser.set_defaults(use_rope=True)
    parser.add_argument("--use-flash", dest="use_flash", action="store_true")
    parser.add_argument("--no-flash", dest="use_flash", action="store_false")
    parser.set_defaults(use_flash=True)

    # 评估与实验记录参数：控制验证集评估、输出目录、断点续训和随机种子。
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)

    # 优化稳定性参数：warmup/cosine decay 控制学习率，grad clipping 限制梯度尖峰。
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=5000)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--no-lr-decay", dest="lr_decay", action="store_false")
    parser.set_defaults(lr_decay=True)

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--input", type=str, default="input.txt")

    return parser.parse_args()

args = parse_args()
torch.manual_seed(args.seed)
print(f"seed: {args.seed}", flush=True)
os.makedirs(args.out_dir, exist_ok=True)
logPath = os.path.join(args.out_dir, "log.csv")
useMps = os.environ.get("USE_MPS") == "1"
device = 'mps' if torch.backends.mps.is_available() and useMps else 'cpu'
print(f"🔥 确认：正在使用 {device} 运行", flush=True)

trainData, valData, vocabularySize, vocabInfo = load_data(
    dataDir=args.data_dir,
    inputPath=args.input,
    trainRatio=args.train_ratio,
)

#构造训练样本
blockSize = args.block_size
batchSize = args.batch_size
maxIters = args.max_iters
evalInterval = args.eval_interval

checkpoint = None
checkpointVocabType = None

if args.resume is not None:
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    checkpointVocabType = checkpoint.get("vocab", {}).get("type", "char")

#实例化
if checkpoint is not None and "config" in checkpoint:
    config = GPTConfig(**checkpoint["config"])
    blockSize = config.blockSize
else:
    config = GPTConfig(
        vocabSize=vocabularySize,
        blockSize=blockSize,
        nEmbd=args.n_embd,
        nLayer=args.n_layer,
        numHeads=args.num_heads,
        numKvHeads=args.num_kv_heads,
        dropout=args.dropout,
        normType=args.norm,
        ffnType=args.ffn,
        useRoPE=args.use_rope,
        useFlashAttention=args.use_flash,
    )
if config.vocabSize != vocabularySize:
    if checkpointVocabType == "tokenizer" and args.data_dir is None:
        raise ValueError(
            "当前 checkpoint 使用 tokenizer 词表，请传入 --data-dir 指向对应的预处理数据目录。"
        )

    raise ValueError(
        f"vocab size 不匹配: checkpoint/config 中是 {config.vocabSize}, "
        f"当前数据中是 {vocabularySize}。请检查 --data-dir 或训练数据是否与 checkpoint 对应。"
    )
print(config, flush=True)
model = BigramLanguageModel(config.vocabSize, config.blockSize, config=config)
model.to(device)

def save_run_config():
    runConfig = {
        "args": vars(args),
        "config": asdict(config),
        "num_params": numParams,
        "device": device,
    }
    path = os.path.join(args.out_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(runConfig, f, indent=2, ensure_ascii=False)
    print(f"saved run config to {path}", flush=True)


numParams = model.get_num_params()
print(f"number of parameters: {numParams / 1e6:.2f}M", flush=True)
save_run_config()

#优化器
optimizer = model.configure_optimizers(
    weightDecay=args.weight_decay,
    learningRate=args.learning_rate,
)
startStep = 0
if checkpoint is not None:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    startStep = checkpoint["step"] + 1
    print(f"resumed from {args.resume} at step {startStep}", flush=True)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            xb, yb = get_batch(
                split,
                trainData,
                valData,
                blockSize,
                batchSize,
                device,
            )
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(step):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "args": vars(args),
        "step": step,
        "vocab": vocabInfo,
    }
    path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save(checkpoint, path)
    print(f"saved checkpoint to {path}", flush=True)

#初始化日志文件
if startStep == 0:
    with open(logPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])

def log_metrics(step, trainLoss, valLoss, lr, tokensPerSec):
    with open(logPath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, trainLoss, valLoss, lr, tokensPerSec])

def get_lr(step):
    if not args.lr_decay:
        return args.learning_rate

    if step < args.warmup_iters:
        return args.learning_rate * step / args.warmup_iters

    if step > args.lr_decay_iters:
        return args.min_lr

    decayRatio = (step - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decayRatio * 3.141592653589793)))
    return args.min_lr + coeff.item() * (args.learning_rate - args.min_lr)

lastTime = time.time()
lastEvalStep = startStep

#训练
for steps in range(startStep, maxIters):
    lr = get_lr(steps)
    for paramGroup in optimizer.param_groups:
        paramGroup["lr"] = lr
    if steps > 0 and steps % args.save_interval == 0:
        save_checkpoint(steps)
    if steps % evalInterval == 0:
        now = time.time()
        if steps == startStep:
            tokensPerSec = 0.0
        else:
            trainedSteps = steps - lastEvalStep
            tokensPerSec = trainedSteps * batchSize * blockSize / (now - lastTime)
        lastTime = now
        lastEvalStep = steps
        losses = estimate_loss()
        print(
            f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6e}, tok/s {tokensPerSec:.0f}",
            flush=True
        )
        log_metrics(steps, losses["train"], losses["val"], lr, tokensPerSec)
    xb, yb = get_batch(
        "train",
        trainData,
        valData,
        blockSize,
        batchSize,
        device,
    )
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    #梯度裁剪
    if args.grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
save_checkpoint(maxIters - 1)
