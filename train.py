import torch
import os
from model import BigramLanguageModel, GPTConfig
import argparse

#argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="rmsnorm")
    parser.add_argument("--ffn", choices=["gelu", "swiglu"], default="swiglu")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--no-rope", dest="use_rope", action="store_false")
    parser.set_defaults(use_rope=True)
    parser.add_argument("--use-flash", dest="use_flash", action="store_true")
    parser.add_argument("--no-flash", dest="use_flash", action="store_false")
    parser.set_defaults(use_flash=True)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    return parser.parse_args()

args = parse_args()
useMps = os.environ.get("USE_MPS") == "1"
device = 'mps' if torch.backends.mps.is_available() and useMps else 'cpu'
print(f"🔥 确认：正在使用 {device} 运行", flush=True)
#引入文本、编码、解码
with open("input.txt","r",encoding = "utf-8") as trainTxt:
    text = trainTxt.read()
chars = sorted(list(set(text)))
vocabularySize = len(chars)
stringToInt = {ch:i for i, ch in enumerate(chars)}
intToString = {i:ch for i, ch in enumerate(chars)}
def encode(s):
    return [stringToInt[c] for c in s]
def decode(In):
    return ''.join([intToString[i] for i in In])

#张量化
data = torch.tensor(
    encode(text),
    dtype = torch.long 
)

#拆分训练集
n = int(args.train_ratio * len(data))
trainData = data[:n]
valData = data[n:]

#构造训练样本
blockSize = args.block_size
batchSize = args.batch_size
maxIters = args.max_iters
evalInterval = args.eval_interval
def getBatch(split):
    sourceData = trainData if split == "train" else valData
    #随机四个起点
    ix = torch.randint(
        len(sourceData) - blockSize,
        (batchSize,)
    )
    #input
    #stack将数据由一维张量堆叠为二维，原本数据是平铺的，现在多了batch作为纵轴
    x = torch.stack([
        sourceData[i:i+blockSize]
        for i in ix
    ])
    #target
    y = torch.stack([
        sourceData[i+1:i+blockSize+1]
        for i in ix
    ])
    x, y = x.to(device), y.to(device)
    return x,y

#实例化
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
print(config, flush=True)
model = BigramLanguageModel(vocabularySize, blockSize, config=config)
model.to(device)
#优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate
)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            xb, yb = getBatch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

#训练
for steps in range(maxIters):
    if steps % evalInterval == 0:
        losses = estimate_loss()
        print(
            f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}",
            flush=True
        )
    xb,yb = getBatch("train")
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#生成
context = torch.zeros(
    (1,1),
    dtype=torch.long,
    device=device
)
generated = model.generate(
    context,
    maxNewTokens=100
)
print(
    decode(
        generated[0].tolist()
    )
)
