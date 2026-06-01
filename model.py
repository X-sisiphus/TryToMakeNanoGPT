import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocabSize: int
    blockSize: int = 256
    nEmbd: int = 384
    nLayer: int = 6
    numHeads: int = 6
    dropout: float = 0.2
    normType: str = "layernorm"
    ffnType: str = "gelu"
    useRoPE: bool = False
    numKvHeads: int = None
    useFlashAttention: bool = True


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rope(q, k):
    _, T, _, headSize = q.shape
    assert headSize % 2 == 0
    position = torch.arange(T, device=q.device)
    dim = torch.arange(0, headSize, 2, device=q.device)
    inv_freq = 1.0 / (10000 ** (dim / headSize))
    freqs = torch.outer(position, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def repeat_kv(x, repeatNum):
    # GQA/MQA 中把较少的 KV heads 扩展到 query head 数量。
    if repeatNum == 1:
        return x
    B, T, numKvHeads, headSize = x.shape
    x = x[:, :, :, None, :].expand(B, T, numKvHeads, repeatNum, headSize)
    return x.reshape(B, T, numKvHeads * repeatNum, headSize)


class RMSNorm(nn.Module):
    def __init__(self, nEmbd, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(nEmbd))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x


def build_norm(config):
    if config.normType == "layernorm":
        return nn.LayerNorm(config.nEmbd)
    if config.normType == "rmsnorm":
        return RMSNorm(config.nEmbd)
    raise ValueError(f"不支持的 normType: {config.normType}")


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.nEmbd % config.numHeads == 0
        self.useFlashAttention = config.useFlashAttention
        self.numHeads = config.numHeads
        self.numKvHeads = config.numKvHeads or config.numHeads
        self.headSize = config.nEmbd // config.numHeads
        assert self.numHeads % self.numKvHeads == 0
        self.query = nn.Linear(config.nEmbd, self.numHeads * self.headSize, bias=False)
        self.key = nn.Linear(config.nEmbd, self.numKvHeads * self.headSize, bias=False)
        self.value = nn.Linear(config.nEmbd, self.numKvHeads * self.headSize, bias=False)
        self.proj = nn.Linear(config.nEmbd, config.nEmbd)
        self.dropout = nn.Dropout(config.dropout)
        self.useRoPE = config.useRoPE
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.blockSize, config.blockSize))
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(B, T, self.numHeads, self.headSize)
        k = k.view(B, T, self.numKvHeads, self.headSize)
        v = v.view(B, T, self.numKvHeads, self.headSize)
        if self.useRoPE:
            q, k = apply_rope(q, k)
        repeatNum = self.numHeads // self.numKvHeads
        k = repeat_kv(k, repeatNum)
        v = repeat_kv(v, repeatNum)
        q = q.transpose(1, 2)  # B, H, T, D
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.useFlashAttention:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            wei = q @ k.transpose(-2, -1) * (self.headSize ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffnType = config.ffnType
        if self.ffnType == "gelu":
            self.net = nn.Sequential(
                nn.Linear(config.nEmbd, 4 * config.nEmbd),
                nn.GELU(),
                nn.Linear(4 * config.nEmbd, config.nEmbd),
                nn.Dropout(config.dropout)
            )
        elif self.ffnType == "swiglu":
            hiddenDim = int(8 * config.nEmbd / 3)
            self.w1 = nn.Linear(config.nEmbd, hiddenDim, bias=False)
            self.w2 = nn.Linear(hiddenDim, config.nEmbd, bias=False)
            self.w3 = nn.Linear(config.nEmbd, hiddenDim, bias=False)
            self.dropout = nn.Dropout(config.dropout)
        else:
            raise ValueError(f"不支持的 ffnType: {self.ffnType}")

    def forward(self, x):
        if self.ffnType == "swiglu":
            x = self.w2(F.silu(self.w1(x)) * self.w3(x))
            return self.dropout(x)
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = build_norm(config)
        self.ln2 = build_norm(config)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocabSize, blockSize=None, config=None):
        super().__init__()
        if config is None:
            if blockSize is None:
                blockSize = 256
            config = GPTConfig(vocabSize=vocabSize, blockSize=blockSize)
        self.config = config

        self.tokenEmbeddingTable = nn.Embedding(
            config.vocabSize,
            config.nEmbd
        )
        if not config.useRoPE:
            self.positionEmbeddingTable = nn.Embedding(config.blockSize, config.nEmbd)
        else:
            self.positionEmbeddingTable = None

        self.languageModelHead = nn.Linear(
            config.nEmbd,
            config.vocabSize
        )

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.nLayer)])

        self.ln_f = build_norm(config)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tokenEmbd = self.tokenEmbeddingTable(idx)
        x = tokenEmbd
        if self.positionEmbeddingTable is not None:
            positionEmbd = self.positionEmbeddingTable(torch.arange(T, device=idx.device))
            x = x + positionEmbd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.languageModelHead(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def configure_optimizers(self, weightDecay, learningRate):
        # GPT 训练中常见做法：矩阵权重 decay，bias/norm 不 decay。
        decayParams = []
        noDecayParams = []

        for _, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if param.dim() >= 2:
                decayParams.append(param)
            else:
                noDecayParams.append(param)

        optimGroups = [
            {"params": decayParams, "weight_decay": weightDecay},
            {"params": noDecayParams, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optimGroups,
            lr=learningRate,
        )

        numDecayParams = sum(p.numel() for p in decayParams)
        numNoDecayParams = sum(p.numel() for p in noDecayParams)
        print(f"num decayed parameter tensors: {len(decayParams)}, with {numDecayParams:,} parameters")
        print(f"num non-decayed parameter tensors: {len(noDecayParams)}, with {numNoDecayParams:,} parameters")
        return optimizer

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def generate(self, idx, maxNewTokens, temperature=1.0, topK=None):
        assert temperature > 0
        if topK is not None:
            assert topK > 0
        for _ in range(maxNewTokens):
            idxCond = idx[:, -self.config.blockSize:]
            logits, loss = self(idxCond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            if topK is not None:
                v, _ = torch.topk(logits, min(topK, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            nextIdx = torch.multinomial(
                probs,
                num_samples=1
            )
            idx = torch.cat((idx, nextIdx), dim=1)
        return idx
