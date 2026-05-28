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

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

def apply_rope(q,k):
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
    
# 旧版单头注意力，当前 GQA 实现不再使用
# class Head(nn.Module):

#     def __init__(self, config, headSize):
#         super().__init__()
#         self.headSize = headSize
#         self.key = nn.Linear(config.nEmbd, headSize, bias=False)
#         self.query = nn.Linear(config.nEmbd, headSize, bias=False)
#         self.value = nn.Linear(config.nEmbd, headSize, bias=False)
#         self.register_buffer('tril',torch.tril(torch.ones(config.blockSize, config.blockSize)))
#         self.dropout = nn.Dropout(config.dropout)
#         self.useRoPE = config.useRoPE
    
#     def forward(self, x):
#         B,T,C = x.shape
#         k = self.key(x)
#         q = self.query(x)
#         #RoPE
#         if self.useRoPE:
#             q, k = apply_rope(q, k)
#         #注意力矩阵，反应了两个token间的注意力
#         wei = q @ k.transpose(-2,-1) * (self.headSize ** -0.5)
#         #mask
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#         #softmax
#         wei = F.softmax(wei, dim=-1)
#         #dropout
#         wei = self.dropout(wei)
#         #value 聚合
#         v = self.value(x)
#         out = wei @ v
#         return out

#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.nEmbd % config.numHeads == 0
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
        wei = q @ k.transpose(-2, -1) * (self.headSize ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out

#FFN
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
#block
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.sa = MultiHeadAttention(config)
        #前馈神经网络
        self.ffwd = FeedForward(config)
        self.ln1 = build_norm(config)
        self.ln2 = build_norm(config)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

class BigramLanguageModel(nn.Module):
    #embedding
    def __init__(self, vocabSize, blockSize=None, config=None):
        super().__init__()
        if config is None:
            if blockSize is None:
                blockSize = 256
            config = GPTConfig(vocabSize=vocabSize, blockSize=blockSize)
        self.config = config
        #生成了一个token对应向量的表，由pytorch随机生成
        #这里的生成的结果直接就是代表了预测值，简化了由特征到预测的过程，也可以说预测的结果本身也是一种特征
        #self.tokenEmbeddingTable = nn.Embedding(
        #    vocabSize,
        #    vocabSize
        
        #原先的写法简化了特征,现在让embedding的结果表示语义而不是预测
        self.tokenEmbeddingTable = nn.Embedding(
            config.vocabSize,
            config.nEmbd
        )

        #增加了位置向量；现在可选是否RoPE
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
        
    #forword
    #idx是二维张量
    def forward(self, idx, targets = None):
        #将idx中的元素替换为对应的随机向量，将idx升维
        B,T = idx.shape
        tokenEmbd = self.tokenEmbeddingTable(idx)
        #对一个batch中T个元素0——T生成位置编码
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
            B,T,C = logits.shape
            #view把原本2维的张量重新排列为一维
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            #下面的方法会先softmax，再计算loss（对数似然损失）
            loss = F.cross_entropy(logits, targets)
        return logits,loss
    
    #generate
    def generate(self,idx,maxNewTokens):
        for _ in range(maxNewTokens):
            #剪切token
            idxCond = idx[:, -self.config.blockSize:]
            logits, loss = self(idxCond)
            logits = logits[:,-1,:]
            probs = torch.softmax(logits, dim=-1)
            nextIdx = torch.multinomial(
            probs,
            num_samples=1
            )
            idx = torch.cat((idx,nextIdx),dim=1)
        return idx
