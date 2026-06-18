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

def apply_rope(q, k, positionOffset=0, positionIds=None):
    _, T, _, headSize = q.shape
    assert headSize % 2 == 0
    dim = torch.arange(0, headSize, 2, device=q.device)
    inv_freq = 1.0 / (10000 ** (dim / headSize))
    if positionIds is None:
        position = torch.arange(positionOffset, positionOffset + T, device=q.device)
        freqs = torch.outer(position, inv_freq)
    else:
        freqs = positionIds.to(q.device).float().unsqueeze(-1) * inv_freq
    cos = freqs.cos()
    sin = freqs.sin()
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    if positionIds is None:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
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
    def forward(
        self,
        x,
        pastKv=None,
        useCache=False,
        positionOffset=None,
        attentionMask=None,
        positionIds=None,
    ):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(B, T, self.numHeads, self.headSize)
        k = k.view(B, T, self.numKvHeads, self.headSize)
        v = v.view(B, T, self.numKvHeads, self.headSize)
        pastLength = 0
        if pastKv is not None:
            pastLength = pastKv[0].shape[1]
        if self.useRoPE:
            if positionOffset is None:
                positionOffset = pastLength
            q, k = apply_rope(q, k, positionOffset=positionOffset, positionIds=positionIds)
        if pastKv is not None:
            pastK, pastV = pastKv
            k = torch.cat((pastK, k), dim=1)
            v = torch.cat((pastV, v), dim=1)
        presentKv = (k, v) if useCache else None
        repeatNum = self.numHeads // self.numKvHeads
        k = repeat_kv(k, repeatNum)
        v = repeat_kv(v, repeatNum)
        q = q.transpose(1, 2)  # B, H, T, D
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.useFlashAttention and attentionMask is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=pastKv is None,
            )
        else:
            wei = q @ k.transpose(-2, -1) * (self.headSize ** -0.5)
            if pastKv is None:
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            if attentionMask is not None:
                keyMask = attentionMask[:, -k.shape[-2]:]
                queryMask = attentionMask[:, -T:]
                keyMask = keyMask[:, None, None, :].bool()
                queryMask = queryMask[:, None, :, None].bool()
                wei = wei.masked_fill(~keyMask, float("-inf"))
                wei = torch.where(queryMask, wei, torch.zeros_like(wei))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        if useCache:
            return out, presentKv
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

    def forward(
        self,
        x,
        pastKv=None,
        useCache=False,
        positionOffset=None,
        attentionMask=None,
        positionIds=None,
    ):

        if useCache:
            attnOut, presentKv = self.sa(
                self.ln1(x),
                pastKv=pastKv,
                useCache=True,
                positionOffset=positionOffset,
                attentionMask=attentionMask,
                positionIds=positionIds,
            )
            x = x + attnOut
        else:
            x = x + self.sa(self.ln1(x), attentionMask=attentionMask, positionIds=positionIds)
            presentKv = None
        x = x + self.ffwd(self.ln2(x))

        if useCache:
            return x, presentKv
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
    def forward(self, idx, targets = None, attentionMask=None):
        #将idx中的元素替换为对应的随机向量，将idx升维
        B,T = idx.shape
        tokenEmbd = self.tokenEmbeddingTable(idx)
        #对一个batch中T个元素0——T生成位置编码
        x = tokenEmbd
        positionIds = None
        if attentionMask is not None:
            positionIds = attentionMask.long().cumsum(dim=1) - 1
            positionIds = positionIds.clamp(min=0)
        if self.positionEmbeddingTable is not None:
            if positionIds is None:
                positionIds = torch.arange(T, device=idx.device)
            positionEmbd = self.positionEmbeddingTable(positionIds)
            x = x + positionEmbd
        if attentionMask is None:
            x = self.blocks(x)
        else:
            for block in self.blocks:
                x = block(x, attentionMask=attentionMask, positionIds=positionIds)
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
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
        return logits,loss

    def forward_with_cache(
            self,
            idx,
            pastKvs=None,
            useCache=True,
            positionOffset=None,
            attentionMask=None,
            positionIds=None,
        ):
        B,T = idx.shape
        tokenEmbd = self.tokenEmbeddingTable(idx)
        x = tokenEmbd

        pastLength = 0
        if pastKvs is not None and len(pastKvs) > 0:
            pastLength = pastKvs[0][0].shape[1]

        if positionOffset is None:
            positionOffset = pastLength

        if positionIds is None and attentionMask is not None:
            if idx.shape[1] == attentionMask.shape[1]:
                positionIds = attentionMask.long().cumsum(dim=1) - 1
                positionIds = positionIds.clamp(min=0)
            else:
                positionIds = attentionMask.long().sum(dim=1, keepdim=True) - idx.shape[1]
                positionIds = positionIds + torch.arange(idx.shape[1], device=idx.device)
                positionIds = positionIds.clamp(min=0)

        if self.positionEmbeddingTable is not None:
            if positionIds is None:
                positions = torch.arange(
                    positionOffset,
                    positionOffset + T,
                    device=idx.device,
                )
            else:
                positions = positionIds
            positionEmbd = self.positionEmbeddingTable(positions)
            x = x + positionEmbd

        if pastKvs is None:
            pastKvs = [None] * len(self.blocks)

        presentKvs = []
        for block, pastKv in zip(self.blocks, pastKvs):
            x, presentKv = block(
                x,
                pastKv=pastKv,
                useCache=useCache,
                positionOffset=positionOffset,
                attentionMask=attentionMask,
                positionIds=positionIds,
            )
            if presentKv is not None:
                presentK, presentV = presentKv
                if presentK.shape[1] > self.config.blockSize:
                    presentKv = (
                        presentK[:, -self.config.blockSize:, :, :],
                        presentV[:, -self.config.blockSize:, :, :],
                    )
            presentKvs.append(presentKv)

        x = self.ln_f(x)
        logits = self.languageModelHead(x)
        return logits, presentKvs
    
    def configure_optimizers(self, weightDecay, learningRate):
        decayParams = []
        noDecayParams = []

        for name, param in self.named_parameters():
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

    #generate
    def generate(
        self,
        idx,
        maxNewTokens,
        temperature=1.0,
        topK=None,
        repetitionPenalty=1.0,
        repetitionStart=0,
        eosTokenId=None,
        useKvCache=False,
        attentionMask=None,
    ):
        assert temperature > 0
        assert repetitionPenalty >= 1.0
        assert repetitionStart >= 0
        if topK is not None:
            assert topK > 0
        canUseKvCache = useKvCache and (
            self.config.useRoPE
            or idx.shape[1] + maxNewTokens <= self.config.blockSize
        )
        pastKvs = None
        cachePosition = 0
        for _ in range(maxNewTokens):
            #剪切token
            if canUseKvCache:
                if pastKvs is None:
                    if self.config.useRoPE:
                        idxCond = idx[:, -self.config.blockSize:]
                        cachePosition = idx.shape[1] - idxCond.shape[1]
                        if attentionMask is not None:
                            attentionMaskCond = attentionMask[:, -self.config.blockSize:]
                        else:
                            attentionMaskCond = None
                    else:
                        idxCond = idx
                        attentionMaskCond = attentionMask
                else:
                    idxCond = idx[:, -1:]
                    attentionMaskCond = attentionMask
                logits, pastKvs = self.forward_with_cache(
                    idxCond,
                    pastKvs=pastKvs,
                    positionOffset=cachePosition,
                    attentionMask=attentionMaskCond,
                )
                cachePosition += idxCond.shape[1]
            else:
                idxCond = idx[:, -self.config.blockSize:]
                if attentionMask is not None:
                    attentionMaskCond = attentionMask[:, -self.config.blockSize:]
                else:
                    attentionMaskCond = None
                logits, loss = self(idxCond, attentionMask=attentionMaskCond)
            logits = logits[:,-1,:]
            if repetitionPenalty > 1.0:
                for batchIdx in range(idx.size(0)):
                    seenTokens = torch.unique(idx[batchIdx, repetitionStart:])
                    if seenTokens.numel() == 0:
                        continue
                    seenLogits = logits[batchIdx, seenTokens]
                    logits[batchIdx, seenTokens] = torch.where(
                        seenLogits < 0,
                        seenLogits * repetitionPenalty,
                        seenLogits / repetitionPenalty,
                    )
            logits = logits / temperature
            if topK is not None:
                v, _ = torch.topk(logits, min(topK, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            nextIdx = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat((idx, nextIdx), dim=1)
            if attentionMask is not None:
                nextMask = torch.ones(
                    (attentionMask.shape[0], 1),
                    dtype=attentionMask.dtype,
                    device=attentionMask.device,
                )
                attentionMask = torch.cat((attentionMask, nextMask), dim=1)

            if eosTokenId is not None:
                if torch.all(nextIdx.squeeze(-1) == eosTokenId):
                    break
        return idx
