import torch
import torch.nn as nn
import torch.nn.functional as F

nEmbd = 32
blockSize = 8

class Head(nn.Module):

    def __init__(self, headSize):
        super().__init__()
        self.headSize = headSize
        self.key = nn.Linear(nEmbd, headSize, bias=False)
        self.query = nn.Linear(nEmbd, headSize, bias=False)
        self.value = nn.Linear(nEmbd, headSize, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(blockSize, blockSize)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #注意力矩阵，反应了两个token间的注意力
        wei = q @ k.transpose(-2,-1) * (self.headSize ** -0.5)
        #mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        #softmax
        wei = F.softmax(wei, dim=-1)
        #value 聚合
        v = self.value(x)
        out = wei @ v
        return out

#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, numHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(headSize)
            for _ in range(numHeads)
        ])
        self.proj = nn.Linear(
            headSize * numHeads,
            nEmbd
        )
    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        out = self.proj(out)
        return out

#FFN
class FeedForward(nn.Module):
    def __init__(self, nEmbd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nEmbd, 4 * nEmbd),
            nn.ReLU(),
            nn.Linear(4 * nEmbd, nEmbd),
        )
    def forward(self, x):
        return self.net(x)

class BigramLanguageModel(nn.Module):
    #embedding
    def __init__(self, vocabSize, blockSize):
        super().__init__()
        #生成了一个token对应向量的表，由pytorch随机生成
        #这里的生成的结果直接就是代表了预测值，简化了由特征到预测的过程，也可以说预测的结果本身也是一种特征
        #self.tokenEmbeddingTable = nn.Embedding(
        #    vocabSize,
        #    vocabSize
        
        #原先的写法简化了特征,现在让embedding的结果表示语义而不是预测
        nEmbd = 32
        self.tokenEmbeddingTable = nn.Embedding(
            vocabSize,
            nEmbd
        )

        #增加了位置向量
        self.positionEmbeddingTable = nn.Embedding(
            blockSize,
            nEmbd
        )

        #前馈神经网络
        self.ffwd = FeedForward(nEmbd)

        numHeads = 4
        headSize = nEmbd // numHeads
        
        self.saHead = MultiHeadAttention(
            numHeads,
            headSize
        )

        self.languageModelHead = nn.Linear(
            nEmbd,
            vocabSize
        )

        #LayerNorm
        self.ln1 = nn.LayerNorm(nEmbd)
        self.ln2 = nn.LayerNorm(nEmbd)
        
        

    
    #forword
    #idx是二维张量
    def forward(self, idx, targets = None):
        #将idx中的元素替换为对应的随机向量，将idx升维
        B,T = idx.shape
        tokenEmbd = self.tokenEmbeddingTable(idx)
        #对一个batch中T个元素0——T生成位置编码
        positionEmbd = self.positionEmbeddingTable(torch.arange(T))
        x = tokenEmbd + positionEmbd
        x = x + self.saHead(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
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
            idxCond = idx[:, -blockSize:]
            logits, loss = self(idxCond)
            logits = logits[:,-1,:]
            probs = torch.softmax(logits, dim=-1)
            nextIdx = torch.multinomial(
            probs,
            num_samples=1
            )
            idx = torch.cat((idx,nextIdx),dim=1)
        return idx
        





