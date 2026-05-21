import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.languageModelHead = nn.Linear(
            nEmbd,
            vocabSize
        )

    
    #forword
    #idx是二维张量
    def forward(self, idx, targets = None):
        #将idx中的元素替换为对应的随机向量，将idx升维
        B,T = idx.shape
        tokenEmbd = self.tokenEmbeddingTable(idx)
        #对一个batch中T个元素0——T生成位置编码
        positionEmbd = self.positionEmbeddingTable(torch.arange(T))
        x = tokenEmbd + positionEmbd
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
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = torch.softmax(logits, dim=-1)
            nextIdx = torch.multinomial(
            probs,
            num_samples=1
            )
            idx = torch.cat((idx,nextIdx),dim=1)
        return idx
        





