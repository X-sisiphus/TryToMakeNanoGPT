import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    #embedding
    def __init__(self, vocabSize):
        super().__init__()
        #生成了一个token对应向量的表，由pytorch随机生成
        #这里的生成的结果直接就是代表了预测值，简化了由特征到预测的过程，也可以说预测的结果本身也是一种特征
        self.tokenEmbeddingTable = nn.Embedding(
            vocabSize,
            vocabSize
        )
    #forword
    #idx是二维张量
    def forward(self, idx, targets = None):
        #将idx中的元素替换为对应的随机向量，将idx升维
        logits = self.tokenEmbeddingTable(idx)
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


