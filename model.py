import torch
import torch.nn as nn


class BigramLanguageModel(nn.Module):
    #embedding
    def __init__(self, vocabSize):
        super().__init__()
        #生成了一个token对应向量的表，由pytorch随机生成
        self.tokenEmbeddingTable = nn.Embedding(
            vocabSize,
            vocabSize
        )
    #forword
    #idx是二维张量
    def forward(self, idx):
        #将idx中的元素替换为对应的随机向量，将idx升维
        logits = self.tokenEmbeddingTable(idx)
        return logits
    