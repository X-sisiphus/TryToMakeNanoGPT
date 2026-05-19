from turtle import forward
import torch
import torch.nn as nn


class BigramLanguageModel(nn.Module):
    #embedding
    def __init__(self, vocabSize):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(
            vocabSize,
            vocabSize
        )
    #forword
    #idx是二维张量
    def forward(self, idx):
        logits = self.tokenEmbeddingTable(idx)
        return logits
    