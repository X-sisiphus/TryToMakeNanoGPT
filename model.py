import torch
import torch.nn as nn

#embedding
class BigramLanguageModel(nn.Module):
    def __init__(self, vocabSize):
        super.__init__
        self.tokenEmbeddingTable = nn.Embedding(
            vocabSize,
            vocabSize
        )