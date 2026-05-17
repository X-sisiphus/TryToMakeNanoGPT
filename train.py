from inspect import stack
from random import randint
import torch

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
#构造训练样本
blockSize = 8
batchSize = 4
def getBatch():
    #随机四个起点
    ix = randint(
        len(data) - blockSize,
        (batchSize,)
    )
    #input
    #stack将数据由一维张量堆叠为二维，原本数据是平铺的，现在多了batch作为纵轴
    x = torch.stack([
        data[i:i+blockSize]
        for i in ix
    ])
    #target
    y = torch.stack([
        data[i+1:i+blockSize+1]
        for i in ix
    ])
    return x,y