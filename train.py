import torch
from model import BigramLanguageModel
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"🔥 确认：正在使用 {device} 运行")
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
blockSize = 256
batchSize = 64
def getBatch():
    #随机四个起点
    ix = torch.randint(
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
    x, y = x.to(device), y.to(device)
    return x,y

#实例化
model = BigramLanguageModel(vocabularySize,blockSize)
model.to(device)
#优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 3e-4
)

#训练
for steps in range(5000):
    xb,yb = getBatch()
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(loss.item())

#生成
context = torch.zeros(
    (1,1),
    dtype=torch.long,
    device=device
)
generated = model.generate(
    context,
    maxNewTokens=100
)
print(
    decode(
        generated[0].tolist()
    )
)


