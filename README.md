# TryToMakeNanoGPT

这是一个从 nanoGPT 出发，逐步改造成 LLaMA-style decoder-only Transformer 的学习项目。

项目目标不是直接训练大模型，而是通过手写和小规模实验，理解现代大语言模型中的核心结构与训练工程。

## 当前特性

- Decoder-only Transformer 语言模型
- LayerNorm / RMSNorm 可切换
- GELU / SwiGLU 可切换
- learned position embedding / RoPE 可切换
- MHA / GQA / MQA 可切换
- 手写 attention / PyTorch scaled_dot_product_attention 可切换
- train / validation loss 评估
- checkpoint 保存与恢复
- resume 时自动从 checkpoint 恢复模型结构
- CSV 训练日志
- temperature 与 top-k 采样
- learning rate warmup + cosine decay
- gradient clipping
- weight decay 参数分组
- 随机种子与参数量统计

## 项目结构

```text
.
├── model.py              # 模型结构、RoPE、GQA、采样生成、优化器分组
├── train.py              # 训练、验证、checkpoint、日志、学习率调度
├── sample.py             # 从 checkpoint 加载模型并生成文本
├── plot_log.py           # 根据 log.csv 绘制 loss 曲线
├── run_ablation.py       # 批量运行结构消融实验
├── requirements-mps.txt  # Apple Silicon / MPS 环境依赖
└── README.md
```

## 环境

推荐环境：

- Python 3.9+
- PyTorch
- Apple Silicon 可使用 MPS 加速

安装依赖：

```bash
python -m pip install -r requirements-mps.txt
```

如果使用项目内的 MPS 虚拟环境：

```bash
source .venv-mps/bin/activate
python -m pip install -r requirements-mps.txt
```

## 快速调试训练

小模型调试命令：

```bash
python train.py \
  --max-iters 5 \
  --eval-interval 1 \
  --eval-iters 1 \
  --batch-size 4 \
  --block-size 16 \
  --n-embd 48 \
  --n-layer 1 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --dropout 0.0 \
  --out-dir out/debug
```

使用 MPS：

```bash
USE_MPS=1 python train.py \
  --max-iters 5 \
  --eval-interval 1 \
  --eval-iters 1 \
  --batch-size 4 \
  --block-size 16 \
  --n-embd 48 \
  --n-layer 1 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --dropout 0.0 \
  --out-dir out/debug
```

## LLaMA-style 配置

默认配置已经偏向现代 decoder-only Transformer：

- RMSNorm
- SwiGLU
- RoPE
- GQA
- PyTorch SDPA

示例命令：

```bash
USE_MPS=1 python train.py \
  --max-iters 5000 \
  --eval-interval 100 \
  --eval-iters 20 \
  --norm rmsnorm \
  --ffn swiglu \
  --use-rope \
  --num-kv-heads 2 \
  --use-flash \
  --out-dir out/modern
```

## Baseline 配置

更接近原始 nanoGPT 的 baseline：

- LayerNorm
- GELU
- learned position embedding
- MHA
- 手写 attention

示例命令：

```bash
python train.py \
  --norm layernorm \
  --ffn gelu \
  --no-rope \
  --num-kv-heads 6 \
  --no-flash \
  --out-dir out/baseline
```

## 恢复训练

训练会保存 checkpoint：

```text
out/xxx/ckpt.pt
```

恢复训练：

```bash
python train.py \
  --resume out/modern/ckpt.pt \
  --max-iters 8000 \
  --out-dir out/modern
```

恢复时，模型结构会自动从 checkpoint 中读取，不需要重新手动指定 `n-embd`、`n-layer`、`num-heads` 等结构参数。

## 生成文本

训练完成后，用 `sample.py` 从 checkpoint 生成文本：

```bash
python sample.py \
  --checkpoint out/modern/ckpt.pt \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40
```

使用 MPS：

```bash
USE_MPS=1 python sample.py \
  --checkpoint out/modern/ckpt.pt \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40
```

说明：

- `temperature < 1`：生成更保守
- `temperature > 1`：生成更发散
- `top-k`：只从概率最高的 k 个 token 中采样

## 消融实验

可以用 `run_ablation.py` 批量运行多组结构对比：

```bash
python run_ablation.py \
  --max-iters 200 \
  --eval-interval 20 \
  --eval-iters 5 \
  --out-dir out/ablation
```

默认会运行：

- `baseline`：LayerNorm + GELU + learned position embedding + MHA + manual attention
- `rmsnorm`
- `swiglu`
- `rope`
- `gqa`
- `sdpa`

每组实验会写入独立目录：

```text
out/ablation/
├── baseline/
├── rmsnorm/
├── swiglu/
├── rope/
├── gqa/
└── sdpa/
```

使用 MPS：

```bash
python run_ablation.py --use-mps --out-dir out/ablation
```

## 输出文件

每次训练会写入 `--out-dir`：

```text
out/xxx/
├── ckpt.pt   # 模型参数、优化器状态、config、vocab、step
└── log.csv   # step, train_loss, val_loss, lr
```

`log.csv` 可用于画 loss 曲线或做消融实验对比。

## 重要参数

模型结构：

- `--n-embd`：embedding 维度
- `--n-layer`：Transformer block 层数
- `--num-heads`：query head 数量
- `--num-kv-heads`：key/value head 数量；等于 `num-heads` 时是 MHA，等于 1 时是 MQA
- `--block-size`：最大上下文长度

架构开关：

- `--norm layernorm|rmsnorm`
- `--ffn gelu|swiglu`
- `--use-rope` / `--no-rope`
- `--use-flash` / `--no-flash`

训练稳定性：

- `--warmup-iters`
- `--lr-decay-iters`
- `--min-lr`
- `--grad-clip`
- `--weight-decay`
- `--seed`

## 学习路线

这个项目记录了从教学版 nanoGPT 到现代 decoder-only Transformer 的改造过程：

1. 从字符级语言模型开始
2. 加入 Transformer block、残差连接、多头注意力
3. 将 LayerNorm / GELU 扩展为 RMSNorm / SwiGLU
4. 将 learned position embedding 扩展为 RoPE
5. 将 MHA 扩展为 GQA / MQA
6. 将手写 attention 扩展为 PyTorch SDPA
7. 加入 train / val 评估
8. 加入 checkpoint 保存与自动恢复
9. 加入独立 sample 脚本
10. 加入学习率调度、梯度裁剪、weight decay 参数分组

后续计划：

- 做系统消融实验
- 记录 loss 曲线和训练速度
- 接入更真实的 tokenizer
- 进入小规模 SFT / DPO 实验
