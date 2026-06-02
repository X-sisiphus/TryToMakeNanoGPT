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
├── data_loader.py        # 字符级数据和 tokenizer 缓存数据的统一加载入口
├── prepare_data.py       # 将原始文本预处理为 tokenizer token 缓存
├── check_data_loader.py  # 检查数据加载和 causal LM batch 构造
├── sample.py             # 从 checkpoint 加载模型并生成文本
├── plot_log.py           # 根据 log.csv 绘制 loss 曲线
├── plot_ablation.py      # 批量绘制消融实验 loss 曲线
├── plot_ablation_summary.py # 绘制消融实验总览图
├── run_ablation.py       # 批量运行结构消融实验
├── run_full_ablation.py  # 一键运行消融、汇总和绘图
├── summarize_ablation.py # 汇总多组消融实验的最终指标
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

## 数据预处理

第一阶段的 `train.py` 直接从 `input.txt` 构造字符级词表。第二阶段开始，需要先把原始文本离线处理成 tokenizer token 缓存：

```bash
python prepare_data.py \
  --input input.txt \
  --out-dir data/tiny \
  --train-ratio 0.9 \
  --encoding gpt2
```

输出文件：

```text
data/tiny/
├── train.bin  # 训练 token ids
├── val.bin    # 验证 token ids
└── meta.json  # tokenizer、vocab_size、字符数、token 数量等元信息
```

当前默认使用 `tiktoken` 的 `gpt2` encoding，词表大小为 50257。`train.bin` 和 `val.bin` 使用 `uint16` 保存，因为 GPT-2 词表大小小于 65535。

`meta.json` 会记录 tokenizer、vocab_size、字符数、token 数、`chars_per_token`、train/val token 数等信息，方便更换语料时快速检查数据规模和 tokenizer 是否正常。

检查数据加载和 batch 构造：

```bash
python check_data_loader.py --data-dir data/tiny
```

这个脚本会同时检查 tokenizer 数据和字符级数据，并验证 causal LM 的 `x/y` 是否满足向后错一位的训练目标。

使用预处理后的 tokenizer 数据训练：

```bash
python train.py \
  --data-dir data/tiny \
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
  --out-dir out/token_debug
```

传入 `--data-dir` 时，`train.py` 会从 `meta.json` 读取 `vocab_size`，并从 `train.bin` / `val.bin` 读取 token id。没有传入 `--data-dir` 时，仍然保留原来的字符级 `input.txt` 训练流程。

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

如果恢复的是 tokenizer 数据训练得到的 checkpoint，需要同时传入对应的 `--data-dir`：

```bash
python train.py \
  --resume out/token_debug/ckpt.pt \
  --data-dir data/tiny \
  --max-iters 8000 \
  --out-dir out/token_debug
```

`--data-dir` 用来重新加载 `train.bin`、`val.bin` 和 `meta.json`。如果 tokenizer checkpoint 恢复时忘记传入 `--data-dir`，脚本会给出明确的中文错误提示。

## 生成文本

训练完成后，用 `sample.py` 从 checkpoint 生成文本：

```bash
python sample.py \
  --checkpoint out/modern/ckpt.pt \
  --prompt "The " \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40
```

使用 MPS：

```bash
USE_MPS=1 python sample.py \
  --checkpoint out/modern/ckpt.pt \
  --prompt "The " \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40
```

说明：

- `temperature < 1`：生成更保守
- `temperature > 1`：生成更发散
- `top-k`：只从概率最高的 k 个 token 中采样
- `prompt`：生成起始文本；没有传入时默认使用换行符

`sample.py` 会根据 checkpoint 中保存的 `vocab.type` 自动选择解码方式：字符级 checkpoint 使用字符表，tokenizer checkpoint 使用 `tiktoken`。

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

也可以一键完成消融训练、结果汇总和绘图：

```bash
python run_full_ablation.py \
  --out-dir out/ablation \
  --max-iters 200 \
  --eval-interval 20 \
  --eval-iters 5
```

汇总消融实验：

```bash
python summarize_ablation.py --root out/ablation
```

会生成：

```text
out/ablation/summary.csv
```

批量绘制每组实验的 loss 曲线：

```bash
python plot_ablation.py --root out/ablation
```

会在每个实验目录下生成：

```text
loss.png
```

绘制消融实验总览图：

```bash
python plot_ablation_summary.py --summary out/ablation/summary.csv
```

会生成：

```text
out/ablation/summary.png
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

数据入口：

- `--input`：字符级训练时读取的原始文本文件，默认是 `input.txt`
- `--data-dir`：读取 tokenizer 预处理后的 `train.bin`、`val.bin` 和 `meta.json`；传入后优先级高于 `--input`

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

## 阶段总结

这个项目的第一阶段，是从教学版 nanoGPT 出发，把一个字符级语言模型逐步改造成更接近现代 LLM 的 decoder-only Transformer 实验框架。

第一阶段已经完成的主线：

1. 从字符级语言模型开始，理解 embedding、block size、自回归训练目标
2. 手写 Transformer block、残差连接、多头注意力和 causal mask
3. 将 LayerNorm / GELU 扩展为 RMSNorm / SwiGLU
4. 将 learned position embedding 扩展为 RoPE
5. 将 MHA 扩展为 GQA / MQA
6. 将手写 attention 扩展为 PyTorch SDPA
7. 加入 train / validation split 和 loss 评估
8. 加入 checkpoint 保存、恢复训练和结构配置自动恢复
9. 加入独立 `sample.py`，支持 temperature 和 top-k 采样
10. 加入 seed、训练日志、学习率调度、梯度裁剪、weight decay 参数分组
11. 加入消融实验、结果汇总和可视化脚本

这一阶段的重点不是把 loss 训练到很低，而是建立一套可读、可改、可验证的小型 LLM 实验骨架。后面进入更真实的数据、tokenizer、SFT、DPO 和部署时，这个项目会作为底层理解和实验记录。

## 第二阶段：小型指令模型

第二阶段的目标，是从“理解模型结构”进入“理解训练流程”。也就是说，不只是知道 Transformer 怎么写，还要完整走一遍现代小模型从 continued pretraining 到 instruction tuning，再到偏好优化和评测的流程。

建议周期：2-3 个月。

### 2.1 Tokenizer 与数据模块

当前项目还是字符级建模，适合学习原理，但不适合进入真实 LLM 训练。第二阶段第一步要把数据入口升级：

- 接入 BPE / SentencePiece / Hugging Face tokenizer
- 把 `input.txt` 式单文件数据，升级为可复用的数据处理脚本
- 支持 train / validation / test 划分
- 支持把原始文本预处理成 token id 缓存文件
- 记录数据来源、清洗规则、token 数量和平均长度

这一小步完成后，项目会从“玩具字符模型”变成“可以吃真实文本数据的小型语言模型框架”。

### 2.2 Continued Pretraining

continued pretraining 是在一个已有模型或已有结构上，继续用领域文本做语言模型训练。这里可以先从小模型开始，不急着追求 100M+ 参数。

可以准备两类数据：

- 通用中文 / 英文文本：用于保持基本语言能力
- 天文、遥感、时空智能相关文本：用于建立你的研究方向特色

重点观察：

- 训练 loss 和验证 loss 是否稳定下降
- 领域数据比例变化是否影响通用生成能力
- 模型是否开始学到领域术语和基本表达
- 小模型容量不够时，错误主要来自数据、模型大小还是训练步数

### 2.3 Instruction Tuning

SFT 的目标，是让模型从“续写文本”变成“按指令回答”。这个阶段可以用小规模高质量数据，不需要一开始就追大数据量。

建议先做三类指令数据：

- 通用问答：让模型学会基本指令格式
- 摘要、改写、解释：训练可控生成能力
- 天文/时空智能相关问答：逐步形成个人方向特色

需要重点理解：

- prompt 格式如何影响回答风格
- 只训练 answer token 和训练完整 conversation 的区别
- 数据质量比数据规模在小模型上有多重要
- SFT 后模型是否牺牲了原本的续写能力

### 2.4 偏好优化：DPO / GRPO

在 SFT 之后，可以进入偏好优化。这个阶段不建议一开始就追复杂 RLHF，而是先理解 DPO。

DPO 适合你的学习路线，因为它能把“偏好数据”和“模型行为改变”联系起来，而且工程复杂度比完整 RLHF 低。

可以先做：

- 构造 chosen / rejected 数据
- 训练一个很小的 DPO 实验
- 比较 SFT 模型和 DPO 模型的回答差异
- 观察 DPO 是否真的让回答更符合偏好

如果后续想靠近 reasoning 或 agent 方向，再继续了解 GRPO。GRPO 更适合和可验证任务、数学推理、代码执行或规则反馈结合。

### 2.5 评测集与实验报告

第二阶段不能只看 loss，要开始建立自己的评测体系。

至少保留这些评测：

- 基础语言能力：困惑度、简单问答、摘要质量
- 指令跟随能力：格式、完整性、是否跑题
- 领域能力：天文、时空智能、遥感/GNSS 相关问答
- 稳定性：不同 seed、不同数据比例、不同训练步数的差异
- 生成质量：人工抽样记录好答案和坏答案

这一阶段的输出应该包括：

- 一个可复现实验配置
- 一组训练曲线
- SFT / DPO 前后的对比样例
- 一个小型领域评测集
- 一份技术报告，记录你为什么这么做、观察到了什么、下一步怎么改

## 第三阶段：推理部署与优化

第三阶段目标，是把训练出来的小模型变成可用系统。

建议周期：1-2 个月。

要做的事情：

- 用 vLLM / SGLang / Transformers serving 部署模型
- 测 latency、throughput、显存占用
- 对比不同 batch size、上下文长度、并发数
- 尝试 INT8 / INT4 量化
- 做一个可以交互的 demo

这一阶段的输出：

- 一个可运行服务
- 一个 demo
- 一份性能报告
- 一组部署参数对比结果

## 第四阶段：研究切口

第四阶段是长期方向选择。结合你的背景，比较自然的路线是“小模型训练工程 + 天文/时空智能领域数据 + 可靠评测”。

可以考虑的研究问题：

- 小模型 reasoning 能力能否通过高质量数据提升
- DPO / GRPO 在小模型上什么时候有效
- 数据质量比数据规模重要到什么程度
- 长上下文训练如何影响短上下文能力
- 量化对 reasoning / 代码 / 领域问答能力的伤害有多大
- RAG 能否弥补小模型知识不足
- 天文和时空智能任务中，哪些知识适合写进参数，哪些更适合用检索增强

短期最推荐的下一步：先完成 tokenizer 和数据模块。它是从第一阶段进入第二阶段的门槛，也是后面 continued pretraining、SFT、DPO 和领域评测的共同地基。
