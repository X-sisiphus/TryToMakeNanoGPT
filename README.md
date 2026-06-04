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
├── inspect_tokenizer.py  # 观察 tokenizer 的 token id 和文本切片
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
  --encoding gpt2 \
  --dtype uint16
```

`--input` 可以是单个 `.txt` 文件，也可以是包含多个 `.txt` 文件的目录。传入目录时，脚本会递归读取所有 `.txt` 文件，按路径排序后拼接。

输出文件：

```text
data/tiny/
├── train.bin  # 训练 token ids
├── val.bin    # 验证 token ids
├── meta.json  # tokenizer、vocab_size、字符数、token 数量等元信息
└── manifest.json  # 每个输入文件的路径和字符数
```

当前默认使用 `tiktoken` 的 `gpt2` encoding，词表大小为 50257。`train.bin` 和 `val.bin` 默认使用 `uint16` 保存，因为 GPT-2 词表大小小于 65535。更大的 tokenizer 可以使用 `--dtype uint32`，避免 token id 超出 `uint16` 上限。

`meta.json` 会记录 input、文件数量、文件列表、manifest 路径、tokenizer、vocab_size、dtype、字符数、token 数、`chars_per_token`、train/val token 数等信息，方便更换语料时快速检查数据规模和 tokenizer 是否正常。数据在磁盘上按 `dtype` 保存，读入训练时会转换为 `torch.long`。

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

Tokenizer 观察：

```bash
python inspect_tokenizer.py --encoding gpt2
```

BPE 的核心思想是先从很小的基本符号开始，不断把语料中最常共同出现的相邻片段合并成新 token。高频英文词、常见空格加单词组合、常见数字或符号片段会更容易被合并；低频词、中文字符和专业术语可能会被拆成更多 token。

当前用 GPT-2 tokenizer 观察到的现象：

- `hello world` 会被切成 `hello` 和 ` world`，空格常常和后面的英文词绑定在一起
- `Transformer` 会被切成 `Trans` 和 `former`，说明词表里不一定有完整词
- 中文在 GPT-2 tokenizer 下会明显更碎，单个 token 解码时还可能出现替换字符
- `GNSS time series and space geodesy` 里，常见英文片段比较稳定，`geodesy` 会被拆成多个子词
- `RA=12h30m, Dec=+45deg` 这类领域符号串会被拆成缩写、数字和符号片段

这说明 tokenizer 会影响有效上下文长度、embedding 参数量和领域文本的建模效率。后续如果进入中文天文/时空智能语料，GPT-2 tokenizer 适合学习流程，但不一定是最终训练中文领域模型的最佳选择。

Tokenizer 小训练实验：

```bash
python train.py \
  --data-dir data/tiny \
  --max-iters 300 \
  --eval-interval 50 \
  --eval-iters 5 \
  --batch-size 8 \
  --block-size 64 \
  --n-embd 96 \
  --n-layer 2 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --dropout 0.1 \
  --learning-rate 3e-4 \
  --warmup-iters 20 \
  --lr-decay-iters 300 \
  --out-dir out/token_300
```

实验记录：

- 模型参数量约 9.90M
- step 0：train loss 11.0051，val loss 11.0085
- step 200：train loss 6.4742，val loss 6.4179
- step 250：train loss 6.3929，val loss 6.5479
- 训练速度约 3900-4200 tokens/s

随机模型在 GPT-2 tokenizer 词表上的初始 loss 接近 `log(50257) ≈ 10.82`，所以 step 0 的 loss 在 11 左右是正常的。训练到 300 step 后，loss 明显下降，说明模型已经学到 token 分布、空格、标点和常见短词模式；但采样结果仍然很碎，主要表现为短词、标点、换行和常见代词的组合，还没有形成稳定语义。

采样观察：

- prompt `The `：能生成英文短词、标点和换行，但句子结构不稳定
- prompt `RA=`：没有学到天文坐标格式，只生成了通用英文碎片

结论：这次实验验证了 tokenizer 数据链路和训练闭环是正确的，但 `data/tiny` 规模、训练步数和模型容量都还不足以产生可读文本。后续 continued pretraining 需要更长训练、更稳定评估，以及更贴近目标方向的语料。

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

`astro_tiny` 小型领域实验：

```bash
python train.py \
  --data-dir data/astro_tiny \
  --max-iters 300 \
  --eval-interval 50 \
  --eval-iters 5 \
  --batch-size 4 \
  --block-size 32 \
  --n-embd 64 \
  --n-layer 2 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --dropout 0.1 \
  --learning-rate 3e-4 \
  --warmup-iters 20 \
  --lr-decay-iters 300 \
  --out-dir out/astro_tiny_300
```

数据规模：

- `data/astro_tiny` 来自 4 个领域说明文本
- 总量约 12160 chars / 2493 tokens
- train tokens 2243，val tokens 250

实验记录：

- 模型参数量约 6.57M
- step 0：train loss 11.0235，val loss 10.9908
- step 150：train loss 7.7700，val loss 8.1631
- step 250：train loss 6.8630，val loss 7.3034
- 训练速度约 3000-3300 tokens/s

采样观察：

- prompt `GNSS `：开始出现 `Earth`、`clock`、`satellite`、`temporal`、`infrastructure`、`measurements` 等领域相关词
- prompt `Space geodesy `：出现 `Earth`、`remote`、`satellite`、`measurements`、`geodesy` 片段
- prompt `A terrestrial reference frame `：出现 `GNSS`、`precise`、`analysis`、`measurements`、`temporal` 等词

结论：小语料 continued pretraining 已经能把采样分布推向领域词，但 2493 tokens 太少，模型主要学到词频和局部片段，还没有稳定句法和可靠语义。这个实验更像“领域小语料过拟合/分布迁移验证”，不能说明模型具备真正的天文或时空智能能力。

通用小模型与 astro 小模型的采样对比记录见 [experiments/token_vs_astro_sampling.md](experiments/token_vs_astro_sampling.md)。

同尺寸控制变量对比记录见 [experiments/controlled_general_vs_astro.md](experiments/controlled_general_vs_astro.md)。

扩大到 41.6K tokens 的 `astro_small_500` 实验记录见 [experiments/astro_small_500.md](experiments/astro_small_500.md)。

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

最小 SFT 数据格式：

```json
{"instruction": "...", "input": "...", "output": "..."}
```

当前可以用脚本生成一个本地 tiny SFT 数据集：

```bash
python scripts/build_astro_sft_tiny.py \
  --out data/sft/astro_sft_tiny.jsonl
```

检查数据格式：

```bash
python check_sft_data.py \
  --path data/sft/astro_sft_tiny.jsonl
```

`astro_sft_tiny.jsonl` 当前包含 30 条样本，覆盖：

- `concept_explanation`：概念解释
- `summary`：段落摘要
- `field_extraction`：GNSS station metadata 字段抽取
- `format_conversion`：RA/Dec 等格式转自然语言
- `qa`：领域问答

Continued pretraining 仍然是普通文本的 next-token prediction；SFT 则把数据组织成 instruction/input/output，让模型学习“看到任务描述后生成目标答案”。后续训练 SFT 时，还需要决定是否只在 `output` 部分计算 loss，这是 instruction tuning 的关键技术点之一。

当前 SFT 模板：

```text
Instruction:
...

Input:
...

Answer:
...
```

编码检查：

```bash
python check_sft_encoding.py \
  --path data/sft/astro_sft_tiny.jsonl \
  --encoding gpt2
```

当前检查结果：

- examples: 30
- avg prompt tokens: 46.7
- avg answer tokens: 31.8
- max total tokens: 114
- prompt 部分 labels 使用 `-100`，不参与 loss
- answer 部分 labels 等于目标 token id，用于 SFT 训练

Batch padding 检查：

```bash
python check_sft_batch.py \
  --path data/sft/astro_sft_tiny.jsonl \
  --encoding gpt2 \
  --batch-size 4
```

当前检查结果：

- `input_ids` shape: `(4, 62)`
- `labels` shape: `(4, 62)`
- `attention_mask` shape: `(4, 62)`
- `input_ids` padding 使用 `PAD_TOKEN_ID=0`
- `labels` padding 使用 `-100`，避免 padding token 参与 loss

SFT debug 训练：

```bash
python train_sft.py \
  --sft-path data/sft/astro_sft_tiny.jsonl \
  --max-iters 20 \
  --eval-interval 5 \
  --batch-size 4 \
  --block-size 128 \
  --n-embd 64 \
  --n-layer 2 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --dropout 0.1 \
  --out-dir out/sft_debug
```

当前 debug 结果：

- step 0：loss 10.9659
- step 5：loss 10.8767
- step 10：loss 10.7385
- step 15：loss 10.4012
- checkpoint 保存到 `out/sft_debug/ckpt.pt`

这一步只验证 SFT 训练链路能跑通。当前 `train_sft.py` 是从随机初始化模型开始训练，20 step 后采样仍然接近随机文本是正常的。真正有意义的 SFT 应该从 continued pretraining checkpoint 初始化，或者使用更大的高质量 SFT 数据。

从 continued pretraining checkpoint 初始化 SFT：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_tiny.jsonl \
  --max-iters 20 \
  --eval-interval 5 \
  --batch-size 4 \
  --block-size 128 \
  --out-dir out/sft_from_astro_debug
```

这里的代码逻辑和随机初始化不同：

- `--init-from` 会读取预训练 checkpoint 中的模型结构和参数
- SFT 仍然使用 `gpt2` tokenizer，因此会检查 checkpoint 词表大小是否匹配
- SFT 样本通常比预训练 block 更长，所以允许用新的 `--block-size`
- 如果 block size 改变，attention 里的 causal mask 形状会变，脚本会跳过这些不匹配的 buffer，让模型重新创建
- prompt 部分的 label 仍然是 `-100`，只在 answer 部分计算 loss

这一步把二阶段主线接起来：

```text
领域文本 continued pretraining -> 指令样本 SFT -> 后续 DPO / 评测
```

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
