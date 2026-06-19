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
- temperature / top-k / repetition penalty 采样
- learning rate warmup + cosine decay
- gradient clipping
- weight decay 参数分组
- 随机种子与参数量统计

## 项目结构

```text
.
├── model.py              # 模型结构、RoPE、GQA、采样生成、优化器分组
├── train.py              # 训练、验证、checkpoint、日志、学习率调度
├── train_sft.py          # SFT 训练入口
├── sample.py             # 从 checkpoint 加载模型并生成文本
├── data_loader.py        # 字符级数据和 tokenizer 缓存数据的统一加载入口
├── sft_data.py           # SFT 样本格式化、编码、padding
├── tools/
│   ├── data/             # tokenizer 数据预处理、数据加载检查
│   ├── sft/              # SFT 数据检查、编码检查、task 筛选
│   ├── eval/             # SFT 采样对比、生成诊断、质量评测
│   ├── plots/            # loss 曲线和消融图表
│   ├── serve/            # 本地 FastAPI 推理服务
│   └── experiments/      # 消融实验运行和汇总
├── scripts/              # SFT / 领域数据生成脚本
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
python tools/data/prepare_data.py \
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
python tools/data/check_data_loader.py --data-dir data/tiny
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
  --top-k 40 \
  --repetition-penalty 1.0
```

使用 MPS：

```bash
USE_MPS=1 python sample.py \
  --checkpoint out/modern/ckpt.pt \
  --prompt "The " \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-k 40 \
  --repetition-penalty 1.0
```

说明：

- `temperature < 1`：生成更保守
- `temperature > 1`：生成更发散
- `top-k`：只从概率最高的 k 个 token 中采样
- `repetition-penalty > 1`：降低已出现 token 再次被采样的概率，缓解重复
- `prompt`：生成起始文本；没有传入时默认使用换行符

`sample.py` 会根据 checkpoint 中保存的 `vocab.type` 自动选择解码方式：字符级 checkpoint 使用字符表，tokenizer checkpoint 使用 `tiktoken`。

## 消融实验

可以用 `run_ablation.py` 批量运行多组结构对比：

```bash
python tools/experiments/run_ablation.py \
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
python tools/experiments/run_ablation.py --use-mps --out-dir out/ablation
```

也可以一键完成消融训练、结果汇总和绘图：

```bash
python tools/experiments/run_full_ablation.py \
  --out-dir out/ablation \
  --max-iters 200 \
  --eval-interval 20 \
  --eval-iters 5
```

汇总消融实验：

```bash
python tools/experiments/summarize_ablation.py --root out/ablation
```

会生成：

```text
out/ablation/summary.csv
```

批量绘制每组实验的 loss 曲线：

```bash
python tools/plots/plot_ablation.py --root out/ablation
```

会在每个实验目录下生成：

```text
loss.png
```

绘制消融实验总览图：

```bash
python tools/plots/plot_ablation_summary.py --summary out/ablation/summary.csv
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
9. 加入独立 `sample.py`，支持 temperature、top-k 和 repetition penalty 采样
10. 加入 seed、训练日志、学习率调度、梯度裁剪、weight decay 参数分组
11. 加入消融实验、结果汇总和可视化脚本

这一阶段的重点不是把 loss 训练到很低，而是建立一套可读、可改、可验证的小型 LLM 实验骨架。后面进入更真实的数据、tokenizer、SFT、DPO 和部署时，这个项目会作为底层理解和实验记录。

## 第二阶段：小型指令模型

第二阶段的目标，是从“理解模型结构”进入“理解训练流程”。也就是说，不只是知道 Transformer 怎么写，还要完整走一遍现代小模型从 continued pretraining 到 instruction tuning，再到偏好优化和评测的流程。

建议周期：2-3 个月。

第二阶段总结见：

```text
experiments/stage2_summary.md
```

该总结收束了 token 数据、continued pretraining、SFT、结构化评测、curriculum、DPO 正负结果，以及为什么下一步进入推理部署阶段。

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
python tools/data/inspect_tokenizer.py --encoding gpt2
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
python tools/sft/check_sft_data.py \
  --path data/sft/astro_sft_tiny.jsonl
```

`astro_sft_tiny.jsonl` 当前包含 30 条样本，覆盖：

- `concept_explanation`：概念解释
- `summary`：段落摘要
- `field_extraction`：GNSS station metadata 字段抽取
- `format_conversion`：RA/Dec 等格式转自然语言
- `qa`：领域问答

tiny 数据集主要用于验证 SFT 编码、batch 和训练链路。进入实际 SFT 对比时，可以生成稍大的 small 数据集：

```bash
python scripts/build_astro_sft_small.py \
  --out data/sft/astro_sft_small.jsonl
```

`astro_sft_small.jsonl` 当前包含 200 条样本，五类任务各 40 条：

- `concept_explanation`：概念解释
- `field_extraction`：字段抽取
- `format_conversion`：格式转换
- `qa`：简短问答
- `summary`：摘要归纳

检查 small 数据：

```bash
python tools/sft/check_sft_data.py \
  --path data/sft/astro_sft_small.jsonl

python tools/sft/check_sft_encoding.py \
  --path data/sft/astro_sft_small.jsonl \
  --encoding gpt2

python tools/sft/check_sft_batch.py \
  --path data/sft/astro_sft_small.jsonl \
  --encoding gpt2 \
  --batch-size 4
```

当前检查结果：

- examples: 200
- avg input chars: 70.2
- avg output chars: 127.1
- avg prompt tokens: 41.2
- avg answer tokens: 30.1
- max total tokens: 102
- end token text: `<|endoftext|>`
- end token ids: `[50256]`
- batch shape: `(4, 85)`

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
python tools/sft/check_sft_encoding.py \
  --path data/sft/astro_sft_tiny.jsonl \
  --encoding gpt2
```

当前检查结果：

- examples: 30
- avg prompt tokens: 46.7
- avg answer tokens: 32.8
- max total tokens: 115
- end token text: `<|endoftext|>`
- end token ids: `[50256]`
- prompt 部分大多数 labels 使用 `-100`，不参与 loss
- prompt 最后一个位置的 label 是 answer 的第一个 token
- answer 部分的 label 右移一位，用于 next-token SFT 训练
- answer 末尾追加 GPT-2 EOS `<|endoftext|>`，用于学习回答结束

Batch padding 检查：

```bash
python tools/sft/check_sft_batch.py \
  --path data/sft/astro_sft_tiny.jsonl \
  --encoding gpt2 \
  --batch-size 4
```

当前检查结果：

- `input_ids` shape: `(4, 63)`
- `labels` shape: `(4, 63)`
- `attention_mask` shape: `(4, 63)`
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
  --eval-iters 3 \
  --batch-size 4 \
  --block-size 128 \
  --out-dir out/sft_eval_log_debug
```

这里的代码逻辑和随机初始化不同：

- `--init-from` 会读取预训练 checkpoint 中的模型结构和参数
- SFT 仍然使用 `gpt2` tokenizer，因此会检查 checkpoint 词表大小是否匹配
- SFT 样本通常比预训练 block 更长，所以允许用新的 `--block-size`
- 如果 block size 改变，attention 里的 causal mask 形状会变，脚本会跳过这些不匹配的 buffer，让模型重新创建
- prompt 部分的 label 仍然是 `-100`，只在 answer 部分计算 loss

当前 `train_sft.py` 也已经支持 SFT 验证集和日志：

- `--train-ratio` 控制 SFT 样本切分比例，默认 0.9
- `--split-mode` 控制 SFT 切分方式，默认 `stratified`
- `--eval-iters` 控制每次评估抽多少个 batch 求平均
- 每隔 `--eval-interval` 输出 train loss 和 val loss
- `out/sft_eval_log_debug/log.csv` 会记录 `step,train_loss,val_loss`

注意：SFT 数据如果按任务顺序生成，不能直接用顺序切分。旧版 `astro_sft_small` 的顺序是 concept、field、format、qa、summary，如果直接 `train[:90%] / val[90%:]`，验证集会几乎全是 summary。当前默认使用 `stratified`，每个 task 都会按比例进入 train 和 val。

一次小规模验证结果：

```text
train sft examples: 27
val sft examples: 3
step 0: train loss 8.1701, val loss 9.5273
step 5: train loss 8.2557, val loss 9.3438
step 10: train loss 7.7982, val loss 9.3768
step 15: train loss 7.1760, val loss 9.1227
```

绘制 SFT loss 曲线：

```bash
python tools/plots/plot_log.py \
  --log out/sft_eval_log_debug/log.csv \
  --out out/sft_eval_log_debug/loss.png
```

`plot_log.py` 同时兼容普通预训练日志和 SFT 日志，因为两者都包含：

```text
step,train_loss,val_loss
```

普通预训练日志里额外的 `lr` 和 `tokens_per_sec` 不影响画 loss 曲线。

当前 SFT 曲线已经保存到：

```text
out/sft_eval_log_debug/loss.png
```

这张图里 train loss 下降更明显，val loss 下降较慢。由于当前验证集只有 3 条样本，这个结果只能说明链路可用，还不能作为可靠泛化结论。后续需要更多 SFT 数据和更稳定的评测集。

对比 SFT 前后采样：

```bash
python tools/eval/compare_sft_samples.py \
  --base-checkpoint out/astro_small_500/ckpt.pt \
  --sft-checkpoint out/sft_eval_log_debug/ckpt.pt \
  --out-dir out/sft_compare_samples \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 40
```

这个脚本会用同一组 instruction prompt 分别采样：

- continued pretraining checkpoint
- SFT checkpoint

并把结果写到：

```text
out/sft_compare_samples/report.md
```

当前比较的是三类任务：

- 概念解释
- 字段抽取
- 格式转换

这一版输出还不稳定，但它建立了一个很重要的观察工具：以后只要更换更好的 SFT checkpoint，就可以直接复用同一批 prompt 观察模型行为变化。

使用 `astro_sft_small` 训练 300 step：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --max-iters 300 \
  --eval-interval 25 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_small_300
```

当前训练结果：

```text
step 0: train loss 8.2933, val loss 7.6560
step 75: train loss 5.4476, val loss 6.0623
step 150: train loss 3.6624, val loss 5.0370
step 225: train loss 2.2215, val loss 4.1491
step 275: train loss 1.6608, val loss 4.0064
```

输出文件：

```text
out/sft_small_300/ckpt.pt
out/sft_small_300/log.csv
out/sft_small_300/loss.png
out/sft_small_compare/report.md
```

这次实验说明：

- small SFT 数据确实让 train loss 和 val loss 明显下降
- 验证集从 tiny 的 3 条增加到 20 条，曲线比 tiny 更有参考价值
- 采样仍然容易出现重复或空行，说明这一版 SFT 数据和采样链路还缺少明确的答案结束机制

这个结果暴露出一个关键问题：需要给 SFT answer 末尾加入明确的结束标记，并让 `sample.py` 在生成结束标记时停止。否则模型不知道一个回答应该在哪里结束，容易一直重复高概率 token。

上一版 GPT-2 EOS 机制实验：

- 当时 `sft_data.py` 在每条 answer 末尾追加 `<|endoftext|>`
- `sample.py` 支持 `--stop-at-eos`
- `compare_sft_samples.py` 支持把 `--stop-at-eos` 传给采样脚本
- 后续诊断发现 GPT-2 EOS 在答案末尾概率极低，因此当前代码已改为可见的 `<END>` 边界实验

当时重新训练 EOS 版 SFT：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --max-iters 300 \
  --eval-interval 25 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_small_eos_300
```

EOS 版训练结果：

```text
step 0: train loss 8.4678, val loss 8.1143
step 75: train loss 5.5900, val loss 6.4452
step 150: train loss 3.4745, val loss 5.0172
step 225: train loss 2.2693, val loss 4.3927
step 275: train loss 1.7174, val loss 3.8776
```

绘图和采样：

```bash
python tools/plots/plot_log.py \
  --log out/sft_small_eos_300/log.csv \
  --out out/sft_small_eos_300/loss.png

python tools/eval/compare_sft_samples.py \
  --base-checkpoint out/astro_small_500/ckpt.pt \
  --sft-checkpoint out/sft_small_eos_300/ckpt.pt \
  --out-dir out/sft_eos_compare \
  --max-new-tokens 80 \
  --temperature 0.7 \
  --top-k 40 \
  --stop-at-eos
```

这一步的结论要分开看：

- 代码机制已经完成，SFT 样本的 answer 末尾确实包含 EOS
- EOS 版 val loss 略低于无 EOS 版，说明训练没有被破坏
- 但 300 step 采样时仍然经常重复或输出空行，说明模型还没有稳定学会主动生成 EOS

这不是失败，而是暴露了下一个训练问题：当前模型和数据还太小，EOS 只是“提供了停止目标”，不等于模型已经学会稳定停下。后续可以用更长训练、改进数据分布、或增加重复惩罚/贪心评估来继续排查。

SFT 生成诊断：

```bash
python tools/eval/diagnose_sft_generation.py \
  --checkpoint out/sft_small_eos_300/ckpt.pt \
  --out-dir out/sft_generation_diagnostics \
  --max-new-tokens 80 \
  --temperatures 0.5,0.7,1.0 \
  --top-ks 20,40 \
  --num-samples 2
```

输出文件：

```text
out/sft_generation_diagnostics/diagnostics.csv
out/sft_generation_diagnostics/report.md
```

诊断指标：

- `eos_rate`：生成内容中是否出现 EOS
- `empty_rate`：是否一开始就生成 EOS，导致空回答
- `avg_completion_tokens`：EOS 截断后的平均回答长度
- `repeat_bigram_ratio`：重复 bigram 占比，越高说明越容易陷入重复
- `max_token_run`：同一个 token 连续重复的最长长度

当前诊断结果：

```text
total samples: 36
eos rate: 0.00% (0/36)
empty completion rate: 0.00% (0/36)
avg completion tokens: 80.0
avg repeated bigram ratio: 0.922
max repeated token run: 80
```

按采样参数看：

```text
temperature 0.5, top_k 20/40: repeated bigram ratio 0.987
temperature 0.7, top_k 20:    repeated bigram ratio 0.968
temperature 0.7, top_k 40:    repeated bigram ratio 0.956
temperature 1.0, top_k 20:    repeated bigram ratio 0.833
temperature 1.0, top_k 40:    repeated bigram ratio 0.802
```

结论：调高 temperature 可以稍微降低重复，但 EOS 仍然完全没有被稳定生成。低 temperature/top-k 会把模型压到最高概率 token 上，反而更容易变成纯换行或同词重复。下一步应该先做重复惩罚或 greedy 对照，再考虑是否继续延长 SFT 训练。

重复惩罚采样：

```bash
python sample.py \
  --checkpoint out/sft_small_eos_300/ckpt.pt \
  --prompt $'Instruction:\nConvert the observation into a compact JSON object.\n\nInput:\nVLBI baseline residual: station=WETTZELL, delay=12 ps, band=X.\n\nAnswer:\n' \
  --max-new-tokens 80 \
  --temperature 1.0 \
  --top-k 40 \
  --repetition-penalty 2.0 \
  --stop-at-eos
```

对 repetition penalty 做诊断：

```bash
python tools/eval/diagnose_sft_generation.py \
  --checkpoint out/sft_small_eos_300/ckpt.pt \
  --out-dir out/sft_generation_penalty_diagnostics \
  --max-new-tokens 80 \
  --temperatures 0.7,1.0 \
  --top-ks 40 \
  --repetition-penalties 1.0,1.2,1.5,2.0 \
  --num-samples 2
```

当前诊断结果：

```text
penalty 1.0: eos 0/12, avg repeated bigram 0.865, max token run 80
penalty 1.2: eos 0/12, avg repeated bigram 0.712, max token run 58
penalty 1.5: eos 0/12, avg repeated bigram 0.369, max token run 29
penalty 2.0: eos 0/12, avg repeated bigram 0.064, max token run 9
```

结论：repetition penalty 能显著降低重复，但没有让模型学会生成 EOS，也没有让回答真正变正确。它是一个解码层面的缓解工具，不是训练质量的根因修复。下一步更应该做 greedy 对照和 EOS 概率诊断，确认模型在 `Answer:` 后到底把哪些 token 排在前面。

EOS 概率 / next-token 诊断：

```bash
python tools/eval/diagnose_next_token.py \
  --checkpoint out/sft_small_eos_300/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --out-dir out/sft_next_token_diagnostics \
  --max-per-task 2 \
  --top-k 15
```

输出文件：

```text
out/sft_next_token_diagnostics/summary.csv
out/sft_next_token_diagnostics/top_tokens.csv
out/sft_next_token_diagnostics/report.md
```

这个脚本检查两个位置：

- `prompt_start`：刚看到 `Answer:\n`，模型应该开始生成答案
- `answer_end`：给出标准答案文本之后，模型应该倾向于生成 EOS

当前诊断结果：

```text
examples inspected: 10
avg prompt-start EOS rank: 865.3
avg prompt-start EOS prob: 0.000004
avg answer-end EOS rank: 848.4
avg answer-end EOS prob: 0.000030
```

进一步观察：

```text
prompt_start top1: 全部是换行 token，概率约 0.88
answer_end top1: 多数是 "."，概率最高约 0.93
answer_end EOS: 平均排名 848，最高也只到 346
```

当时的阶段性结论：EOS 不是“采样时没碰巧采到”，而是模型在标准答案末尾也几乎不认为 EOS 应该出现。这个现象先引出了结束模板排查，但后续进一步定位发现，更深层的根因是 SFT label 没有按 next-token 方式右移。

显式 `<END>` 边界实验，历史排查记录：

当前 `sft_data.py` 会在每条 answer 末尾追加：

```text
<END>
```

实际进入 GPT-2 BPE 的结束序列是：

```text
[198, 27, 10619, 29]  # "\n", "<", "END", ">"
```

重新训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --max-iters 300 \
  --eval-interval 25 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_small_end_300
```

当前 `<END>` 版训练结果：

```text
step 0: train loss 8.4780, val loss 8.1896
step 75: train loss 5.4926, val loss 5.8375
step 150: train loss 3.5132, val loss 4.4519
step 225: train loss 2.1727, val loss 3.7576
step 275: train loss 1.6163, val loss 3.3365
```

采样和诊断：

```bash
python sample.py \
  --checkpoint out/sft_small_end_300/ckpt.pt \
  --prompt $'Instruction:\nExtract the station, signal, value, and unit from the text.\n\nInput:\nStation BJFS shows a vertical velocity of 2.4 mm/yr from space geodetic observations.\n\nAnswer:\n' \
  --max-new-tokens 80 \
  --temperature 1.0 \
  --top-k 40 \
  --repetition-penalty 1.5 \
  --stop-at-text "<END>"

python tools/eval/diagnose_next_token.py \
  --checkpoint out/sft_small_end_300/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --out-dir out/sft_next_token_end_diagnostics \
  --max-per-task 2 \
  --top-k 15

python tools/eval/diagnose_sft_generation.py \
  --checkpoint out/sft_small_end_300/ckpt.pt \
  --out-dir out/sft_generation_end_diagnostics \
  --max-new-tokens 80 \
  --temperatures 0.7,1.0 \
  --top-ks 40 \
  --repetition-penalties 1.0,1.5 \
  --stop-text "<END>" \
  --num-samples 2
```

`<END>` 版生成诊断结果：

```text
total samples: 24
stop-text rate: 0.00% (0/24)
avg repeated bigram ratio: 0.645
max repeated token run: 80
```

next-token 诊断结果：

```text
prompt_start avg END-first rank: 1.0
prompt_start avg END-sequence max rank: 338.3
answer_end avg END-first rank: 5.1
answer_end avg END-sequence max rank: 341.2
```

当时的解释：`<END>` 的第一个 token 是换行，而模型本来就非常偏好换行，所以 `END-first rank` 看起来很好。但完整序列的第二步 `<` 通常排在 300 多名，模型仍然没有真正学会输出完整 `<END>`。

进一步修复：SFT label 右移对齐

普通 causal LM 训练不是“当前位置预测当前位置”，而是“当前位置预测下一个 token”。`train.py` 的普通预训练已经通过 `x` 和 `y=x+1` 做了这个对齐；但旧版 SFT 数据把 answer token 直接放在同一个位置的 label 上，导致模型训练时可以看着 answer token 预测自己。

旧版错误对齐：

```text
input:  prompt_token answer_1 answer_2 ... END
label:  -100         answer_1 answer_2 ... END
```

正确 SFT 对齐：

```text
input:  prompt_token answer_1 answer_2 ... END
label:  answer_1     answer_2 ... END   -100
```

当前 `sft_data.py` 的做法是：

```python
labels = [IGNORE_INDEX] * len(inputIds)
answerStart = len(promptIds) - 1
labels[answerStart:answerStart + len(answerIds)] = answerIds
```

先把结束标记换成 GPT-2 BPE 里的普通单 token 做对照：

```text
 END -> [23578]
```

重新训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --max-iters 300 \
  --eval-interval 25 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_small_shifted_end_300
```

正确对齐后的训练结果：

```text
step 0: train loss 8.1694, val loss 7.8326
step 75: train loss 6.1057, val loss 6.9887
step 150: train loss 4.4361, val loss 6.4496
step 225: train loss 3.4031, val loss 6.3138
step 275: train loss 2.8531, val loss 6.1267
```

这里的 val loss 比旧实验更高，但不能直接和旧实验比较。旧实验有 label 泄漏，loss 偏乐观；修复后才是真正的 next-token SFT loss。

next-token 诊断结果：

```text
answer_end avg END-first rank: 1.0
answer_end avg END-first prob: 0.311218
answer_end avg END-sequence max rank: 1.0
```

生成诊断结果：

```text
total samples: 24
stop-text rate: 95.83% (23/24)
avg repeated bigram ratio: 0.067
max repeated token run: 2
```

结论：结束问题在当前阶段可以解决。真正的关键是保证 SFT labels 按 causal LM 的 next-token 目标正确右移；` END` 只是一个帮助排查的普通单 token stop marker。当前模型的回答质量仍然一般，但“能不能停”这个问题已经基本修复。

回到 GPT-2 EOS：

修复 label 右移以后，可以正常用回标准 EOS：

```text
<|endoftext|> -> [50256]
```

当前 `sft_data.py` 已经重新切回 EOS：

```python
END_TOKEN = EOS_TOKEN
answerIds = enc.encode(
    answer + END_TOKEN,
    allowed_special={EOS_TOKEN},
)
```

重新训练 EOS shifted 版：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --max-iters 300 \
  --eval-interval 25 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_small_shifted_eos_300
```

EOS shifted 训练结果：

```text
step 0: train loss 8.1626, val loss 7.8043
step 75: train loss 6.0997, val loss 6.9557
step 150: train loss 4.4207, val loss 6.4315
step 225: train loss 3.3870, val loss 6.2960
step 275: train loss 2.8398, val loss 6.1140
```

EOS next-token 诊断：

```text
answer_end avg EOS rank: 1.0
answer_end avg EOS prob: 0.290772
```

EOS 生成诊断：

```text
total samples: 24
eos rate: 83.33% (20/24)
avg repeated bigram ratio: 0.071
max repeated token run: 2
```

对比结论：

```text
旧 EOS + 错误 label 对齐：answer_end EOS rank 约 848，生成不停止
END + 正确 label 对齐：stop-text rate 95.83%
EOS + 正确 label 对齐：eos rate 83.33%，answer_end EOS rank 1
```

所以当前项目可以正常用回 EOS。` END` 版可以保留为历史对照；主线推荐用 EOS，因为它是 tokenizer 原生结束符，`sample.py` 可以直接通过 `--stop-at-eos` 截断生成。

分层切分重新训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --split-mode stratified \
  --max-iters 300 \
  --eval-interval 25 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_small_stratified_eos_300
```

分层切分会得到更均衡的 train/val：

```text
train tasks: {'concept_explanation': 36, 'field_extraction': 36, 'format_conversion': 36, 'qa': 36, 'summary': 36}
val tasks: {'concept_explanation': 4, 'field_extraction': 4, 'format_conversion': 4, 'qa': 4, 'summary': 4}
```

分层训练结果：

```text
step 0: train loss 8.1069, val loss 8.4972
step 75: train loss 6.0291, val loss 6.2559
step 150: train loss 4.4121, val loss 4.9197
step 225: train loss 3.2598, val loss 3.4966
step 275: train loss 2.8882, val loss 3.0201
```

分层 EOS 诊断：

```text
answer_end avg EOS rank: 1.0
answer_end avg EOS prob: 0.349163
generation eos rate: 95.83% (23/24)
avg repeated bigram ratio: 0.066
```

SFT 质量评测：

新增脚本：

```bash
python tools/eval/evaluate_sft_quality.py \
  --checkpoint out/sft_small_stratified_eos_300/ckpt.pt \
  --sft-path data/sft/astro_sft_small.jsonl \
  --split val \
  --split-mode stratified \
  --out-dir out/sft_quality_stratified_eos_val \
  --temperature 0.0 \
  --max-new-tokens 80
```

当前质量指标：

```text
examples: 20
eos rate: 30.00%
exact match: 0.00%
avg token F1: 0.112
avg target recall: 0.108
avg char similarity: 0.158
avg repeated bigram ratio: 0.629
```

这个结果说明：模型已经基本学会“在哪里停”，但还没有学会“答对”。下一步的主要矛盾不是 EOS，而是 SFT 数据质量、任务混合方式和模型容量。

单任务 field extraction 对照：

为了判断“回答质量差”到底是多任务混合导致，还是模型整体能力不足，可以先只训练一个最结构化的任务：`field_extraction`。

新增筛选脚本：

```bash
python tools/sft/filter_sft_by_task.py \
  --input data/sft/astro_sft_small.jsonl \
  --out data/sft/astro_sft_field.jsonl \
  --task field_extraction
```

生成结果：

```text
loaded examples: 200
saved examples: 40
tasks: {'field_extraction': 40}
```

检查结果：

```text
examples: 40
avg prompt tokens: 50.7
avg answer tokens: 23.7
max total tokens: 81
end token ids: [50256]
```

训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_field.jsonl \
  --split-mode stratified \
  --max-iters 300 \
  --eval-interval 25 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_field_300
```

训练结果：

```text
train tasks: {'field_extraction': 36}
val tasks: {'field_extraction': 4}
step 0: train loss 6.9630, val loss 6.9324
step 75: train loss 4.0012, val loss 4.1432
step 150: train loss 2.2541, val loss 2.4979
step 225: train loss 1.3465, val loss 1.5070
step 275: train loss 0.9024, val loss 1.0884
```

质量评测：

```bash
python tools/eval/evaluate_sft_quality.py \
  --checkpoint out/sft_field_300/ckpt.pt \
  --sft-path data/sft/astro_sft_field.jsonl \
  --split val \
  --split-mode stratified \
  --out-dir out/sft_quality_field_val \
  --temperature 0.0 \
  --max-new-tokens 80
```

结果：

```text
examples: 4
eos rate: 100.00%
exact match: 0.00%
avg token F1: 0.453
avg target recall: 0.325
avg char similarity: 0.486
avg repeated bigram ratio: 0.000
```

对比：

```text
多任务 SFT 的 field_extraction token F1: 0.303
单任务 field_extraction token F1: 0.453
```

结论：单任务训练明显改善了结构化抽取能力，说明多任务混合确实会干扰这个小模型。但预测仍然经常把 station/value 记错，例如目标是 `KOKEE`，模型输出 `BJFS` 或 `WETTZELL`。这说明下一步不应急着上 DPO，而应先扩大和重构 SFT 数据，尤其是字段抽取这种可验证任务。

扩充 field extraction 到 500 条：

新增脚本：

```bash
python scripts/build_field_sft.py \
  --out data/sft/astro_sft_field_500.jsonl \
  --num-examples 500
```

这个脚本系统组合：

```text
station: BJFS / WETTZELL / KOKEE / NYALES20 / HOBART12 / ONSA / ...
signal: vertical velocity / east displacement / clock bias / zenith wet delay / ...
value: 正数、负数、小数
unit: mm/yr / mm / ns / ps
```

数据检查：

```text
examples: 500
avg prompt tokens: 47.5
avg answer tokens: 23.3
max total tokens: 81
end token ids: [50256]
```

训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_field_500.jsonl \
  --split-mode stratified \
  --max-iters 500 \
  --eval-interval 50 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_field_500
```

训练结果：

```text
train examples: 450
val examples: 50
step 0: train loss 7.1243, val loss 6.9742
step 150: train loss 2.5977, val loss 2.5498
step 300: train loss 0.9741, val loss 0.9779
step 450: train loss 0.5595, val loss 0.5853
```

质量评测：

```bash
python tools/eval/evaluate_sft_quality.py \
  --checkpoint out/sft_field_500/ckpt.pt \
  --sft-path data/sft/astro_sft_field_500.jsonl \
  --split val \
  --split-mode stratified \
  --out-dir out/sft_quality_field_500_val \
  --temperature 0.0 \
  --max-new-tokens 80
```

结果：

```text
examples: 50
eos rate: 100.00%
exact match: 0.00%
avg token F1: 0.729
avg target recall: 0.732
avg char similarity: 0.869
avg repeated bigram ratio: 0.008
```

对比：

```text
field_40 token F1: 0.453
field_500 token F1: 0.729
```

结论：扩大高质量、可验证的单任务数据显著提升了小模型的结构化抽取能力。它已经学会了输出字段格式、命中 EOS、减少重复，但 exact match 仍为 0，最低分样例里还会把 `clock bias` 预测成 `north displacement`，或把数值预测成训练集中常见的其他值。下一步应该增加字段级评测，例如分别计算 station、signal、value、unit 的准确率，而不是只看 token F1。

字段级准确率评测：

新增脚本：

```bash
python tools/eval/evaluate_field_accuracy.py \
  --results out/sft_quality_field_500_val/results.csv \
  --out-dir out/field_accuracy_field_500
```

这个脚本会解析 target 和 prediction 中的四个字段：

```text
station
signal
value
unit
```

field_500 结果：

```text
total: 50
station accuracy: 20.00%
signal accuracy: 76.00%
value accuracy: 2.00%
unit accuracy: 68.00%
all fields accuracy: 0.00%
avg correct fields: 1.66/4
```

解释：

- `signal` 和 `unit` 相对较高，说明模型已经学会了一部分字段类型和单位模式
- `station` 很低，说明模型没有稳定复制输入里的站名，而是在训练集中常见站名之间猜
- `value` 几乎为 0，说明模型没有学会精确复制数值，经常生成另一个训练集中出现过的数值
- `all fields accuracy` 为 0，说明 token F1 的提升主要来自格式和部分字段相似，不等于真正完成字段抽取

这个结果把下一步方向变得很清楚：继续扩数据不是唯一重点，还要让任务更强调“从输入复制字段”。后续可以做两类改进：一是加入更多数字和站名组合，二是把输出格式改成更严格的 JSON，再做字段级 exact-match 评测。

复制型 field extraction 对照：

为了排查上一版自然语言输入是否太难，新增 copy 型数据。输入直接写成结构化字段：

```text
signal=zenith wet delay; station=YEBES40M; unit=mm; value=38.5
```

输出仍然保持原格式：

```text
station: YEBES40M
signal: zenith wet delay
value: 38.5
unit: mm
```

新增脚本：

```bash
python scripts/build_field_copy_sft.py \
  --out data/sft/astro_sft_field_copy_500.jsonl \
  --num-examples 500
```

数据检查：

```text
examples: 500
avg prompt tokens: 50.6
avg answer tokens: 23.4
max total tokens: 82
end token ids: [50256]
```

训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_field_copy_500.jsonl \
  --split-mode stratified \
  --max-iters 500 \
  --eval-interval 50 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_field_copy_500
```

训练结果：

```text
train examples: 450
val examples: 50
step 0: train loss 7.1168, val loss 7.2219
step 150: train loss 2.6001, val loss 2.6686
step 300: train loss 0.9770, val loss 1.0082
step 450: train loss 0.5620, val loss 0.5942
```

质量评测：

```text
examples: 50
eos rate: 100.00%
exact match: 0.00%
avg token F1: 0.738
avg target recall: 0.741
avg char similarity: 0.860
avg repeated bigram ratio: 0.001
```

字段级评测：

```bash
python tools/eval/evaluate_field_accuracy.py \
  --results out/sft_quality_field_copy_500_val/results.csv \
  --out-dir out/field_accuracy_field_copy_500
```

结果：

```text
station accuracy: 32.00%
signal accuracy: 62.00%
value accuracy: 0.00%
unit accuracy: 90.00%
all fields accuracy: 0.00%
avg correct fields: 1.84/4
```

和自然语言 field_500 对比：

```text
自然语言 field_500:
station 20%, signal 76%, value 2%, unit 68%, all fields 0%

复制型 field_copy_500:
station 32%, signal 62%, value 0%, unit 90%, all fields 0%
```

结论：把输入改成结构化 copy 格式，并没有真正解决精确复制问题。它让 `station` 小幅提升、`unit` 明显提升，但 `value` 仍然几乎完全失败。说明当前瓶颈不只是“自然语言输入太难”，而是这个小模型在当前训练规模下对数字和实体的逐字复制能力不足。下一步更应该做 copy 机制压力测试：从最简单的 `value=2.4 -> value: 2.4` 开始，只训练一个字段，确认模型到底能不能复制数字。

单字段 value copy 压力测试：

为了排除“四字段输出太复杂”的干扰，进一步把任务压缩到最小形式：

```text
Input:
value=38.5

Answer:
value: 38.5
```

新增脚本：

```bash
python scripts/build_value_copy_sft.py \
  --out data/sft/value_copy_500.jsonl \
  --num-examples 500
```

数据检查：

```text
examples: 500
avg prompt tokens: 28.2
avg answer tokens: 6.2
max total tokens: 37
end token ids: [50256]
```

训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/value_copy_500.jsonl \
  --split-mode stratified \
  --max-iters 500 \
  --eval-interval 50 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 64 \
  --learning-rate 3e-4 \
  --out-dir out/sft_value_copy_500
```

训练结果：

```text
train examples: 450
val examples: 50
step 0: train loss 8.2263, val loss 8.1218
step 150: train loss 2.4863, val loss 2.4400
step 300: train loss 1.0728, val loss 1.0816
step 450: train loss 0.8958, val loss 0.8245
```

质量评测：

```text
examples: 50
eos rate: 100.00%
exact match: 6.00%
avg token F1: 0.530
avg target recall: 0.530
avg char similarity: 0.833
avg repeated bigram ratio: 0.000
```

字段级评测：

```bash
python tools/eval/evaluate_field_accuracy.py \
  --results out/sft_quality_value_copy_500_val/results.csv \
  --out-dir out/field_accuracy_value_copy_500
```

结果：

```text
value accuracy: 6.00%
```

典型错误：

```text
target:     value: 5.6
prediction: value: 12.5

target:     value: -4.4
prediction: value: -0.5

target:     value: 52.2
prediction: value: 12.5
```

结论：即使任务简化到只复制一个 value，模型也只有 6% exact match。这说明当前 6.57M 参数的小模型、当前预训练和 SFT 设置下，数字复制是明确瓶颈。后续如果要继续研究这个方向，应该优先尝试三件事：增加模型容量、增加训练步数/数据重复、或把数字拆成更稳定的字符级/格式化 token 任务，而不是直接进入 DPO。

value copy 过拟合诊断：

为了区分“模型完全不会复制数字”和“模型会记忆但不会泛化复制”，先评测 `value_copy_500` 的 train split：

```bash
python tools/eval/evaluate_sft_quality.py \
  --checkpoint out/sft_value_copy_500/ckpt.pt \
  --sft-path data/sft/value_copy_500.jsonl \
  --split train \
  --split-mode stratified \
  --out-dir out/sft_quality_value_copy_500_train \
  --temperature 0.0 \
  --max-new-tokens 20

python tools/eval/evaluate_field_accuracy.py \
  --results out/sft_quality_value_copy_500_train/results.csv \
  --out-dir out/field_accuracy_value_copy_500_train
```

结果：

```text
train examples: 450
train exact match: 7.11%
train value accuracy: 7.11%
```

这说明 500 条训练集本身也没有被稳定记住。

接着构造 20 条小样本：

```bash
python scripts/build_value_copy_sft.py \
  --out data/sft/value_copy_20.jsonl \
  --num-examples 20
```

过拟合训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/value_copy_20.jsonl \
  --split-mode stratified \
  --max-iters 2000 \
  --eval-interval 200 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 64 \
  --learning-rate 3e-4 \
  --out-dir out/sft_value_copy_20_overfit
```

训练结果：

```text
train examples: 18
val examples: 2
step 0: train loss 8.3131, val loss 8.9043
step 600: train loss 0.1790, val loss 3.2299
step 1200: train loss 0.0198, val loss 2.2073
step 1800: train loss 0.0072, val loss 3.8369
```

过拟合评测：

```text
train exact match: 100.00%
train value accuracy: 100.00%
val exact match: 50.00%
val value accuracy: 50.00%
```

结论：模型可以记住 20 条 value copy 样本，因此不是“完全不能输出数字”。但在 500 条组合上，训练集 accuracy 也只有约 7%，说明当前设置没有学到稳定的复制规则，而是在少量样本上可以记忆。下一步可以做两个方向的对照：一是增加模型容量看复制规则是否出现，二是把数字复制改成字符级任务，减少 tokenizer 对小数和负号的干扰。

数值 tokenizer 诊断：

为了确认数字复制为什么难，新增脚本观察 GPT-2 tokenizer 如何切分小数和负数：

```bash
python tools/data/inspect_value_tokens.py \
  --encoding gpt2
```

典型结果：

```text
text: -10.5
ids: [12, 940, 13, 20]
pieces: ['-', '10', '.', '5']
num tokens: 4

text: 12.5
ids: [1065, 13, 20]
pieces: ['12', '.', '5']
num tokens: 3

text: 52.2
ids: [4309, 13, 17]
pieces: ['52', '.', '2']
num tokens: 3
```

带输入/输出前缀后，token 边界还会变化：

```text
text: value: -10.5
ids: [8367, 25, 532, 940, 13, 20]
pieces: ['value', ':', ' -', '10', '.', '5']
num tokens: 6

text: value=-10.5
ids: [8367, 10779, 940, 13, 20]
pieces: ['value', '=-', '10', '.', '5']
num tokens: 5
```

解释：

- 负号、小数点、整数部分和小数部分经常被拆成多个 token
- `value=-10.5` 和 `value: -10.5` 的 token 边界不同，输入里的 `=-` 到输出里的 ` -` 不是同一个 token
- 模型要学的不是简单复制字符串，而是把一种 token 切分转换成另一种 token 切分

这解释了为什么 value copy 很难。下一步更合理的实验是 digit-spaced copy，例如：

```text
Input:  value = - 1 0 . 5
Answer: value: - 1 0 . 5
```

这样可以把任务改成更接近字符级复制，减少 GPT-2 BPE 对数字边界的干扰。

digit-spaced value copy：

为了验证 tokenizer 数字切分是否是核心问题之一，构造空格拆分数字的数据：

```text
Input:
reported value is 8 . 0

Answer:
value: 8 . 0
```

新增脚本：

```bash
python scripts/build_digit_spaced_value_copy_sft.py \
  --out data/sft/value_copy_spaced_500.jsonl \
  --num-examples 500
```

数据检查：

```text
examples: 500
avg prompt tokens: 29.0
avg answer tokens: 6.7
max total tokens: 40
end token ids: [50256]
```

训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/value_copy_spaced_500.jsonl \
  --split-mode stratified \
  --max-iters 500 \
  --eval-interval 50 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 64 \
  --learning-rate 3e-4 \
  --out-dir out/sft_value_copy_spaced_500
```

训练结果：

```text
train examples: 450
val examples: 50
step 0: train loss 9.2266, val loss 9.2176
step 150: train loss 2.4809, val loss 2.4978
step 300: train loss 1.0153, val loss 1.0698
step 450: train loss 0.5667, val loss 0.5860
```

质量评测：

```text
examples: 50
eos rate: 100.00%
exact match: 50.00%
avg token F1: 0.864
avg target recall: 0.868
avg char similarity: 0.944
avg repeated bigram ratio: 0.000
```

字段级评测：

```bash
python tools/eval/evaluate_field_accuracy.py \
  --results out/sft_quality_value_copy_spaced_500_val/results.csv \
  --out-dir out/field_accuracy_value_copy_spaced_500
```

结果：

```text
value accuracy: 50.00%
```

对比：

```text
普通 value_copy_500:
value accuracy: 6.00%

digit-spaced value_copy_500:
value accuracy: 50.00%
```

结论：把数字拆成更接近字符级的形式后，value exact match 从 6% 提升到 50%。这说明 GPT-2 BPE 对数字的切分确实是复制困难的重要来源。它还没有到 100%，说明小模型的复制规则仍然不稳，但方向已经成立。下一步可以做 curriculum：先 digit-spaced，再逐步回到普通数字格式。

digit-spaced -> normal curriculum：

为了验证 digit-spaced 学到的复制能力能否迁移回普通数字格式，做两阶段 curriculum：

```text
Stage 1: digit-spaced value copy
Stage 2: normal value copy
```

训练：

```bash
python train_sft.py \
  --init-from out/sft_value_copy_spaced_500/ckpt.pt \
  --sft-path data/sft/value_copy_500.jsonl \
  --split-mode stratified \
  --max-iters 500 \
  --eval-interval 50 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 64 \
  --learning-rate 3e-4 \
  --out-dir out/sft_value_copy_curriculum_500
```

评测：

```bash
python tools/eval/evaluate_sft_quality.py \
  --checkpoint out/sft_value_copy_curriculum_500/ckpt.pt \
  --sft-path data/sft/value_copy_500.jsonl \
  --split val \
  --split-mode stratified \
  --out-dir out/sft_quality_value_copy_curriculum_500_val \
  --temperature 0.0 \
  --max-new-tokens 20

python tools/eval/evaluate_field_accuracy.py \
  --results out/sft_quality_value_copy_curriculum_500_val/results.csv \
  --out-dir out/field_accuracy_value_copy_curriculum_500
```

结果：

```text
eos rate: 100.00%
exact match: 80.00%
avg token F1: 0.900
avg target recall: 0.900
value accuracy: 80.00%
```

对比：

```text
normal value_copy_500:
value accuracy: 6.00%

digit-spaced value_copy_500:
value accuracy: 50.00%

digit-spaced -> normal curriculum:
value accuracy: 80.00%
```

结论：curriculum 明显有效。先让模型在 digit-spaced 任务中学习更接近字符级的复制，再切回普通数字格式，可以把普通 value copy 的准确率从 6% 提升到 80%。这说明小模型不是完全不能学数字复制，而是需要更合适的数据路径和中间任务。

四字段 digit-spaced field copy：

为了把 value-only curriculum 迁移到完整字段抽取，构造了一个四字段 copy 任务，只把 value 改成 digit-spaced，其他字段保持正常：

```text
station / signal / value / unit
value: 38.5 -> value: 3 8 . 5
```

数据构造：

```bash
python scripts/build_digit_spaced_field_copy_sft.py \
  --out data/sft/astro_sft_field_copy_spaced_500.jsonl \
  --num-examples 500
```

训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/astro_sft_field_copy_spaced_500.jsonl \
  --split-mode stratified \
  --max-iters 500 \
  --eval-interval 50 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_field_copy_spaced_500
```

训练曲线：

```text
step 0: train loss 7.4365, val loss 7.5259
step 150: train loss 2.7229, val loss 2.7283
step 300: train loss 0.9989, val loss 1.0235
step 450: train loss 0.5787, val loss 0.6121
```

评测：

```bash
python tools/eval/evaluate_sft_quality.py \
  --checkpoint out/sft_field_copy_spaced_500/ckpt.pt \
  --sft-path data/sft/astro_sft_field_copy_spaced_500.jsonl \
  --split val \
  --split-mode stratified \
  --out-dir out/sft_quality_field_copy_spaced_500_val \
  --temperature 0.0 \
  --max-new-tokens 80

python tools/eval/evaluate_field_accuracy.py \
  --results out/sft_quality_field_copy_spaced_500_val/results.csv \
  --out-dir out/field_accuracy_field_copy_spaced_500
```

结果：

```text
eos rate: 100.00%
exact match: 0.00%
avg token F1: 0.634
avg target recall: 0.643

station accuracy: 10.00%
signal accuracy: 20.00%
value accuracy: 0.00%
unit accuracy: 66.00%
all fields accuracy: 0.00%
avg correct fields: 0.96/4
```

观察：虽然 loss 降得很顺，但生成时出现了模式坍缩，模型经常输出训练中常见的字段组合，例如 `NYALES20 / zenith wet delay / 2 . 5 / mm`。对 train split 抽样评测也是 0% exact match，说明这不是验证集泛化问题，而是四字段 copy 对当前小模型和训练设置仍然偏难。

结论：value-only curriculum 成功，但直接扩展到 500 条、10 个 station、7 类 signal 的四字段任务跨度太大。下一步应该加一层更小的课程，例如 20 条过拟合或 tiny field copy，先验证模型能否记住并复制完整四字段格式。

tiny field copy overfit：

为了判断模型是否具备四字段复制能力，先做一个极小数据集，只包含 3 个 station、2 类 signal、20 条 digit-spaced field copy 样本。这一步不测泛化，只测模型能否在训练集上过拟合。

数据构造：

```bash
python scripts/build_tiny_field_copy_sft.py \
  --out data/sft/tiny_field_copy_spaced_20.jsonl \
  --num-examples 20 \
  --digit-spaced
```

训练：

```bash
python train_sft.py \
  --init-from out/astro_small_500/ckpt.pt \
  --sft-path data/sft/tiny_field_copy_spaced_20.jsonl \
  --split-mode shuffle \
  --train-ratio 0.95 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_tiny_field_copy_spaced_20_overfit
```

训练曲线：

```text
step 0: train loss 6.9762, val loss 6.9690
step 300: train loss 0.4275, val loss 0.4979
step 600: train loss 0.0840, val loss 0.1923
step 900: train loss 0.0184, val loss 0.1812
```

训练集评测：

```bash
python tools/eval/evaluate_sft_quality.py \
  --checkpoint out/sft_tiny_field_copy_spaced_20_overfit/ckpt.pt \
  --sft-path data/sft/tiny_field_copy_spaced_20.jsonl \
  --split train \
  --split-mode shuffle \
  --train-ratio 0.95 \
  --out-dir out/sft_quality_tiny_field_copy_spaced_20_train \
  --temperature 0.0 \
  --max-new-tokens 80

python tools/eval/evaluate_field_accuracy.py \
  --results out/sft_quality_tiny_field_copy_spaced_20_train/results.csv \
  --out-dir out/field_accuracy_tiny_field_copy_spaced_20_train
```

结果：

```text
examples: 19
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

结论：模型具备四字段复制能力，20 条 tiny digit-spaced field copy 可以完全过拟合。500 条失败不是因为模型完全不会格式，而是因为从 20 条可记忆任务到 500 条多类别组合任务跨度太大。下一步应该做 100 条 small field copy，观察能力从记忆走向泛化时在哪个规模开始下降。

small field copy 100：

为了继续扩大 curriculum 台阶，从 20 条 tiny field copy 扩展到 100 条 small field copy。这个数据集包含 5 个 station、3 类 signal、3 种输入模板，value 仍然使用 digit-spaced 格式。

数据构造：

```bash
python scripts/build_small_field_copy_sft.py \
  --out data/sft/small_field_copy_spaced_100.jsonl \
  --num-examples 100 \
  --digit-spaced
```

训练从 tiny overfit checkpoint 继续：

```bash
python train_sft.py \
  --init-from out/sft_tiny_field_copy_spaced_20_overfit/ckpt.pt \
  --sft-path data/sft/small_field_copy_spaced_100.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_small_field_copy_spaced_100
```

训练曲线：

```text
step 0: train loss 2.3850, val loss 2.4909
step 300: train loss 0.1201, val loss 0.1306
step 600: train loss 0.0270, val loss 0.0437
step 900: train loss 0.0116, val loss 0.0140
```

训练集结果：

```text
examples: 90
exact match: 97.78%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 97.78%
unit accuracy: 100.00%
all fields accuracy: 97.78%
```

验证集结果：

```text
examples: 10
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

观察：small 100 已经基本学会四字段 digit-spaced copy。训练集中少量错误集中在 value 上，station、signal、unit 都达到 100%。这说明课程台阶有效，模型能力断点不在 20 到 100 之间，而更可能出现在 100 到 500 之间，或者出现在类别数、模板数、数值组合同时扩张时。

medium field copy 250：

继续把样本数扩大到 250 条。这里复用完整四字段 digit-spaced 数据构造脚本，类别范围回到 10 个 station、7 类 signal、完整 value 集合，但样本数先控制在 250。

数据构造：

```bash
python scripts/build_digit_spaced_field_copy_sft.py \
  --out data/sft/medium_field_copy_spaced_250.jsonl \
  --num-examples 250
```

训练从 small 100 checkpoint 继续：

```bash
python train_sft.py \
  --init-from out/sft_small_field_copy_spaced_100/ckpt.pt \
  --sft-path data/sft/medium_field_copy_spaced_250.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_medium_field_copy_spaced_250
```

训练曲线：

```text
step 0: train loss 3.7820, val loss 3.1516
step 300: train loss 0.3794, val loss 0.4244
step 600: train loss 0.1432, val loss 0.2371
step 900: train loss 0.0673, val loss 0.1091
```

训练集结果：

```text
examples: 225
exact match: 77.33%
station accuracy: 88.44%
signal accuracy: 95.11%
value accuracy: 90.22%
unit accuracy: 96.89%
all fields accuracy: 77.33%
```

验证集结果：

```text
examples: 25
exact match: 60.00%
station accuracy: 88.00%
signal accuracy: 92.00%
value accuracy: 72.00%
unit accuracy: 100.00%
all fields accuracy: 60.00%
```

观察：medium 250 已经明显比 full 500 的直接训练好，但相比 small 100 出现下降。错误主要集中在 value，其次是 station 和 signal 混淆。unit 最稳定，因为它的取值空间最小，并且和 signal 有强绑定关系。

结论：当前 curriculum 的断点基本定位在 100 到 250 之间。模型可以学会四字段 digit-spaced copy，但当类别组合和数值组合扩大后，value 复制最先退化。下一步可以选择两条路线：一是继续训练 medium 250 更久，看是否只是训练步数不足；二是把 250 拆成更细的课程，例如先扩大 station，再扩大 signal，最后扩大 value。

factor field copy 250：

为了判断到底是谁把模型压垮，构造三组 factor 实验。三组都使用 250 条 digit-spaced field copy，并且都从 `out/sft_small_field_copy_spaced_100/ckpt.pt` 继续训练。

三组变量：

```text
station factor: 10 个 station，保持 3 类 signal
signal factor: 5 个 station，扩大到 7 类 signal
value factor: 5 个 station，3 类 signal，但扩大 value 取值
```

数据构造：

```bash
python scripts/build_factor_field_copy_sft.py \
  --mode station \
  --out data/sft/factor_station_field_copy_spaced_250.jsonl \
  --num-examples 250 \
  --digit-spaced

python scripts/build_factor_field_copy_sft.py \
  --mode signal \
  --out data/sft/factor_signal_field_copy_spaced_250.jsonl \
  --num-examples 250 \
  --digit-spaced

python scripts/build_factor_field_copy_sft.py \
  --mode value \
  --out data/sft/factor_value_field_copy_spaced_250.jsonl \
  --num-examples 250 \
  --digit-spaced
```

训练命令结构一致：

```bash
python train_sft.py \
  --init-from out/sft_small_field_copy_spaced_100/ckpt.pt \
  --sft-path data/sft/factor_${mode}_field_copy_spaced_250.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_factor_${mode}_field_copy_spaced_250
```

验证集结果：

```text
station factor:
exact match: 92.00%
station accuracy: 96.00%
signal accuracy: 96.00%
value accuracy: 96.00%
unit accuracy: 96.00%
all fields accuracy: 92.00%

signal factor:
exact match: 88.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 88.00%
unit accuracy: 100.00%
all fields accuracy: 88.00%

value factor:
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

对比 medium 250：

```text
medium 250 full combination:
all fields accuracy: 60.00%
value accuracy: 72.00%

station-only expansion:
all fields accuracy: 92.00%

signal-only expansion:
all fields accuracy: 88.00%

value-only expansion:
all fields accuracy: 100.00%
```

结论：压垮模型的不是单独扩大 value。相反，只在固定 station/signal 空间里增加 value，模型仍然可以 100% 复制。单独扩大 station 或 signal 也能保持较高准确率。真正的问题是组合空间同时扩大：更多 station、更多 signal、更多 value 同时出现时，小模型容易把字段组合混在一起，尤其表现为 value 复制错误和 station/signal 混淆。

这说明后续 curriculum 不能只按样本数递增，而应该按组合复杂度递增。例如：

```text
small 100
-> station factor
-> signal factor
-> value factor
-> station + signal
-> signal + value
-> full medium
-> full 500
```

double factor field copy 250：

继续做双因素实验，判断两个变量同时扩大时模型是否开始明显退化。

新增两组：

```text
station_signal: 10 个 station + 7 类 signal
signal_value: 5 个 station + 7 类 signal + 更多 value
```

数据构造：

```bash
python scripts/build_factor_field_copy_sft.py \
  --mode station_signal \
  --out data/sft/factor_station_signal_field_copy_spaced_250.jsonl \
  --num-examples 250 \
  --digit-spaced

python scripts/build_factor_field_copy_sft.py \
  --mode signal_value \
  --out data/sft/factor_signal_value_field_copy_spaced_250.jsonl \
  --num-examples 250 \
  --digit-spaced
```

训练仍然从 small 100 checkpoint 继续：

```bash
python train_sft.py \
  --init-from out/sft_small_field_copy_spaced_100/ckpt.pt \
  --sft-path data/sft/factor_${mode}_field_copy_spaced_250.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_factor_${mode}_field_copy_spaced_250
```

验证集结果：

```text
station_signal:
exact match: 72.00%
station accuracy: 88.00%
signal accuracy: 96.00%
value accuracy: 76.00%
unit accuracy: 100.00%
all fields accuracy: 72.00%

signal_value:
exact match: 68.00%
station accuracy: 100.00%
signal accuracy: 88.00%
value accuracy: 72.00%
unit accuracy: 88.00%
all fields accuracy: 68.00%
```

和单因素对比：

```text
station-only: 92.00%
signal-only: 88.00%
value-only: 100.00%

station + signal: 72.00%
signal + value: 68.00%

full medium 250: 60.00%
```

结论：双因素组合已经明显比单因素更难。`signal + value` 组合尤其容易让 value 出错，因为 signal、unit、value 分布之间存在绑定关系，模型需要同时记住“这个 signal 通常对应什么 unit 和什么 value 空间”，再从 input 里复制具体 value。小模型在这个组合泛化上开始不稳。

目前更准确的判断是：模型不是被 value 字符串本身压垮，而是被跨字段组合关系压垮。value 是最终出错最多的字段，但根因是 station/signal/value 同时组合时的干扰。

full 500 after factor curriculum：

在定位到组合泛化问题之后，尝试把 factor 实验作为 curriculum 中间台阶，再回到 full 500 digit-spaced field copy。这里选择 `signal_value` checkpoint 作为起点，因为它最接近 value 出错机制。

训练：

```bash
python train_sft.py \
  --init-from out/sft_factor_signal_value_field_copy_spaced_250/ckpt.pt \
  --sft-path data/sft/astro_sft_field_copy_spaced_500.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_full_field_copy_spaced_500_from_signal_value
```

训练曲线：

```text
step 0: train loss 1.6094, val loss 1.6983
step 300: train loss 0.1219, val loss 0.1669
step 600: train loss 0.0398, val loss 0.0412
step 900: train loss 0.0190, val loss 0.0264
```

验证集结果：

```text
examples: 50
exact match: 90.00%
station accuracy: 94.00%
signal accuracy: 100.00%
value accuracy: 96.00%
unit accuracy: 100.00%
all fields accuracy: 90.00%
```

训练集结果：

```text
examples: 450
station accuracy: 91.56%
signal accuracy: 100.00%
value accuracy: 99.11%
unit accuracy: 100.00%
all fields accuracy: 90.89%
```

和最早的 full 500 直接训练对比：

```text
full 500 direct:
all fields accuracy: 0.00%

full 500 after factor curriculum:
all fields accuracy: 90.00%
```

观察：factor curriculum 明显解决了 full 500 的坍缩问题。剩余错误主要集中在 station 字符串，例如把 `YEBES40M` 生成成 `YEBES20`；signal、unit 已经稳定，value 也从最早的主要错误来源变成了少量错误。

结论：这一步验证了前面的诊断。模型不是没有四字段复制能力，也不是单纯被数字压垮，而是需要通过 curriculum 逐步学习组合关系。把 `signal + value` 作为中间台阶之后，full 500 digit-spaced field copy 从 0% 提升到 90%。

normal full 500 after digit-spaced curriculum：

在 full 500 digit-spaced field copy 达到 90% 后，继续回到普通数字格式，验证 curriculum 学到的字段结构和组合关系能否迁移到真实 value 格式。

训练：

```bash
python train_sft.py \
  --init-from out/sft_full_field_copy_spaced_500_from_signal_value/ckpt.pt \
  --sft-path data/sft/astro_sft_field_copy_500.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_full_field_copy_normal_500_from_spaced
```

训练曲线：

```text
step 0: train loss 1.7106, val loss 1.6202
step 300: train loss 0.1736, val loss 0.1680
step 600: train loss 0.1106, val loss 0.1101
step 900: train loss 0.0514, val loss 0.0537
```

验证集结果：

```text
examples: 50
exact match: 76.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 76.00%
unit accuracy: 100.00%
all fields accuracy: 76.00%
```

训练集结果：

```text
examples: 450
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 77.78%
unit accuracy: 100.00%
all fields accuracy: 77.78%
```

观察：从 digit-spaced 回到普通数字格式后，station、signal、unit 全部稳定到 100%，说明字段结构和组合关系已经迁移成功。剩余错误几乎都集中在 value，例如把 `4.7` 生成成 `4.4`，或把 `12.0` 生成成 `25.0`。

结论：curriculum 成功把 full field copy 从 digit-spaced 迁移回普通格式，但 GPT-2 BPE 下的普通数字复制仍然是主要瓶颈。和最早普通 full field copy 失败相比，这已经从 0% 提升到 76%，说明中间课程确实有效。

normal value repair：

为了继续修复普通数字格式下的 value 错误，做两段训练：

```text
Stage 1: 在普通 value_copy_500 上做局部 value 修复
Stage 2: 回到普通 field_copy_500，刷新完整四字段格式
Stage 3: 使用更低学习率继续微调普通 field_copy_500
```

第一段 value 修复：

```bash
python train_sft.py \
  --init-from out/sft_full_field_copy_normal_500_from_spaced/ckpt.pt \
  --sft-path data/sft/value_copy_500.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 8 \
  --block-size 64 \
  --learning-rate 3e-4 \
  --out-dir out/sft_value_repair_normal_from_field_500
```

第二段回到四字段：

```bash
python train_sft.py \
  --init-from out/sft_value_repair_normal_from_field_500/ckpt.pt \
  --sft-path data/sft/astro_sft_field_copy_500.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_full_field_copy_normal_500_after_value_repair
```

第二段结果：

```text
validation:
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 80.00%
unit accuracy: 100.00%
all fields accuracy: 80.00%

train:
value accuracy: 87.33%
all fields accuracy: 87.33%
```

第三段低学习率继续微调：

```bash
python train_sft.py \
  --init-from out/sft_full_field_copy_normal_500_after_value_repair/ckpt.pt \
  --sft-path data/sft/astro_sft_field_copy_500.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 1e-4 \
  --out-dir out/sft_full_field_copy_normal_500_after_value_repair_long
```

第三段结果：

```text
validation:
exact match: 88.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 88.00%
unit accuracy: 100.00%
all fields accuracy: 88.00%

train:
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 95.78%
unit accuracy: 100.00%
all fields accuracy: 95.78%
```

对比：

```text
normal full field copy direct: 0.00%
after digit-spaced curriculum: 76.00%
after value repair: 80.00%
after low-lr field refresh: 88.00%
```

观察：value repair 能继续提升普通数字格式下的 value 复制，且不会破坏 station、signal、unit。低学习率 field refresh 比单纯 value repair 更有效，说明 value-only 修复后需要回到完整格式中重新对齐输出分布。

结论：普通数字复制仍然是最后瓶颈，但经过 curriculum 和 value repair，普通 full field copy 已经从 0% 提升到 88%。剩余错误仍然是同一 signal 下的数值混淆，例如 `25.0 -> 12.0`、`1.2 -> 5.2`。

natural field extraction from copy：

完成普通四字段 copy 后，继续迁移到更自然的句子抽取任务。这里不再使用 `station=...; value=...` 这种显式模板，而是使用自然语言句子：

```text
YEBES40M has a reported zenith wet delay of 38.5 mm in the latest solution.
For TSKB, the estimated tropospheric delay equals 18.5 ps.
Station WETTZELL shows a clock bias of 1.2 ns from space geodetic observations.
```

训练：

```bash
python train_sft.py \
  --init-from out/sft_full_field_copy_normal_500_after_value_repair_long/ckpt.pt \
  --sft-path data/sft/astro_sft_field_500.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_natural_field_500_from_copy
```

训练曲线：

```text
step 0: train loss 0.3480, val loss 0.4986
step 300: train loss 0.0166, val loss 0.0118
step 600: train loss 0.0052, val loss 0.0085
step 900: train loss 0.0051, val loss 0.0089
```

验证集结果：

```text
examples: 50
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

训练集结果：

```text
examples: 450
station accuracy: 99.56%
signal accuracy: 99.11%
value accuracy: 98.00%
unit accuracy: 100.00%
all fields accuracy: 96.67%
```

观察：copy curriculum 成功迁移到自然模板抽取。验证集达到 100%，训练集仍有少量 value 和 signal 错误，说明验证集在这个 split 下可能更容易，不能把 100% 过度解释为真实泛化已经完全解决。

结论：从 copy 到自然句子抽取的迁移是有效的。前面学习到的字段格式、组合关系和普通数字复制能力，为自然 field extraction 提供了很好的初始化。

rich natural field extraction：

为了测试模型是否只适应原来的 5 个自然模板，构造 20 个自然语言模板、1000 条样本：

```text
The zenith wet delay entry for station YEBES40M reads 38.5 mm.
At YEBES40M, the processing chain found tropospheric delay to be 12.0 ps.
TSKB -- tropospheric delay: 18.5 ps.
In the geodetic report, GOLD is associated with 52.2 mm of zenith wet delay.
```

数据构造：

```bash
python scripts/build_rich_field_sft.py \
  --out data/sft/astro_sft_field_rich_1000.jsonl \
  --num-examples 1000
```

训练从自然字段抽取 checkpoint 继续：

```bash
python train_sft.py \
  --init-from out/sft_natural_field_500_from_copy/ckpt.pt \
  --sft-path data/sft/astro_sft_field_rich_1000.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_rich_field_1000_from_natural
```

训练曲线：

```text
step 0: train loss 0.0035, val loss 0.0055
step 300: train loss 0.0041, val loss 0.0067
step 600: train loss 0.0040, val loss 0.0031
step 900: train loss 0.0010, val loss 0.0007
```

验证集结果：

```text
examples: 100
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

训练集结果：

```text
examples: 900
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

观察：在 20 个模板的合成自然语言分布内，模型已经完全掌握字段抽取。这个结果比 5 模板自然抽取更有说服力，但仍然属于模板生成分布内的泛化，不等同于真实开放文本泛化。

结论：copy curriculum -> natural extraction -> rich natural extraction 的迁移链路跑通。下一步如果继续提升研究价值，应该做 held-out 模板评测，也就是训练时不用某些模板，评测时专门看未见过模板。

held-out template natural extraction：

为了验证模型是否真的学到抽取规则，而不是只记住模板，构造 held-out template 实验：

```text
total rich templates: 20
train templates: 0-15
held-out templates: 16-19
```

训练集只包含前 16 个模板，评测集只包含后 4 个未见过模板。

数据构造：

```bash
python scripts/build_heldout_template_field_sft.py \
  --train-out data/sft/field_rich_train_templates_1000.jsonl \
  --heldout-out data/sft/field_rich_heldout_templates_200.jsonl \
  --train-examples 1000 \
  --heldout-examples 200 \
  --heldout-templates 4
```

训练：

```bash
python train_sft.py \
  --init-from out/sft_natural_field_500_from_copy/ckpt.pt \
  --sft-path data/sft/field_rich_train_templates_1000.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_heldout_template_train_1000
```

训练模板验证集：

```text
examples: 100
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

held-out 模板评测：

```text
examples: 200
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

观察：模型在未见过的 4 个模板上仍达到 100%，说明它不是单纯记住训练模板，而是已经学到了这个合成分布内的字段抽取规则。

结论：在当前合成模板体系内，结构化抽取阶段已经彻底跑通。下一步如果继续挑战泛化，应转向更开放的扰动，例如加入无关背景句、多个数值干扰、字段缺失、同一句多 station，或真实论文/报告句子。

distractor robustness：

为了测试模型在多个数字干扰下的鲁棒性，构造 distractor 数据。每条样本只有一个目标 value，但 input 中会出现多个无关数字，例如 epoch、previous solution、network average、uncertainty、iteration count：

```text
At YEBES40M, zenith wet delay equals 38.5 mm. The previous solution listed 52.2 mm, and the quality flag is 1.

The tropospheric delay at HOBART12 is not 12.0 ps; the accepted estimate is 8.0 ps after 3 iterations.

Report 101: YEBES40M has vertical velocity = -8.5 mm/yr; the network average for this product is 5.6 mm/yr.
```

数据构造：

```bash
python scripts/build_distractor_field_sft.py \
  --out data/sft/field_distractor_500.jsonl \
  --num-examples 500
```

先用 held-out template 模型直接评测 distractor 数据：

```text
zero-shot distractor:
exact match: 13.20%
station accuracy: 23.20%
signal accuracy: 72.80%
value accuracy: 42.20%
unit accuracy: 94.20%
all fields accuracy: 13.20%
```

观察：这是一次真正的鲁棒性断崖。模型在 held-out 模板上 100%，但遇到多个干扰数字后，输出格式和字段选择都会明显坏掉。

然后在 distractor 数据上训练：

```bash
python train_sft.py \
  --init-from out/sft_heldout_template_train_1000/ckpt.pt \
  --sft-path data/sft/field_distractor_500.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_distractor_500_from_heldout
```

训练后 distractor 结果：

```text
distractor all:
exact match: 88.40%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 88.40%
unit accuracy: 100.00%
all fields accuracy: 88.40%
```

训练后 held-out template 回测：

```text
held-out template after distractor training:
exact match: 97.50%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 97.50%
unit accuracy: 100.00%
all fields accuracy: 97.50%
```

观察：distractor training 把干扰集从 13.20% 提升到 88.40%，但 held-out template 从 100% 轻微下降到 97.50%。剩余错误几乎都集中在 value，尤其是从多个候选数字中选错目标数字，例如把 previous solution 或 rejected value 当成目标 value。

结论：模板泛化不等于抗干扰泛化。模型已经能处理未见模板，但当句子中出现多个数字时，还需要专门的 distractor curriculum。下一步可以继续做更细的干扰类型拆解：previous value、negative statement、network average、uncertainty，分别看哪类干扰最难。

distractor type breakdown：

为了判断哪类数字干扰最难，继续构造五类单独的 distractor 评测集：

```text
previous: 当前值 + previous / last week value
negative: not / reject value + accepted value
network: station value + network average
uncertainty: target value + formal uncertainty
metadata: target value + window / samples / epoch 等元数据数字
```

数据构造：

```bash
for t in previous negative network uncertainty metadata; do
  python scripts/build_distractor_type_field_sft.py \
    --distractor-type $t \
    --out data/sft/field_distractor_${t}_200.jsonl \
    --num-examples 200
done
```

held-out 模型 zero-shot：

```text
previous:
all fields accuracy: 16.50%
value accuracy: 30.50%

negative:
all fields accuracy: 37.00%
value accuracy: 49.00%

network:
all fields accuracy: 28.00%
value accuracy: 42.00%

uncertainty:
all fields accuracy: 39.50%
value accuracy: 51.50%

metadata:
all fields accuracy: 85.50%
value accuracy: 97.00%
```

distractor training 后：

```text
previous:
all fields accuracy: 53.00%
value accuracy: 53.00%

negative:
all fields accuracy: 52.00%
value accuracy: 52.00%

network:
all fields accuracy: 63.50%
value accuracy: 63.50%

uncertainty:
all fields accuracy: 90.50%
value accuracy: 90.50%

metadata:
all fields accuracy: 99.00%
value accuracy: 99.00%
```

观察：metadata 数字最容易，因为它们的语义和单位通常不像目标字段；uncertainty 中等偏容易。真正困难的是 previous、negative、network，这三类都包含同单位、同 signal 空间内的合法候选值。模型已经能稳定抽取 station、signal、unit，但经常在多个候选 value 之间选错。

典型错误：

```text
Input:
The zenith wet delay at YEBES40M is not 71.4 mm; the accepted estimate is 38.5 mm.

Target value:
38.5

Prediction:
71.4
```

结论：剩余瓶颈已经不是字段抽取格式，而是目标选择能力。模型需要学会使用语义线索，例如 `accepted estimate`、`current`、`station has`，同时忽略 `not`、`previous`、`network average` 等非目标数值。

hard distractor curriculum：

基于类型拆解结果，继续只针对最难的三类干扰做 curriculum：

```text
previous
negative
network
```

构造每类 300 条样本，合并成 900 条 hard distractor 训练集：

```bash
for t in previous negative network; do
  python scripts/build_distractor_type_field_sft.py \
    --distractor-type $t \
    --out data/sft/field_distractor_hard_${t}_300.jsonl \
    --num-examples 300
done

cat data/sft/field_distractor_hard_previous_300.jsonl \
    data/sft/field_distractor_hard_negative_300.jsonl \
    data/sft/field_distractor_hard_network_300.jsonl \
    > data/sft/field_distractor_hard_types_900.jsonl
```

训练：

```bash
python train_sft.py \
  --init-from out/sft_distractor_500_from_heldout/ckpt.pt \
  --sft-path data/sft/field_distractor_hard_types_900.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_distractor_hard_types_900
```

hard-types 训练前后对比：

```text
previous:
53.00% -> 99.00%

negative:
52.00% -> 96.00%

network:
63.50% -> 99.00%

uncertainty:
90.50% -> 95.00%

metadata:
99.00% -> 99.00%
```

整体 distractor 回测：

```text
after general distractor training:
all fields accuracy: 88.40%
value accuracy: 88.40%

after hard distractor curriculum:
all fields accuracy: 93.80%
value accuracy: 93.80%
```

held-out template 回测：

```text
after general distractor training:
all fields accuracy: 97.50%

after hard distractor curriculum:
all fields accuracy: 98.00%
```

观察：针对瓶颈类型训练，比泛泛增加 distractor 数据更有效。previous、negative、network 三类从 52-63% 提升到 96-99%，整体 distractor 从 88.40% 提升到 93.80%，且没有破坏 held-out template 能力。

结论：目标 value 选择能力可以通过针对性 curriculum 明显修复。剩余错误仍集中在极相似数值或同一 signal 的合法候选值之间，例如 `accepted estimate`、`final value` 和 rejected/previous value 的竞争。

multi-station binding：

继续测试同一句多 station / 多测量值绑定。每条样本包含两个 station 和两组 measurement，并明确指定 requested / target station：

```text
The report lists two stations: KOKEE with seasonal amplitude 3.6 mm; YEBES40M with zenith wet delay 38.5 mm. Extract YEBES40M.

Extract the measurement for TSKB. In the same solution, GOLD has tropospheric delay of 12.0 ps, and TSKB has tropospheric delay of 18.5 ps.

ONSA: vertical velocity, 2.4 mm/yr. GOLD: vertical velocity, 5.6 mm/yr. The requested record is ONSA.
```

数据构造：

```bash
python scripts/build_multi_station_field_sft.py \
  --out data/sft/field_multi_station_500.jsonl \
  --num-examples 500
```

用 hard distractor 模型直接评测：

```text
zero-shot multi-station:
exact match: 18.20%
station accuracy: 54.80%
signal accuracy: 51.60%
value accuracy: 38.20%
unit accuracy: 66.60%
all fields accuracy: 18.20%
```

观察：这是另一个鲁棒性断崖。模型经常抽到另一个 station 的 measurement，说明它还没有稳定掌握 station 和 measurement 的绑定关系。

在 multi-station 数据上训练：

```bash
python train_sft.py \
  --init-from out/sft_distractor_hard_types_900/ckpt.pt \
  --sft-path data/sft/field_multi_station_500.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 3e-4 \
  --out-dir out/sft_multi_station_500_from_hard
```

训练后结果：

```text
multi-station:
exact match: 62.80%
station accuracy: 91.80%
signal accuracy: 72.00%
value accuracy: 70.00%
unit accuracy: 78.60%
all fields accuracy: 62.80%

distractor回测:
all fields accuracy: 81.20%
value accuracy: 82.40%

held-out template回测:
all fields accuracy: 100.00%
```

观察：multi-station 从 18.20% 提升到 62.80%，但还没有解决；同时 distractor 从 93.80% 下降到 81.20%，说明 multi-station 绑定训练和 value 干扰鲁棒性之间存在能力冲突。held-out template 保持 100%，说明简单模板抽取没有被破坏。

结论：新的瓶颈是 entity-measurement binding，也就是同一句多个 station 时，如何把 station、signal、value、unit 作为一组绑定起来。下一步应该做混合训练，把 hard distractor 和 multi-station 样本混在一起，避免修一个能力时忘掉另一个能力。

mixed hard distractor + multi-station：

为了同时保住 hard distractor 的 value 选择能力和 multi-station 的 entity-measurement binding，构造混合训练集：

```text
hard distractor: 900
multi-station: 500
total: 1400
```

训练：

```bash
cat data/sft/field_distractor_hard_types_900.jsonl \
    data/sft/field_multi_station_500.jsonl \
    > data/sft/field_mixed_hard_multi_1400.jsonl

python train_sft.py \
  --init-from out/sft_distractor_hard_types_900/ckpt.pt \
  --sft-path data/sft/field_mixed_hard_multi_1400.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 1200 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 2e-4 \
  --out-dir out/sft_mixed_hard_multi_1400
```

结果对比：

```text
multi-station zero-shot:
all fields accuracy: 18.20%

multi-station after multi-only:
all fields accuracy: 62.80%

multi-station after mixed:
all fields accuracy: 48.40%
```

```text
distractor after hard:
all fields accuracy: 93.80%

distractor after multi-only:
all fields accuracy: 81.20%

distractor after mixed:
all fields accuracy: 93.60%
```

```text
held-out after mixed:
all fields accuracy: 99.00%
```

观察：混合训练保住了 hard distractor 能力，但 multi-station 只有 48.40%，低于 multi-only 的 62.80%。这说明简单拼接数据不是自动解决能力冲突的办法。由于 hard distractor 样本更多，而且初始化来自 hard distractor checkpoint，训练可能仍然偏向 value 选择，而没有充分学习 station-measurement binding。

结论：混合训练揭示了新的工程问题：数据比例和采样策略很重要。下一步可以提高 multi-station 占比，或做阶段式训练，例如 `multi-station -> mixed low-lr refresh`，而不是一次性拼接。

multi-station -> mixed low-lr refresh：

上一步的 simple mixed 是从 hard distractor checkpoint 出发，直接喂混合数据。结果说明模型更容易保住旧能力，而不是优先学新绑定能力。

所以这里换成两阶段顺序：

```text
Stage 1: hard distractor -> multi-station
Stage 2: multi-station checkpoint -> mixed hard+multi low-lr refresh
```

直觉是：先让模型集中学会 station-measurement binding，再用较低学习率混合刷新，把 hard distractor 能力补回来，减少遗忘。

训练：

```bash
python train_sft.py \
  --init-from out/sft_multi_station_500_from_hard/ckpt.pt \
  --sft-path data/sft/field_mixed_hard_multi_1400.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 800 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 1e-4 \
  --out-dir out/sft_multi_then_mixed_refresh_1400
```

结果对比：

```text
multi-station zero-shot:
all fields accuracy: 18.20%

multi-station after multi-only:
all fields accuracy: 62.80%

multi-station after simple mixed:
all fields accuracy: 48.40%

multi-station after multi -> mixed low-lr refresh:
all fields accuracy: 70.00%
```

```text
distractor after hard:
all fields accuracy: 93.80%

distractor after multi-only:
all fields accuracy: 81.20%

distractor after simple mixed:
all fields accuracy: 93.60%

distractor after multi -> mixed low-lr refresh:
all fields accuracy: 93.80%
```

```text
held-out after multi -> mixed low-lr refresh:
all fields accuracy: 99.00%
```

观察：这次 staged refresh 同时改善了两个目标。multi-station 从 multi-only 的 62.80% 提升到 70.00%，同时 distractor 回到 93.80%，没有继续牺牲 held-out template。

结论：阶段顺序和学习率确实重要。直接混合训练会被旧任务和样本比例牵着走；先学绑定，再低学习率混合刷新，更适合这种小模型多能力叠加。当前新的 frontier 是 multi-station binding 的 70.00%，还没有彻底解决，但已经找到比 simple mixed 更好的训练策略。

multi-station error analysis：

继续分析 70% 剩下的错误，不再只看总 accuracy。先用字段评测结果做错误归类：

```bash
python tools/eval/analyze_field_errors.py \
  --field-accuracy out/field_accuracy_multi_station_500_after_refresh/field_accuracy.csv \
  --out-dir out/field_error_multi_station_after_refresh
```

这个工具不会重新生成模型预测，只读取已有的 `field_accuracy.csv`，把错误分成几类：

```text
wrong_station_or_record:
模型选中了输入中的另一个 station 或另一条 record

target_station_wrong_measurement:
station 选对了，但 signal/value/unit 搬成了另一个 measurement

target_station_field_noise:
station 选对了，但其他字段不是明显来自上下文另一组 measurement

missing_field:
输出缺字段
```

三组模型对比：

```text
multi-only:
all correct: 62.80%
target_station_wrong_measurement: 26.80%
wrong_station_or_record: 8.20%

simple mixed:
all correct: 48.40%
target_station_wrong_measurement: 29.40%
wrong_station_or_record: 17.80%

multi -> mixed low-lr refresh:
all correct: 70.00%
target_station_wrong_measurement: 21.20%
wrong_station_or_record: 6.40%
```

结论：staged refresh 不只是让总分变高，它确实减少了两类绑定错误。但剩余错误里最大的仍然是 `target_station_wrong_measurement`：模型经常知道要抽哪个 station，却把另一个 station 的 signal/value/unit 绑定过来。

这说明下一步数据不应该泛泛增加，而应该专门构造“目标 station 与干扰 station 使用相同 signal / 相同 unit / 相近 value / 位置反转”的 hard binding curriculum。

hard binding curriculum：

新增 hard binding 数据生成脚本：

```bash
python scripts/build_hard_binding_field_sft.py \
  --out data/sft/field_hard_binding_800.jsonl \
  --num-examples 800 \
  --seed 1337
```

这个数据集专门制造更容易混淆的双 station 样本：

```text
same_signal: 两个 station 使用相同 signal，但 value 不同
same_unit: 两个 station 使用相同 unit，但 signal/value 不同
near_value: 两个 station 的数值接近
target_first: 目标 station 在前
target_second: 目标 station 在后
```

先做 zero-shot：

```text
hard binding zero-shot:
all fields accuracy: 40.12%
```

再从 `multi -> mixed low-lr refresh` checkpoint 单独训练 hard binding：

```bash
python train_sft.py \
  --init-from out/sft_multi_then_mixed_refresh_1400/ckpt.pt \
  --sft-path data/sft/field_hard_binding_800.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 800 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 1e-4 \
  --out-dir out/sft_hard_binding_800_from_refresh
```

结果：

```text
hard binding after hard-only:
all fields accuracy: 60.50%

multi-station after hard-only:
all fields accuracy: 59.00%

distractor after hard-only:
all fields accuracy: 89.00%
```

观察：hard-only 训练能把 hard binding 从 40.12% 提到 60.50%，说明数据方向有效；但它把原 multi-station 从 70.00% 拉低到 59.00%，distractor 也从 93.80% 降到 89.00%。这说明 hard binding 难度跳跃太大，单独训练会改变模型的能力分布。

所以继续做混合刷新。新增通用混合脚本：

```bash
python scripts/mix_sft_jsonl.py \
  --inputs \
    data/sft/field_hard_binding_800.jsonl \
    data/sft/field_multi_station_500.jsonl \
    data/sft/field_distractor_hard_types_900.jsonl \
  --out data/sft/field_mixed_binding_multi_hard_2200.jsonl \
  --shuffle \
  --seed 1337
```

再从 `multi -> mixed low-lr refresh` checkpoint 训练：

```bash
python train_sft.py \
  --init-from out/sft_multi_then_mixed_refresh_1400/ckpt.pt \
  --sft-path data/sft/field_mixed_binding_multi_hard_2200.jsonl \
  --split-mode shuffle \
  --train-ratio 0.9 \
  --max-iters 800 \
  --eval-interval 100 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 1e-4 \
  --out-dir out/sft_mixed_binding_multi_hard_2200
```

结果：

```text
hard binding after mixed binding:
all fields accuracy: 52.62%

multi-station after mixed binding:
all fields accuracy: 71.40%

distractor after mixed binding:
all fields accuracy: 95.40%

held-out after mixed binding:
all fields accuracy: 99.00%
```

错误分析：

```text
multi-station after refresh:
target_station_wrong_measurement: 21.20%
wrong_station_or_record: 6.40%

multi-station after mixed binding:
target_station_wrong_measurement: 21.00%
wrong_station_or_record: 5.00%
```

结论：hard binding-only 是一个负结果，说明“更难的数据”不能直接灌给小模型。mixed binding 是温和正结果：hard binding 自身从 40.12% 到 52.62%，multi-station 从 70.00% 到 71.40%，distractor 从 93.80% 到 95.40%。这次提升不大，但说明正确方向是“硬样本混合巩固”，而不是单独硬训。

field extraction scorecard：

新增 scorecard 脚本，用来从多个字段评测报告中自动抽取指标：

```bash
python tools/eval/summarize_field_scorecard.py \
  --out-csv out/field_scorecard.csv \
  --out-md out/field_scorecard.md \
  multi_zero=out/field_accuracy_multi_station_500_zero_shot/report.md \
  multi_only=out/field_accuracy_multi_station_500_after_train/report.md \
  simple_mixed=out/field_accuracy_multi_station_500_after_mixed/report.md \
  staged_refresh=out/field_accuracy_multi_station_500_after_refresh/report.md \
  hard_only=out/field_accuracy_multi_station_500_after_hard_binding/report.md \
  mixed_binding=out/field_accuracy_multi_station_500_after_mixed_binding/report.md \
  hard_binding_zero=out/field_accuracy_hard_binding_800_zero_shot/report.md \
  hard_binding_only=out/field_accuracy_hard_binding_800_after_train/report.md \
  hard_binding_mixed=out/field_accuracy_hard_binding_800_after_mixed_binding/report.md \
  distractor_refresh=out/field_accuracy_distractor_500_after_refresh/report.md \
  distractor_hard_only=out/field_accuracy_distractor_500_after_hard_binding/report.md \
  distractor_mixed_binding=out/field_accuracy_distractor_500_after_mixed_binding/report.md \
  heldout_refresh=out/field_accuracy_heldout_template_after_refresh/report.md \
  heldout_hard_only=out/field_accuracy_heldout_template_after_hard_binding/report.md \
  heldout_mixed_binding=out/field_accuracy_heldout_template_after_mixed_binding/report.md
```

关键总表：

```text
checkpoint              multi-station   hard-binding   distractor   held-out
multi_zero              18.20%          -              -            -
multi_only              62.80%          -              81.20%       100.00%
simple_mixed            48.40%          -              93.60%       99.00%
staged_refresh          70.00%          -              93.80%       99.00%
hard_binding_only       59.00%          60.50%         89.00%       98.00%
mixed_binding           71.40%          52.62%         95.40%       99.00%
```

当前综合最好的 checkpoint：

```text
out/sft_mixed_binding_multi_hard_2200/ckpt.pt
```

选择它不是因为它在 hard binding 单项最高，而是因为它的整体 trade-off 最好：

```text
multi-station: 71.40%
distractor: 95.40%
held-out: 99.00%
hard binding: 52.62%
```

阶段收束结论：这个 mini research 已经完成了从失败定位、课程构造、错误分析到综合模型选择的闭环。接下来继续提升 multi-station 当然可以，但更有学习价值的是进入 DPO / preference optimization，用已经建立好的字段抽取评测体系来观察偏好优化是否真的改善模型行为。

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

DPO step 1: preference data：

SFT 数据只有一个标准答案：

```text
prompt -> answer
```

DPO 数据有一对答案：

```text
prompt -> chosen
prompt -> rejected
```

其中 `chosen` 是正确字段抽取结果，`rejected` 是刻意构造的错误结果，例如 wrong station、wrong signal group、wrong value。

新增数据模块：

```text
dpo_data.py
```

它负责：

```text
format_dpo_example
encode_dpo_example
pad_dpo_batch
```

新增偏好数据构造脚本：

```bash
python scripts/build_field_dpo.py \
  --sft-path data/sft/field_mixed_binding_multi_hard_2200.jsonl \
  --out data/dpo/field_mixed_binding_multi_hard_2200.jsonl \
  --seed 1337
```

生成结果：

```text
saved 2200 examples
wrong_signal_group: 552
wrong_station: 534
wrong_value_from_input: 568
wrong_value_same_signal: 546
```

这一步的关键是：我们不是让模型“随便更喜欢人类偏好的回答”，而是利用字段抽取任务天然可验证的特点，构造可控的 chosen / rejected 对。

DPO step 2: trainer smoke test：

新增训练脚本：

```text
train_dpo.py
```

DPO 的核心计算是：

```text
policy_margin = logp_policy(chosen) - logp_policy(rejected)
reference_margin = logp_ref(chosen) - logp_ref(rejected)

loss = -logsigmoid(beta * (policy_margin - reference_margin))
```

直觉：

```text
如果 policy 相比 reference 更偏向 chosen，loss 会下降。
如果 policy 没有比 reference 更偏向 chosen，loss 接近 0.693。
```

smoke test：

```bash
python train_dpo.py \
  --init-from out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --dpo-path data/dpo/field_mixed_binding_multi_hard_2200.jsonl \
  --split-mode stratified \
  --train-ratio 0.9 \
  --max-iters 3 \
  --eval-interval 1 \
  --eval-iters 1 \
  --batch-size 2 \
  --block-size 128 \
  --learning-rate 1e-5 \
  --beta 0.1 \
  --out-dir out/dpo_smoke_test
```

结果：

```text
step 0: train loss 0.6931, val loss 0.6931
step 1: train loss 0.6930, val loss 0.6929
step 2: train loss 0.6918, val loss 0.6931
saved checkpoint to out/dpo_smoke_test/ckpt.pt
```

观察：step 0 是正常的 0.693 起点，因为 policy 和 reference 来自同一个 checkpoint，初始时二者完全一样。smoke test 的目的不是证明 DPO 有效果，而是证明数据、logprob、reference model、loss、反传、保存 checkpoint 全链路可运行。

DPO step 3: small run and scorecard：

正式小跑 100 step：

```bash
python train_dpo.py \
  --init-from out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --dpo-path data/dpo/field_mixed_binding_multi_hard_2200.jsonl \
  --split-mode stratified \
  --train-ratio 0.9 \
  --max-iters 100 \
  --eval-interval 10 \
  --eval-iters 5 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 1e-5 \
  --beta 0.1 \
  --out-dir out/dpo_field_100
```

训练日志显示 DPO 目标正常起效：

```text
step 0: train loss 0.6931, val loss 0.6931
step 90: train loss 0.6879, val loss 0.6742, val pref acc 90.00%
```

但 DPO 是否真的有用，不能只看 preference accuracy，而要回到字段评测。和 DPO 前的 mixed binding checkpoint 对比：

```text
checkpoint          multi-station   hard-binding   distractor   held-out
SFT mixed binding   71.40%          52.62%         95.40%       99.00%
DPO 100             69.80%          53.62%         95.20%       99.00%
```

观察：

```text
multi-station: 71.40% -> 69.80%
hard binding: 52.62% -> 53.62%
distractor: 95.40% -> 95.20%
held-out: 99.00% -> 99.00%
```

结论：这次 DPO 是温和 mixed result。它确实优化了偏好目标，也让 hard binding 小幅上升，但没有带来整体字段抽取提升；multi-station 还小幅下降。说明当前 DPO 数据更像是在修 hard binding/value preference，而不是全面提高结构化抽取能力。

这一步的学习重点是：DPO 不是“必然提升模型”的魔法。它优化的是 chosen/rejected 偏好边界，最终是否改善任务，要靠独立评测集判断。对于这个小模型，DPO 的收益很容易表现为局部 trade-off。

DPO step 4: preference eval by type：

新增评测脚本：

```text
tools/eval/evaluate_dpo_preference.py
```

这个脚本不生成文本，只计算模型给 chosen / rejected 的答案 logprob，并按 `preference_type` 分组统计 chosen 胜率。

对比 SFT baseline 和 DPO 100：

```text
preference type          SFT acc   DPO acc
all                      96.68%    96.91%
wrong_signal_group       98.91%    98.73%
wrong_station            99.81%    100.00%
wrong_value_from_input   91.37%    92.08%
wrong_value_same_signal  96.89%    97.07%
```

平均 margin：

```text
SFT: 10.0275
DPO: 10.4898
```

结论：DPO 确实把 preference margin 往正确方向推了一点，但 SFT baseline 本来已经能达到 96.68% preference accuracy，所以这批 DPO 数据对当前模型太容易，训练信号偏弱。这解释了为什么 DPO 100 的字段 scorecard 只是小幅 trade-off，而不是明显提升。

下一轮如果继续研究 DPO，应该构造更难的 rejected，例如只改 value 的最后一位、交换同 signal 同 unit 的两个 station、或使用模型自己真实生成的错误作为 rejected。

DPO step 5: hard value rejected：

上一批 DPO 数据对 SFT baseline 太容易：

```text
SFT preference accuracy: 96.68%
```

所以新增 hard value DPO 构造脚本：

```text
scripts/build_hard_value_dpo.py
```

它只做一种错误：

```text
station 正确
signal 正确
unit 正确
value 替换成 input 中另一个数值
```

构造命令：

```bash
python scripts/build_hard_value_dpo.py \
  --sft-path data/sft/field_hard_binding_800.jsonl \
  --out data/dpo/field_hard_value_800.jsonl \
  --seed 1337
```

生成结果：

```text
loaded 800 sft examples
saved 788 dpo examples
preference type: hard_wrong_value_from_input
```

这里有一个重要实现细节：不能用普通数字正则直接抓所有数字，因为 station 名里也有数字，例如 `YEBES40M`、`NYALES20`。脚本用边界约束避免把 station 名中的 `40`、`20` 当成候选 value。

baseline 难度评估：

```bash
python tools/eval/evaluate_dpo_preference.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --dpo-path data/dpo/field_hard_value_800.jsonl \
  --out-dir out/dpo_pref_eval_hard_value_sft
```

结果：

```text
SFT baseline:
accuracy: 75.00%
avg margin: 3.1764

DPO 100:
accuracy: 75.63%
avg margin: 3.2077
```

结论：这批 hard value rejected 明显比上一批 easy preference 更有训练价值。SFT baseline 从 96.68% 降到 75.00%，说明模型还不能稳定区分“同字段格式、只错 value”的 chosen/rejected。旧 DPO 100 只提升到 75.63%，说明上一轮 DPO 没有真正解决 hard value preference。

DPO step 6: hard value DPO 100：

用 hard value preference 数据训练：

```bash
python train_dpo.py \
  --init-from out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --dpo-path data/dpo/field_hard_value_800.jsonl \
  --split-mode stratified \
  --train-ratio 0.9 \
  --max-iters 100 \
  --eval-interval 10 \
  --eval-iters 5 \
  --batch-size 4 \
  --block-size 128 \
  --learning-rate 1e-5 \
  --beta 0.1 \
  --out-dir out/dpo_hard_value_100
```

训练日志：

```text
step 0: train loss 0.6931, val loss 0.6931
step 80: train loss 0.6786, val loss 0.6822, val pref acc 80.00%
step 90: train loss 0.6846, val loss 0.6900, val pref acc 55.00%
```

由于 `eval-iters=5`、`batch-size=4`，每次只抽 20 条样本，val pref acc 抖动很大。因此最终仍然要看完整 preference eval：

```text
hard value preference:
SFT baseline: 75.00%, avg margin 3.1764
DPO hard value 100: 74.37%, avg margin 3.4113
```

字段 scorecard：

```text
checkpoint              multi-station   hard-binding   distractor   held-out
SFT mixed binding       71.40%          52.62%         95.40%       99.00%
DPO hard value 100      69.80%          53.37%         95.80%       99.00%
```

结论：这是一个负结果 / mixed result。hard value DPO 没有提升 hard value preference accuracy，虽然 avg margin 略升；字段任务上 hard binding 和 distractor 小幅提升，但 multi-station 下滑。

这说明“rejected 更难”是必要条件，但不是充分条件。当前 DPO 设置可能仍然太弱、样本太少，或者 hard value preference 与 multi-station binding 存在 trade-off。下一步如果继续 DPO，应该做超参对照，而不是直接加训练步数。

DPO step 7: hard value DPO ablation：

对 hard value DPO 做最小超参对照：

```text
A: lr 1e-5, beta 0.1, steps 100
B: lr 5e-6, beta 0.1, steps 100
C: lr 1e-5, beta 0.05, steps 100
D: lr 5e-6, beta 0.05, steps 100
```

完整 hard value preference eval：

```text
checkpoint              preference acc   avg margin
SFT baseline            75.00%           3.1764
A lr1e-5 beta0.1        74.37%           3.4113
B lr5e-6 beta0.1        74.49%           3.2955
C lr1e-5 beta0.05       74.37%           3.4152
D lr5e-6 beta0.05       74.37%           3.2972
```

没有任何一组超过 SFT baseline。说明当前问题不是简单调低学习率或 beta 就能解决。

选择表现相对最稳的 B 组做字段回测：

```text
checkpoint              multi-station   hard-binding   distractor   held-out
SFT mixed binding       71.40%          52.62%         95.40%       99.00%
B lr5e-6 beta0.1        70.20%          53.12%         95.60%       99.00%
```

结论：B 组比 A 组稍稳，但仍然是 mixed result。hard binding 和 distractor 小幅提升，multi-station 下降，held-out 不变；hard value preference accuracy 仍低于 SFT baseline。

阶段判断：在当前 6.57M 小模型、当前 hard value DPO 数据和当前训练规模下，DPO 没有带来稳定净收益。它可以推动 margin，但不能稳定提高 hard value preference accuracy，也不能避免 multi-station trade-off。

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

- 先为当前自写 nanoGPT checkpoint 建立本地推理基准
- 再做一个可以交互的本地 demo / API
- 之后再学习 vLLM / SGLang / Transformers serving 部署标准模型
- 测 latency、throughput、显存占用
- 对比不同 batch size、上下文长度、并发数
- 尝试 INT8 / INT4 量化
- 做一个可以交互的 demo

### 3.1 Generation Benchmark

当前 checkpoint 是自写 nanoGPT 结构，不能直接交给 vLLM / SGLang 加载。因此第三阶段第一步先测本地生成速度。

新增脚本：

```text
tools/eval/benchmark_generation.py
```

运行 CPU benchmark：

```bash
python tools/eval/benchmark_generation.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --max-new-tokens 80 \
  --num-runs 5 \
  --warmup-runs 1 \
  --temperature 1.0 \
  --out-dir out/generation_benchmark_sft_mixed_binding_cpu
```

运行 MPS benchmark：

```bash
USE_MPS=1 python tools/eval/benchmark_generation.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --max-new-tokens 80 \
  --num-runs 5 \
  --warmup-runs 1 \
  --temperature 1.0 \
  --out-dir out/generation_benchmark_sft_mixed_binding_mps
```

结果：

```text
device   avg latency   avg tokens/s
cpu      0.4556 s      175.58
mps      1.1890 s       67.36
```

观察：在当前 6.57M 小模型、逐 token decode 的场景下，CPU 比 MPS 更快。原因是模型太小，MPS 调度和同步开销会抵消并行计算收益。

这一步引入两个部署指标：

```text
latency:
一次请求从开始到结束的耗时。

throughput:
单位时间生成多少 token，这里用 tokens/s 表示。
```

当前 benchmark 还暴露了一个 demo/API 必须解决的问题：模型生成 `<|endoftext|>` 后底层 `generate()` 仍会继续采样。现在 `generate()` 已经支持传入 `eosTokenId`，服务和采样脚本都可以选择在 EOS 处停止。

### 3.2 FastAPI 本地推理服务

`sample.py` 是一次性生成脚本，每次运行都会重新加载 checkpoint。服务化以后，模型在启动时只加载一次，之后每个请求只负责编码 prompt、生成 token、解码文本和返回指标。

新增脚本：

```text
tools/serve/serve_fastapi.py
```

启动服务：

```bash
python tools/serve/serve_fastapi.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --port 8010
```

检查服务状态：

```bash
curl http://127.0.0.1:8010/health
```

调用生成接口：

```bash
curl -X POST http://127.0.0.1:8010/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Instruction:\nExtract the station, signal, value, and unit from the text.\n\nInput:\nONSA reports vertical velocity of 2.4 mm/yr.\n\nAnswer:\n",
    "max_new_tokens": 40,
    "temperature": 0.8,
    "top_k": 40,
    "stop_at_eos": true
  }'
```

接口返回内容包括：

```text
text:
完整输出文本，包含 prompt 和生成部分。

completion_text:
只包含模型新生成的部分。

latency_sec:
这次请求的生成耗时。

tokens_per_sec:
这次请求的生成速度。
```

这一步的关键变化是：模型不再只是“能在命令行生成”，而是变成了一个可以被外部程序调用的本地推理服务。

### 3.3 API Benchmark

`benchmark_generation.py` 测的是模型内部生成耗时，`benchmark_api.py` 测的是 HTTP 接口的端到端耗时。后者更接近真实使用时的体验，因为它包含请求发送、JSON 解析、模型生成和结果返回。

新增脚本：

```text
tools/eval/benchmark_api.py
```

先启动服务：

```bash
python tools/serve/serve_fastapi.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --port 8010
```

再运行 API benchmark：

```bash
python tools/eval/benchmark_api.py \
  --url http://127.0.0.1:8010/generate \
  --num-runs 5 \
  --warmup-runs 1 \
  --max-new-tokens 40 \
  --temperature 0.8 \
  --top-k 40 \
  --stop-at-eos \
  --out-dir out/api_benchmark_sft_mixed_binding_cpu
```

本次 CPU 结果：

```text
avg end-to-end latency:       0.0980 sec
avg server generation latency: 0.0962 sec
avg HTTP overhead:            0.0018 sec
avg tokens/s:                 228.61
```

观察：当前模型很小，本地 HTTP 额外开销约 0.002 秒，主要耗时仍然来自逐 token 生成。对于本项目这种本地小模型 demo，API 封装本身不是主要瓶颈。

### 3.4 API Concurrency Benchmark

单请求 benchmark 只能说明一个请求的延迟，并发 benchmark 用来观察多个请求同时进入服务时的排队和吞吐变化。

新增脚本：

```text
tools/eval/benchmark_api_concurrency.py
```

运行方式：

```bash
python tools/eval/benchmark_api_concurrency.py \
  --url http://127.0.0.1:8010/generate \
  --concurrency 1,2,4 \
  --requests-per-level 8 \
  --warmup-runs 1 \
  --max-new-tokens 40 \
  --temperature 0.8 \
  --top-k 40 \
  --stop-at-eos \
  --out-dir out/api_concurrency_sft_mixed_binding_cpu
```

本次 CPU 结果：

```text
concurrency   req/s   output tok/s   avg latency   p95 latency
1             11.03   242.70         0.0905s       0.0979s
2             19.01   420.63         0.1044s       0.1107s
4             23.90   525.74         0.1638s       0.1755s
```

观察：并发从 1 提到 4 后，整体吞吐从 11.03 req/s 提升到 23.90 req/s，但平均延迟也从 0.0905 秒上升到 0.1638 秒。这说明当前服务在本地小模型场景下可以通过并发提高吞吐，但请求之间会出现排队和资源竞争。部署时需要在吞吐和延迟之间做取舍。

### 3.5 Context Length Benchmark

上下文长度 benchmark 用来观察 prompt 变长时，固定输出长度下的延迟变化。真实部署中，长 prompt 会增加 prefill 成本，也会让后续逐 token decode 时处理更长的历史上下文。

新增脚本：

```text
tools/eval/benchmark_context_length.py
```

运行方式：

```bash
python tools/eval/benchmark_context_length.py \
  --url http://127.0.0.1:8010/generate \
  --context-lengths 32,64,96,112 \
  --num-runs 5 \
  --warmup-runs 1 \
  --max-new-tokens 16 \
  --temperature 0.8 \
  --top-k 40 \
  --stop-at-eos \
  --out-dir out/context_length_sft_mixed_binding_cpu
```

本次 CPU 结果：

```text
target ctx   actual prompt   avg latency   avg tok/s
32           43              0.0664s       247.16
64           69              0.0767s       213.41
96           107             0.0954s       171.06
112          121             0.0953s       171.12
```

观察：在固定最多生成 16 个 token 的情况下，prompt 从 43 token 增加到 121 token，平均延迟从 0.0664 秒上升到约 0.095 秒，生成速度从 247 tok/s 下降到约 171 tok/s。当前模型没有 KV cache，每生成一个 token 都会重新计算最近 `block_size` 范围内的上下文，所以 prompt 越长，单步生成越慢。

### 3.6 Output Length Benchmark

输出长度 benchmark 用来观察生成 token 数变多时的延迟变化。为了避免 EOS 提前停止影响横向比较，这次不启用 `--stop-at-eos`，让每组尽量生成固定数量的 token。

新增脚本：

```text
tools/eval/benchmark_output_length.py
```

运行方式：

```bash
python tools/eval/benchmark_output_length.py \
  --url http://127.0.0.1:8010/generate \
  --output-lengths 8,16,32,64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --temperature 0.8 \
  --top-k 40 \
  --out-dir out/output_length_sft_mixed_binding_cpu
```

本次 CPU 结果：

```text
target output   avg new tokens   avg latency   avg tok/s
8               8.0              0.0340s       248.85
16              16.0             0.0703s       237.24
32              32.0             0.1385s       233.91
64              64.0             0.2961s       217.62
```

观察：输出 token 数从 8 增加到 64 后，总延迟从 0.0340 秒增加到 0.2961 秒，基本呈线性增长。原因是当前生成是逐 token decode，每多生成一个 token，就要多做一次模型 forward。tokens/s 略有下降，是因为序列越来越长，每一步 attention 看到的上下文也会变长。

### 3.7 KV Cache

普通自回归生成时，每生成一个新 token，模型都会把最近一整段上下文重新送入 Transformer。这样会重复计算历史 token 的 key/value。KV cache 的思路是：历史 token 的 key/value 一旦算出来，就保存在 cache 里；下一步生成时只计算新 token 的 query/key/value，再让新 token 的 query 去 attend 过去缓存的 key/value。

本项目中新增了缓存生成路径：

```text
MultiHeadAttention.forward(..., pastKv, useCache)
Block.forward(..., pastKv, useCache)
BigramLanguageModel.forward_with_cache(...)
BigramLanguageModel.generate(..., useKvCache=True)
```

`sample.py`、FastAPI 服务和 benchmark 脚本都支持打开 KV cache：

```bash
python sample.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --prompt "Instruction:\nExtract the station, signal, value, and unit from the text.\n\nInput:\nONSA reports vertical velocity of 2.4 mm/yr.\n\nAnswer:\n" \
  --max-new-tokens 64 \
  --top-k 40 \
  --use-kv-cache
```

API 请求中也可以传入：

```json
{
  "prompt": "...",
  "max_new_tokens": 64,
  "use_kv_cache": true
}
```

正确性检查：同一段输入下，普通 full forward 和 cached incremental forward 的最后一个位置 logits 最大误差约为 `6.7e-06`，属于浮点计算顺序导致的正常微小差异。

速度对比：

```text
mode        avg latency   avg tok/s
no cache    0.2903s       220.49
kv cache    0.1553s       412.17
```

观察：在 prompt 42 token、生成 64 token 的 CPU 测试中，KV cache 让生成速度从 220.49 tok/s 提升到 412.17 tok/s，接近 1.9 倍。

当前 RoPE 模型已经支持 sliding-window KV cache：当生成长度超过 `block_size` 时，cache 只保留最近 `block_size` 个 token 的 key/value，旧 token 会被裁掉。非 RoPE 模型仍然要求 `prompt_tokens + max_new_tokens <= block_size`，因为 learned position embedding 没有超过 `block_size` 的位置表。

超过 `block_size` 的长生成测试：

```text
prompt tokens: 42
max new tokens: 160
block_size: 128

mode                  avg latency   avg tok/s
no cache              0.9682s       165.26
sliding kv cache      0.3821s       418.79
```

FastAPI 也验证了 `max_new_tokens=160,use_kv_cache=true` 的请求可以正常返回，此时 `total_tokens=202`，已经超过 `block_size=128`。

把 KV cache 接入 FastAPI 后，服务链路也有明显收益：

```text
case                       no cache        kv cache
API avg latency            0.0980s         0.0627s
API avg tok/s              228.61          363.64
concurrency=1 req/s        11.03           16.13
concurrency=2 req/s        19.01           25.46
concurrency=4 req/s        23.90           24.62
output 64 avg latency      0.2961s         0.1570s
output 64 avg tok/s        217.62          411.96
```

观察：KV cache 对单请求、较低并发和长输出最明显；当并发升到 4 时，总吞吐提升变小，说明瓶颈开始转向 CPU 资源竞争和请求排队。

### 3.8 Transformers Baseline

为了和成熟推理框架做一个最小对照，新增 Hugging Face Transformers benchmark。默认使用 `sshleifer/tiny-gpt2`，这是一个极小测试模型，hidden size 只有 2，适合验证标准 `generate()` 链路，但不能和本项目 checkpoint 做生成质量上的公平比较。

新增脚本：

```text
tools/eval/benchmark_transformers_generation.py
```

运行方式：

```bash
python tools/eval/benchmark_transformers_generation.py \
  --model-name sshleifer/tiny-gpt2 \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --out-dir out/transformers_tiny_gpt2_cpu
```

本次 CPU 结果：

```text
model                 avg latency   avg tok/s
sshleifer/tiny-gpt2   0.0502s       1275.78
```

对照观察：

```text
model/path                         avg latency   avg tok/s
self nanoGPT no cache, 64 tokens    0.2903s       220.49
self nanoGPT kv cache, 64 tokens    0.1553s       412.17
Transformers tiny-gpt2, 64 tokens   0.0502s       1275.78
```

这个结果主要说明两点。第一，成熟框架的 `generate()` 已经内置了高效缓存路径，哪怕是 CPU 上的小模型也很快。第二，这不是同规模模型的严格公平对比，因为 `tiny-gpt2` 极小；真正公平的下一步应该选择规模更接近的 Hugging Face 模型，或者把自写 checkpoint 转换成标准模型格式后再对比。

为避免 Hugging Face 下载不稳定，也支持本地随机初始化 GPT-2 配置。这个模式不比较生成质量，只比较同等参数量附近的框架生成速度。

运行方式：

```bash
python tools/eval/benchmark_transformers_generation.py \
  --random-gpt2 \
  --n-embd 112 \
  --n-layer 2 \
  --n-head 4 \
  --n-positions 128 \
  --max-new-tokens 64 \
  --num-runs 5 \
  --warmup-runs 1 \
  --out-dir out/transformers_random_gpt2_6m_cpu
```

本次 CPU 结果：

```text
model/path                              params   avg latency   avg tok/s
self nanoGPT kv cache, 64 tokens        6.57M    0.1553s       412.17
Transformers random GPT-2, 64 tokens    5.95M    0.0821s       779.46
```

这个对照更接近参数规模，但仍然不是严格同构比较：Transformers GPT-2 使用的是成熟框架里的 GPT-2 实现和 `generate()` 调度，本项目使用的是自写 LLaMA-style 结构、RoPE/GQA/RMSNorm/SwiGLU。它的价值在于说明：即使参数量接近，成熟框架的推理路径仍然有明显工程优势。

### 3.9 Batch Serving

并发请求只是让多个请求同时进入服务；batch serving 则是把多个 prompt 合成一个 batch，让一次模型 forward 同时处理多条请求。当前实现新增了 `/generate_batch` 接口，并支持不同长度 prompt 的 left padding + padding attention mask。为了避免 left padding 改变 RoPE 和位置 embedding 的语义，模型会根据 `attention_mask` 重新计算每条样本自己的 `position_ids`。现在不同长度 batch 也可以开启 `use_kv_cache=True`，生成阶段会根据每条样本的真实长度维护位置。

新增脚本：

```text
tools/eval/benchmark_batch_api.py
```

运行方式：

```bash
python tools/eval/benchmark_batch_api.py \
  --base-url http://127.0.0.1:8010 \
  --batch-sizes 1,2,4,8 \
  --num-runs 3 \
  --warmup-runs 1 \
  --max-new-tokens 40 \
  --temperature 0.8 \
  --top-k 40 \
  --stop-at-eos \
  --use-kv-cache \
  --out-dir out/batch_api_kv_cache_sft_mixed_binding_cpu
```

本次 CPU 结果：

```text
mode        batch   avg latency   avg tok/s   avg req/s
sequential  1       0.0585s       376.35      17.11
batched     1       0.0580s       379.54      17.25
sequential  2       0.1206s       365.13      16.60
batched     2       0.0801s       549.60      24.98
sequential  4       0.2361s       372.66      16.94
batched     4       0.1175s       749.51      34.07
sequential  8       0.4740s       371.38      16.88
batched     8       0.1987s       886.25      40.28
```

观察：逐条请求时，req/s 基本停在 17 左右；batch size 增加到 8 后，batched 模式达到 40.28 req/s，输出吞吐达到 886.25 tok/s。这说明合并请求能显著提高模型计算利用率，也是 vLLM / SGLang 等推理系统要做 batching 的原因。

不同长度 prompt 的 batch 测试：

```bash
python tools/eval/benchmark_batch_api.py \
  --base-url http://127.0.0.1:8010 \
  --batch-sizes 1,2,4,8 \
  --num-runs 3 \
  --warmup-runs 1 \
  --max-new-tokens 20 \
  --temperature 0.8 \
  --top-k 40 \
  --stop-at-eos \
  --vary-prompts \
  --use-kv-cache \
  --out-dir out/batch_api_varlen_kv_cache_sft_mixed_binding_cpu
```

关闭 KV cache 的结果：

```text
mode        batch   avg latency   avg tok/s   avg req/s
batched     8       0.6687s       239.28      11.96
sequential  8       0.7619s       210.04      10.50
```

开启 KV cache 的结果：

```text
mode        batch   avg latency   avg tok/s   avg req/s
batched     1       0.0556s       360.11      18.01
batched     2       0.0849s       470.99      23.55
batched     4       0.1138s       703.46      35.17
batched     8       0.1903s       812.14      42.05
sequential  1       0.0529s       378.40      18.92
sequential  2       0.1173s       341.30      17.06
sequential  4       0.2061s       360.67      19.64
sequential  8       0.4403s       363.43      18.17
```

观察：变长 prompt 已经可以进入同一个 batch，并且可以和 KV cache 结合。batch size 为 8 时，关闭 KV cache 的 batched 模式是 11.96 req/s；开启 KV cache 后达到 42.05 req/s，输出吞吐从 239.28 tok/s 提升到 812.14 tok/s。为了验证 mask 和位置修正没有改变真实 token 的语义，做过一次 padded/unpadded 等价性检查，最后一个真实 token 的 logits 最大误差约为 `3.34e-06`。这也解释了真实推理系统为什么需要 bucketing、动态 batching、paged KV cache 和更复杂的调度。

### 3.10 Dynamic Batching

手动 batch serving 需要客户端直接调用 `/generate_batch`，并一次传入多条 prompt。dynamic batching 则保持单请求接口：用户调用 `/generate_dynamic`，服务端把短时间窗口内到达的多个单请求暂存起来，再内部转成一次 batch 生成。

当前教学版实现：

- 等待窗口：5 ms
- 最大 batch size：8
- 只合并采样参数一致的请求
- 内部复用 `/generate_batch` 的 padding mask 和 KV cache 路径

验证时，同时发送 8 个 `/generate_dynamic` 请求，返回结果中的 `dynamic_batch_size` 都为 8，说明这些单请求确实被服务端合并成了一次 batch。接口 `/dynamic_stats` 会返回累计调度统计，包括总请求数、总 batch 数、平均 batch size、平均排队时间、batch latency 和 batch size 分布。

并发 benchmark 对比普通 `/generate` 和 `/generate_dynamic`，两者都开启 KV cache：

```text
endpoint           concurrency   req/s   output tok/s   avg latency   p95 latency   avg batch   avg wait
/generate          1             18.84   376.86         0.0530s       0.0537s      1.00        0.00ms
/generate_dynamic  1             16.19   323.76         0.0617s       0.0632s      1.00        5.73ms
/generate          4             32.98   659.70         0.1196s       0.1249s      1.00        0.00ms
/generate_dynamic  4             35.04   700.89         0.1139s       0.1153s      4.00        5.47ms
/generate          8             17.88   357.57         0.4426s       0.4648s      1.00        0.00ms
/generate_dynamic  8             42.85   857.01         0.1858s       0.1895s      8.00        5.06ms
```

观察：低并发时，dynamic batching 会额外等待约 5 ms，所以单请求延迟略高；高并发时，它把多个请求合并到一次模型 forward，concurrency=8 时平均合批大小达到 8，吞吐从 17.88 req/s 提升到 42.85 req/s，平均延迟也从 0.4426 秒降到 0.1858 秒。这体现了推理系统里的核心取舍：用很小的等待窗口换更高的硬件利用率。

### 3.11 Length Bucketing

变长 prompt 合批时，batch 内所有样本会 pad 到最长 prompt。如果短 prompt 和长 prompt 混在一起，padding token 会变多，注意力计算里就会出现更多无效位置。length bucketing 的思路是：flush 请求时，先按 prompt token 长度排序，再把长度相近的请求放进同一个 batch。

当前实现会在 `/generate_dynamic` 入队时记录 prompt token 长度，并在调度统计中记录：

- batch 内 prompt 长度跨度
- batch padding token 数
- batch padding ratio

使用 `--vary-prompts` 发送不同长度 prompt 的结果：

```text
endpoint           concurrency   req/s   output tok/s   avg latency   p95 latency   avg batch   avg wait   avg padding
/generate_dynamic  4             38.62   617.92         0.1034s       0.1242s      4.00        5.31ms     5.06%
/generate_dynamic  8             47.58   761.28         0.1676s       0.1762s      8.00        5.10ms     12.01%
/generate_dynamic  12            48.49   775.87         0.2152s       0.2469s      6.67        83.72ms    5.69%
```

观察：并发 8 时每次刚好凑成一个满 batch，padding ratio 为 12.01%。并发 12 时，调度器会拆成 8 和 4 两批，长度排序让平均 padding ratio 降到 5.69%，但第二批要等第一批推理结束，所以平均等待时间升到 83.72 ms。这说明 bucketing 可以减少 padding 浪费，但调度器还需要处理“多批串行导致尾部等待”的问题。

### 3.12 Concurrent Batch Workers

为了解决同一次 flush 里多个 batch 串行执行的问题，`/generate_dynamic` 新增了参数：

```bash
python tools/serve/serve_fastapi.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --port 8010 \
  --dynamic-max-concurrent-batches 2
```

它的含义是：同一个 flush 如果拆出多批，最多允许几个 batch 同时进入推理。为了避免事件循环绑定问题，实现上没有长期 worker 或全局 semaphore，而是把 batch 列表按 `max_concurrent_batches` 切成小组，再逐组 `gather`。

缩小版验证结果，设置 concurrency=12、requests=12、max-new-tokens=8：

```text
workers   req/s    output tok/s   avg latency   p95 latency   avg wait
1         67.55    512.24         0.1422s       0.1749s       41.22ms
2         102.73   821.87         0.1095s       0.1152s       3.95ms
```

观察：2 workers 把第二批等待显著压低，平均等待从 41.22 ms 降到 3.95 ms，吞吐也提升了。但这不是无限增加 worker 的理由，因为多个 batch 同时跑会争抢 CPU/GPU 资源；真实系统需要根据硬件、模型大小和请求分布调节并行度。

### 3.13 Adaptive Wait

固定等待窗口的问题是：低压时可能多等，高压时可能又不够灵活。当前加入了可选自适应等待：

```bash
python tools/serve/serve_fastapi.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --port 8010 \
  --dynamic-max-concurrent-batches 2 \
  --dynamic-adaptive-wait \
  --dynamic-min-wait-ms 1 \
  --dynamic-max-wait-ms 8
```

逻辑是：先等最小时间；如果队列还没有达到一个 batch，并且没有超过最大等待时间，就继续短暂等待。实现时还修复了一个调度器生命周期问题：如果当前 flush 正在推理时又来了新请求，flush 结束后会自动启动下一轮，否则这些请求可能留在队列里不被处理。

缩小版 burst 场景对比，设置 concurrency=12、requests=12、max-new-tokens=4：

```text
strategy           req/s    output tok/s   avg latency   p95 latency   avg queue wait   avg flush wait
fixed 5ms          104.34   417.37         0.0958s       0.1130s       3.63ms           5.49ms
adaptive 1-8ms     144.94   567.70         0.0747s       0.0811s       2.10ms           5.59ms
```

观察：这次 burst 场景里 adaptive wait 有更低的平均延迟和更高吞吐，但它不是一定优于固定等待。自适应等待真正要解决的是请求到达不均匀的问题；在稳定高并发、请求几乎同时到达时，固定等待也可能表现很好。

### 3.14 Dynamic Scheduler Benchmark Runner

前面每次比较调度策略，都需要手动启动服务、跑 benchmark、查看 `/dynamic_stats`、再关闭服务。现在新增一个总控脚本：

```text
tools/eval/run_dynamic_scheduler_benchmark.py
```

它会自动完成：

- 按配置启动 FastAPI 服务
- 等待 `/health` 可用
- 调用 `benchmark_api_concurrency.py` 压测 `/generate_dynamic`
- 读取 `/dynamic_stats`
- 关闭当前服务
- 汇总所有配置到 `scheduler_summary.csv` 和 `report.md`

完整运行示例：

```bash
python tools/eval/run_dynamic_scheduler_benchmark.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --configs fixed_w1,fixed_w2,adaptive_w2 \
  --concurrency 4,8,12 \
  --requests-per-level 12 \
  --max-new-tokens 8 \
  --stop-at-eos \
  --use-kv-cache \
  --vary-prompts \
  --out-dir out/dynamic_scheduler_benchmark
```

其中三组内置配置分别是：

```text
fixed_w1      固定等待窗口，1 个 batch worker
fixed_w2      固定等待窗口，2 个 batch worker
adaptive_w2   自适应等待窗口，2 个 batch worker
```

缩小版验证命令：

```bash
python tools/eval/run_dynamic_scheduler_benchmark.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --port 8012 \
  --configs fixed_w1,adaptive_w2 \
  --concurrency 4 \
  --requests-per-level 4 \
  --max-new-tokens 2 \
  --warmup-runs 1 \
  --stop-at-eos \
  --use-kv-cache \
  --vary-prompts \
  --out-dir out/dynamic_scheduler_smoke
```

输出文件：

```text
out/dynamic_scheduler_smoke/fixed_w1/summary.csv
out/dynamic_scheduler_smoke/adaptive_w2/summary.csv
out/dynamic_scheduler_smoke/scheduler_summary.csv
out/dynamic_scheduler_smoke/report.md
```

阶段性部署报告见：

```text
experiments/deployment_report.md
```

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

短期最推荐的下一步：先把当前自写模型做成本地可交互 demo，并补齐 EOS 停止、latency 记录和基础错误处理。
