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
