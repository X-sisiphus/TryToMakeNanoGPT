# 第二阶段总结：从 SFT 到 DPO 的小型研究闭环

## 1. 阶段目标

第二阶段的目标不是把一个小模型训练成“好用的大模型”，而是完整走通现代 LLM 训练流程中最关键的几段：

```text
真实 tokenizer
-> 领域语料 continued pretraining
-> instruction tuning / SFT
-> 结构化评测
-> curriculum 数据设计
-> DPO 偏好优化
-> 结果回测和阶段判断
```

这一阶段最重要的学习目标，是把训练从“loss 降了没有”推进到“模型到底学会了什么、在哪些分布上坏掉、如何用评测解释行为”。

## 2. 已完成内容

### 2.1 Tokenizer 与真实 token 数据

第一阶段主要是字符级训练，适合理解 Transformer，但不适合模拟真实 LLM 训练。

第二阶段接入 GPT-2 BPE tokenizer 后，项目开始支持 token 数据：

```text
data/tiny
data/astro_tiny
```

并补充了：

```text
data_loader.py
batch loader
prepare data stats
tokenizer inspection
dtype / device 处理
manifest 记录
```

这一步让项目从“教学 nanoGPT”进入“可以吃真实文本数据的小模型训练框架”。

关键观察：

```text
tokenizer 会影响有效上下文长度、数字复制难度、领域术语切分和中文文本效率。
```

GPT-2 tokenizer 可以用于学习流程，但如果未来真正训练中文天文/时空智能领域模型，它未必是最合适的 tokenizer。

### 2.2 Continued Pretraining

我们构造了一个小型天文/时空智能相关语料，并训练 token-level language model。

早期训练现象：

```text
token_300:
loss 明显下降，但 sample 仍然碎片化

astro_tiny_300:
领域词如 GNSS / Space geodesy 开始出现，但语义仍不稳定
```

这一步的价值不是生成质量，而是理解：

```text
小数据 + 小模型 + 短训练可以学到 token 分布，但距离稳定领域表达还很远。
```

它为后面的 SFT 提供了一个现实背景：预训练本身不能直接带来可靠任务能力，指令数据和评测体系必须跟上。

### 2.3 SFT 数据与训练

我们实现了：

```text
sft_data.py
train_sft.py
evaluate_sft_quality.py
evaluate_field_accuracy.py
compare_sft_samples.py
```

SFT 数据格式统一为：

```text
Instruction:
...

Input:
...

Answer:
...
```

并使用 label mask 只在 answer 部分计算 loss。

核心技术点：

```text
prompt token 不参与 supervised loss
answer token 参与 supervised loss
END token 也要作为 answer 的一部分被预测
```

这一步之后，项目具备了完整的 instruction tuning 最小闭环：

```text
jsonl 指令数据
-> 编码
-> batch
-> SFT 训练
-> 生成评测
-> 字段级准确率
```

## 3. 结构化抽取研究线

### 3.1 为什么选择字段抽取

我们没有直接做开放问答，而是把任务压成可验证的结构化抽取：

```text
station
signal
value
unit
```

原因是它有清晰评测：

```text
station accuracy
signal accuracy
value accuracy
unit accuracy
all fields accuracy
```

这让每次训练结果都能被解释，而不是只看生成样例的主观好坏。

### 3.2 从失败到 curriculum

普通四字段 copy 直接训练失败：

```text
all fields accuracy: 0.00%
```

后续实验逐步拆解：

```text
value-only
tiny field copy
small field copy
medium field copy
factor field copy
digit-spaced curriculum
normal value repair
```

核心发现：

```text
模型不是不会输出格式，也不是单独不会复制 value。
真正瓶颈是 station / signal / value / unit 的组合泛化。
```

digit-spaced 和 factor curriculum 让模型从失败任务中恢复：

```text
normal full field copy after value repair: 88.00%
```

### 3.3 自然模板与泛化

在 copy curriculum 之后，我们迁移到自然句子抽取：

```text
natural field extraction: 100.00% validation
rich natural field extraction: 100.00%
held-out template extraction: 100.00%
```

这说明模型已经不只是死记固定模板，而能在当前合成模板体系内做一定泛化。

### 3.4 Distractor 鲁棒性

加入多个数字干扰后，模型出现明显断崖：

```text
distractor zero-shot: 13.20%
distractor after training: 88.40%
hard distractor curriculum: 93.80%
```

类型拆解发现最难的是：

```text
previous value
negative statement
network average
```

这一步说明：

```text
模板泛化不等于抗干扰泛化。
```

模型能处理未见模板，不代表能从多个候选数字中稳定选对目标 value。

### 3.5 Multi-station Binding

多 station 场景暴露了新的瓶颈：

```text
multi-station zero-shot: 18.20%
multi-station after multi-only: 62.80%
multi -> mixed low-lr refresh: 70.00%
mixed binding: 71.40%
```

错误分析显示，剩余最大错误不是找不到目标 station，而是：

```text
station 选对了，但 signal/value/unit 绑定成了另一个 station 的 measurement。
```

这个瓶颈可以概括为：

```text
entity-measurement binding
```

当前综合最佳 checkpoint 是：

```text
out/sft_mixed_binding_multi_hard_2200/ckpt.pt
```

综合表现：

```text
multi-station: 71.40%
hard binding: 52.62%
distractor: 95.40%
held-out: 99.00%
```

它不是单项最强模型，而是整体 trade-off 最稳的模型。

## 4. DPO 实验线

### 4.1 DPO 实现

我们实现了：

```text
dpo_data.py
train_dpo.py
scripts/build_field_dpo.py
tools/eval/evaluate_dpo_preference.py
scripts/build_hard_value_dpo.py
```

DPO 数据格式：

```text
prompt -> chosen
prompt -> rejected
```

DPO loss 的核心：

```text
policy_margin = logp_policy(chosen) - logp_policy(rejected)
reference_margin = logp_ref(chosen) - logp_ref(rejected)
loss = -logsigmoid(beta * (policy_margin - reference_margin))
```

关键理解：

```text
reference model 是锚点。
DPO 不是无约束推高 chosen，而是让 policy 相比 reference 更偏好 chosen。
```

### 4.2 Easy Rejected 结果

第一批 rejected 包含：

```text
wrong_station
wrong_signal_group
wrong_value_from_input
wrong_value_same_signal
```

但 preference eval 显示：

```text
SFT baseline preference accuracy: 96.68%
DPO 100 preference accuracy: 96.91%
```

结论：

```text
这批 rejected 对 SFT baseline 太容易，DPO 信号很弱。
```

字段回测：

```text
checkpoint          multi-station   hard-binding   distractor   held-out
SFT mixed binding   71.40%          52.62%         95.40%       99.00%
DPO 100             69.80%          53.62%         95.20%       99.00%
```

DPO 让 hard binding 小涨，但 multi-station 小跌，没有稳定净收益。

### 4.3 Hard Value Rejected

为了构造更有训练价值的偏好数据，我们只替换 value，保持其他字段正确：

```text
station: correct
signal: correct
value: wrong value from input
unit: correct
```

这把 preference 难度明显拉高：

```text
hard value DPO data:
SFT preference accuracy: 75.00%
```

但 DPO 100 仍没有提升 preference accuracy：

```text
SFT baseline: 75.00%
DPO hard value 100: 74.37%
```

字段回测：

```text
checkpoint              multi-station   hard-binding   distractor   held-out
SFT mixed binding       71.40%          52.62%         95.40%       99.00%
DPO hard value 100      69.80%          53.37%         95.80%       99.00%
```

### 4.4 DPO 超参对照

最小 grid：

```text
A: lr 1e-5, beta 0.1, steps 100
B: lr 5e-6, beta 0.1, steps 100
C: lr 1e-5, beta 0.05, steps 100
D: lr 5e-6, beta 0.05, steps 100
```

结果：

```text
checkpoint              preference acc   avg margin
SFT baseline            75.00%           3.1764
A lr1e-5 beta0.1        74.37%           3.4113
B lr5e-6 beta0.1        74.49%           3.2955
C lr1e-5 beta0.05       74.37%           3.4152
D lr5e-6 beta0.05       74.37%           3.2972
```

没有任何一组超过 SFT baseline。

B 组字段回测：

```text
checkpoint              multi-station   hard-binding   distractor   held-out
SFT mixed binding       71.40%          52.62%         95.40%       99.00%
B lr5e-6 beta0.1        70.20%          53.12%         95.60%       99.00%
```

阶段结论：

```text
在当前 6.57M 小模型、当前 hard value DPO 数据和 100 step 训练规模下，
DPO 没有带来稳定净收益。
```

## 5. 关键负结果

这一阶段最有价值的不是某个单项分数，而是几个负结果：

```text
1. 直接 full field copy 失败，不代表模型完全不会，而是任务组合太难。
2. 模板泛化成功，不代表 distractor 鲁棒性成功。
3. hard binding-only 训练会提升硬任务，但破坏旧能力。
4. easy rejected DPO 几乎没有训练信号。
5. hard rejected 变难后，DPO 仍不一定带来净收益。
```

这些负结果共同说明：

```text
小模型训练中，能力不是线性叠加的。
数据难度、训练顺序、任务比例、评测口径都会改变最终行为。
```

## 6. 当前最佳模型与产物

当前推荐 baseline checkpoint：

```text
out/sft_mixed_binding_multi_hard_2200/ckpt.pt
```

核心脚本：

```text
train_sft.py
train_dpo.py
sft_data.py
dpo_data.py
data_loader.py
sample.py
```

关键评测工具：

```text
tools/eval/evaluate_sft_quality.py
tools/eval/evaluate_field_accuracy.py
tools/eval/analyze_field_errors.py
tools/eval/evaluate_dpo_preference.py
tools/eval/summarize_field_scorecard.py
```

关键报告：

```text
experiments/field_copy_curriculum_report.md
experiments/stage2_summary.md
```

## 7. 阶段结论

第二阶段已经完成了一个小型但完整的研究闭环：

```text
构建数据
-> 训练 SFT
-> 发现失败
-> 拆解错误
-> 构造 curriculum
-> 选择综合 checkpoint
-> 实现 DPO
-> 构造偏好数据
-> 做 preference eval
-> 得到 DPO 负结果
```

DPO 暂时收束，不是因为它不重要，而是因为当前实验已经回答了一个明确问题：

```text
在这个小模型和这组结构化抽取任务上，
当前 DPO 数据与超参没有带来稳定净收益。
```

继续深挖 DPO 的下一步应该是：

```text
使用模型真实生成错误构造 rejected
增加模型容量
增加 preference 数据规模
尝试更长训练和更系统的 beta/lr grid
引入更接近真实任务的人工偏好标准
```

但从学习效率看，现在更值得进入第三阶段：

```text
推理、部署、latency / throughput、服务化、量化和 demo。
```

原因是我们已经有了一个可运行 checkpoint 和完整评测体系。部署阶段可以继续复用这些评测，学习模型从“训练产物”变成“可用系统”的过程。

