# 四字段复制 Curriculum 实验报告

## 1. 问题背景

这个阶段的目标是让一个小型 GPT 模型完成结构化四字段复制任务：

```text
station: ...
signal: ...
value: ...
unit: ...
```

最初直接训练普通四字段 copy 任务时，模型几乎失败：

```text
station accuracy: 10.00%
signal accuracy: 20.00%
value accuracy: 0.00%
unit accuracy: 66.00%
all fields accuracy: 0.00%
```

如果只看这个结果，很容易得到一个粗糙结论：小模型能力不够。但后续实验表明，这个判断太早了。真正的问题不是“模型完全不会”，而是训练任务同时包含了多个困难：

- 四字段输出格式
- station / signal / unit 的字段绑定
- value 的数字复制
- 多字段组合泛化
- GPT-2 BPE 对普通数字的切分不稳定

因此，这个阶段的核心不是单纯调参数，而是把失败任务拆开，逐步定位瓶颈。

## 2. 核心方法：Curriculum Learning

Curriculum learning 的思想是先给模型更简单的任务，再逐步过渡到真实任务。

在这个项目中，课程设计大致是：

```text
value-only copy
-> tiny field copy
-> small field copy
-> factor field copy
-> full digit-spaced field copy
-> normal field copy
-> normal value repair
```

这里的关键技巧是 digit-spaced value：

```text
38.5 -> 3 8 . 5
-10.5 -> - 1 0 . 5
```

这样做没有改 tokenizer，而是在文本进入 tokenizer 之前降低数字复制难度。原本 GPT-2 BPE 可能把不同上下文里的数字切成不同 token 组合；digit-spaced 让数字更接近字符级复制任务。

代码层面的核心变化是：

```python
def space_value(value):
    return " ".join(list(str(value)))
```

然后在数据构造时同时改 input 和 output：

```python
inputText = template.format(
    station=station,
    signal=signal,
    value=spacedValue,
    unit=unit,
)

outputText = (
    f"station: {station}\n"
    f"signal: {signal}\n"
    f"value: {spacedValue}\n"
    f"unit: {unit}"
)
```

## 3. 实验链路与结果

### 3.1 Value-only 实验

最早普通 value copy 很差：

```text
normal value_copy_500:
value accuracy: 6.00%
```

使用 digit-spaced 后：

```text
digit-spaced value_copy_500:
value accuracy: 50.00%
```

再从 digit-spaced checkpoint 迁移回普通 value：

```text
digit-spaced -> normal curriculum:
value accuracy: 80.00%
```

这一步说明：小模型不是完全不会复制数字，数字 tokenization 和训练路径会显著影响结果。

### 3.2 Full field copy 直接训练失败

直接训练 500 条 digit-spaced 四字段 copy，loss 虽然下降，但生成时出现常见答案坍缩：

```text
station accuracy: 10.00%
signal accuracy: 20.00%
value accuracy: 0.00%
unit accuracy: 66.00%
all fields accuracy: 0.00%
```

典型现象是模型反复生成某些常见字段组合，而不是按 input 复制。

这说明 loss 降低不等于结构化抽取正确。对于这种任务，字段级评测比单看 loss 更重要。

### 3.3 Tiny / Small 过拟合实验

为了判断模型是否具备四字段复制能力，先做极小数据：

```text
20 条 tiny field copy:
all fields accuracy: 100.00%
```

然后扩大到 100 条：

```text
100 条 small field copy:
validation all fields accuracy: 100.00%
```

这两步说明：模型会四字段格式，也能在小规模数据上学会复制。最初 500 条失败不是因为模型完全没有能力，而是任务跨度太大。

### 3.4 Medium 250 暴露断点

扩大到 250 条后：

```text
medium 250:
validation all fields accuracy: 60.00%
station accuracy: 88.00%
signal accuracy: 92.00%
value accuracy: 72.00%
unit accuracy: 100.00%
```

断点出现在 100 到 250 之间。value 最先明显下降，但这还不能说明 value 本身是根因。

### 3.5 Factor 实验定位瓶颈

为了判断哪个因素压垮模型，构造三组单因素实验：

```text
station-only expansion:
all fields accuracy: 92.00%

signal-only expansion:
all fields accuracy: 88.00%

value-only expansion:
all fields accuracy: 100.00%
```

这个结果很关键：单独扩大 value 并不会压垮模型。

随后做双因素实验：

```text
station + signal:
all fields accuracy: 72.00%

signal + value:
all fields accuracy: 68.00%

full medium 250:
all fields accuracy: 60.00%
```

结论是：瓶颈不是单字段复制，而是跨字段组合泛化。value 看起来最常错，但根因是 station / signal / value 同时组合时，模型把组合关系混淆了。

### 3.6 Factor Curriculum 修复 full 500

把 factor 实验作为中间课程，再回到 full 500 digit-spaced：

```text
full 500 direct:
all fields accuracy: 0.00%

full 500 after factor curriculum:
all fields accuracy: 90.00%
station accuracy: 94.00%
signal accuracy: 100.00%
value accuracy: 96.00%
unit accuracy: 100.00%
```

这是这个阶段最重要的结果。它证明了任务失败可以通过更合理的 curriculum 修复。

### 3.7 回到普通数字格式

从 digit-spaced full 500 checkpoint 迁移回普通数字：

```text
normal full field copy after digit-spaced curriculum:
all fields accuracy: 76.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 76.00%
unit accuracy: 100.00%
```

字段结构已经完全迁移成功，剩余错误集中在普通数字 value 上。

### 3.8 Normal Value Repair

最后做普通 value 修复：

```text
after value repair:
all fields accuracy: 80.00%

after low-lr field refresh:
all fields accuracy: 88.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 88.00%
unit accuracy: 100.00%
```

最终普通四字段 copy 从 0% 提升到 88%。

### 3.9 迁移到自然句子抽取

在普通四字段 copy 稳定后，继续迁移到自然语言模板：

```text
YEBES40M has a reported zenith wet delay of 38.5 mm in the latest solution.
For TSKB, the estimated tropospheric delay equals 18.5 ps.
Station WETTZELL shows a clock bias of 1.2 ns from space geodetic observations.
```

从 `normal field copy` checkpoint 初始化后：

```text
validation:
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%

train:
station accuracy: 99.56%
signal accuracy: 99.11%
value accuracy: 98.00%
unit accuracy: 100.00%
all fields accuracy: 96.67%
```

这一步说明前面的 copy curriculum 不只是让模型记住固定格式，也能作为自然 field extraction 的初始化。需要注意的是，验证集 100% 不代表真实泛化彻底解决，因为当前自然模板仍然来自有限模板集合，且这个 split 下验证样本可能更容易。

### 3.10 Rich 自然模板扩展

为了测试模型是否只适应原来的 5 个自然模板，进一步构造 20 个自然语言模板、1000 条样本，例如：

```text
The zenith wet delay entry for station YEBES40M reads 38.5 mm.
At YEBES40M, the processing chain found tropospheric delay to be 12.0 ps.
TSKB -- tropospheric delay: 18.5 ps.
In the geodetic report, GOLD is associated with 52.2 mm of zenith wet delay.
```

从自然字段抽取 checkpoint 继续训练后：

```text
validation:
exact match: 100.00%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%

train:
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 100.00%
unit accuracy: 100.00%
all fields accuracy: 100.00%
```

这说明在模板生成分布内，模型已经稳定掌握自然字段抽取。但这仍不等于真实开放文本泛化，因为训练和验证来自同一模板集合。下一步更严格的评测应该使用 held-out templates。

### 3.11 Held-out Template 评测

为了判断模型是否只是记住模板，进一步构造 held-out template 实验：

```text
total rich templates: 20
train templates: 0-15
held-out templates: 16-19
```

训练集只包含前 16 个模板，评测集只包含后 4 个未见模板。模型从 5 模板自然字段抽取 checkpoint 初始化，在训练模板上训练 1000 steps。

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

这个结果比普通 rich train/val 更有说服力：在当前合成模板体系内，模型确实学到了字段抽取规则，而不是只记住训练模板。

### 3.12 多数字干扰鲁棒性

Held-out template 说明模型能处理未见句式，但它仍然可能不具备真实文本中的抗干扰能力。为此构造 distractor 数据：每条样本只有一个目标 value，但 input 中包含多个无关数字，例如 previous solution、network average、epoch、uncertainty、iteration count。

例子：

```text
At YEBES40M, zenith wet delay equals 38.5 mm. The previous solution listed 52.2 mm, and the quality flag is 1.

The tropospheric delay at HOBART12 is not 12.0 ps; the accepted estimate is 8.0 ps after 3 iterations.
```

先用 held-out template 模型直接评测：

```text
zero-shot distractor:
exact match: 13.20%
station accuracy: 23.20%
signal accuracy: 72.80%
value accuracy: 42.20%
unit accuracy: 94.20%
all fields accuracy: 13.20%
```

这说明模板泛化不等于抗干扰泛化。模型能处理未见模板，但遇到多个数字时，会抓错字段、抓错 value，甚至破坏输出格式。

经过 distractor training 后：

```text
distractor all:
exact match: 88.40%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 88.40%
unit accuracy: 100.00%
all fields accuracy: 88.40%
```

回测 held-out template：

```text
held-out template after distractor training:
exact match: 97.50%
station accuracy: 100.00%
signal accuracy: 100.00%
value accuracy: 97.50%
unit accuracy: 100.00%
all fields accuracy: 97.50%
```

这一步说明 distractor curriculum 可以显著提升抗干扰能力，但会对原本简单模板分布带来轻微 trade-off。剩余错误依然集中在 value，特别是从多个候选数字中选错目标数字。

### 3.13 干扰类型拆解

进一步把 distractor 拆成五类：

```text
previous: 当前值 + previous / last week value
negative: not / reject value + accepted value
network: station value + network average
uncertainty: target value + formal uncertainty
metadata: target value + window / samples / epoch 等元数据数字
```

held-out 模型 zero-shot：

```text
previous: all 16.50%, value 30.50%
negative: all 37.00%, value 49.00%
network: all 28.00%, value 42.00%
uncertainty: all 39.50%, value 51.50%
metadata: all 85.50%, value 97.00%
```

distractor training 后：

```text
previous: all 53.00%, value 53.00%
negative: all 52.00%, value 52.00%
network: all 63.50%, value 63.50%
uncertainty: all 90.50%, value 90.50%
metadata: all 99.00%, value 99.00%
```

这个结果说明，模型并不是简单地怕“句子里有多个数字”。metadata 数字对它影响很小，因为这些数字的语义角色明显不是目标测量值。真正困难的是 previous、negative 和 network：它们都包含同单位、同 signal 空间里的合法候选 value。

因此，剩余瓶颈可以更精确地表述为目标选择能力：模型需要依据语义线索选择 accepted/current/station-specific value，并忽略 previous、rejected、network-average value。

### 3.14 Hard Distractor Curriculum

基于类型拆解结果，继续只针对最难的 previous、negative、network 三类做训练。每类构造 300 条样本，合并为 900 条 hard distractor 训练集。

hard-types 训练前后对比：

```text
previous: 53.00% -> 99.00%
negative: 52.00% -> 96.00%
network: 63.50% -> 99.00%
uncertainty: 90.50% -> 95.00%
metadata: 99.00% -> 99.00%
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

这说明，针对已定位瓶颈类型做 curriculum，比泛泛增加数据更有效。模型开始学会利用 `current`、`accepted estimate`、`station has` 等语义线索来选择目标 value，同时忽略 previous、rejected 和 network-average value。

### 3.15 Multi-station Binding

继续测试同一句多 station / 多测量值绑定。每条样本包含两个 station 和两组 measurement，并明确指定 requested / target station。

zero-shot：

```text
multi-station:
exact match: 18.20%
station accuracy: 54.80%
signal accuracy: 51.60%
value accuracy: 38.20%
unit accuracy: 66.60%
all fields accuracy: 18.20%
```

这说明，即使模型已经具备 held-out template 和 hard distractor 能力，遇到两个 station 时仍然容易抽错 measurement 组。

在 multi-station 数据上训练后：

```text
multi-station:
exact match: 62.80%
station accuracy: 91.80%
signal accuracy: 72.00%
value accuracy: 70.00%
unit accuracy: 78.60%
all fields accuracy: 62.80%

distractor 回测:
all fields accuracy: 81.20%
value accuracy: 82.40%

held-out template 回测:
all fields accuracy: 100.00%
```

这一步暴露了新的 trade-off：multi-station 绑定能力可以被训练起来，但会削弱之前 hard distractor 获得的 value 干扰鲁棒性。新的瓶颈可以称为 entity-measurement binding，即如何把 station、signal、value、unit 作为一组绑定起来，而不是只识别局部字段。

### 3.16 Mixed Hard Distractor + Multi-station

为了同时保住 hard distractor 和 multi-station 能力，构造混合训练集：

```text
hard distractor: 900
multi-station: 500
total: 1400
```

从 hard distractor checkpoint 初始化后训练，结果如下：

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

这个结果是一个有价值的负结果。简单拼接数据保住了 hard distractor 能力，但没有充分学会 multi-station binding。原因可能是 hard distractor 样本更多，且初始化 checkpoint 已经偏向 hard distractor 解法；multi-station 绑定需要更高比例或更专门的阶段式训练。

这一步提醒：多能力训练不是把数据拼起来就自动解决，采样比例、训练顺序和学习率都会影响最终能力分布。

### 3.17 Multi-station -> Mixed Low-lr Refresh

基于 3.16 的负结果，继续尝试阶段式训练：

```text
Stage 1:
hard distractor -> multi-station

Stage 2:
multi-station checkpoint -> mixed hard+multi low-lr refresh
```

具体做法是从 `out/sft_multi_station_500_from_hard/ckpt.pt` 初始化，使用同一个 hard distractor + multi-station 混合数据集，但把学习率降到 `1e-4`，训练 800 step。

结果如下：

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

这一步说明，训练顺序比“是否混合数据”更关键。先让模型集中学习 multi-station binding，再用低学习率混合刷新，可以在保住 hard distractor 的同时，把 multi-station 从 62.80% 推到 70.00%。

但这还不是彻底解决。multi-station 的剩余错误说明模型仍然没有稳定掌握“同一句多个实体时，把 station、signal、value、unit 绑定成同一组 measurement”的规则。当前 frontier 可以记录为：multi-station binding 70.00%，distractor 93.80%，held-out template 99.00%。

### 3.18 Multi-station Error Analysis

为了避免盲目继续加数据，对 multi-station 错误做字段级归因。新增分析工具：

```bash
python tools/eval/analyze_field_errors.py \
  --field-accuracy out/field_accuracy_multi_station_500_after_refresh/field_accuracy.csv \
  --out-dir out/field_error_multi_station_after_refresh
```

错误类型定义：

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

三组结果对比：

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

这说明 staged refresh 的提升不是偶然的：它同时降低了“选错 station”和“station 对但 measurement 绑定错”两类错误。

但最大残留错误仍然是 `target_station_wrong_measurement`。也就是说，模型多数时候已经能读懂“目标 station 是谁”，但还不能稳定把这个 station 后面的 signal/value/unit 作为一个整体绑定起来。

下一步更合理的课程不是继续扩大普通 multi-station 数据，而是构造 hard binding 数据：

```text
same signal, different value
same unit, different signal
nearby numeric values
target station sometimes first, sometimes second
explicit contrastive wording
```

这一步把问题从“模型抽取能力不足”进一步缩小到“实体与 measurement 组的绑定不足”。

### 3.19 Hard Binding Curriculum

基于 3.18 的错误分析，构造 hard binding 数据：

```text
same signal, different value
same unit, different signal
nearby numeric values
target station first / second position reversal
explicit contrastive wording
```

新增脚本：

```bash
python scripts/build_hard_binding_field_sft.py \
  --out data/sft/field_hard_binding_800.jsonl \
  --num-examples 800 \
  --seed 1337
```

zero-shot 结果：

```text
hard binding zero-shot:
all fields accuracy: 40.12%
```

从 `multi -> mixed low-lr refresh` checkpoint 单独训练 hard binding 后：

```text
hard binding after hard-only:
all fields accuracy: 60.50%

multi-station after hard-only:
all fields accuracy: 59.00%

distractor after hard-only:
all fields accuracy: 89.00%
```

这是一个重要负结果。hard-only 训练确实让 hard binding 自身从 40.12% 提升到 60.50%，但同时破坏了原 multi-station 和 distractor 能力。说明“更难的数据”不能直接作为单独课程灌给小模型，否则会把模型推向一个更窄的新分布。

随后构造混合集：

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

混合刷新结果：

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

multi-station 错误类型对比：

```text
after refresh:
target_station_wrong_measurement: 21.20%
wrong_station_or_record: 6.40%

after mixed binding:
target_station_wrong_measurement: 21.00%
wrong_station_or_record: 5.00%
```

这个结果是温和正结果。hard binding 自身没有 hard-only 高，但没有造成能力崩塌；multi-station 和 distractor 反而略有提升。它说明当前更合适的做法不是单独 hard binding 训练，而是把 hard binding 作为混合巩固样本，和旧能力一起复习。

### 3.20 Field Extraction Scorecard

为了收束这个 mini research，新增 scorecard 脚本，从多个 `field_accuracy/report.md` 中自动抽取指标：

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

关键结果：

```text
checkpoint              multi-station   hard-binding   distractor   held-out
multi_zero              18.20%          -              -            -
multi_only              62.80%          -              81.20%       100.00%
simple_mixed            48.40%          -              93.60%       99.00%
staged_refresh          70.00%          -              93.80%       99.00%
hard_binding_only       59.00%          60.50%         89.00%       98.00%
mixed_binding           71.40%          52.62%         95.40%       99.00%
```

当前综合最好的 checkpoint 是：

```text
out/sft_mixed_binding_multi_hard_2200/ckpt.pt
```

它不是 hard binding 单项最高，但整体 trade-off 最好：

```text
multi-station: 71.40%
distractor: 95.40%
held-out: 99.00%
hard binding: 52.62%
```

这一步把阶段结论从“某个任务提升了多少”推进到“如何选择综合最优模型”。这也是后续 DPO / preference optimization 的前提：必须先有稳定评测集和 baseline checkpoint，否则无法判断偏好优化到底是在改善模型，还是在破坏已有能力。

### 3.21 DPO Preference Optimization Smoke Test

基于当前最优 checkpoint：

```text
out/sft_mixed_binding_multi_hard_2200/ckpt.pt
```

构造 DPO 偏好数据：

```text
chosen: 正确字段抽取答案
rejected: 人工构造的错误答案
```

rejected 类型分布：

```text
wrong_signal_group: 552
wrong_station: 534
wrong_value_from_input: 568
wrong_value_same_signal: 546
```

DPO 100 step 训练日志：

```text
step 0: train loss 0.6931, val loss 0.6931
step 90: train loss 0.6879, val loss 0.6742, val pref acc 90.00%
```

step 0 的 0.6931 是正确起点，因为 policy 和 reference 初始完全相同：

```text
loss = -logsigmoid(0) = 0.6931
```

字段回测结果：

```text
checkpoint          multi-station   hard-binding   distractor   held-out
SFT mixed binding   71.40%          52.62%         95.40%       99.00%
DPO 100             69.80%          53.62%         95.20%       99.00%
```

结论：DPO 100 是温和 mixed result。它优化了偏好目标，hard binding 小幅提升，但 multi-station 小幅下降，distractor 基本持平，held-out 不变。

这说明 DPO 的效果必须通过独立任务评测判断，不能只看 DPO loss 或 preference accuracy。当前偏好数据更像是在修 hard binding/value preference，而不是全面提升结构化抽取能力。

### 3.22 DPO Preference Evaluation by Type

为了判断 DPO 到底学会了哪类偏好，新增脚本：

```text
tools/eval/evaluate_dpo_preference.py
```

该脚本不生成文本，只计算：

```text
logp(chosen answer)
logp(rejected answer)
margin = chosen_logp - rejected_logp
```

然后按 `preference_type` 汇总。

结果：

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

这说明 DPO 100 确实增强了一点 chosen/rejected margin，但提升很小。关键原因是：SFT baseline 已经能以 96.68% 的准确率选中 chosen，所以当前 rejected 对模型来说太容易，DPO 训练信号不够强。

这也解释了 3.21 的 mixed result：DPO loss 在下降，preference accuracy 看起来不错，但独立字段 scorecard 没有明显提升。下一步如果继续做 DPO，需要构造 harder rejected，而不是继续增加同类 easy preference。

## 4. 关键技术收获

### 4.1 不要只看 loss

很多实验里 loss 降得很顺，但字段准确率仍然很差。结构化任务必须做任务级评测：

```text
station accuracy
signal accuracy
value accuracy
unit accuracy
all fields accuracy
```

这比只看 train loss / val loss 更能解释模型行为。

### 4.2 Tokenizer 会改变任务难度

普通数字在 GPT-2 BPE 下可能被切成不同 token 组合。小模型在这种情况下容易出现数值混淆：

```text
25.0 -> 12.0
1.2 -> 5.2
4.7 -> 4.4
```

digit-spaced 并不是最终目标，而是训练课程中的中间表示。

### 4.3 小模型的瓶颈经常是组合泛化

单独复制 station、signal、value 都不一定难。难的是：

```text
station / signal / value / unit 同时变化
```

模型需要学的不只是字段名，还包括字段之间的绑定关系和输入到输出的组合映射。

### 4.4 Curriculum 不是玄学，而是实验控制

这个阶段的 curriculum 不是随便“从简单到困难”，而是通过实验逐步定位：

```text
能否过拟合 20 条？
能否学会 100 条？
250 条为什么掉？
是 station、signal 还是 value？
双因素是否更难？
回 full 500 是否改善？
普通数字是否还能迁移？
```

每一步都在回答一个具体问题。

## 5. 阶段结论

这个阶段没有把普通四字段 copy 做到 100%，但核心问题已经解决：

```text
普通 full field copy direct: 0.00%
digit-spaced factor curriculum: 90.00%
normal full field copy after curriculum: 76.00%
normal full field copy after value repair: 88.00%
natural field extraction after copy curriculum: 100.00% validation
rich natural field extraction: 100.00% validation
held-out template extraction: 100.00%
distractor zero-shot: 13.20%
distractor after training: 88.40%
distractor type hardest cases: previous / negative / network
hard distractor curriculum: 93.80%
multi-station zero-shot: 18.20%
multi-station after training: 62.80%
mixed hard + multi: multi-station 48.40%, distractor 93.60%
multi -> mixed low-lr refresh: multi-station 70.00%, distractor 93.80%, held-out 99.00%
multi-station remaining largest error: target_station_wrong_measurement 21.20%
hard binding zero-shot: 40.12%
hard binding after hard-only: 60.50%, but multi-station drops to 59.00%
mixed binding: hard binding 52.62%, multi-station 71.40%, distractor 95.40%
current best checkpoint: out/sft_mixed_binding_multi_hard_2200/ckpt.pt
DPO 100: hard binding 53.62%, multi-station 69.80%, distractor 95.20%, held-out 99.00%
DPO preference eval: SFT 96.68%, DPO 96.91%
```

最终结论：

```text
对于小模型结构化字段抽取，直接训练失败主要来自组合泛化和数字 tokenization。
通过 digit-spaced curriculum、factor curriculum 和 normal value repair，
可以把普通四字段 copy 从 0% 提升到 88%，进一步迁移到自然模板抽取、rich 模板抽取和 held-out 模板抽取，并通过 distractor curriculum 提升抗干扰能力。
针对 previous / negative / network 的 hard distractor curriculum 又能把整体 distractor 从 88.40% 提升到 93.80%。
多 station 场景暴露了新的 entity-measurement binding 瓶颈，单独训练 multi-station 会提升绑定能力，但会削弱 distractor 鲁棒性。
简单混合训练保住了 distractor，却没有充分提升 multi-station，说明多能力训练需要控制数据比例和训练顺序。
multi-station -> mixed low-lr refresh 比 simple mixed 更有效，说明 curriculum 的核心不是把数据都放进去，而是控制模型先学什么、后巩固什么。
错误分析显示，剩余瓶颈不是“找不到目标 station”，而是“station 对了但 measurement 组绑定错了”。
hard binding 说明难样本不能直接单独灌入；对小模型来说，难样本更适合作为混合复习的一部分。
当前最优模型不是单项最高模型，而是 trade-off 最稳的 mixed binding checkpoint。
DPO 可以优化偏好目标，但不保证主任务 accuracy 全面提升，必须用独立 scorecard 回测。
DPO 数据如果对 SFT baseline 已经过于容易，DPO 只能带来很弱的边际收益。
```

剩余 12% 主要是普通数字 value 的细粒度混淆。继续追 100% 可以做，但学习收益已经低于进入下一阶段。

## 6. 下一步建议

这个阶段建议收束，不再继续死磕最后 12%。更值得继续的是：

- 把当前实验链整理成正式技术报告
- 拆解 distractor 类型，例如 previous value、negative statement、network average、uncertainty
- 针对 previous / negative / network 做更细的 contrastive curriculum
- 混合 hard distractor 和 multi-station 数据，减少能力互相覆盖
- 构造 hard binding curriculum，专门处理 station 对但 measurement 绑定错的问题
- 尝试更强 tokenizer 或数字专用表示
- 尝试更大模型或更长训练，观察是否自然缓解 value 混淆
- 进入 DPO / 偏好优化前，先建立稳定的结构化评测集

对于学习目标来说，这个阶段最重要的收获不是 88% 这个数字，而是完整走通了一次小型研究流程：

```text
发现失败
-> 拆解变量
-> 设计对照实验
-> 定位瓶颈
-> 设计 curriculum
-> 回到真实任务
-> 分析剩余误差
```
