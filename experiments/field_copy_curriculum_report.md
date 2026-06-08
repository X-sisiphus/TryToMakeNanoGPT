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
```

最终结论：

```text
对于小模型结构化字段抽取，直接训练失败主要来自组合泛化和数字 tokenization。
通过 digit-spaced curriculum、factor curriculum 和 normal value repair，
可以把普通四字段 copy 从 0% 提升到 88%，并进一步迁移到自然模板抽取。
```

剩余 12% 主要是普通数字 value 的细粒度混淆。继续追 100% 可以做，但学习收益已经低于进入下一阶段。

## 6. 下一步建议

这个阶段建议收束，不再继续死磕最后 12%。更值得继续的是：

- 把当前实验链整理成正式技术报告
- 增加更多自然语言模板，验证是否仍能保持高准确率
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
