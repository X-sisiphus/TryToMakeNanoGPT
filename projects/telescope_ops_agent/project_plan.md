# TelescopeOps-Agent 项目计划 v0.1

## 项目定位

项目名：

```text
TelescopeOps-Agent
```

中文名：

```text
面向望远镜系统运维日志的多工具诊断 Agent
```

一句话目标：

```text
给定观测日志、设备状态、天气记录和历史故障案例，系统能够定位异常、调用合适工具、给出证据链和诊断建议。
```

这个项目不是单纯 RAG。RAG 只是 agent 的一个基础工具。完整路线是：

```text
Level 1: RAG
检索手册、日志、历史案例并回答问题。

Level 2: Tool-Augmented QA
能调用日志查询、时间序列查询、天气查询、历史案例检索等工具。

Level 3: Diagnostic Agent
能规划诊断步骤，决定先查什么、后查什么，并基于中间结果继续行动。

Level 4: OPD Agent
让小模型 agent 自己诊断，再由强 teacher 修正它的错误轨迹，用 On-Policy Distillation 提升诊断能力。
```

## 为什么做这个

这个方向适合当前背景：

- 和望远镜系统、观测系统、光学方向有直接联系。
- 和大模型 agent、RAG、工具调用、评估、OPD 都能自然结合。
- 比普通领域问答更实用，因为它围绕“异常诊断”而不是“泛泛问答”。
- 可以做成可演示系统，也可以做成有研究味道的评测项目。

核心研究问题：

```text
在望远镜运维诊断任务中，RAG、工具调用和 agent 规划分别带来多少收益？
On-Policy Distillation 能否提升小模型 agent 在故障诊断中的工具选择、证据引用和最终诊断准确率？
```

## 任务定义

输入：

```text
用户问题：
昨晚 02:10 到 02:40 观测质量下降，请帮我定位可能原因。

可用数据：
1. 观测日志
2. 设备状态日志
3. CCD/制冷/电源/圆顶/指向系统记录
4. 天气和 seeing 记录
5. 观测计划
6. 历史故障案例
7. 维护手册或说明文档
```

输出：

```json
{
  "diagnosis": "可能由 CCD 温度异常和湿度升高共同导致观测质量下降",
  "fault_type": "detector_temperature_or_environment",
  "time_window": "02:10-02:40",
  "affected_subsystem": ["CCD", "weather"],
  "evidence": [
    {
      "source": "device_log",
      "time": "02:17",
      "text": "CCD temperature rose from -80C to -61C"
    },
    {
      "source": "weather_log",
      "time": "02:20",
      "text": "humidity increased to 88%"
    }
  ],
  "recommended_actions": [
    "检查 CCD 制冷状态",
    "检查湿度阈值和圆顶开合记录",
    "对比历史相似故障案例"
  ],
  "uncertainty": "缺少制冷机电流记录，无法确认是否为硬件故障"
}
```

## 数据设计

第一版先不依赖真实望远镜日志，先做一个可控 benchmark。后续如果能拿到脱敏真实日志，再替换或混合。

数据分三层：

```text
Layer 1: 合成日志
构造可控故障场景，明确 gold diagnosis 和 evidence。

Layer 2: 半真实文档
使用公开望远镜/仪器手册、观测说明、天气影响说明，作为 RAG 文档。

Layer 3: 脱敏真实日志
如果导师或实验室允许，加入真实观测日志、报警记录、运维工单。
```

第一版故障类型：

```text
1. weather_humidity
2. seeing_degradation
3. ccd_temperature
4. dome_tracking
5. pointing_error
6. focus_drift
7. power_or_network
8. schedule_conflict
```

样本字段 schema：

```json
{
  "case_id": "case_0001",
  "question": "昨晚 02:10 到 02:40 观测质量下降，请定位原因。",
  "logs": [
    {"source": "observation_log", "time": "02:12", "text": "..."},
    {"source": "weather_log", "time": "02:20", "text": "..."}
  ],
  "gold": {
    "fault_type": "ccd_temperature",
    "time_window": "02:10-02:40",
    "affected_subsystem": ["CCD"],
    "evidence_ids": ["log_03", "log_07"],
    "diagnosis": "CCD 制冷异常导致图像质量下降"
  }
}
```

## 系统路线

### Baseline 0：Direct LLM

直接把问题和相关日志塞给模型，不做检索和工具调用。

用途：

```text
看模型在无系统设计时能做到什么程度。
```

### Baseline 1：Naive RAG

把日志和手册切块，向量检索 top-k，再让模型回答。

用途：

```text
验证 RAG 是否提升证据命中率和诊断准确率。
```

### Baseline 2：RAG + Reranker

先向量检索，再用 reranker 排序证据。

用途：

```text
减少检索噪声，观察 evidence precision 是否提升。
```

### System 1：Tool-Augmented QA

提供工具：

```text
search_logs(time_window, keywords)
query_timeseries(metric, start, end)
search_manual(query)
retrieve_cases(query)
summarize_evidence(evidence)
```

模型根据问题选择工具，但流程可以半固定。

### System 2：Diagnostic Agent

Agent 自己规划：

```text
1. 识别问题中的时间窗口和症状
2. 决定先查哪些日志
3. 根据结果决定是否查天气、设备状态、历史案例
4. 汇总证据
5. 输出诊断和不确定性
```

### System 3：OPD Agent

流程：

```text
1. student agent 自己完成诊断轨迹
2. teacher 检查工具选择、证据、诊断是否正确
3. 生成 correction trace
4. 把 student 自己犯错的轨迹转成训练数据
5. 继续训练或偏好优化 student
6. 和非 OPD agent 对比
```

## OPD 设计

这里的 OPD 先做工程可行版本：

```text
On-policy correction distillation
```

不要求 teacher 提供 token-level logits，而是让 teacher 给结构化反馈：

```json
{
  "student_trace_id": "trace_0001",
  "error_types": [
    "wrong_tool_order",
    "missed_weather_evidence",
    "unsupported_diagnosis"
  ],
  "correction": {
    "better_tool_sequence": [
      "search_logs",
      "query_timeseries",
      "search_manual",
      "retrieve_cases"
    ],
    "missing_evidence": ["weather_log_02:20"],
    "correct_diagnosis": "humidity-related degradation"
  }
}
```

训练数据形式：

```text
输入：问题 + student 错误轨迹 + teacher 反馈
输出：修正后的诊断轨迹 / 最终诊断
```

## 评估指标

诊断结果：

```text
fault_type accuracy
affected_subsystem F1
time_window IoU / accuracy
diagnosis correctness
```

证据质量：

```text
evidence recall
evidence precision
citation accuracy
unsupported claim rate
```

Agent 行为：

```text
tool call accuracy
tool order accuracy
average tool calls
redundant tool call rate
failed tool call rate
```

OPD 特有指标：

```text
self-generated error recovery rate
trace correction accuracy
post-OPD improvement over pre-OPD
```

## 对比和消融

主对比：

```text
Direct LLM
Naive RAG
RAG + Reranker
Tool-Augmented QA
Diagnostic Agent
Diagnostic Agent + OPD
```

消融实验：

```text
去掉 RAG 文档
去掉天气工具
去掉时间序列工具
去掉历史案例检索
去掉 reranker
去掉 evidence constraint
去掉 OPD correction
不同 student 模型大小
不同 teacher 模型
不同 OPD 轮数
```

## 第一周计划

第一周只做最小闭环，不追求复杂 agent。

### Day 1：任务和 schema 固化

输出：

```text
projects/telescope_ops_agent/project_plan.md
projects/telescope_ops_agent/schema.md
```

### Day 2：构造第一版合成日志数据

目标：

```text
20 个 case
每个 case 5-15 条日志
每个 case 有 gold fault_type、evidence_ids、diagnosis
```

### Day 3：写数据检查脚本

检查：

```text
字段是否完整
evidence_id 是否存在
fault_type 是否在枚举内
时间格式是否规范
```

### Day 4：实现 Direct LLM / rule baseline

先不接真实大模型也可以，用规则 baseline 打通评估。

### Day 5：实现评估脚本

指标：

```text
fault_type accuracy
evidence recall
affected_subsystem F1
```

### Day 6-7：做第一版 RAG

用日志文本和手册片段建立检索，比较：

```text
Direct baseline vs RAG baseline
```

## 近期不做什么

为了避免项目一开始失控，先不做：

```text
不接真实望远镜控制系统
不做自动执行运维动作
不让 agent 修改设备状态
不一开始就训练大模型
不一开始就做完整 OPD
```

第一阶段只做：

```text
可控数据 + 可复现评估 + RAG/Agent 雏形
```

等评估闭环稳定后，再进入：

```text
小模型 SFT / DPO / OPD
```
