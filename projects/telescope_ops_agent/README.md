# TelescopeOps-Agent

面向望远镜系统运维日志的多工具诊断 Agent。

当前阶段目标不是马上训练模型，而是先建立：

```text
可控数据
可复现评估
RAG baseline
Tool-Augmented QA
Agent 诊断轨迹
OPD 扩展路线
```

核心文档：

```text
project_plan.md
schema.md
```

项目路线：

```text
RAG -> Tool-Augmented QA -> Diagnostic Agent -> OPD Agent
```

第一周最小闭环：

```text
1. 固化 schema
2. 构造 20 个合成诊断 case
3. 写数据检查脚本
4. 写评估脚本
5. 跑 Direct baseline 和 RAG baseline
```
