# 第三阶段部署实验报告

本阶段的目标是把已经训练好的小型 nanoGPT checkpoint 从“能离线采样”推进到“能作为本地服务被调用”，并初步观察推理服务中的几个关键性能指标。

本次使用的 checkpoint：

```text
out/sft_mixed_binding_multi_hard_2200/ckpt.pt
```

模型规模约 6.57M 参数，使用 GPT-2 tokenizer，运行设备为 CPU。服务接口基于 FastAPI 实现，测试任务是字段抽取：

```text
Input:
ONSA reports vertical velocity of 2.4 mm/yr.

Answer:
station: ONSA
signal: vertical velocity
value: 2.4
unit: mm/yr
```

## 实验内容

本阶段主要完成了七类实验。第一类是直接测试模型内部生成速度，用来观察纯生成性能。第二类是测试 HTTP API 单请求延迟，用来观察服务封装带来的额外开销。第三类是并发请求测试，用来观察多个请求同时到达时吞吐和延迟的变化。第四类是上下文长度测试，用来观察 prompt 变长时的成本。第五类是输出长度测试，用来观察生成 token 数增加时的成本。第六类是 KV cache 测试，用来观察缓存历史 key/value 后对逐 token 生成速度的提升。第七类是 Transformers baseline，用来和成熟框架的标准 `generate()` 链路做初步对照。

这些实验对应的脚本如下：

```text
tools/eval/benchmark_generation.py
tools/serve/serve_fastapi.py
tools/eval/benchmark_api.py
tools/eval/benchmark_api_concurrency.py
tools/eval/benchmark_context_length.py
tools/eval/benchmark_output_length.py
tools/eval/benchmark_transformers_generation.py
```

## 结果汇总

纯模型生成 benchmark 中，prompt 长度为 42 token，最多生成 80 token。CPU 平均延迟为 0.4556 秒，平均生成速度为 175.58 tokens/s。

HTTP API 单请求 benchmark 中，最多生成 40 token，并启用 EOS 截断。平均端到端延迟为 0.0980 秒，服务端生成延迟为 0.0962 秒，HTTP 额外开销约 0.0018 秒，平均速度为 228.61 tokens/s。

并发 benchmark 的结果如下：

```text
concurrency   req/s   output tok/s   avg latency   p95 latency
1             11.03   242.70         0.0905s       0.0979s
2             19.01   420.63         0.1044s       0.1107s
4             23.90   525.74         0.1638s       0.1755s
```

上下文长度 benchmark 固定最多生成 16 token，改变 prompt 长度：

```text
actual prompt   avg latency   avg tok/s
43              0.0664s       247.16
69              0.0767s       213.41
107             0.0954s       171.06
121             0.0953s       171.12
```

输出长度 benchmark 固定 prompt，改变生成长度：

```text
target output   avg latency   avg tok/s
8               0.0340s       248.85
16              0.0703s       237.24
32              0.1385s       233.91
64              0.2961s       217.62
```

KV cache benchmark 固定 prompt 为 42 token，生成 64 token：

```text
mode        avg latency   avg tok/s
no cache    0.2903s       220.49
kv cache    0.1553s       412.17
```

sliding-window KV cache benchmark 固定 prompt 为 42 token，生成 160 token。此时总 token 数已经超过 `block_size=128`，缓存路径会保留最近 128 个 token 的 key/value：

```text
mode                  avg latency   avg tok/s
no cache              0.9682s       165.26
sliding kv cache      0.3821s       418.79
```

Transformers baseline 使用 `sshleifer/tiny-gpt2`，生成 64 token：

```text
model                 avg latency   avg tok/s
sshleifer/tiny-gpt2   0.0502s       1275.78
```

需要注意，`sshleifer/tiny-gpt2` 是极小测试模型，hidden size 只有 2，不能和本项目 6.57M 参数 checkpoint 做质量或同规模性能的严格公平比较。这个实验主要用于验证成熟框架的生成链路和缓存优化形态。

KV cache 接入 FastAPI 后，服务链路 benchmark 的对比如下：

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

上下文长度 benchmark 中，启用 KV cache 后，固定生成 16 token 的结果如下：

```text
actual prompt   no cache latency   kv cache latency
43              0.0664s            0.0433s
69              0.0767s            0.0435s
107             0.0954s            0.0528s
```

## 结果分析

首先，FastAPI 封装本身在本地小模型场景下不是主要瓶颈。API 单请求 benchmark 中，端到端延迟约 0.0980 秒，服务端生成延迟约 0.0962 秒，HTTP 额外开销只有约 0.0018 秒。这说明当前系统的主要耗时仍然来自模型逐 token 生成，而不是 JSON 解析或 HTTP 请求。

其次，并发会提升整体吞吐，但会拉高单请求延迟。并发从 1 增加到 4 时，req/s 从 11.03 提升到 23.90，output tok/s 从 242.70 提升到 525.74；但平均延迟也从 0.0905 秒增加到 0.1638 秒。这体现了部署中常见的吞吐和延迟取舍：如果希望系统同时处理更多请求，就要接受一部分排队和资源竞争。

第三，上下文长度会影响生成速度。在固定输出长度时，prompt 从 43 token 增加到 121 token，平均延迟从 0.0664 秒增加到约 0.095 秒，tokens/s 从 247.16 下降到约 171。原因是当前自写模型没有 KV cache，每生成一个 token 都会重新计算最近 `block_size` 内的上下文，prompt 越长，每一步 forward 的注意力计算越重。

第四，输出长度和总延迟基本呈线性关系。生成 8 token 时平均延迟约 0.0340 秒，生成 64 token 时平均延迟约 0.2961 秒。因为当前生成过程是逐 token decode，每多生成一个 token 就要多执行一次模型 forward。tokens/s 略有下降，是因为随着生成继续，参与 attention 的序列长度也在增加。

第五，KV cache 能显著减少重复计算。普通生成路径每一步都会重新计算整段上下文的 key/value；缓存路径只在第一步处理 prompt，之后每次只处理新增 token，并复用历史 key/value。在本次 CPU 测试中，生成 64 token 的平均延迟从 0.2903 秒降到 0.1553 秒，平均速度从 220.49 tok/s 提升到 412.17 tok/s。

第六，KV cache 在服务链路中同样有效。单请求 API 平均延迟从 0.0980 秒下降到 0.0627 秒，平均 tokens/s 从 228.61 提升到 363.64。长输出场景收益更明显，生成 64 token 时平均延迟从 0.2961 秒下降到 0.1570 秒。并发场景下，concurrency=1 和 concurrency=2 都有明显提升，但 concurrency=4 时 req/s 只从 23.90 增加到 24.62，说明此时系统更接近 CPU 资源竞争或请求排队瓶颈。

第七，sliding-window KV cache 让缓存路径可以支持超过 `block_size` 的长生成。在生成 160 token 时，普通路径平均延迟为 0.9682 秒，sliding-window KV cache 平均延迟为 0.3821 秒，平均速度从 165.26 tok/s 提升到 418.79 tok/s。当前实现对 RoPE 模型启用滑动窗口缓存；非 RoPE 的 learned position embedding 模型仍然不能超过位置表长度。

第八，Transformers baseline 展示了成熟框架的标准 `generate()` 链路。`sshleifer/tiny-gpt2` 生成 64 token 的平均速度为 1275.78 tok/s，明显快于本项目自写模型。但该模型极小，不能直接说明框架一定比当前实现快多少；更准确的理解是，Transformers 已经默认使用缓存生成，而本项目通过手写 KV cache 正在逐步复现这种推理优化机制。

## 阶段结论

到这里，当前项目已经从训练和采样推进到了一个最小可用的本地推理服务。这个服务可以通过 HTTP 接收 prompt，返回生成结果、延迟、tokens/s 等指标，也可以通过 benchmark 脚本观察单请求、并发、上下文长度和输出长度对性能的影响。

目前最重要的结论是：对于这个 6.57M 参数的小模型，服务框架开销很小，性能瓶颈主要在逐 token 生成；并发可以提高吞吐，但会增加延迟；长上下文和长输出都会明显增加推理成本；KV cache 可以显著降低生成阶段的重复计算，并且这种收益能传递到 FastAPI 服务层。加入 sliding-window 之后，RoPE 模型的缓存路径已经可以支持超过 `block_size` 的长生成。Transformers baseline 说明成熟框架的生成链路已经默认包含类似缓存优化。KV cache 是推理优化中最核心的机制之一，但它不能消除所有瓶颈，高并发时仍然会受到 CPU 资源和请求调度限制。

## 后续方向

下一步可以继续围绕推理优化做两件事。第一是选择规模更接近的标准模型做 Transformers 对照，或者把当前自写 checkpoint 转换成标准模型格式后再比较。第二是进一步做 batch serving，让一次 forward 同时处理多个请求，观察吞吐和延迟如何变化，并为理解 vLLM / SGLang 的 batching、paged KV cache 和调度策略打基础。
