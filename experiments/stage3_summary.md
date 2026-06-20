# 第三阶段总结：推理部署与优化

## 阶段目标

第三阶段的目标，是把前面训练出来的小模型从“能在脚本里生成”推进到“能作为本地服务被调用、被压测、被分析”。

当前 checkpoint：

```text
out/sft_mixed_binding_multi_hard_2200/ckpt.pt
```

模型规模约 6.57M 参数，任务是结构化字段抽取。

## 当前交付物

- 本地 FastAPI 推理服务：`tools/serve/serve_fastapi.py`
- Demo client：`tools/serve/demo_client.py`
- 单请求 benchmark：`tools/eval/benchmark_api.py`
- 并发 benchmark：`tools/eval/benchmark_api_concurrency.py`
- 上下文长度 benchmark：`tools/eval/benchmark_context_length.py`
- 输出长度 benchmark：`tools/eval/benchmark_output_length.py`
- batch serving benchmark：`tools/eval/benchmark_batch_api.py`
- Transformers baseline benchmark：`tools/eval/benchmark_transformers_generation.py`
- dynamic scheduler 总控 benchmark：`tools/eval/run_dynamic_scheduler_benchmark.py`
- 内存 benchmark：`tools/eval/benchmark_memory.py`
- 动态 INT8 量化 benchmark：`tools/eval/benchmark_quantization.py`
- OpenAI-compatible benchmark：`tools/eval/benchmark_openai_compatible.py`
- 完整部署报告：`experiments/deployment_report.md`

## Demo 运行方式

启动服务：

```bash
python tools/serve/serve_fastapi.py \
  --checkpoint out/sft_mixed_binding_multi_hard_2200/ckpt.pt \
  --port 8010 \
  --dynamic-max-concurrent-batches 2 \
  --dynamic-adaptive-wait \
  --dynamic-min-wait-ms 1 \
  --dynamic-max-wait-ms 8
```

调用普通生成：

```bash
python tools/serve/demo_client.py \
  --base-url http://127.0.0.1:8010 \
  --endpoint generate \
  --max-new-tokens 40 \
  --stop-at-eos \
  --use-kv-cache
```

调用 dynamic batching：

```bash
python tools/serve/demo_client.py \
  --base-url http://127.0.0.1:8010 \
  --endpoint generate_dynamic \
  --max-new-tokens 40 \
  --stop-at-eos \
  --use-kv-cache
```

## 关键技术点

- FastAPI 服务化：模型启动时只加载一次，后续请求复用同一个模型实例。
- KV cache：生成时复用历史 token 的 key/value，减少重复计算。
- Sliding-window KV cache：RoPE 模型在超过 `block_size` 后保留最近窗口，支持更长生成。
- Batch serving：把多个 prompt 合到一次 forward，提高吞吐。
- Padding attention mask：让不同长度 prompt 可以进入同一个 batch，同时不让 padding 影响真实 token。
- Dynamic batching：客户端仍发送单请求，服务端在短时间窗口内自动合批。
- Length bucketing：按 prompt 长度排序合批，降低 padding 浪费。
- Concurrent batch workers：同一轮 flush 拆出多批时，允许多个 batch 并行推理，减少尾部等待。
- Adaptive wait：根据队列压力在最小等待和最大等待之间调整 flush 时机。
- Memory benchmark：记录 RSS、peak RSS，以及 CUDA/MPS 可用时的设备侧内存。
- Dynamic INT8 quantization：把 Linear 动态量化为 INT8，对比模型体积和生成速度。
- OpenAI-compatible API：为后续接 vLLM / SGLang / 标准 serving 服务保留统一 benchmark 入口。

## 代表性结果

KV cache 在生成 64 token 时的 CPU 对比：

```text
mode        avg latency   avg tok/s
no cache    0.2903s       220.49
kv cache    0.1553s       412.17
```

batch serving 在 batch size=8、开启 KV cache 时：

```text
mode        avg latency   avg tok/s   avg req/s
sequential  0.4740s       371.38      16.88
batched     0.1987s       886.25      40.28
```

dynamic scheduler 正式对比中，concurrency=12：

```text
strategy       req/s    output tok/s   avg latency   avg wait
fixed_w1       86.47    691.75         0.1034s       31.38ms
fixed_w2       115.09   920.71         0.0961s       4.00ms
adaptive_w2    126.49   1011.91        0.0900s       1.34ms
```

内存 smoke test，CPU、KV cache、生成 16 token：

```text
stage          rss MB   peak rss MB
before_load    178.73   178.77
after_load     262.08   271.11
after_warmup   269.08   271.11
after_run_1    269.50   271.11
```

动态 INT8 量化 smoke test，CPU、KV cache、生成 8 token：

```text
mode           state dict   param/buffer   avg latency   avg tok/s
fp32           25.21 MB     25.20 MB       0.0209s       383.41
dynamic_int8   15.77 MB     12.40 MB       0.0269s       296.96
```

## 阶段结论

这一阶段最重要的收获是：推理部署不是简单地把 `sample.py` 包一层 HTTP。

真正影响服务性能的是生成路径和调度路径：

- 生成路径看 KV cache、上下文长度、输出长度。
- batch 路径看 padding mask、position ids、变长 batch 和缓存兼容。
- 调度路径看等待窗口、batch size、请求到达分布、worker 数和排队时间。
- 资源路径看 RSS、峰值内存、设备 allocated/reserved memory。
- 量化路径看模型体积、低精度 kernel 支持，以及速度和质量是否真的改善。
- 框架路径看模型格式、OpenAI-compatible API、硬件后端和 serving 框架支持范围。

当前项目已经具备一个最小可用的本地推理系统：可以启动服务，可以发 demo 请求，可以跑 benchmark，可以输出性能报告，也可以解释不同优化为什么有效。

## 下一步

第三阶段里，当前自写 checkpoint 不直接上 vLLM / SGLang。更合理的下一步是在标准模型上体验 vLLM-Metal 或 CUDA/Linux 上的 vLLM/SGLang，再用 OpenAI-compatible benchmark 做统一对比。

第四阶段可以从部署工程转向研究切口。比较自然的方向是：

- 小模型在结构化天文/时空任务上的数据效率
- SFT 和 DPO 在小模型字段抽取任务上的边界
- RAG 是否能弥补小模型领域知识不足
- 长上下文、量化和推理调度对结构化任务准确率的影响
