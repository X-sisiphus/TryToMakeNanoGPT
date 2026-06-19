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

本阶段主要完成了八类实验。第一类是直接测试模型内部生成速度，用来观察纯生成性能。第二类是测试 HTTP API 单请求延迟，用来观察服务封装带来的额外开销。第三类是并发请求测试，用来观察多个请求同时到达时吞吐和延迟的变化。第四类是上下文长度测试，用来观察 prompt 变长时的成本。第五类是输出长度测试，用来观察生成 token 数增加时的成本。第六类是 KV cache 测试，用来观察缓存历史 key/value 后对逐 token 生成速度的提升。第七类是 Transformers baseline，用来和成熟框架的标准 `generate()` 链路做初步对照。第八类是 batch serving，用来观察把多条请求合并到一次模型 forward 后的吞吐变化。

这些实验对应的脚本如下：

```text
tools/eval/benchmark_generation.py
tools/serve/serve_fastapi.py
tools/eval/benchmark_api.py
tools/eval/benchmark_api_concurrency.py
tools/eval/benchmark_context_length.py
tools/eval/benchmark_output_length.py
tools/eval/benchmark_transformers_generation.py
tools/eval/benchmark_batch_api.py
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

为了做更接近参数规模的框架对照，新增了本地随机初始化 GPT-2 baseline。该模型不需要下载权重，配置为 `n_embd=112, n_layer=2, n_head=4, n_positions=128`，参数量为 5.95M：

```text
model/path                              params   avg latency   avg tok/s
self nanoGPT kv cache, 64 tokens        6.57M    0.1553s       412.17
Transformers random GPT-2, 64 tokens    5.95M    0.0821s       779.46
```

这个 baseline 不能比较生成质量，因为它是随机初始化模型；它的作用是比较接近参数规模下，Transformers 标准 `generate()` 路径和本项目自写生成路径的速度差异。曾尝试下载 `roneneldan/TinyStories-8M` 和 `roneneldan/TinyStories-1M` 做预训练小模型对照，但本机 Hugging Face 下载出现超时和 DNS 解析失败，因此没有作为本阶段主结果。

batch serving benchmark 对比逐条请求和合并请求。第一组使用等长 prompt，并开启 KV cache：

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

随后加入 left padding + padding attention mask，使 `/generate_batch` 支持不同长度 prompt。模型会根据 `attention_mask` 为每条样本重新计算 `position_ids`，避免 padding 改变 RoPE 和位置 embedding 的语义。变长 batch 先关闭 KV cache 测试：

```text
mode        batch   avg latency   avg tok/s   avg req/s
batched     8       0.6687s       239.28      11.96
sequential  8       0.7619s       210.04      10.50
```

随后把变长 batch 和 KV cache 合并。`forward_with_cache` 可以接收 `attentionMask` 和每行独立的 `positionIds`，生成阶段每新增一个 token 也同步扩展 mask：

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

最后实现教学版 dynamic batching。客户端仍然发送单条 `/generate_dynamic` 请求，服务端等待 5 ms，把同一时间窗口内采样参数一致的请求自动合并成 batch。接口 `/dynamic_stats` 用来观察累计调度统计。并发 benchmark 对比如下：

```text
endpoint           concurrency   req/s   output tok/s   avg latency   p95 latency   avg batch   avg wait
/generate          1             18.84   376.86         0.0530s       0.0537s      1.00        0.00ms
/generate_dynamic  1             16.19   323.76         0.0617s       0.0632s      1.00        5.73ms
/generate          4             32.98   659.70         0.1196s       0.1249s      1.00        0.00ms
/generate_dynamic  4             35.04   700.89         0.1139s       0.1153s      4.00        5.47ms
/generate          8             17.88   357.57         0.4426s       0.4648s      1.00        0.00ms
/generate_dynamic  8             42.85   857.01         0.1858s       0.1895s      8.00        5.06ms
```

随后加入 length bucketing。服务端在 `/generate_dynamic` 入队时记录 prompt token 长度，flush 时按长度排序，让长度相近的请求优先进入同一个 batch。使用不同长度 prompt 的结果如下：

```text
endpoint           concurrency   req/s   output tok/s   avg latency   p95 latency   avg batch   avg wait   avg padding
/generate_dynamic  4             38.62   617.92         0.1034s       0.1242s      4.00        5.31ms     5.06%
/generate_dynamic  8             47.58   761.28         0.1676s       0.1762s      8.00        5.10ms     12.01%
/generate_dynamic  12            48.49   775.87         0.2152s       0.2469s      6.67        83.72ms    5.69%
```

针对同一次 flush 拆出多批时的串行等待问题，继续加入 `--dynamic-max-concurrent-batches` 参数。缩小版验证设置为 concurrency=12、requests=12、max-new-tokens=8：

```text
workers   req/s    output tok/s   avg latency   p95 latency   avg wait
1         67.55    512.24         0.1422s       0.1749s       41.22ms
2         102.73   821.87         0.1095s       0.1152s       3.95ms
```

最后加入可选 adaptive wait。固定 wait 的调度器只等待一个固定窗口；adaptive wait 会先等待最小时间，如果队列还没有达到一个 batch，并且没有超过最大等待时间，就继续短暂等待。实现时还修复了 flush 生命周期问题：当前 flush 推理期间新进入队列的请求，会在本轮 flush 结束后自动启动下一轮处理。缩小版 burst 场景设置为 concurrency=12、requests=12、max-new-tokens=4：

```text
strategy           req/s    output tok/s   avg latency   p95 latency   avg queue wait   avg flush wait
fixed 5ms          104.34   417.37         0.0958s       0.1130s       3.63ms           5.49ms
adaptive 1-8ms     144.94   567.70         0.0747s       0.0811s       2.10ms           5.59ms
```

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

第八，Transformers baseline 展示了成熟框架的标准 `generate()` 链路。`sshleifer/tiny-gpt2` 生成 64 token 的平均速度为 1275.78 tok/s，但该模型极小，不能直接说明框架一定比当前实现快多少。更接近参数规模的随机 GPT-2 baseline 为 5.95M 参数，生成速度为 779.46 tok/s，仍然明显快于本项目 6.57M 自写模型的 412.17 tok/s。这说明 Transformers 的推理路径、缓存管理和生成循环已经有较强工程优化，而本项目通过手写 KV cache 正在逐步复现这些机制。

第九，batch serving 能显著提高吞吐。等长 prompt 且开启 KV cache 时，逐条请求的 req/s 基本停在 17 左右；batch size 为 8 时，合并请求达到 40.28 req/s，输出吞吐达到 886.25 tok/s。这说明将多个请求合并到同一次 forward 可以提高模型计算利用率。

第十，padding attention mask 让不同长度 prompt 可以进入同一个 batch。变长 prompt、关闭 KV cache 时，batch size 为 8 的合并请求从 sequential 的 10.50 req/s 提升到 11.96 req/s，收益有限。把变长 batch 和 KV cache 合并后，batch size 为 8 的合并请求达到 42.05 req/s，输出吞吐达到 812.14 tok/s。实现时还需要同步修正 `position_ids`，否则 left padding 会让真实 token 的位置编号整体后移，影响 RoPE 和 learned position embedding。padded/unpadded 等价性检查中，最后一个真实 token 的 logits 最大误差约为 `3.34e-06`。这正是实际推理系统需要 bucketing、动态 batching、paged KV cache 和调度策略的原因。

第十一，dynamic batching 把 batching 从客户端责任变成了服务端调度能力。低并发时，5 ms 等待窗口会让单请求略慢；但 concurrency=8 时，平均合批大小达到 8，普通 `/generate` 的吞吐为 17.88 req/s，`/generate_dynamic` 达到 42.85 req/s，平均延迟也从 0.4426 秒下降到 0.1858 秒。这说明在请求足够密集时，短暂排队可以换来更高的模型计算利用率。新增的 `/dynamic_stats` 和 benchmark 调度字段可以直接观察平均 batch size、平均排队时间和 batch latency，避免只看端到端吞吐。

第十二，length bucketing 开始处理变长 prompt 的 padding 浪费。并发 8 时，平均 padding ratio 为 12.01%；并发 12 时，请求被拆成 8 和 4 两批，长度排序后平均 padding ratio 降到 5.69%。但因为当前教学版调度器串行执行同一次 flush 里的多个 batch，第二批会等待第一批推理结束，平均等待时间升到 83.72 ms。这说明 bucketing 能减少无效计算，但真实推理系统还需要更细的调度策略，例如并行 worker、最大等待时间、按队列压力动态决定 batch size。

第十三，并行 batch worker 可以缓解多批串行造成的尾部等待。缩小版验证中，`dynamic-max-concurrent-batches=2` 将平均等待从 41.22 ms 降到 3.95 ms，吞吐从 67.55 req/s 提升到 102.73 req/s。但这不是无限增加 worker 的理由，因为多个 batch 同时推理会竞争同一份 CPU/GPU 资源。真实系统需要根据硬件利用率、模型大小、batch 大小和延迟目标调节并行度。

第十四，adaptive wait 让调度器开始根据队列压力调整等待时间。缩小版 burst 测试中，adaptive 1-8 ms 相比固定 5 ms，平均延迟从 0.0958 秒降到 0.0747 秒，吞吐从 104.34 req/s 提升到 144.94 req/s。与此同时，这次实现还修复了一个重要调度 bug：如果 flush 推理期间有新请求入队，旧版本不会自动开启下一轮 flush，可能导致尾部请求挂起。修复后，每轮 flush 结束都会检查是否还有 pending 请求，并自动续跑。

第十五，调度实验已经整理成总控 benchmark。`tools/eval/run_dynamic_scheduler_benchmark.py` 会自动按不同配置启动 FastAPI 服务，等待 `/health`，压测 `/generate_dynamic`，读取 `/dynamic_stats`，关闭服务，并把结果汇总到 `scheduler_summary.csv` 和 `report.md`。这样可以稳定比较 fixed wait、adaptive wait、不同 worker 数和不同并发，不再依赖手动重复操作。

缩小版 smoke test 使用 `fixed_w1` 和 `adaptive_w2` 两组配置，concurrency=4、requests=4、max-new-tokens=2：

```text
strategy       req/s    output tok/s   avg latency   p95 latency
fixed_w1       114.90   229.81         0.0339s       0.0341s
adaptive_w2    113.23   226.46         0.0345s       0.0347s
```

这组结果不是为了证明某个策略更优，而是验证完整实验链路已经打通：服务可以自动拉起和关闭，压测脚本可以拿到动态调度字段，最终可以生成跨配置汇总表。后续只需要扩大并发、请求数和输出长度，就能得到更有代表性的部署对比。

正式对比使用 concurrency=4,8,12、requests-per-level=12、max-new-tokens=8，结果如下：

```text
strategy       concurrency   req/s    output tok/s   avg latency   p95 latency   avg wait
fixed_w1       4             62.30    472.48         0.0639s       0.0665s       5.40ms
fixed_w1       8             67.15    537.17         0.0988s       0.1190s       4.97ms
fixed_w1       12            86.47    691.75         0.1034s       0.1361s       31.38ms
fixed_w2       4             61.76    494.05         0.0644s       0.0673s       5.52ms
fixed_w2       8             66.26    530.08         0.1003s       0.1211s       5.01ms
fixed_w2       12            115.09   920.71         0.0961s       0.1025s       4.00ms
adaptive_w2    4             65.85    526.79         0.0605s       0.0636s       7.89ms
adaptive_w2    8             75.04    600.28         0.0861s       0.0995s       3.69ms
adaptive_w2    12            126.49   1011.91        0.0900s       0.0933s       1.34ms
```

这组正式对比说明：并发 12 时，单 worker 的 fixed wait 会把第二批请求压在队列里，平均等待达到 31.38 ms；改成 2 个 worker 后，等待降到 4.00 ms，吞吐提升到 115.09 req/s；在同样 2 个 worker 下，adaptive wait 进一步把等待降到 1.34 ms，吞吐达到 126.49 req/s。也就是说，dynamic batching 的收益不仅来自合批本身，还来自调度策略能否及时处理被拆出来的后续 batch。

第十六，补充了一个最小 demo client。`tools/serve/demo_client.py` 会先读取 `/health`，再调用 `/generate` 或 `/generate_dynamic`，最后打印生成结果、延迟、tokens/s；如果走 dynamic batching，还会额外打印 `/dynamic_stats`。短输出验证中，默认字段抽取 prompt 生成了 `station: ONSA`。这一步的意义是把推理服务从“可以被 benchmark 调用”推进到“可以被人直接演示和检查”。

## 阶段结论

到这里，当前项目已经从训练和采样推进到了一个最小可用的本地推理服务。这个服务可以通过 HTTP 接收 prompt，返回生成结果、延迟、tokens/s 等指标，也可以通过 benchmark 脚本观察单请求、并发、上下文长度和输出长度对性能的影响。

目前最重要的结论是：对于这个 6.57M 参数的小模型，服务框架开销很小，性能瓶颈主要在逐 token 生成；并发可以提高吞吐，但会增加延迟；长上下文和长输出都会明显增加推理成本；KV cache 可以显著降低生成阶段的重复计算，并且这种收益能传递到 FastAPI 服务层。加入 sliding-window 之后，RoPE 模型的缓存路径已经可以支持超过 `block_size` 的长生成。Transformers baseline 说明成熟框架的生成链路已经默认包含类似缓存优化。batch serving 进一步说明，请求合并可以提升整体吞吐。padding attention mask 让变长 prompt batch 成为可能；继续接入 KV cache 后，变长 batch 也能获得明显吞吐提升。dynamic batching 进一步把 batch 合并从手动 API 变成服务端自动调度。length bucketing 则开始处理变长 prompt 的 padding 浪费。并行 batch worker 可以缓解同一轮调度中多批串行导致的尾部等待。adaptive wait 让等待窗口开始随队列压力变化。总控 benchmark 则把这些调度策略变成可重复比较的实验。demo client 让服务具备了直接演示入口。KV cache、batching、mask 和调度是推理优化中最核心的机制，但它们不能消除所有瓶颈，高并发时仍然会受到 CPU 资源、padding 浪费和请求分布限制。

## 后续方向

下一步可以继续围绕推理优化做两件事。第一是扩大总控 benchmark 的实验规模，比较更多并发、输出长度和等待窗口参数。第二是在网络条件稳定时选择规模更接近的预训练标准模型做 Transformers 对照，或者把当前自写 checkpoint 转换成标准模型格式后再比较。
