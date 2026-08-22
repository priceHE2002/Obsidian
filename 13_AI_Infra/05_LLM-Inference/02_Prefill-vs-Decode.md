---
tags: [llm, prefill, decode]
---
# Prefill vs Decode

## Prefill

一次性处理整个 prompt，并建立初始 KV Cache。

典型关注：

- TTFT
- prompt length
- large matrix operations
- compute utilization

## Decode

每轮通常只加入一个新 token，重复读取模型权重和历史 KV。

典型关注：

- TPOT
- memory bandwidth
- KV Cache access
- batch size

## 为什么不能把两者当成同一种 workload？

如果一个超长 Prefill 独占一次调度窗口，正在 Decode 的请求可能出现明显 TPOT 抖动。因此后续需要 [[13_AI_Infra/07_Scheduling/02_Chunked-Prefill|Chunked Prefill]]。

## 面试结论

不要只说“Prefill compute-bound、Decode memory-bound”作为绝对结论。更严谨地说：它们的计算/访存特征明显不同，具体瓶颈要结合模型、batch、sequence length、硬件和 kernel 实现实际 profile。
