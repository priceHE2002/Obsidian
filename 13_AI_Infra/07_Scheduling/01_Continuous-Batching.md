---
tags: [scheduler, continuous-batching, serving]
---
# Continuous Batching

## Static Batching 的问题

不同请求生成长度不同。短请求提前结束后，固定 batch 中会出现空洞，浪费 GPU capacity。

## Continuous Batching

核心思想：**每个 decoding iteration 重新调度**。

```text
Iter 1: [A B C]
Iter 2: [A B C]
B done
Iter 3: [A D C]
```

## Request State Machine

```text
WAITING → PREFILL → DECODING → FINISHED
```

## Scheduler 每轮负责

1. 回收已完成请求
2. 释放对应 KV blocks
3. 接纳 waiting requests
4. 遵守 token / sequence budget
5. 构建本轮 `SchedulerBatch`

## 关键参数

- `max_num_seqs`
- `max_num_batched_tokens`
- prefill budget
- decode priority

## 需要观察的系统 trade-off

$$
\text{Throughput} \leftrightarrow \text{Latency}
$$

以及：

- fairness
- starvation
- head-of-line blocking
- queueing delay

下一步：[[13_AI_Infra/07_Scheduling/02_Chunked-Prefill|Chunked Prefill]]
