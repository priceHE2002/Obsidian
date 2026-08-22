---
tags: [scheduler, chunked-prefill, serving]
---
# Chunked Prefill

## 问题

一个超长 prompt 的 Prefill 可能持续占用较大的 token budget，导致正在 Decode 的请求 TPOT 抖动。

## 设计

把一个长 Prefill 拆成多个 chunk：

```text
2048 tokens → 512 + 512 + 512 + 512
```

每轮混合：

```text
Prefill A(512) + Decode B + Decode C + Decode D
```

## 核心 trade-off

- chunk 太大：Decode latency 容易被拖慢
- chunk 太小：调度和 kernel launch overhead 增加

## 实验设计

固定 workload，对比 chunk size：

- 128
- 256
- 512
- 1024

记录：

- TTFT
- TPOT P50/P95
- throughput
- GPU utilization

## 面试问题

> 如果追求吞吐最大化与交互延迟最小化，chunk size 应该如何选？

答案不应是固定数字，而应基于 workload 和 SLO。
