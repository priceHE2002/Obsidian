---
tags: [benchmark, profiling, performance]
---
# Benchmark and Profiling

## MiniServe 必须测的指标

### TTFT

$$
TTFT=t_{first\ token}-t_{arrival}
$$

### TPOT

单个输出 token 的平均/分位延迟。

### Throughput

```text
tokens / second
```

### Tail Latency

- P50
- P95
- P99（可选）

### GPU Memory

- peak VRAM
- KV block utilization
- free/used block count

## 公平 Benchmark 纪律

- 同模型
- 同精度
- 同 prompt/output distribution
- 同 GPU
- warmup
- 固定软件版本
- 明确 concurrency

## Workload Matrix

### Concurrency

1 / 2 / 4 / 8 / 16 / 32

### Prompt Length

128 / 512 / 1024 / 2048

### Output Length

64 / 128 / 256 / 512

## Ablation

Baseline → +KV Cache → +Continuous Batching → +Paged KV → +Prefix Cache → +Triton

## 重要原则

不要问“MiniServe 有没有比 vLLM 快”。优先问：

> 在什么 workload 下差距最大？差距来自哪里？下一步应该优化什么？

日志模板：[[13_AI_Infra/15_Templates/Experiment-Log|Experiment Log]]
