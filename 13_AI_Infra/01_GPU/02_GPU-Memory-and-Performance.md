---
tags: [gpu, memory, performance]
---
# GPU Memory and Performance

## 记忆层次

Registers → Shared Memory → L2 → HBM / VRAM → Host Memory

原则：越靠近计算单元越快，但容量越小。

## Arithmetic Intensity

$$
\text{Arithmetic Intensity}=\frac{\text{FLOPs}}{\text{Bytes moved}}
$$

用途：帮助判断一个 kernel 更可能受到算力还是内存带宽限制。

## 需要做实验而不是背结论

每个 kernel 至少记录：

- latency
- effective bandwidth
- achieved occupancy
- register usage
- memory throughput
- SM utilization

## 和 MiniServe 的关系

KV Cache 的读取/写入、本地化、block layout 都会直接影响 GPU memory traffic。最终你要能解释：

> Scheduler 产生的 batch shape 与 KV block layout，如何进一步影响 kernel 的访存行为？
