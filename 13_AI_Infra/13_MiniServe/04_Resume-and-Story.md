---
tags: [miniserve, resume, interview]
---
# MiniServe Resume and Project Story

## Resume Bullet 模板

> 数字必须替换成真实实验结果。

- Built a lightweight LLM inference runtime supporting continuous batching, paged KV-cache management, prefix caching and dynamic request scheduling.
- Designed a block-based GPU KV-memory allocator to improve memory utilization under variable-length concurrent workloads.
- Implemented Triton/CUDA kernels for transformer inference operators and profiled GPU bottlenecks using Nsight Systems/Compute.
- Developed a reproducible benchmark suite covering TTFT, TPOT, throughput, P50/P95 latency and GPU memory usage, and compared against HuggingFace/vLLM.

## 项目故事结构

### 1. Problem

Naive generation 对并发、动态 KV 生命周期和 GPU utilization 处理很差。

### 2. Design

Scheduler + Paged KV + GPU kernels。

### 3. Hardest Part

提前准备一个真实 debugging / correctness / performance case。

### 4. Evidence

展示 benchmark 与 ablation，而不是只展示 feature list。

### 5. Limitations

主动讲：

- single GPU
- limited model support
- no TP/PP
- simple scheduler
- no production reliability guarantees

### 6. Next Step

根据 profile 结果说明下一项最值得投入的优化。
