---
tags: [gpu, fundamentals]
---
# GPU Mental Model

## 先建立这条执行链

Python → PyTorch → CUDA Runtime → CUDA Kernel → GPU

不要把“GPU”理解成一个更快的 CPU。GPU 的核心优势来自大规模并行吞吐，而代价是对内存访问、并行粒度和同步更敏感。

## 必须掌握的层次

- Thread / Warp / Block / Grid
- SM（Streaming Multiprocessor）
- Register / Shared Memory / L2 / HBM(VRAM)
- Memory Coalescing
- Occupancy
- Kernel Launch Overhead
- Compute-bound vs Memory-bound

## 面试必须能解释

### 为什么 Decode 常常更容易 memory-bound？

单步 Decode 只处理极少的新 token，但仍需要读取大量模型权重和历史 KV；可供摊销的算术量较少，内存访问占比更突出。

### 为什么 Kernel Fusion 可能更快？

减少：

- 中间结果写回/重新读取 HBM
- Kernel launch 次数
- 不必要的同步

但 fusion 也可能增加 register pressure，导致 occupancy 下降。

## 关联

- [[13_AI_Infra/01_GPU/02_GPU-Memory-and-Performance|GPU Memory & Performance]]
- [[13_AI_Infra/02_CUDA/01_CUDA-Learning-Path|CUDA Learning Path]]
- [[13_AI_Infra/11_Performance/01_Benchmark-and-Profiling|Benchmark & Profiling]]
