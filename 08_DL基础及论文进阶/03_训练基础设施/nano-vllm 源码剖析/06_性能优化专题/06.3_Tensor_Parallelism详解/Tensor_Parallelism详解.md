---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - 张量并行
created: 2026-07-07
up: "[[性能优化专题总览|性能优化专题总览]]"
---

# Tensor Parallelism 详解

nano-vllm 的张量并行（Tensor Parallelism, TP）支持 1 到 8 个 GPU。它采用 Megatron-LM 风格的列并行 + 行并行方案，但使用独特的 `SharedMemory + pickle` 进程间通信替代了标准的 `torchrun` 方案。

## TP 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      rank 0 (主进程)                          │
│  LLMEngine + Scheduler + ModelRunner(rank=0)                 │
│                                                             │
│  schedule() → write_shm("run", seqs, is_prefill)            │
│                   │                                         │
│      ┌────────────┼────────────┐                            │
│      ▼            ▼            ▼                            │
│  SharedMemory (1MB) + Event                                 │
│      │            │            │                            │
│      ▼            ▼            ▼                            │
│  rank 1        rank 2        rank 3                         │
│  ModelRunner   ModelRunner   ModelRunner                    │
│  loop()        loop()        loop()                         │
│                                                             │
│  dist.init_process_group("nccl", ...)                        │
│  ← NCCL 用于权重同步和数据并行计算 →                            │
└─────────────────────────────────────────────────────────────┘
```

## ModelRunner 中的 TP 初始化

```python
def __init__(self, config, rank, event):
    dist.init_process_group("nccl", "tcp://localhost:2333",
                            world_size=self.world_size, rank=rank)
    torch.cuda.set_device(rank)    # rank 0 → GPU 0, rank 1 → GPU 1, ...

    # rank 0 创建 SharedMemory
    if self.world_size > 1:
        if rank == 0:
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            dist.barrier()   # 等所有进程初始化完成
        else:
            dist.barrier()   # 等 rank 0 创建 shm
            self.shm = SharedMemory(name="nanovllm")
            self.loop()      # Worker 进入无限循环
```

### SharedMemory + Event 通信 vs torchrun

| 方案 | nano-vllm | torchrun |
|------|-----------|----------|
| 启动方式 | 单 Python 脚本，内部 spawn | 需要 `torchrun --nproc_per_node=N` |
| 参数传递 | SharedMemory + pickle | 无（需自行实现） |
| 同步机制 | multiprocessing.Event | dist.barrier() |
| 灵活性 | 高（可在代码中动态启停 worker） | 中（严格的 SPMD 模型） |

## 线性层的 TP 切分策略

完整分析见 [[../04_模型层实现/04.3_线性层与张量并行/线性层与张量并行]]。此处总结关键数学原理。

### ColumnParallel 切分

对权重矩阵 W 按列（输出维度）切分：

```
W = [W_0 | W_1]   (2 GPU)
y = x @ W^T = x @ [W_0 | W_1]^T = [x @ W_0^T | x @ W_1^T] = [y_0 | y_1]
```

每个 GPU 持有部分权重，计算部分输出。输入 x 需要完全相同（通过上一层的 all_reduce 保证）。

### RowParallel 切分

对权重矩阵 W 按行（输入维度）切分：

```
W = [W_0; W_1]   (2 GPU, stacked vertically)
y = [x_0 | x_1] @ [W_0; W_1]^T = x_0 @ W_0^T + x_1 @ W_1^T
```

每个 GPU 持有部分权重和部分输入，计算部分结果后通过 all_reduce 求和。

### Column + Row 配对

```
输入 x [完整] → ColumnParallel → [y_0 | y_1] (分片)
                              → RowParallel  → all_reduce → 输出 [完整]
```

ColumnParallel 产生分片输出，RowParallel 接收分片输入并 all_reduce 回完整结果。两者配对使用，确保每层开始和结束时所有 GPU 有完整数据。

## Embedding 层的 TP

详细分析见 [[../04_模型层实现/04.5_Embed与LM_Head/Embed与LM_Head]]。

```
VocabParallelEmbedding:
  每个 GPU: 持有 vocab/tp 个词的嵌入
  forward: mask + local lookup + all_reduce

ParallelLMHead:
  每个 GPU: 持有 vocab/tp 个词的输出投影
  forward: local matmul + dist.gather → rank 0 得到完整 logits
```

## TP 的通信开销

| 操作 | 类型 | 频率 | 通信量 |
|------|------|------|--------|
| ColumnParallel forward | 无 | 每层 | 0（输入已在所有 GPU） |
| RowParallel forward | all_reduce | 每层 | `batch × hidden × dtype_bytes` |
| VocabParallelEmbedding | all_reduce | 每步 | `tokens × hidden × dtype_bytes` |
| ParallelLMHead | dist.gather | 每步 | `(batch × vocab) / tp × dtype_bytes` |

RowParallel 的 `all_reduce` 是最大的通信开销。对于 Qwen3-0.6B (hidden=1024, bf16):
- batch=512 时：512 × 1024 × 2B = 1MB per all_reduce
- 28 层 × 2 (attn o_proj + mlp down_proj) = 56 次 all_reduce
- 总计约 56MB / step

这在高带宽 NVLink 上可以忽略，但在 PCIe 上可能成为瓶颈。

## TP 的显存收益

TP 的主要价值不是加速（有时甚至会因通信开销而略微减速），而是**显存分摊**：

```
Qwen3-0.6B 推理显存占用（估算）：
  模型参数: ~1.2GB (bf16)
  KV-Cache (block_size=256, 4096 context): ~1.5GB
  CUDA Graph + 中间激活: ~1.5GB
  总计: ~4.2GB

TP=1: 每个 GPU 需要 ~4.2GB
TP=2: 每个 GPU 需要 ~2.7GB (参数减半，KV-Cache 减半)
TP=4: 每个 GPU 需要 ~1.8GB
```

TP=2 时 8GB 显存的笔记本 GPU 就能运行 + 更长 context 的模型，这是最实用的场景。
