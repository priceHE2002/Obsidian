---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - CUDA_Graph
created: 2026-07-07
up: "[[性能优化专题总览|性能优化专题总览]]"
---

# CUDA Graph 详解

CUDA Graph 是 nano-vllm 最重要的 Decode 加速技术。核心思想：将 GPU 上的整个前向推理过程记录为一个图，后续只需一次 `graph.replay()` 调用即可执行完整的前向传播，消除大量 CPU-GPU 同步和 kernel launch 开销。

## 为什么 Decode 适合 CUDA Graph？

Decode 阶段的特征是**固定的计算模式**：每轮只处理一个 token，batch size 变化但计算图结构不变。这与 CUDA Graph 的使用场景完美匹配——一旦图被捕获，输入数据变化只需更新 Tensor 内容，不需要修改图结构。

Prefill 阶段则不适合：序列长度变化大，导致计算图结构（矩阵大小、kernel 配置）频繁变化，每个图只能用一次，捕获开销超过收益。

## 完整实现：capture_cudagraph()

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

    # 1. 预分配固定大小的 Tensor
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    # 2. 定义 batch size 分档
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    # 3. 从大到小捕获（共享 memory pool）
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs])

        # warmup: 确保所有 kernel 已编译
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # capture: 记录图形
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        if self.graph_pool is None:
            self.graph_pool = graph.pool()   # 第一个图创建 pool，后续共享
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    # 4. 保存 Tensor 引用（replay 时更新数据用）
    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
```

## 关键设计决策

### 1. Batch Size 分档策略

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

分档采用了**小 batch 密集、大 batch 稀疏**的策略：
- 1, 2, 4, 8：小 batch 时精确匹配（Decode 初期序列数较少）
- 16, 32, 48, ..., 512：步长 16（`max_num_seqs=512` 时共 33 个图）

为什么步长 16？权衡内存和效率。每个 CUDA Graph 占用约等于一次推理的 GPU 显存（intermediate activations）。33 个图 × ~50MB ≈ 1.65GB，在 8GB 显存中是可接受的。

### 2. Reversed Capture — Memory Pool 共享

```python
for bs in reversed(self.graph_bs):
    with torch.cuda.graph(graph, self.graph_pool):
        ...
    if self.graph_pool is None:
        self.graph_pool = graph.pool()
```

从最大的 batch size 开始捕获，第一个图创建 memory pool，后续更小的图**复用同一个 pool**——因为大图需要的内存量比小图多，pool 由最大图决定。`reversed` 确保 pool 足够容纳所有图。

### 3. Static Tensors + In-place Update

CUDA Graph 的硬性限制：**不能改变 Tensor 大小**。解决方案：
- 预分配 `max_bs` 大小的 Tensor
- Replay 前通过 `[:bs]` 索引更新有效数据
- 用 `fill_(-1)` / `zero_()` 清理无效位置

```python
# replay 时的数据更新（在 run_model 中）
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph_vars["slot_mapping"].fill_(-1)          # 先全部标记为无效
graph_vars["slot_mapping"][:bs] = context.slot_mapping  # 再覆盖有效部分
graph_vars["context_lens"].zero_()
graph_vars["context_lens"][:bs] = context.context_lens
```

## Graph Replay 过程

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
graph.replay()
return self.model.compute_logits(graph_vars["outputs"][:bs])
```

`next(x for x in self.graph_bs if x >= bs)` 找到 ≥ 当前 batch size 的最小预捕获图。例如：
- 当前 bs=3 → 使用 bs=4 的图
- 当前 bs=18 → 使用 bs=32 的图

选择 ≥ 而非精确匹配，是因为 CUDA Graph 中填充的无效 token 会被 Triton kernel 的 `slot == -1` skip 掉，几乎零开销。

注意 `compute_logits` 在 graph replay **之后**调用，不在图内。原因：LM Head 使用 `dist.gather`（TP 时），这类集合通信操作不适合放入 CUDA Graph。

## 性能收益

没有 CUDA Graph 时，每次 Decode 包含：
- 数十个 kernel launch（每个线性层、注意力、归一化都是独立 kernel）
- 每次 launch 有 ~5-10μs 的 CPU-GPU 同步开销
- PyTorch 框架的 dispatch 和 autograd 开销

CUDA Graph 将所有这些压缩为一次 `graph.replay()`：
- 1 次 kernel launch（整个图作为一个 CUDA graph node）
- 消除 PyTorch dispatch 开销
- 消除 CPU-GPU 同步点（直到 `torch.cuda.synchronize()`）

在 batch size=1 时，CUDA Graph 可将 Decode 延迟从 ~15ms 降至 ~8ms（约 50% 提升），在 batch size 较大时提升约 30%。
