---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - ModelRunner
created: 2026-07-07
up: "[[模型执行引擎总览|模型执行引擎总览]]"
---

# ModelRunner 详解

`ModelRunner` 是整个项目中最大的单个文件（~340 行），管理着 GPU 推理的完整生命周期：从序列到 Tensor 的转换、前向推理、CUDA Graph 管理、到 Tensor Parallel 通信。

## __init__() — 初始化流程

```python
def __init__(self, config, rank, event):
    # 1. Tensor Parallel 初始化
    dist.init_process_group("nccl", "tcp://localhost:2333",
                            world_size=self.world_size, rank=rank)
    torch.cuda.set_device(rank)    # 每个 rank 绑定一个 GPU

    # 2. 设置默认数据类型和设备
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.dtype)   # 如 bfloat16
    torch.set_default_device("cuda")           # 模型创建直接在 GPU 上

    # 3. 创建模型和采样器
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)       # 加载 HuggingFace 权重
    self.sampler = Sampler()

    # 4. 预热 + KV-Cache 分配 + CUDA Graph 捕获
    self.warmup_model()
    self.allocate_kv_cache()
    if not self.enforce_eager:
        self.capture_cudagraph()

    # 5. 恢复默认设置
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)

    # 6. 如果是 Worker 进程，进入循环等待
    if self.world_size > 1:
        if rank == 0:
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name="nanovllm")
            self.loop()    # Worker 进程进入无限循环
```

### 为什么设置 `torch.set_default_device("cuda")`？

这是 nano-vllm 最优雅的设计之一。通过将默认设备设为 CUDA，所有 `nn.Parameter(torch.empty(...))` 创建的参数都会直接在 GPU 上分配，避免了"先在 CPU 创建再 `.to(cuda)`"的冗余内存操作。

注意 `__init__` 末尾恢复为 CPU——这确保后续在 LLMEngine 中创建的 tensor（如 tokenizer 输出）仍在 CPU 上，不会污染 GPU 内存。

## warmup_model() — 模型预热

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = ...
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)            # 一次完整 Prefill 前向
    torch.cuda.empty_cache()
```

预热的目的：
1. **触发 CUDA kernel 编译**：PyTorch、FlashAttention、Triton kernel 的首次运行会触发 JIT 编译，预热吸收这部分开销
2. **初始化 cuBLAS/cuDNN 工作区**：CUDA 库首次调用时会分配内部缓冲区
3. **获得准确的 peak memory**：`reset_peak_memory_stats` + 预热后，`memory_stats()["allocated_bytes.all.peak"]` 准确反映模型推理的显存峰值，用于后续 KV-Cache 分配计算

## allocate_kv_cache() — KV-Cache 显存分配

```python
def allocate_kv_cache(self):
    # 计算可用显存
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # 计算每个 Block 的字节数
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim",
                       hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * \
                  num_kv_heads * head_dim * hf_config.dtype.itemsize
    # 2: K + V, num_hidden_layers: 每层独立 KV-Cache
    # block_size: 每个 Block 的 token 数
    # num_kv_heads x head_dim: 每个 token 的 KV 维度

    # 动态计算可分配的 Block 数
    config.num_kvcache_blocks = int(
        total * config.gpu_memory_utilization - used - peak + current
    ) // block_bytes

    # 创建 KV-Cache Tensor [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers,
                                 config.num_kvcache_blocks,
                                 self.block_size, num_kv_heads, head_dim)

    # 将 KV-Cache 绑定到各 Attention 层
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

显存布局的精妙之处：**全局连续分配**。整个 KV-Cache 是一个 6D Tensor `[2, L, B, T, H, D]`，各层通过**视图**（view）引用各自的 segment。这比每层独立分配更高效——减少内存碎片、简化管理。

显存计算公式：
```
可用显存 = total * gpu_memory_utilization - used - (peak - current)
```
`(peak - current)` 是模型推理峰值的残差（如果当前模型占用比峰值少，这部分被预留）。

## Prefill 输入准备：prepare_prefill()

```python
def prepare_prefill(self, seqs):
    input_ids, positions = [], []
    cu_seqlens_q, cu_seqlens_k = [0], [0]
    max_seqlen_q = max_seqlen_k = 0
    slot_mapping = []

    for seq in seqs:
        start = seq.num_cached_tokens          # 从缓存后的位置开始
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        seqlen_k = end                          # 历史 + 当前
        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        if not seq.block_table: continue    # warmup 中没有 block_table

        # 计算 slot_mapping：将逻辑 token 位置映射到物理 KV-Cache slot
        start_block = start // self.block_size
        end_block = (end + self.block_size - 1) // self.block_size
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * self.block_size
            if i == start_block:
                slot_start += start % self.block_size
            if i != end_block - 1:
                slot_end = seq.block_table[i] * self.block_size + self.block_size
            else:
                slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
            slot_mapping.extend(range(slot_start, slot_end))

    # 前缀缓存检测
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:   # K序列长度 > Q序列长度
        block_tables = self.prepare_block_tables(seqs)

    # 转换为 Tensor（pin_memory 加速 CPU→GPU 拷贝）
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    ...
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables)
    return input_ids, positions
```

### slot_mapping 的核心作用

`slot_mapping` 解决了 PagedAttention 的核心问题：逻辑 token 位置到物理 KV-Cache 位置的映射。

```
序列 A 的 block_table = [3, 7, 1]  (使用 Block 3, 7, 1)
token 位置 0..255    → slot 3*256+0  .. 3*256+255   (Block 3)
token 位置 256..511  → slot 7*256+0  .. 7*256+255   (Block 7)
token 位置 512..767  → slot 1*256+0  .. 1*256+255   (Block 1)
```

Triton `store_kvcache_kernel` 读取 slot_mapping，将 K/V 写入正确的物理位置。

### 前缀缓存检测

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
    block_tables = self.prepare_block_tables(seqs)
```

当 `cu_seqlens_k`（K 的总长度）> `cu_seqlens_q`（Q 的总长度）时，说明有序列使用了前缀缓存（K 比 Q 长，因为包含了共享前缀的 KV-Cache）。此时需要传入 `block_tables` 给 FlashAttention，让它通过 `block_table` 查找前缀缓存中的 K/V。

## Decode 输入准备：prepare_decode()

```python
def prepare_decode(self, seqs):
    input_ids, positions, slot_mapping, context_lens = [], [], [], []
    for seq in seqs:
        input_ids.append(seq.last_token)      # 只需要最后一个 token
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        # 最后一个 token 的 slot
        slot_mapping.append(seq.block_table[-1] * self.block_size +
                           seq.last_block_num_tokens - 1)

    ...
    block_tables = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping,
                context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

Decode 阶段每个序列只处理一个 token（`seq.last_token`）。`context_lens` 是 FlashAttention 的 `cache_seqlens`——告诉 kernel 每个序列的完整 KV-Cache 长度。

## run_model() — 推理分支选择

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # CUDA Graph 路径
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

分支选择逻辑：
1. **Prefill**：直接用 `model(input_ids, positions)`——Prefill 序列长度变化大，CUDA Graph 不适合
2. **enforce_eager=True**：用户明确禁用 CUDA Graph
3. **input_ids.size(0) > 512**：Batch size 太大，超出了 CUDA Graph 捕获的范围
4. **Decode + batch ≤ 512**：走 CUDA Graph 路径，选择能覆盖当前 bs 的最小预捕获 graph

### CUDA Graph 的变量更新策略

CUDA Graph 捕获后不能修改 Tensor 大小，所以预分配了最大尺寸的 Tensor。每次 replay 前：
- `fill_(-1)` + 索引赋值：清零无效 slot_mapping（-1 在 Triton kernel 中表示"跳过"）
- `zero_()` + 索引赋值：清零无效 context_lens
- 只更新有效行：`[:bs]` 覆盖

这比每次创建新 Tensor 快了数十倍。

## run() — 主入口

```python
def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill \
                           else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

采样只在 rank 0 执行——Worker 进程完成后不需要采样结果。

## TP 通信：write_shm() / read_shm()

```python
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")    # 4字节长度前缀
    self.shm.buf[4:n+4] = data                      # pickle 序列化的数据
    for event in self.event:
        event.set()                                  # 唤醒所有 Worker

def read_shm(self):
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

简单的长度前缀 + pickle 协议。SharedMemory 大小固定为 2^20 = 1MB——对于传输方法名和序列参数来说绰绰有余。Sequence 的 `__getstate__` 优化（Decode 只传 last_token）确保了不会超出这个限制。

## capture_cudagraph() 详解

见 [[06_性能优化专题/06.1_CUDA_Graph/CUDA Graph]]。
