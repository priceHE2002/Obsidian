---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - FlashAttention
  - Triton
created: 2026-07-07
up: "[[模型层实现总览|模型层实现总览]]"
---

# 注意力机制与 FlashAttention

`attention.py` 是 nano-vllm 推理性能的核心——它结合了 Triton 自定义 kernel（KV-Cache 写入）和 FlashAttention（高效注意力计算）。

## Triton Kernel: store_kvcache_kernel

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,         # [N, num_heads, head_dim] — 当前计算的 K
    key_stride,      # key.stride(0)
    value_ptr,       # [N, num_heads, head_dim] — 当前计算的 V
    value_stride,
    k_cache_ptr,     # [num_blocks * block_size, num_heads * head_dim] — 展平的 KV-Cache
    v_cache_ptr,
    slot_mapping_ptr,# [N] — 每个 token 对应的物理 slot
    D: tl.constexpr, # num_heads * head_dim
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return             # -1 表示不存储（CUDA Graph 填充位）

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

这个 kernel 的职责很简单：将模型刚计算出的 K 和 V 写入物理 KV-Cache。每个 Triton program（对应一个 CUDA thread block）处理一个 token。

为什么用 Triton 而不是纯 PyTorch？标准 PyTorch 的 `k_cache[slot_mapping] = k` 会为每个 token 分别 launch 一个 scatter kernel，当 token 数量多时有大量 kernel launch 开销。Triton 将整个 scatter 操作融合为一个 kernel，减少了调度开销。

### 关键设计决策：slot_mapping == -1 的 skip

```python
if slot == -1: return
```

在 CUDA Graph replay 时，`slot_mapping` 中超出当前 batch size 的位置被填充为 -1。Triton kernel 直接 return 跳过，避免了无效的显存读写。这是 CUDA Graph 和自定义 kernel 协同工作的优雅例子。

## wrap 函数：store_kvcache()

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1     # 最后一维连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # KV-Cache 展平
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                                k_cache, v_cache, slot_mapping, D)
```

`assert` 验证 stride 是为了确保 Triton kernel 中的偏移计算正确。KV-Cache 的 `stride(1) == D` 说明它被 view 为 `[num_blocks * block_size, D]`（2D 展平视图），而不是原始的 6D 形状。

## Attention 模块前向传播

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])  # 占位，运行时替换

    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 步骤 1: 将 K/V 写入 KV-Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 步骤 2: 注意力计算
        if context.is_prefill:
            if context.block_tables is not None:  # 前缀缓存场景
                k, v = k_cache, v_cache           # 从完整 KV-Cache 读取
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,  # PagedAttention!
            )
        else:  # Decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),                    # [batch, 1, heads, dim]
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,  # PagedAttention!
                softmax_scale=self.scale,
                causal=True,
            )
        return o
```

## 两个 FlashAttention API 的对比

| 特性 | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
|------|--------------------------|---------------------------|
| 用途 | Prefill（可变长度序列） | Decode（单 token + KV-Cache） |
| Q 形状 | `[total_q, heads, dim]` | `[batch, 1, heads, dim]` |
| K/V 来源 | 刚计算的 K/V（或缓存） | 全部从 KV-Cache 读取 |
| 批次组织 | `cu_seqlens` 累积长度 | 固定 batch size |
| `block_table` | 可选（前缀缓存时使用） | 必需（PagedAttention） |

## 前缀缓存时的 K/V 切换

```python
if context.block_tables is not None:
    k, v = k_cache, v_cache    # 从缓存读取完整的 K/V（含前缀部分）
```

当有前缀缓存命中时，当前计算的 K/V 只是非缓存部分的 token，但 FlashAttention 需要完整的 K/V 序列（包括前缀部分）。此时将 K/V 指向完整的 `k_cache, v_cache`，FlashAttention 通过 `block_table` 找到所有相关的 Block。

这里的关键是时序：`store_kvcache` 已经将当前计算的 K/V 写入了 `k_cache`，然后 `k, v = k_cache, v_cache` 让 FlashAttention 从缓存中读取完整的序列（prefix cache + 刚写入的新 token）。

## Prefill vs Decode 的 Attention 模式

**Prefill**：Q 长度为 N（本轮处理的 token 数），K 长度为 N+M（M 为前缀缓存 token 数）。FlashAttention 的 `varlen` API 通过 `cu_seqlens` 处理可变长度序列，`causal=True` 确保每个 token 只看自己和之前的 token。

**Decode**：Q 长度为 1（单个 token），K 长度为完整序列长度。`flash_attn_with_kvcache` 专为这种场景优化——它直接从 KV-Cache 读取，通过 `cache_seqlens` 知道每个序列的有效 KV-Cache 长度，避免处理未使用的 Block。
