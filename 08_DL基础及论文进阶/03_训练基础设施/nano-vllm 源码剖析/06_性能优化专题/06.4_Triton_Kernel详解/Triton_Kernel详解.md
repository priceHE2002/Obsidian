---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - Triton
created: 2026-07-07
up: "[[性能优化专题总览|性能优化专题总览]]"
---

# Triton Kernel 详解

nano-vllm 只使用了一个自定义 Triton kernel——`store_kvcache_kernel`——但它展示了 Triton 在推理引擎中的典型用法：批量 scatter 操作的融合加速。

## 为什么需要自定义 Kernel？

标准 PyTorch 的 KV-Cache 写入通常用索引赋值：

```python
k_cache[slot_mapping] = k    # scatter 写入
v_cache[slot_mapping] = v
```

这会为每个 token launch 一个独立的 scatter kernel，当 token 数量很多（如 Prefill 512 tokens）时，kernel launch 开销变得显著。Triton kernel 将所有 scatter 操作融合为一个 GPU kernel。

## store_kvcache_kernel 详解

完整代码见 [[../04_模型层实现/04.2_注意力机制与FlashAttention/注意力机制与FlashAttention]]。

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                          k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
                          D: tl.constexpr):
    idx = tl.program_id(0)                    # 当前处理的 token 索引
    slot = tl.load(slot_mapping_ptr + idx)    # 该 token 对应的物理 slot
    if slot == -1: return                     # CUDA Graph 填充位，跳过

    key_offsets = idx * key_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)      # 加载当前 token 的 K
    value = tl.load(value_ptr + value_offsets) # 加载当前 token 的 V

    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key) # 写入 KV-Cache
    tl.store(v_cache_ptr + cache_offsets, value)
```

### 性能特征

| 方案 | Kernel Launch 次数 | 内存访问模式 |
|------|---------------------|-------------|
| PyTorch scatter | N 次（每个 token 一次） | 随机写入（scatter） |
| Triton fused | 1 次 | 合并的加载 + 散射写入 |

在 Prefill 阶段（N=512 tokens），Triton 方案减少约 512 次 kernel launch，节省约 2-5ms 的 CPU-GPU 同步开销。

### Triton Program 的并行模型

```python
store_kvcache_kernel[(N,)]    # 启动 N 个 program
```

每个 program（对应一个 CUDA thread block）处理一个 token。Triton 的 `tl.program_id(0)` 返回当前 program 的索引（0 到 N-1），这是 Triton SPMD 模型的核心——类似于 CUDA 的 `blockIdx.x`。

### 为什么用 tl.constexpr？

```python
D: tl.constexpr,
```

`D`（num_heads × head_dim）在编译时确定，允许 Triton 为其生成优化的加载/存储指令。对于 Qwen3-0.6B（num_kv_heads=8, head_dim=64），D=512，Triton 可以为 512 个 float16 的连续加载生成高效的向量化指令。

### KV-Cache 的 strided 访问

```python
k_cache.stride(1) == D  # 确保 KV-Cache 最后一维连续
```

`k_cache` 的 6D 形状被 view 为 `[num_blocks * block_size, D]`（2D），stride 的验证确保 Triton kernel 的偏移计算不会出错。如果 stride 不匹配，`tl.store(k_cache_ptr + cache_offsets, ...)` 会写入错误位置。

## Torch Compile 优化的算子

除了 Triton kernel，nano-vllm 还用 `@torch.compile` 优化了三个小算子：

### 1. RMSNorm::rms_forward / add_rms_forward

```
原始 PyTorch: pow → mean → rsqrt → mul → mul
Torch Inductor 将 5 个操作融合为 1 个 kernel
收益: 减少 ~4 次全局显存读写 / 每次调用
```

### 2. SiluAndMul::forward

```
原始 PyTorch: chunk → silu → mul
融合后: 单 kernel 完成分块 + 激活 + 乘法
收益: 减少 ~2 次中间 tensor 的显存分配
```

### 3. Sampler::forward

```
原始 PyTorch: div → softmax → exponential → div → argmax
融合后: 多步融合 + softmax 的 online 计算
收益: 减少 ~3 次中间 tensor 分配
```

## 为何不多用 Triton？

nano-vllm 的设计哲学是"尽可能复用现有库，只在必要时手写 kernel"。FlashAttention 已经提供了高度优化的 attention kernel，`torch.compile` 覆盖了小算子融合，Triton 只用在 scatter 这种 PyTorch 确实做不好的操作上。这个判断非常务实——在 ~1200 行代码中保持如此少的自定义 kernel 是项目简洁性的关键。
