---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - RoPE
  - 位置编码
created: 2026-07-07
up: "[[模型层实现总览|模型层实现总览]]"
---

# RoPE 旋转位置编码

`rotary_embedding.py` 实现了标准的 Rotary Position Embedding (RoPE)，支持通过 `@lru_cache` 缓存实例避免重复创建。

## apply_rotary_emb — 核心旋转变换

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

RoPE 的数学原理：将 head_dim 维的向量两两组对，对每组 `(x_{2i}, x_{2i+1})` 施加 2D 旋转：

```
[x_{2i}']   [cos θ_i  -sin θ_i] [x_{2i}]
[x_{2i+1}'] = [sin θ_i   cos θ_i] [x_{2i+1}]

其中 θ_i = base^{-2i/d}, base 通常为 10000（Qwen3 为 1,000,000）
```

实现中使用 `float()` 提升精度做旋转计算，再转回原始 dtype——这是 RoPE 实现的标准实践，避免低精度下的旋转精度损失。

## RotaryEmbedding — 缓存预计算

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float)
                                   / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rotary_dim/2]

        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)  # [max_pos, 1, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)
```

关键细节：
- `persistent=False`：这个 buffer 不会被保存到 state_dict，因为它是纯计算缓存，可以从配置重新生成
- `unsqueeze_(1)` 插入的维度用于支持 `[seq_len, num_heads, dim]` 的广播
- `torch.einsum("i,j -> ij", t, inv_freq)` 计算所有 `t * θ_i` 的外积

## @torch.compile 加速的 forward

```python
@torch.compile
def forward(self, positions, query, key):
    cos_sin = self.cos_sin_cache[positions]    # 按位置索引
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = apply_rotary_emb(query, cos, sin)
    key = apply_rotary_emb(key, cos, sin)
    return query, key
```

`@torch.compile` 在这里特别有效——RoPE 的 forward 包含 `chunk`、索引、逐元素运算，这些都是 Torch Inductor 擅长融合的操作。编译后可以将多次内存读写合并为单次 kernel。

## LRU 缓存工厂函数

```python
@lru_cache(1)
def get_rope(head_size, rotary_dim, max_position, base):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

`@lru_cache(1)` 的设计很精妙：通常一个模型只有一种 RoPE 配置（所有层共享同一组参数），所以缓存最近 1 个实例刚好够用。如果将来需要支持不同层的不同 RoPE 配置，改 `lru_cache(maxsize=N)` 即可。

## RoPE 在推理中的位置索引

```python
# Prefill 阶段
positions = range(start, end)    # 如 [0, 1, 2, ..., 511]
# Decode 阶段
positions = [len(seq) - 1]       # 如 [512]（单个位置）
```

Decode 阶段只对最新的一个 token 做 RoPE——历史 token 的 KV 已经包含在 KV-Cache 中，不需要重新计算 RoPE。
