---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - Qwen3
  - 模型架构
created: 2026-07-07
up: "[[模型层实现总览|模型层实现总览]]"
---

# Qwen3 模型架构

nano-vllm 目前只支持 Qwen3 架构（最简洁的现代 Decoder-only LLM），但通过模块化的层设计，添加新模型只需重新组合现有组件。

## Qwen3ForCausalLM — 顶层模型

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),       # HuggingFace 名称 → nanovllm 名称
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:       # 权重绑定
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

`forward()` 和 `compute_logits()` 分开调用的原因：CUDA Graph 捕获时，需要在 replay 后才计算 logits（因为 LM Head 涉及 `dist.gather`，不适合放入 CUDA Graph）。

### packed_modules_mapping

用于 `load_model()` 中兼容 HuggingFace 格式的 safetensors 权重。HuggingFace 的 Qwen3 将 QKV 分开存储（`q_proj`, `k_proj`, `v_proj`），但 nano-vllm 将它们合并为一个 `QKVParallelLinear` 以减少 kernel launch 次数。`packed_modules_mapping` 告诉 loader 如何映射：

```
HF weight: model.layers.0.self_attn.q_proj.weight
    映射为: model.layers.0.self_attn.qkv_proj.weight 的 "q" 部分

HF weight: model.layers.0.mlp.gate_proj.weight
    映射为: model.layers.0.mlp.gate_up_proj.weight 的 shard 0
```

## Qwen3Model — 主体模型

```python
class Qwen3Model(nn.Module):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

### Residual 流设计

与标准 Transformer 不同，nano-vllm 的 Qwen3 使用**带 fused residual add 的 RMSNorm**：

```python
# DecoderLayer.forward() 中：
if residual is None:
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
# ... self_attn ...
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
# ... mlp ...
return hidden_states, residual
```

这个设计的关键优势是 **fused kernel**——RMSNorm 的 `add_rms_forward` 将 residual add 和 RMSNorm 融合为一个操作，减少了内存读写：

```python
# layernorm.py 中的 fused 操作
def add_rms_forward(self, x, residual):
    x = x.float().add_(residual.float())  # 先原地 add
    residual = x.to(orig_dtype)           # 保存新的 residual
    # ... 然后 RMSNorm ...
    return x, residual
```

## Qwen3DecoderLayer — Decoder 层

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        self.self_attn = Qwen3Attention(...)
        self.mlp = Qwen3MLP(...)
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
```

每层 = Pre-Attention RMSNorm → Self-Attention → Post-Attention RMSNorm → MLP，标准的 Pre-Norm Transformer。

## Qwen3Attention — 注意力模块

```python
class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
        # TP 下划分 heads
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size

        # QKV 合并投影
        self.qkv_proj = QKVParallelLinear(hidden_size, head_dim, num_heads, num_kv_heads)
        # 输出投影（Row Parallel = 输入分片 + all_reduce）
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size)
        # RoPE
        self.rotary_emb = get_rope(head_dim, ...)
        # FlashAttention
        self.attn = Attention(self.num_heads, head_dim, scaling, num_kv_heads)
        # QK Norm (Qwen3 特有：无 bias 时的 QK 归一化)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
```

`q_norm` 和 `k_norm` 是 Qwen3 的一个特殊设计：当不使用 QKV bias 时，对 Q 和 K 分别做 RMSNorm，稳定注意力计算的数值范围。

## Qwen3MLP — FFN 模块

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()          # gate 用 SiLU × up

    def forward(self, x):
        gate_up = self.gate_up_proj(x)     # [gate | up] 拼接
        x = self.act_fn(gate_up)           # SiLU(gate) * up
        x = self.down_proj(x)
        return x
```

SwiGLU 激活的标准实现：`MergeColumnParallelLinear` 将 gate 和 up 投影合并为一个矩阵乘法，然后 `SiluAndMul` 做 SiLU(gate) × up。
