---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - Embedding
  - LM_Head
created: 2026-07-07
up: "[[模型层实现总览|模型层实现总览]]"
---

# Embedding 与 LM Head

`embed_head.py` 实现了两个紧密相关的组件：`VocabParallelEmbedding`（词嵌入）和 `ParallelLMHead`（语言模型输出头）。它们共享同一个基类并通过 `tie_word_embeddings` 共享权重。

## VocabParallelEmbedding — 词汇并行嵌入

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(num_embeddings_per_partition, embedding_dim))
```

**词表分片示意**（TP=2, vocab_size=151936）：

```
完整词表: 0..151935 个嵌入
  ├─ GPU 0: 0..75967   (vocab_start=0,     vocab_end=75968)
  └─ GPU 1: 75968..151935 (vocab_start=75968, vocab_end=151936)
```

### 前向传播：mask + all_reduce 实现并行

```python
def forward(self, x):
    if self.tp_size > 1:
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        x = mask * (x - self.vocab_start_idx)           # 将全局 ID 转换为本地 ID
    y = F.embedding(x, self.weight)
    if self.tp_size > 1:
        y = mask.unsqueeze(1) * y                       # 只保留本 GPU 有嵌入的 token
        dist.all_reduce(y)                               # 跨 GPU 求和
    return y
```

并行 Embedding 的核心逻辑分为三步：

1. **分发**：每个 token 根据其词汇 ID 所在的分片范围，只在一个 GPU 上触发嵌入查找。`mask` 用于标记哪些 token 属于当前 GPU
2. **查找**：`F.embedding` 在本地词表分片中查找（OOV token 被 mask 清零导致查找结果无效，但会被置零）
3. **合并**：`mask * y` 将不属于本 GPU 的 token 的嵌入置零，然后 `all_reduce` 求和——只有正确的 GPU 会产生非零值

**示例**：输入 token_ids = [100, 76000]，TP=2, vocab_per_partition=75968

```
GPU 0 (vocab 0..75967):
  token 100: mask=True,  local_id=100,     embedding=[e100]  ✓
  token 76000: mask=False, local_id=32,     embedding=[e32]   → mask×y=0
  输出: [e100, 0]

GPU 1 (vocab 75968..151935):
  token 100: mask=False, local_id=100,     embedding=[e100'] → mask×y=0
  token 76000: mask=True,  local_id=32,     embedding=[e32']  ✓
  输出: [0, e32']

all_reduce: [e100, 0] + [0, e32'] = [e100, e32']  ← 正确结果
```

### weight_loader — 权重加载

```python
def weight_loader(self, param, loaded_weight):
    shard_size = param_data.size(0)
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
    param_data.copy_(loaded_weight)
```

每个 GPU 只加载自己份额的嵌入权重（如 `[75968, dim]`），与 ColumnParallelLinear 的加载逻辑一致。

## ParallelLMHead — 语言模型输出头

```python
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings, embedding_dim, bias=False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x):
        context = get_context()
        if context.is_prefill:
            # Prefill: 只取每个序列的最后一个 token 的 hidden state
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits)
                          for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)     # 聚集到 rank 0
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

### Prefill 阶段的 last token 处理

与 Embedding 层不同，LM Head 在 Prefill 阶段只需要**每个序列的最后一个 token** 的 logits（因为只有最后一个 token 用于预测下一个 token）。它通过 `cu_seqlens_q` 找到每个序列的结束位置：

```
cu_seqlens_q = [0, 512, 1024, 1536]  (3 个序列，长度分别为 512, 512, 512)
last_indices = [511, 1023, 1535]     (每个序列最后一个 token 的位置)
```

Decode 阶段不需要这个优化，因为此时输入只有一个 token。

### TP 下的 logits 收集

LM Head 的输出需要完整词表的 logits 才能采样。在 TP 模式下：

```
每个 GPU 计算: logits_partial = hidden @ W_partial^T  [batch, vocab/tp]
使用 dist.gather 将所有 GPU 的部分 logits 聚合到 rank 0
rank 0: logits = cat([logits_gpu0, logits_gpu1, ...])  [batch, vocab]
```

`dist.gather` 比 `dist.all_gather` 更高效——只需要 rank 0 获得完整 logits（因为采样只在 rank 0 进行），不需要在每个 GPU 上都复制一份。

## Weight Tying（权重绑定）

```python
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

Qwen3 默认开启 `tie_word_embeddings`，即输入嵌入层和输出投影层共享权重。nano-vllm 通过直接将 `lm_head.weight.data` 指向 `embed_tokens.weight.data` 实现——注意是 `.data` 赋值而非 `=`，确保两个参数共享同一个 Tensor 存储，避免了额外的显存开销。
