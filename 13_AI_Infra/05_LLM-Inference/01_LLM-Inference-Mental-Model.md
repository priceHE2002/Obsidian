---
tags: [llm, inference, miniserve]
---
# LLM Inference Mental Model

## 一次在线推理

Request → Tokenizer → Prefill → KV Cache → Decode Loop → Sampling → Streaming

## 你必须能从 Tensor Shape 解释系统

常见逻辑形状：

```text
[batch, num_heads, seq_len, head_dim]
```

重点不是背 shape，而是理解 batch、sequence length、KV heads 的变化如何影响显存与计算。

## 推理系统的三个核心维度

1. Scheduling：GPU 每一轮跑谁？
2. Memory：每个 request 的 KV 放哪里？
3. Compute：这些算子怎样在 GPU 上跑得更快？

这就是 [[13_AI_Infra/13_MiniServe/00_MiniServe-Architecture|MiniServe]] 的主线。
