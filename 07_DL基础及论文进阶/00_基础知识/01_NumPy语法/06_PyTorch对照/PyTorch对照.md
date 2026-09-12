---
title: PyTorch对照实现
tags:
  - NumPy
  - PyTorch
  - 对照
  - 基础知识
created: 2026-07-01
up: "[[01_NumPy语法|NumPy语法]]"
---

# 6. PyTorch 对照实现

将 `00_PyTorch语法` 中的核心操作逐个用 NumPy 实现。理解这些底层细节对面试（手写 BN、手写卷积、手写反向传播）非常有帮助。

> **使用说明**：每个子文件中的代码有跨文件依赖（如 SwiGLU 依赖 `sigmoid_np`、Encoder Block 依赖 `dropout_forward_np`）。如需完整可运行代码，请按 6.1→6.2→6.3→6.4→6.5→6.6 的顺序组合使用。

## 子文件目录

- [[06.1_基础操作对照|6.1 基础操作对照]] — 张量创建、形状变换、激活函数、交叉熵/MSE 损失
- [[06.2_基础层对照|6.2 基础层对照]] — 全连接层、BatchNorm、卷积（im2col）、最大池化、Dropout
- [[06.3_优化器对照|6.3 优化器对照]] — SGD、Adam
- [[06.4_注意力机制对照|6.4 注意力机制对照]] — Sinusoidal PE、RoPE、缩放点积注意力、GQA、MHA
- [[06.5_Transformer层对照|6.5 Transformer 层对照]] — FFN、SwiGLU、LayerNorm、RMSNorm
- [[06.6_架构组装对照|6.6 架构组装对照]] — 原始 Encoder Block、LLaMA Block、多层堆叠、架构对比

## 对照速查表

| 操作 | PyTorch | NumPy |
|---|---|---|
| 创建全零 | `torch.zeros(3,4)` | `np.zeros((3,4))` |
| 正态随机 | `torch.randn(2,3)` | `np.random.randn(2,3)` |
| 变形 | `x.view(-1,2)` | `x.reshape(-1,2)` |
| 增删维度 | `x.unsqueeze(0)` | `np.expand_dims(x,0)` |
| 矩阵乘法 | `a @ b` | `a @ b` |
| ReLU | `F.relu(x)` | `np.maximum(0, x)` |
| Softmax | `F.softmax(x,dim=-1)` | (见 6.1) |
| Sigmoid | `torch.sigmoid(x)` | (见 6.1) |
| SiLU/Swish | `F.silu(x)` | (见 6.1) |
| CE Loss | `nn.CrossEntropyLoss()` | (见 6.1) |
| Linear | `nn.Linear(in,out)` | `x @ w.T + b` |
| BatchNorm | `nn.BatchNorm1d` | (见 6.2) |
| Conv2d | `nn.Conv2d` | (见 6.2, im2col + matmul) |
| MaxPool2d | `nn.MaxPool2d` | (见 6.2) |
| Dropout | `nn.Dropout(p)` | (见 6.2, inverted dropout) |
| SGD | `optim.SGD` | (见 6.3) |
| Adam | `optim.Adam` | (见 6.3) |
| Sinusoidal PE | `torch.sin`/`torch.cos` 手动 | (见 6.4) |
| RoPE | 自实现 | (见 6.4, rope_apply_np) |
| Scaled Dot-Product Attention | `F.scaled_dot_product_attention` | (见 6.4) |
| Multi-Head Attention | `nn.MultiheadAttention` | (见 6.4) |
| GQA | 自实现 | (见 6.4) |
| FFN | `nn.Linear+ReLU+nn.Linear` | (见 6.5) |
| SwiGLU FFN | 自实现 | (见 6.5) |
| LayerNorm | `nn.LayerNorm` | (见 6.5) |
| RMSNorm | `nn.RMSNorm`（自实现） | (见 6.5) |
| Transformer Encoder Block | `nn.TransformerEncoderLayer` | (见 6.6) |
| LLaMA Block | 自实现 | (见 6.6) |
| Transformer Encoder | `nn.TransformerEncoder` | (见 6.6) |
| GPU | `.to(device)` / `.cuda()` | 不支持 |
| autograd | `loss.backward()` | 需手写反向传播 |
