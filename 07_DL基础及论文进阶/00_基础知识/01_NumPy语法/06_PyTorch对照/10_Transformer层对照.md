---
title: Transformer层对照
tags:
  - NumPy
  - PyTorch
  - Transformer
  - 基础知识
created: 2026-07-01
up: "[[01_NumPy语法|NumPy语法]]"
---

# 6.5 Transformer 层对照

FFN、SwiGLU、LayerNorm、RMSNorm 的 NumPy 实现。

## 6.5.1 前馈网络（Position-wise Feed-Forward Network）

```python
class FeedForward_Numpy:
    """
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    等价于 PyTorch:
        nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    """

    def __init__(self, d_model=512, d_ff=2048):
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)

        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        self.x = x
        self.hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return self.hidden @ self.W2 + self.b2

    def backward(self, dout):
        N, seq_len, d_model = dout.shape

        dout_flat = dout.reshape(-1, d_model)
        hidden_flat = self.hidden.reshape(-1, self.hidden.shape[-1])
        self.dW2 = hidden_flat.T @ dout_flat
        self.db2 = np.sum(dout_flat, axis=0)
        dhidden = dout_flat @ self.W2.T

        dhidden_flat = dhidden.reshape(self.hidden.shape)
        dhidden_flat[self.hidden <= 0] = 0

        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dhidden2d = dhidden_flat.reshape(-1, dhidden_flat.shape[-1])
        self.dW1 = x_flat.T @ dhidden2d
        self.db1 = np.sum(dhidden2d, axis=0)
        dx = dhidden2d @ self.W1.T

        return dx.reshape(N, seq_len, -1)

    def step(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


# ── 测试 ──
ffn = FeedForward_Numpy(d_model=512, d_ff=2048)
x = np.random.randn(4, 10, 512)
out = ffn.forward(x)
print("FFN 输出形状:", out.shape)  # (4, 10, 512)
```

## 6.5.2 SwiGLU（SwiGLU 激活前馈网络）

SwiGLU 是 LLaMA、PaLM 等现代大模型使用的 FFN 变体。公式：

`SwiGLU(x) = (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down`

```python
class SwiGLU_FFN_Numpy:
    """
    SwiGLU 前馈网络（LLaMA 风格）
    结构：x → Linear(d_model, d_ff) + Linear(d_model, d_ff)（双路）
            → Swish 门控 × 上投影 → Linear(d_ff, d_model)

    PyTorch 等价（LLaMA 源码）:
        gate = F.silu(x @ w1)
        up = x @ w3
        out = (gate * up) @ w2
    """

    def __init__(self, d_model=512, d_ff=2048):
        # SwiGLU 的 d_ff 设为 8/3 * d_model
        # 双路投影，hidden_dim 调整为 256 倍数（硬件友好）
        hidden_dim = int(2 * d_ff / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        scale = np.sqrt(2.0 / d_model)
        self.hidden_dim = hidden_dim

        self.W_gate = np.random.randn(d_model, hidden_dim) * scale
        self.W_up = np.random.randn(d_model, hidden_dim) * scale
        self.W_down = np.random.randn(hidden_dim, d_model) * scale

        self.b_gate = np.zeros(hidden_dim)
        self.b_up = np.zeros(hidden_dim)
        self.b_down = np.zeros(d_model)

    def silu_grad_np(self, x):
        s = sigmoid_np(x)  # 需导入 6.1.3 的 sigmoid_np
        return s + x * s * (1 - s)

    def forward(self, x):
        self.x = x
        self.gate_pre = x @ self.W_gate + self.b_gate
        self.gate = x * sigmoid_np(self.gate_pre)  # SiLU
        self.up = x @ self.W_up + self.b_up
        self.hidden = self.gate * self.up            # 门控融合
        return self.hidden @ self.W_down + self.b_down

    def backward(self, dout):
        N, seq_len, d_model = dout.shape
        dout_flat = dout.reshape(-1, d_model)
        hidden_flat = self.hidden.reshape(-1, self.hidden_dim)
        self.dW_down = hidden_flat.T @ dout_flat
        self.db_down = np.sum(dout_flat, axis=0)

        dhidden = dout_flat @ self.W_down.T
        dhidden = dhidden.reshape(N, seq_len, self.hidden_dim)

        dgate = dhidden * self.up
        dup = dhidden * self.gate

        up_flat = dup.reshape(-1, self.hidden_dim)
        x_flat = self.x.reshape(-1, d_model)
        self.dW_up = x_flat.T @ up_flat
        self.db_up = np.sum(up_flat, axis=0)

        dgate_pre_flat = (dgate * self.silu_grad_np(self.gate_pre)).reshape(-1, self.hidden_dim)
        self.dW_gate = x_flat.T @ dgate_pre_flat
        self.db_gate = np.sum(dgate_pre_flat, axis=0)

        dx = up_flat @ self.W_up.T + dgate_pre_flat @ self.W_gate.T
        return dx.reshape(N, seq_len, d_model)

    def step(self, lr):
        for name in ['W_gate', 'W_up', 'W_down', 'b_gate', 'b_up', 'b_down']:
            grad_name = 'd' + name
            if hasattr(self, grad_name):
                setattr(self, name, getattr(self, name) - lr * getattr(self, grad_name))


# ── 测试 ──
# 注意：sigmoid_np 需要从 6.1.3 导入
# 此处省略 import，完整可运行代码见 6.6 章
print("SwiGLU 实现就绪（需配合 sigmoid_np 使用）")
```

**SwiGLU 为什么比标准 FFN 好？**（面试要点）

1. **门控机制**：`gate * up` 相当于可学习的"信息过滤器"——gate 决定哪些特征通过。相比 ReLU 的硬截断（<0 直接归零），Sigmoid 的软门控保留更多信息。
2. **平滑梯度**：SiLU 全域可导且梯度光滑（不像 ReLU 在 0 处一阶导数不连续），训练更稳定。
3. **实证效果**：PaLM 论文表明白样参数量下 SwiGLU 的困惑度显著低于 ReLU FFN。

## 6.5.3 Layer Normalization

```python
class LayerNorm_Numpy:
    """
    torch.nn.LayerNorm 的 NumPy 实现
    对每个样本独立地在最后一个维度上归一化
    与 BN 的关键区别：不依赖 batch 统计，每个样本独立计算
    """

    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def forward(self, x):
        self.mu = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.x_normalized = (x - self.mu) * self.std_inv
        return self.gamma * self.x_normalized + self.beta

    def backward(self, dout):
        D = dout.shape[-1]
        self.dgamma = np.sum(dout * self.x_normalized,
                             axis=tuple(range(dout.ndim - 1)))
        self.dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
        dx_norm = dout * self.gamma

        dx = (1.0 / D) * self.std_inv * (
            D * dx_norm
            - np.sum(dx_norm, axis=-1, keepdims=True)
            - self.x_normalized * np.sum(
                dx_norm * self.x_normalized, axis=-1, keepdims=True)
        )
        return dx

    def step(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


# ── 测试 ──
ln = LayerNorm_Numpy(normalized_shape=512)
x = np.random.randn(4, 10, 512)
out = ln.forward(x)
print("LayerNorm 输出均值:", out.mean(axis=-1)[0, 0])  # 接近 0
print("LayerNorm 输出方差:", out.var(axis=-1)[0, 0])   # 接近 1
```

## 6.5.4 RMSNorm（RMS 归一化）

```python
class RMSNorm_Numpy:
    """
    RMSNorm: y = x / RMS(x) * gamma
    RMS(x) = sqrt(mean(x^2) + eps)

    PyTorch 等价（LLaMA 实现）
    注意：只有 gamma（weight），没有 beta（bias）！
    """

    def __init__(self, dim, eps=1e-6):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones(dim)

    def forward(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        self.x_normalized = x / rms
        self.rms = rms
        self.x = x
        return self.gamma * self.x_normalized

    def backward(self, dout):
        self.dgamma = np.sum(dout * self.x_normalized,
                             axis=tuple(range(dout.ndim - 1)))
        dx_norm = dout * self.gamma
        correction = np.mean(dx_norm * self.x_normalized, axis=-1, keepdims=True)
        dx = (dx_norm - self.x_normalized * correction) / self.rms
        return dx

    def step(self, lr):
        self.gamma -= lr * self.dgamma


# ── RMSNorm vs LayerNorm 对比 ──
rms_norm = RMSNorm_Numpy(dim=512)
ln = LayerNorm_Numpy(normalized_shape=512)
x = np.random.randn(4, 10, 512)

out_rms = rms_norm.forward(x)
out_ln = ln.forward(x)

print("RMSNorm 输出 RMS(最后维):", np.sqrt(np.mean(out_rms[0, 0] ** 2)))  # 接近 1
print("RMSNorm 输出均值:", out_rms.mean(axis=-1)[0, 0])   # 不一定是 0!
print("LayerNorm 输出均值:", out_ln.mean(axis=-1)[0, 0])  # 接近 0
```

## 6.5.5 归一化层对比

| 操作 | LayerNorm | RMSNorm | BatchNorm |
|---|---|---|---|
| 中心化（减均值） | ✅ | ❌ | ✅ |
| 缩放（除以标准差/RMS） | ✅（方差） | ✅（RMS） | ✅（方差） |
| 平移（beta） | ✅ | ❌ | ✅ |
| 缩放（gamma） | ✅ | ✅ | ✅ |
| 计算量 | mean+var（2次reduce） | mean(x²)（1次reduce） | mean+var（2次reduce） |
| batch依赖 | 无 | 无 | 有 |
| 大模型首选 | 原始Transformer | ✅ LLaMA/Gemma | ❌（不适合序列） |

**为什么大模型首选 RMSNorm？**

1. **更快**：少一次 reduce，对大模型累积节省显著。
2. **更少参数**：没有 beta，内存更小。
3. **效果相当**：去掉零均值约束对训练质量几乎没有影响。
