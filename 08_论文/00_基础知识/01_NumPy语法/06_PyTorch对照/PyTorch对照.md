---
title: PyTorch对照实现
tags:
  - NumPy
  - PyTorch
  - 对照
  - 基础知识
created: 2026-07-01
up: "[[00_基础知识/01_NumPy语法/01_NumPy语法|NumPy语法]]"
---

# 6. PyTorch 对照实现

本章将 `00_PyTorch语法` 中的核心操作逐个用 NumPy 实现。理解这些底层细节对面试（手写 BN、手写卷积、手写反向传播）非常有帮助。

## 6.1 张量创建对照

```python
import numpy as np

# ── 对照表 ──
# torch.zeros(3, 4)           → np.zeros((3, 4))
# torch.ones(2, 3)            → np.ones((2, 3))
# torch.rand(2, 3)            → np.random.random((2, 3))
# torch.randn(2, 3)           → np.random.randn(2, 3)
# torch.randint(0, 10, (3,4)) → np.random.randint(0, 10, (3, 4))
# torch.full((2,3), 7.0)      → np.full((2, 3), 7.0)
# torch.arange(0, 10, 2)      → np.arange(0, 10, 2)
# torch.linspace(0, 1, 5)     → np.linspace(0, 1, 5)
# torch.eye(4)                → np.eye(4)
# torch.from_numpy(arr)       → 本身就是 numpy 数组
# t.numpy()                   → 本身就是 numpy 数组

# torch.tensor([[1,2],[3,4]]) → np.array([[1, 2], [3, 4]])
# x.shape                     → x.shape
# x.dtype                     → x.dtype
# x.device                    → NumPy 无 GPU，只有 CPU
# x.to(device)                → NumPy 无此概念
```

## 6.2 形状变换对照

```python
# torch.view / torch.reshape → np.reshape（NumPy 无 view/reshape 区别）
x = np.random.randn(4, 4)

y = x.reshape(16)
y = x.reshape(-1, 2)

# 展平
y = x.reshape(-1)          # torch.view(-1)
y = x.flatten()            # torch.flatten()
y = x.ravel()              # 无完全对应，ravel 返回视图

# 增删维度
y = np.expand_dims(x, 0)   # torch.unsqueeze(x, 0)
y = np.expand_dims(x, -1)  # torch.unsqueeze(x, -1)
y = np.squeeze(y)          # torch.squeeze(y)

# 转置
y = x.T                    # torch.t()
y = np.transpose(x, (1,0)) # torch.permute(1, 0)

# torch.repeat              → np.tile
y = np.tile(x, (2, 1))     # (4,4) → (8,4)
```

## 6.3 激活函数

```python
def sigmoid_np(x):
    """torch.sigmoid — 注意数值稳定性"""
    # 裁剪避免溢出
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def relu_np(x):
    """torch.relu / F.relu"""
    return np.maximum(0, x)


def leaky_relu_np(x, negative_slope=0.01):
    """F.leaky_relu"""
    return np.where(x > 0, x, negative_slope * x)


def tanh_np(x):
    """torch.tanh"""
    return np.tanh(x)  # NumPy 自带


def softmax_np(x, axis=-1):
    """F.softmax — 数值稳定版（减去 max）"""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x_shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def gelu_np(x):
    """F.gelu — 近似版本"""
    return 0.5 * x * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    ))


# ── 测试 ──
x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("Sigmoid:", sigmoid_np(x))
print("ReLU:   ", relu_np(x))
print("Leaky:  ", leaky_relu_np(x))
print("Softmax:", softmax_np(x))
print("GELU:   ", gelu_np(x))
```

## 6.4 交叉熵损失

`torch.nn.CrossEntropyLoss` = Softmax + NLLLoss 的组合。下面用 NumPy 实现。

```python
def cross_entropy_loss_np(logits, targets):
    """
    logits: (N, C) — 原始logit，未经过 softmax
    targets: (N,) — 整数标签
    返回: 标量 loss
    等价于 torch.nn.CrossEntropyLoss()
    """
    N = logits.shape[0]
    # 数值稳定的 softmax
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits_shifted)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # 取正确类别的负对数似然
    correct_log_probs = -np.log(probs[np.arange(N), targets])
    loss = np.mean(correct_log_probs)
    return loss


def cross_entropy_grad_np(logits, targets):
    """CrossEntropyLoss 对 logits 的梯度"""
    N = logits.shape[0]
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits_shifted)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # 梯度 = softmax概率 - one_hot(label)
    grad = probs.copy()
    grad[np.arange(N), targets] -= 1
    grad /= N
    return grad


# ── 测试 ──
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.0, 0.3]])
targets = np.array([0, 1])
print("CrossEntropy Loss:", cross_entropy_loss_np(logits, targets))
print("CrossEntropy Grad:\n", cross_entropy_grad_np(logits, targets))
```

## 6.5 MSE 损失

```python
def mse_loss_np(pred, target):
    """torch.nn.MSELoss()"""
    return np.mean((pred - target) ** 2)


def mse_grad_np(pred, target):
    """MSE 对 pred 的梯度"""
    N = pred.shape[0]
    return 2 * (pred - target) / N
```

## 6.6 全连接层

```python
def linear_forward_np(x, w, b):
    """torch.nn.Linear 的 forward"""
    return x @ w.T + b     # x: (N, in), w: (out, in), b: (out,)


def linear_backward_np(dout, x, w):
    """
    线性层反向传播
    dout: 上游梯度 (N, out)
    返回: dx, dw, db
    """
    dx = dout @ w            # (N, out) @ (out, in) → (N, in)
    dw = dout.T @ x          # (out, N) @ (N, in) → (out, in)
    db = np.sum(dout, axis=0)# (out,)
    return dx, dw, db


# ── 测试 ──
x = np.random.randn(4, 10)
w = np.random.randn(5, 10) * 0.01
b = np.zeros(5)

out = linear_forward_np(x, w, b)
print("Linear output shape:", out.shape)  # (4, 5)
```

## 6.7 批量归一化（BatchNorm）

```python
class BatchNorm1d_Numpy:
    """手写 nn.BatchNorm1d —— 面试高频"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # 缓存：反向传播用
        self.x_centered = None
        self.x_normalized = None
        self.inv_std = None

    def forward(self, x, training=True):
        """
        x: (N, D)
        training: True 时用 batch 统计，False 时用 running 统计
        """
        if training:
            mu = np.mean(x, axis=0)          # (D,)
            var = np.var(x, axis=0)          # (D,)
            std = np.sqrt(var + self.eps)

            # 前向
            self.x_centered = x - mu
            self.x_normalized = self.x_centered / std
            out = self.gamma * self.x_normalized + self.beta

            # 保存反向传播用
            self.inv_std = 1.0 / std
            self.mu = mu
            self.var = var

            # 更新 running 统计（滑动平均）
            self.running_mean = (1 - self.momentum) * self.running_mean \
                                + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var \
                               + self.momentum * var
        else:
            # 推理：用训练期间累计的 running 统计
            x_normalized = (x - self.running_mean) / \
                           np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):
        """
        完整的 BN 反向传播
        dout: 上游梯度，shape (N, D)
        返回: dx — 对输入的梯度
        """
        N = dout.shape[0]

        # 对 gamma 和 beta 的梯度
        dgamma = np.sum(dout * self.x_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)
        self.dgamma = dgamma
        self.dbeta = dbeta

        # 对 x_normalized 的梯度
        dx_norm = dout * self.gamma

        # 通过标准化层传播
        dx = (1.0 / N) * self.inv_std * (
            N * dx_norm
            - np.sum(dx_norm, axis=0)
            - self.x_normalized * np.sum(dx_norm * self.x_normalized, axis=0)
        )

        return dx

    def step(self, lr):
        """更新 gamma 和 beta"""
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


# ── 测试 ──
np.random.seed(42)
x = np.random.randn(8, 4)   # batch=8, features=4
bn = BatchNorm1d_Numpy(4)

# 训练模式
out_train = bn.forward(x, training=True)
print("BN 训练输出均值:", out_train.mean(axis=0))  # 接近 0
print("BN 训练输出方差:", out_train.var(axis=0))  # 接近 1

# 模拟反向传播
dout = np.random.randn(8, 4)
dx = bn.backward(dout)
bn.step(lr=0.1)

# 推理模式
out_eval = bn.forward(x, training=False)
print("BN 推理输出 shape:", out_eval.shape)
```

## 6.8 卷积操作（im2col 实现）

```python
def im2col_np(x, kernel_h, kernel_w, stride, padding):
    """
    将图像展开为列矩阵，方便用矩阵乘法实现卷积
    x: (N, C, H, W)
    返回: (N * out_h * out_w, C * kernel_h * kernel_w)
    """
    N, C, H, W = x.shape

    # padding
    if padding > 0:
        x_padded = np.pad(x,
            ((0,0), (0,0), (padding,padding), (padding,padding)),
            mode='constant')
    else:
        x_padded = x

    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    cols = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x_i in range(kernel_w):
            x_max = x_i + stride * out_w
            cols[:, :, y, x_i, :, :] = \
                x_padded[:, :, y:y_max:stride, x_i:x_max:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3)
    cols = cols.reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


def col2im_np(cols, N, C, H, W, kernel_h, kernel_w, stride, padding):
    """im2col 的逆运算（用于反向传播）"""
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    cols_reshaped = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    cols_reshaped = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * padding + stride - 1,
                    W + 2 * padding + stride - 1))

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x_i in range(kernel_w):
            x_max = x_i + stride * out_w
            img[:, :, y:y_max:stride, x_i:x_max:stride] += \
                cols_reshaped[:, :, y, x_i, :, :]

    return img[:, :, padding:H+padding, padding:W+padding]


def conv2d_forward_np(x, weight, bias=None, stride=1, padding=0):
    """
    简化版 Conv2d forward
    x: (N, C_in, H, W)
    weight: (C_out, C_in, K, K)
    返回: (N, C_out, out_h, out_w)
    """
    N, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape

    out_h = (H + 2 * padding - K) // stride + 1
    out_w = (W + 2 * padding - K) // stride + 1

    # im2col
    cols, _, _ = im2col_np(x, K, K, stride, padding)  # (N*OH*OW, C_in*K*K)

    # 卷积核展开
    w_rows = weight.reshape(C_out, -1)  # (C_out, C_in*K*K)

    # 矩阵乘法 → 卷积！
    out = cols @ w_rows.T  # (N*OH*OW, C_out)
    out = out.reshape(N, out_h, out_w, C_out)
    out = out.transpose(0, 3, 1, 2)  # (N, C_out, OH, OW)

    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)

    return out


# ── 测试 ──
x = np.random.randn(2, 3, 32, 32)  # 模拟图像
w = np.random.randn(16, 3, 3, 3) * 0.01  # 16 个 3x3 卷积核

out = conv2d_forward_np(x, w, stride=1, padding=1)
print("NumPy Conv2d 输出:", out.shape)  # (2, 16, 32, 32)
```

## 6.9 最大池化

```python
def maxpool2d_np(x, kernel_size=2, stride=2):
    """
    MaxPool2d
    x: (N, C, H, W)
    返回: (N, C, H/2, W/2) 和 最大值位置索引（用于反向传播）
    """
    N, C, H, W = x.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    out = np.zeros((N, C, out_h, out_w))
    max_indices = np.zeros((N, C, out_h, out_w, 2), dtype=int)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + kernel_size
            w_start = j * stride
            w_end = w_start + kernel_size

            patch = x[:, :, h_start:h_end, w_start:w_end]  # (N, C, K, K)
            patch_flat = patch.reshape(N, C, -1)

            max_vals = np.max(patch_flat, axis=-1)          # (N, C)
            max_pos = np.argmax(patch_flat, axis=-1)        # (N, C)
            out[:, :, i, j] = max_vals

            # 记录位置（用于反向传播）
            max_indices[:, :, i, j, 0] = max_pos // kernel_size
            max_indices[:, :, i, j, 1] = max_pos % kernel_size

    return out, max_indices


# ── 测试 ──
x = np.random.randn(2, 4, 32, 32)
pooled, _ = maxpool2d_np(x, kernel_size=2, stride=2)
print("MaxPool 输出:", pooled.shape)  # (2, 4, 16, 16)
```

## 6.10 Dropout

```python
def dropout_forward_np(x, p=0.5, training=True):
    """
    torch.nn.Dropout
    training=True: 随机置零；training=False: 透传
    """
    if not training:
        return x

    mask = (np.random.random(x.shape) > p) / (1 - p)
    return x * mask


# ── 测试 ──
x = np.ones((1000,))
out = dropout_forward_np(x, p=0.5, training=True)
print(f"Dropout 后非零比例: {(out > 0).mean():.2f}")  # 约 0.5
print(f"输出均值: {out.mean():.3f}")                 # 约 1.0（inverted dropout 保持期望）
```

## 6.11 SGD 优化器

```python
class SGD_Numpy:
    """torch.optim.SGD 简化版"""

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        """
        params: list of dict: [{'value': ndarray, 'grad': ndarray}, ...]
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p['value']) for p in params]

    def zero_grad(self):
        for p in self.params:
            p['grad'] = np.zeros_like(p['value'])

    def step(self):
        for i, p in enumerate(self.params):
            grad = p['grad'].copy()

            # weight decay（L2 正则）
            if self.weight_decay > 0:
                grad += self.weight_decay * p['value']

            # momentum
            if self.momentum > 0:
                self.velocities[i] = (self.momentum * self.velocities[i]
                                      + (1 - self.momentum) * grad)
                grad = self.velocities[i]

            p['value'] -= self.lr * grad


# ── 测试 ──
params = [{'value': np.array([1.0, 2.0]), 'grad': None}]
sgd = SGD_Numpy(params, lr=0.1)  # 只创建，需要外部赋值 grad
```

## 6.12 Adam 优化器

```python
class Adam_Numpy:
    """torch.optim.Adam 简化版"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p['value']) for p in params]
        self.v = [np.zeros_like(p['value']) for p in params]
        self.t = 0  # 时间步

    def zero_grad(self):
        for p in self.params:
            p['grad'] = np.zeros_like(p['value'])

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p['grad'].copy()

            if self.weight_decay > 0:
                grad += self.weight_decay * p['value']

            # 一阶矩和二阶矩
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p['value'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

## 6.13 DataLoader 等价实现

```python
def numpy_dataloader(x, y, batch_size, shuffle=True):
    """
    torch.utils.data.DataLoader 的 NumPy 简化版
    每次 yield 一个 (batch_x, batch_y) 元组
    """
    N = len(x)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]
```

## 6.14 对照速查表

| 操作 | PyTorch | NumPy |
|---|---|---|
| 创建全零 | `torch.zeros(3,4)` | `np.zeros((3,4))` |
| 正态随机 | `torch.randn(2,3)` | `np.random.randn(2,3)` |
| 变形 | `x.view(-1,2)` | `x.reshape(-1,2)` |
| 展平 | `x.view(-1)` | `x.reshape(-1)` |
| 加维度 | `x.unsqueeze(0)` | `np.expand_dims(x,0)` |
| 转置 | `x.permute(1,0)` | `np.transpose(x,(1,0))` |
| 矩阵乘法 | `a @ b` | `a @ b` |
| ReLU | `F.relu(x)` | `np.maximum(0, x)` |
| Softmax | `F.softmax(x,dim=-1)` | (见 6.3 softmax_np) |
| Sigmoid | `torch.sigmoid(x)` | (见 6.3 sigmoid_np) |
| CE Loss | `nn.CrossEntropyLoss()` | (见 6.4 cross_entropy_loss_np) |
| Linear | `nn.Linear(in,out)` | `x @ w.T + b` |
| BatchNorm | `nn.BatchNorm1d` | (见 6.7 BatchNorm1d_Numpy) |
| Conv2d | `nn.Conv2d` | (见 6.8 im2col + matmul) |
| MaxPool2d | `nn.MaxPool2d` | (见 6.9 maxpool2d_np) |
| Dropout | `nn.Dropout(p)` | `mask * x / (1-p)` |
| SGD | `optim.SGD` | (见 6.11 SGD_Numpy) |
| Adam | `optim.Adam` | (见 6.12 Adam_Numpy) |
| GPU | `.to(device)` / `.cuda()` | 不支持 |
| autograd | `loss.backward()` | 需手写反向传播 |
