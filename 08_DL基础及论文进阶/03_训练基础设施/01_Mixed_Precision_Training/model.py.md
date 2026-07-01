---
tags: [代码, PyTorch]
created: 2026-07-01
---
# Mixed Precision Training - 代码实现
> 本文档包含 PyTorch/NumPy 教学实现。

```python
"""
Mixed Precision Training (AMP) 教学实现
=========================================
论文: "Mixed Precision Training" (Micikevicius et al., ICLR 2018)

核心思想:
  - FP32 主权重副本: 权重更新必须在 FP32 中累积，FP16 精度不足以累加微小梯度
  - FP16 前向/反向: 利用 Tensor Cores 加速矩阵乘法
  - Loss Scaling: 放大 loss 以防止 FP16 梯度下溢

关键流程:
  FP32 主权重 W32 --cast--> FP16 权重 W16
  --> FP16 前向计算 --> FP32 loss
  --> loss * scale --> FP16 反向传播 --> FP16 梯度
  --> 梯度 / scale (反缩放) --> cast 到 FP32
  --> FP32 优化器更新 W32
"""

import numpy as np


class MixedPrecisionTrainer:
    """
    混合精度训练器模拟
    
    模拟 FP32 主权重 + FP16 前向/反向 + Loss Scaling 的完整流程。
    注意: 这里用 NumPy 模拟数据类型行为，并非真正调用 GPU Tensor Cores。
    """
    
    def __init__(self, dim_in=64, dim_out=10, initial_scale=2**15):
        """
        Args:
            dim_in: 输入维度
            dim_out: 输出维度
            initial_scale: 初始 loss scale，默认 32768
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # 核心: FP32 主权重副本 —— 权重更新的"真实来源"
        # 所有参数更新在 FP32 中进行，保证累加精度
        self.W32 = np.random.randn(dim_in, dim_out).astype(np.float32) * 0.01
        self.b32 = np.zeros(dim_out, dtype=np.float32)
        
        # FP32 优化器状态 (Adam 动量)
        # 优化器状态同样必须在 FP32 中维护
        self.m_W32 = np.zeros_like(self.W32)
        self.v_W32 = np.zeros_like(self.W32)
        self.m_b32 = np.zeros_like(self.b32)
        self.v_b32 = np.zeros_like(self.b32)
        
        # Loss scale: 动态 loss scaling
        self.loss_scale = np.float32(initial_scale)
        self.growth_interval = 2000  # 连续 N 次无溢出则 scale 翻倍
        self.steps_since_overflow = 0
        
        # 统计
        self.overflow_count = 0
        self.total_steps = 0
        
    def fp32_to_fp16(self, x):
        """
        模拟 FP32 -> FP16 截断
        
        FP16 规格: 5 位指数 + 10 位尾数
        - 最大可表示值: 65504
        - 最小正规数: 2^(-14) ≈ 6.1 × 10^(-5)
        - 最小次正规数: 2^(-24) ≈ 5.96 × 10^(-8)
        
        这个模拟近似 IEEE 754 half-precision 的行为。
        """
        x = np.asarray(x, dtype=np.float32)
        
        # 模拟 FP16 最大/最小值约束
        x = np.clip(x, -65504.0, 65504.0)
        
        # 模拟尾数截断: FP16 尾数 10 位 ≈ 3.3 位十进制
        # 对非常小的值模拟下溢
        abs_x = np.abs(x)
        tiny_mask = (abs_x > 0) & (abs_x < 6e-8)  # 小于 FP16 最小正数
        x[tiny_mask] = 0.0  # 下溢为 0
        
        # 模拟尾数精度损失 (FP16 相对精度约 9.8 × 10^(-4))
        # 通过舍入到 ~2048 个可表示级别
        scale = 2048.0 / (np.max(abs_x) + 1e-8)
        x = np.round(x * scale) / (scale + 1e-8)
        
        return x
    
    def fp16_to_fp32(self, x):
        """FP16 -> FP32 转换 (模拟扩展精度)"""
        return np.asarray(x, dtype=np.float32)
    
    def forward(self, X, training=True):
        """
        FP16 前向传播
        
        前向在 FP16 中进行以获得 Tensor Cores 加速，
        但 BatchNorm/LayerNorm 应在 FP32 中计算（这里省略）。
        """
        if training:
            # FP32 主权重 -> FP16 副本
            W16 = self.fp32_to_fp16(self.W32)
            b16 = self.fp32_to_fp16(self.b32)
            X16 = self.fp32_to_fp16(X)
        else:
            W16, b16, X16 = self.W32, self.b32, X
        
        # FP16 线性层
        logits = X16 @ W16 + b16
        return logits
    
    def compute_loss(self, logits, targets):
        """
        计算 FP32 loss (分类任务)
        
        Loss 计算必须在 FP32 中进行，因为 Softmax + CrossEntropy
        涉及 exp/sum 等需要高精度的操作。
        """
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64)
        
        # Softmax (FP32)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        
        # Cross-entropy (FP32)
        eps = 1e-8
        batch_size = logits.shape[0]
        loss = -np.mean(np.log(probs[np.arange(batch_size), targets] + eps))
        
        return np.float32(loss)
    
    def backward(self, X, targets):
        """
        FP16 反向传播模拟
        
        关键: 梯度在 FP16 中计算，然后反缩放并转换为 FP32。
        """
        X = np.asarray(X, dtype=np.float32)
        
        # 重新计算 FP16 前向以获取中间值
        W16 = self.fp32_to_fp16(self.W32)
        X16 = self.fp32_to_fp16(X)
        logits_fp32 = X16.astype(np.float32) @ W16.astype(np.float32)
        
        # Softmax 梯度 (FP32 中间计算，因为 softmax 精度敏感)
        shifted = logits_fp32 - np.max(logits_fp32, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        
        batch_size = X.shape[0]
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), targets] -= 1
        grad_logits /= batch_size
        
        # 梯度在 FP16 中计算
        grad_W16 = X16.T @ self.fp32_to_fp16(grad_logits)
        grad_b16 = np.sum(self.fp32_to_fp16(grad_logits), axis=0)
        
        # *** 关键步骤: 反缩放 (Undo Loss Scaling) ***
        # 因为前向时 loss 乘了 scale，梯度也被放大了，
        # 现在必须除以 scale 恢复真实梯度
        grad_W16 = grad_W16 / self.loss_scale
        grad_b16 = grad_b16 / self.loss_scale
        
        # FP16 梯度 -> FP32
        grad_W32 = self.fp16_to_fp32(grad_W16)
        grad_b32 = self.fp16_to_fp32(grad_b16)
        
        return grad_W32, grad_b32
    
    def check_for_overflow(self, grads):
        """
        检查梯度是否包含 Inf/NaN
        
        在动态 loss scaling 中，如果检测到溢出，
        该步不进行权重更新，且 scale 减半。
        """
        for g in grads:
            if np.any(np.isinf(g)) or np.any(np.isnan(g)):
                return True
        return False
    
    def optimizer_step(self, grad_W32, grad_b32, lr=1e-3, beta1=0.9, beta2=0.999):
        """
        FP32 优化器更新 (Adam)
        
        权重更新和优化器状态在 FP32 中进行，
        这是混合精度训练与纯 FP16 训练的核心区别。
        """
        eps = 1e-8
        
        # Adam 更新 W (全部在 FP32 中)
        self.m_W32 = beta1 * self.m_W32 + (1 - beta1) * grad_W32
        self.v_W32 = beta2 * self.v_W32 + (1 - beta2) * grad_W32 ** 2
        self.W32 -= lr * self.m_W32 / (np.sqrt(self.v_W32) + eps)
        
        # Adam 更新 b
        self.m_b32 = beta1 * self.m_b32 + (1 - beta1) * grad_b32
        self.v_b32 = beta2 * self.v_b32 + (1 - beta2) * grad_b32 ** 2
        self.b32 -= lr * self.m_b32 / (np.sqrt(self.v_b32) + eps)
    
    def train_step(self, X, targets, lr=1e-3):
        """
        完整的混合精度训练步骤
        
        1. FP32 主权重 -> FP16
        2. FP16 前向
        3. FP32 loss * scale
        4. FP16 反向 (梯度含 scale)
        5. 梯度反缩放 + 溢出检查
        6. FP32 优化器更新
        7. 动态调整 loss scale
        """
        self.total_steps += 1
        X = np.asarray(X, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64)
        
        # Step 1-2: FP16 前向
        logits = self.forward(X, training=True)
        
        # Step 3: FP32 loss 计算 + Loss Scaling
        loss_fp32 = self.compute_loss(logits, targets)
        scaled_loss = loss_fp32 * self.loss_scale  # 放大!
        
        # Step 4-5: FP16 反向 + 反缩放
        grad_W32, grad_b32 = self.backward(X, targets)
        
        # Step 6: 溢出检查
        has_overflow = self.check_for_overflow([grad_W32, grad_b32])
        
        if has_overflow:
            # 发现溢出: 跳过本次更新，减小 loss scale
            self.overflow_count += 1
            self.loss_scale = max(self.loss_scale / 2, 1.0)
            self.steps_since_overflow = 0
        else:
            # 无溢出: 正常更新 FP32 主权重
            self.optimizer_step(grad_W32, grad_b32, lr=lr)
            self.steps_since_overflow += 1
            
            # 动态调整: 连续 N 步无溢出 → scale 翻倍
            if self.steps_since_overflow >= self.growth_interval:
                self.loss_scale = min(self.loss_scale * 2, 2**24)  # 上限 ~1.67e7
                self.steps_since_overflow = 0
        
        return loss_fp32, has_overflow


# ============================================================
# __main__: 演示混合精度训练
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("Mixed Precision Training (AMP) 模拟演示")
    print("=" * 60)
    
    # 创建训练器
    trainer = MixedPrecisionTrainer(dim_in=64, dim_out=10)
    
    print(f"\n初始 Loss Scale: {trainer.loss_scale}")
    print(f"FP32 主权重形状: {trainer.W32.shape}")
    print(f"FP32 主权重范围: [{trainer.W32.min():.6f}, {trainer.W32.max():.6f}]")
    
    # 生成模拟数据
    num_samples = 128
    X = np.random.randn(num_samples, 64).astype(np.float32)
    targets = np.random.randint(0, 10, size=num_samples)
    
    # 检查 FP16 截断效果
    print("\n--- FP16 截断效果验证 ---")
    tiny_grad = np.array([1e-9, 1e-7, 1e-5, 0.001, 0.1, 1.0, 10.0, 1000.0], dtype=np.float32)
    fp16_grad = trainer.fp32_to_fp16(tiny_grad)
    for orig, truncated in zip(tiny_grad, fp16_grad):
        if truncated == 0 and orig != 0:
            print(f"  原始值 {orig:.2e} → FP16 下溢为 0 (需要 Loss Scaling!)")
        elif orig != truncated:
            print(f"  原始值 {orig:.6f} → FP16 截断为 {truncated:.6f}")
    
    # 训练循环
    print("\n--- 训练开始 ---")
    print(f"{'Step':>6s} | {'Loss':>10s} | {'Scale':>10s} | {'Overflow':>10s}")
    print("-" * 45)
    
    for step in range(100):
        # 每次随机抽样模拟 mini-batch
        idx = np.random.choice(num_samples, size=32)
        X_batch = X[idx]
        y_batch = targets[idx]
        
        loss, overflow = trainer.train_step(X_batch, y_batch, lr=0.01)
        
        if step % 10 == 0 or overflow:
            marker = " [OVERFLOW!]" if overflow else ""
            print(f"{step:6d} | {loss:10.4f} | {trainer.loss_scale:10.0f} | {'Yes' if overflow else 'No':>10s}{marker}")
    
    print(f"\n总步数: {trainer.total_steps}")
    print(f"溢出次数: {trainer.overflow_count}")
    print(f"最终 Loss Scale: {trainer.loss_scale:.0f}")
    print(f"溢出率: {trainer.overflow_count / trainer.total_steps * 100:.2f}%")
    
    # 对比: 如果不使用 Master Weights (纯 FP16 更新)
    print("\n--- 纯 FP16 权重更新的问题 ---")
    # 模拟 FP16 精度不足以累加小更新
    w_init = np.float16(1.0)
    update = np.float16(0.0001)  # 微小梯度更新
    for i in range(100):
        w_init += update  # FP16 累加
    w_fp32 = np.float32(1.0)
    update_fp32 = np.float32(0.0001)
    for i in range(100):
        w_fp32 += update_fp32
    print(f"FP16 累加 100 次小更新: {float(w_init):.8f} (应为 1.0100)")
    print(f"FP32 累加 100 次小更新: {float(w_fp32):.8f} (应为 1.0100)")
    print("结论: FP16 累加丢失精度 → Master Weights 是必需的!")
    
    print("\n✅ Mixed Precision Training 模拟完成")
```
