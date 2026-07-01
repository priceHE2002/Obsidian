---
tags: [代码, PyTorch]
created: 2026-07-01
---
# Gradient Checkpointing - 代码实现
> 本文档包含 PyTorch/NumPy 教学实现。

```python
"""
Gradient Checkpointing 教学实现
===============================
论文: "Training Deep Nets with Sublinear Memory Cost" (Chen et al., NeurIPS 2016)

核心思想:
  - 用 O(sqrt(n)) 激活值显存代替 O(n) 显存
  - 前向时只保存部分层的激活值（检查点），其余丢弃
  - 反向时从最近的检查点重算缺失的激活值

关键流程:
  前向: 每 sqrt(n) 层存一个检查点 → 其余激活值丢弃
  反向: 从检查点开始重新前向 → 算梯度 → 释放激活值
  本质: "以计算换显存"——增加约 30% 计算，节省 90%+ 显存
"""

import numpy as np


class CheckpointFunction:
    """
    模拟 PyTorch 的 torch.utils.checkpoint 行为
    
    当调用 checkpoint(fn, *args) 时:
    - 前向: 正常执行 fn，但不保存中间激活（只存输出）
    - 反向: 重新执行 fn 以获得中间激活，再算梯度
    """
    
    def __init__(self, fn):
        self.fn = fn
        # 保存反向重算所需的输入
        self.saved_inputs = None
    
    def forward(self, *args):
        """前向: 执行 fn，保存输入用于后续重算"""
        self.saved_inputs = args
        return self.fn(*args)
    
    def backward(self, grad_output, recompute_fn=None):
        """
        反向: 重算 fn 的前向以获得中间激活，然后算梯度
        
        这是 checkpoint 的核心: 不存储中间激活，
        而是在反向时重新计算它们。
        """
        if recompute_fn is not None:
            self.fn = recompute_fn
        # 重算前向
        output = self.fn(*self.saved_inputs)
        return output  # 仅作示意，实际 autograd 会计算梯度


class GradientCheckpointingLayer:
    """
    模拟带 checkpoint 的单层网络
    
    一层由 Linear + ReLU 组成
    """
    
    def __init__(self, in_dim, out_dim, name="layer"):
        self.name = name
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.01
        self.b = np.zeros(out_dim, dtype=np.float32)
        # 前向缓存: 存储激活值供反向使用
        self.cached_input = None
        self.cached_pre_activation = None
    
    def forward(self, x, save_for_backward=True):
        """
        前向传播
        
        Args:
            x: 输入
            save_for_backward: 是否缓存激活值（checkpoint=False 时缓存）
        """
        pre_act = x @ self.W + self.b
        output = np.maximum(0, pre_act)  # ReLU
        
        if save_for_backward:
            # 标准模式: 缓存激活值 → 增加显存
            self.cached_input = x
            self.cached_pre_activation = pre_act
        else:
            # Checkpoint 模式: 丢弃激活值 → 节省显存
            self.cached_input = None
            self.cached_pre_activation = None
        
        return output
    
    def backward(self, grad_output, recompute_input=None):
        """
        反向传播
        
        Args:
            grad_output: 上游梯度
            recompute_input: 如果提供了，说明需要先从该输入重算前向
        """
        if recompute_input is not None:
            # Checkpoint 模式: 先重算前向
            self.forward(recompute_input, save_for_backward=True)
        
        # ReLU 反向
        relu_grad = grad_output * (self.cached_pre_activation > 0)
        
        # 线性层反向
        grad_W = self.cached_input.T @ relu_grad
        grad_b = np.sum(relu_grad, axis=0)
        grad_input = relu_grad @ self.W.T
        
        return grad_input, grad_W, grad_b


class DeepNetworkWithCheckpoint:
    """
    深度网络 + 梯度检查点
    
    n 层网络，每 seg_size 层保存一个检查点。
    均匀分割策略: seg_size ≈ sqrt(n)
    """
    
    def __init__(self, dim=64, num_layers=12, seg_size=None):
        """
        Args:
            dim: 每层维度
            num_layers: 总层数
            seg_size: 每段层数 (None 则自动取 sqrt(num_layers))
        """
        self.dim = dim
        self.num_layers = num_layers
        
        # 自动计算段大小: k = sqrt(n)
        if seg_size is None:
            seg_size = max(1, int(np.sqrt(num_layers)))
        self.seg_size = seg_size
        
        # 创建层
        self.layers = [
            GradientCheckpointingLayer(dim, dim, name=f"layer_{i}")
            for i in range(num_layers)
        ]
        
        # 检查点位置: 每 seg_size 层的起点
        self.checkpoint_indices = list(range(0, num_layers, seg_size))
        
        # 显存统计
        self.activation_memory_standard = 0  # 标准模式总激活值显存
        self.activation_memory_checkpoint = 0  # checkpoint 模式总激活值显存
    
    def forward_standard(self, x):
        """标准前向: 缓存所有中间激活值 → O(n) 显存"""
        activations = [x]
        current = x
        for layer in self.layers:
            current = layer.forward(current, save_for_backward=True)
            activations.append(current)
        return activations
    
    def backward_standard(self, activations):
        """标准反向: 从最后一层开始逐层反向"""
        num_layers = len(self.layers)
        grad = np.ones_like(activations[-1]) * 0.1  # 模拟上游梯度
        
        for i in range(num_layers - 1, -1, -1):
            grad, _, _ = self.layers[i].backward(grad)
        
        return grad
    
    def forward_with_checkpoint(self, x):
        """
        Checkpoint 前向: 只在检查点处保存激活值 → O(sqrt(n)) 显存
        
        策略:
        - 在检查点层之前: 保存该层输入
        - 在非检查点层: 运行前向但丢弃激活值
        """
        # 只保存检查点位置的激活值
        checkpoints = {}
        current = x
        
        for i, layer in enumerate(self.layers):
            if i in self.checkpoint_indices:
                # 检查点层: 保存输入
                checkpoints[i] = current.copy()
            
            # 非检查点层: 不保存中间激活
            current = layer.forward(current, save_for_backward=False)
        
        # 最终输出总是保存
        return current, checkpoints
    
    def backward_with_checkpoint(self, final_output, checkpoints, original_input):
        """
        Checkpoint 反向: 从检查点重算缺失的激活值
        
        算法:
        1. 找到最近的检查点
        2. 从检查点开始重新前向到需要反算的层
        3. 计算该层梯度
        4. 释放重算的激活值
        """
        num_layers = len(self.layers)
        grad = np.ones_like(final_output) * 0.1
        
        # 从最后一层向第一层处理
        layer_idx = num_layers - 1
        
        while layer_idx >= 0:
            # 找到离当前层最近的（前面的）检查点
            cp_idx = max(cp for cp in self.checkpoint_indices if cp <= layer_idx)
            
            # 从检查点开始重算: 重算 [cp_idx, layer_idx] 之间的所有层
            current = checkpoints[cp_idx]
            for i in range(cp_idx, layer_idx + 1):
                # 注意: 这里故意用 save_for_backward=True
                # 因为重算时需要临时缓存激活值来计算梯度
                current = self.layers[i].forward(current, save_for_backward=True)
            
            # 对 layer_idx 层算反向
            grad, _, _ = self.layers[layer_idx].backward(grad)
            
            # 移动到上一层
            layer_idx -= 1
        
        return grad
    
    def compute_total_activation_memory(self, batch_size, activations_per_element=4):
        """
        计算激活值显存占用
        
        Args:
            batch_size: 批量大小
            activations_per_element: 每个元素占的字节数 (FP32 = 4)
        """
        # 标准模式: 存储所有层的激活值
        memory_per_layer = batch_size * self.dim * activations_per_element
        standard_memory = self.num_layers * 2 * memory_per_layer  # input + output per layer
        
        # Checkpoint 模式: 只存储检查点 + 当前重算段
        num_checkpoints = len(self.checkpoint_indices)
        checkpoint_memory = num_checkpoints * 2 * memory_per_layer
        seg_memory = self.seg_size * 2 * memory_per_layer  # 当前重算段
        total_checkpoint_memory = checkpoint_memory + seg_memory
        
        return standard_memory, total_checkpoint_memory


# ============================================================
# __main__: 演示梯度检查点
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)
    
    print("=" * 60)
    print("Gradient Checkpointing 模拟演示")
    print("=" * 60)
    
    # 配置
    dim = 32
    num_layers = 16
    batch_size = 8
    
    # seg_size = sqrt(16) = 4
    network = DeepNetworkWithCheckpoint(dim=dim, num_layers=num_layers)
    
    print(f"\n网络配置: {num_layers} 层, 维度 {dim}")
    print(f"检查点间隔: 每 {network.seg_size} 层")
    print(f"检查点位置: {network.checkpoint_indices}")
    
    # 显存分析
    standard_mem, ckpt_mem = network.compute_total_activation_memory(batch_size)
    print(f"\n--- 激活值显存分析 (batch_size={batch_size}) ---")
    print(f"标准模式 (全存):    {standard_mem / 1024:.2f} KB")
    print(f"Checkpoint 模式:     {ckpt_mem / 1024:.2f} KB")
    savings = (1 - ckpt_mem / standard_mem) * 100
    print(f"显存节省:            {savings:.1f}%")
    
    # 计算开销分析
    # 理论: 额外前向 ≈ seg_size × 每层前向
    extra_forward = network.seg_size
    total_forward_no_ckpt = num_layers  # 标准模式 1 次前向
    overhead = extra_forward / total_forward_no_ckpt
    print(f"\n--- 计算开销分析 ---")
    print(f"标准模式前向量:      1× ({num_layers} 层)")
    print(f"Checkpoint 额外前向:  ~{extra_forward} 层 (最坏情况每段重算)")
    print(f"理论计算开销:        {overhead:.1%} (额外前向)")
    print(f"实际开销 (经验):     ~25-33% (因为前向比反向快)")
    
    # 运行演示
    print(f"\n--- 前向/反向演示 ---")
    X = np.random.randn(batch_size, dim).astype(np.float32)
    
    # 方式1: 标准模式 (O(n) 显存)
    print("\n[标准模式] 前向: 缓存所有激活值...")
    activations = network.forward_standard(X)
    print(f"  缓存的激活值数量: {len(activations)} 个张量")
    print("  [标准模式] 反向: 使用缓存激活值...")
    _ = network.backward_standard(activations)
    print("  完成 (无需重算)")
    
    # 方式2: Checkpoint 模式 (O(sqrt(n)) 显存)
    print("\n[Checkpoint 模式] 前向: 只在检查点处保存...")
    final_out, checkpoints = network.forward_with_checkpoint(X)
    print(f"  保存的检查点数量: {len(checkpoints)} (vs 标准模式 {num_layers})")
    print(f"  检查点位于层: {sorted(checkpoints.keys())}")
    print("  [Checkpoint 模式] 反向: 从检查点重算...")
    _ = network.backward_with_checkpoint(final_out, checkpoints, X)
    print("  完成 (含重算开销)")
    
    # 不同 seg_size 的显存-计算 tradeoff
    print(f"\n--- seg_size 对显存-计算的影响 ---")
    print(f"{'seg_size':>10s} | {'显存(相对)':>12s} | {'额外计算':>10s} | {'适用场景':>20s}")
    print("-" * 65)
    for seg in [1, 2, 4, 8, num_layers]:
        net = DeepNetworkWithCheckpoint(dim=dim, num_layers=num_layers, seg_size=seg)
        s_mem, c_mem = net.compute_total_activation_memory(batch_size)
        ratio = c_mem / s_mem
        extra = seg / num_layers
        desc = ""
        if seg == 1:
            desc = "极致显存节省"
        elif seg == 4:
            desc = "最优平衡 (sqrt(n))"
        elif seg == num_layers:
            desc = "无节省 (标准模式)"
        print(f"{seg:10d} | {ratio:11.1%} | {extra:9.0%} | {desc:>20s}")
    
    print("\n✅ Gradient Checkpointing 模拟完成")
```
