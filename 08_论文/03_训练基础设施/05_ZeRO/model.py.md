---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# ZeRO (Zero Redundancy Optimizer) 三级分片模拟 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# ZeRO (Zero Redundancy Optimizer) 三级分片模拟 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
ZeRO (Zero Redundancy Optimizer) 三级分片模拟
==============================================
将优化器状态、梯度和参数分片到各数据并行进程,
使显存消耗从 O(16Ψ) 降至 O(16Ψ/N_d)。

论文: [[ZeRO]] (Rajbhandari et al., SC 2020)
核心思想:
  - 在数据并行中, 所有 GPU 存储完全相同的优化器状态和梯度——这是显存冗余
  - 通过分片消除冗余, 同时保持通信量不变 (只改变通信模式)

三阶段分片:
  ZeRO-1 (OS): 仅分片优化器状态 —— 显存减少 4×
  ZeRO-2 (OS+G): 分片优化器状态 + 梯度 —— 显存减少 8×
  ZeRO-3 (OS+G+P): 分片优化器状态 + 梯度 + 参数 —— 显存减少 N_d×

与 [[FSDP]] (PyTorch 原生 ZeRO-3) 和 [[Megatron-LM]] (TP) 互补.
"""

import numpy as np
from typing import List, Tuple, Dict


# ============================================================
# 参数管理 (模拟分片)
# ============================================================

class ParameterShard:
    """管理一个参数的分布式分片."""

    def __init__(self, name: str, shape: Tuple[int, ...], world_size: int, rank: int):
        """
        Args:
            name: 参数名
            shape: 完整参数的形状
            world_size: 数据并行度 N_d
            rank: 当前进程编号
        """
        self.name = name
        self.shape = shape
        self.total_params = int(np.prod(shape))
        self.world_size = world_size
        self.rank = rank

        # 完整参数 (FP16 工作副本, 仅 ZeRO-1/2 持有)
        self.full_param = np.random.randn(*shape).astype(np.float32) * 0.02

        # 分片大小 (向上取整, 均匀分配)
        self.shard_size = (self.total_params + world_size - 1) // world_size
        self.flat_full = self.full_param.flatten()
        start = rank * self.shard_size
        end = min(start + self.shard_size, self.total_params)
        self.shard_range = (start, end)

    def get_shard(self, flat_data: np.ndarray) -> np.ndarray:
        """从扁平化的完整数据中提取当前 rank 的分片."""
        start, end = self.shard_range
        return flat_data[start:end].copy()

    def get_full_from_shards(self, shards: List[np.ndarray]) -> np.ndarray:
        """从所有 rank 的分片重建完整数据."""
        full = np.zeros(self.total_params, dtype=np.float32)
        for i, shard in enumerate(shards):
            start = i * self.shard_size
            end = min(start + self.shard_size, self.total_params)
            full[start:end] = shard
        return full.reshape(self.shape)


# ============================================================
# ZeRO 优化器基类
# ============================================================

class AdamOptimizer:
    """标准 Adam 优化器 (FP32 状态).

    存储需求: 每参数 8 字节 (momentum) + 8 字节 (variance) = 16B FP32 = 4× 参数.
    """

    def __init__(self, params: List[ParameterShard],
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # 优化器状态 (完整, 每个进程都有) —— 这就是 ZeRO 要消除的冗余!
        self.m = {}  # 一阶矩
        self.v = {}  # 二阶矩
        for p in params:
            self.m[p.name] = np.zeros(p.total_params, dtype=np.float32)
            self.v[p.name] = np.zeros(p.total_params, dtype=np.float32)

    def step(self, grads: Dict[str, np.ndarray]):
        """标准 Adam 更新 (所有参数完整更新)."""
        self.t += 1
        for p in self.params:
            g = grads[p.name]
            self.m[p.name] = self.beta1 * self.m[p.name] + (1 - self.beta1) * g.flatten()
            self.v[p.name] = self.beta2 * self.v[p.name] + (1 - self.beta2) * (g.flatten() ** 2)
            m_hat = self.m[p.name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p.name] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.full_param = (p.full_param.flatten() - update).reshape(p.shape)

    def memory_per_param_bytes(self) -> int:
        """每参数优化器状态字节数 (FP32)."""
        return 2 * 4  # m (4B) + v (4B)


# ============================================================
# ZeRO-1: 优化器状态分片 (OS)
# ============================================================

class ZeRO1Optimizer:
    """ZeRO-1: 只分片优化器状态, 参数和梯度保持完整.

    显存节省: 4× (标准 DP 的优化器状态从 4Ψ 降至 4Ψ/N_d)

    通信流:
      1. 梯度 all-reduce (同标准 DP)
      2. 各进程只更新本地分片内的优化器状态和参数
      3. 下一轮前向时, 各进程 broadcast 更新后的参数分片

    为什么通信量几乎不变:
      - all-reduce 通信量 = 2Ψ (与 DP 相同)
      - 参数 broadcast = Ψ·(N_d-1)/N_d (新增, 但相对于梯度通信很小)
    """

    def __init__(self, params: List[ParameterShard], world_size: int, rank: int,
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        self.params = params
        self.world_size = world_size
        self.rank = rank
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # 优化器状态仅存储本地分片 (关键节省!)
        self.m = {}  # 一阶矩 (仅自己分片)
        self.v = {}  # 二阶矩 (仅自己分片)
        for p in params:
            shard_len = p.shard_range[1] - p.shard_range[0]
            self.m[p.name] = np.zeros(shard_len, dtype=np.float32)
            self.v[p.name] = np.zeros(shard_len, dtype=np.float32)

    def step(self, grads: Dict[str, np.ndarray]):
        """ZeRO-1 更新: 仅更新本地分片内的参数.

        流程:
          1. 梯度 all-reduce (外部调用前已完成)
          2. 从完整梯度中提取本地分片
          3. 更新本地分片的优化器状态
          4. 更新本地分片的参数
          5. (下一轮) broadcast 参数分片
        """
        self.t += 1
        for p in self.params:
            g_full = grads[p.name].flatten()
            # 只取本地分片的梯度
            g_shard = p.get_shard(g_full)

            # 更新本地分片的优化器状态
            self.m[p.name] = self.beta1 * self.m[p.name] + (1 - self.beta1) * g_shard
            self.v[p.name] = self.beta2 * self.v[p.name] + (1 - self.beta2) * g_shard ** 2
            m_hat = self.m[p.name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p.name] / (1 - self.beta2 ** self.t)

            # 更新本地分片的参数
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param_flat = p.full_param.flatten()
            start, end = p.shard_range
            param_flat[start:end] -= update
            p.full_param = param_flat.reshape(p.shape)

    def broadcast_params(self) -> Dict[str, np.ndarray]:
        """模拟参数 broadcast: 各 rank 广播自己的分片.

        使所有进程重新持有完整参数 (为下一轮前向做准备).
        """
        return {p.name: p.full_param.copy() for p in self.params}

    def memory_per_param_bytes(self) -> int:
        return (2 * 4) // self.world_size  # m(4B) + v(4B) / N_d


# ============================================================
# ZeRO-2: 优化器状态 + 梯度分片 (OS + G)
# ============================================================

class ZeRO2Optimizer:
    """ZeRO-2: 分片优化器状态和梯度, 参数保持完整.

    显存节省: 8× (标准 DP 的 优化器状态(4Ψ) + 梯度(1Ψ) 均分片)

    通信流:
      1. 反向传播后, 各进程只保留自己分区的梯度
      2. 使用 reduce-scatter (而非 all-reduce) 聚合梯度
      3. reduce-scatter 通信量 = 2Ψ (与 all-reduce 相同!)
      4. 各进程更新本地参数分片
      5. broadcast 参数

    关键: reduce-scatter 替换 all-reduce, 通信量相同但内存减半.
    """

    def __init__(self, params: List[ParameterShard], world_size: int, rank: int,
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        self.params = params
        self.world_size = world_size
        self.rank = rank
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = {}
        self.v = {}
        for p in params:
            shard_len = p.shard_range[1] - p.shard_range[0]
            self.m[p.name] = np.zeros(shard_len, dtype=np.float32)
            self.v[p.name] = np.zeros(shard_len, dtype=np.float32)

    def reduce_scatter(self, param_name: str,
                       all_grads: List[np.ndarray]) -> np.ndarray:
        """模拟 reduce-scatter: 对所有 rank 的梯度求和, 再分发各 rank 对应的分片.

        在真实场景中, reduce-scatter 由 NCCL 原子化执行,
        通信量 = 2Ψ (同 all-reduce), 但每个进程只获得自己需要的分片.
        """
        # Step 1: 求和 (reduce 阶段, 实际上分布式执行)
        summed = sum(g.flatten() for g in all_grads)

        # Step 2: 分发到各 rank (scatter 阶段)
        p = self._find_param(param_name)
        shard = p.get_shard(summed)
        return shard

    def _find_param(self, name: str) -> ParameterShard:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def step_single_param(self, name: str, shard_grad: np.ndarray):
        """对单个参数的本地分片执行更新."""
        p = self._find_param(name)
        self.t += 1

        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * shard_grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * shard_grad ** 2
        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1 - self.beta2 ** self.t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        param_flat = p.full_param.flatten()
        start, end = p.shard_range
        param_flat[start:end] -= update
        p.full_param = param_flat.reshape(p.shape)

    def memory_per_param_bytes(self) -> int:
        return (2 * 4 + 2) // self.world_size  # m(4B) + v(4B) + grad(2B) / N_d


# ============================================================
# ZeRO-3: 优化器状态 + 梯度 + 参数全分片 (OS + G + P)
# ============================================================

class ZeRO3Optimizer:
    """ZeRO-3: 分片优化器状态、梯度和参数——最激进的阶段.

    显存节省: N_d× (每设备显存 = 16Ψ/N_d)

    前向通信: 在计算第 l 层前, all-gather 收集完整参数分片
    反向通信: reduce-scatter 聚合梯度到对应分区

    通信量: 3Ψ (原始 DP 的 1.5 倍), 但显存减至 1/N_d.

    典型场景:
      GPT-3 175B: 2.8TB → 44GB (N_d=64), 正好在单块 A100 80GB 上.
    """

    def __init__(self, params: List[ParameterShard], world_size: int, rank: int,
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        self.params = params
        self.world_size = world_size
        self.rank = rank
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # 优化器状态 (仅本地分片)
        self.m = {}
        self.v = {}
        # 参数也仅存储本地分片
        self.param_shards = {}

        for p in params:
            shard_len = p.shard_range[1] - p.shard_range[0]
            self.m[p.name] = np.zeros(shard_len, dtype=np.float32)
            self.v[p.name] = np.zeros(shard_len, dtype=np.float32)
            # 参数分片 (从完整参数中提取)
            self.param_shards[p.name] = p.get_shard(p.full_param.flatten())

    def all_gather_params(self, param_name: str,
                          all_shards: List[np.ndarray]) -> np.ndarray:
        """模拟 all-gather: 从所有 rank 收集参数分片, 重建完整参数.

        ZeRO-3 中, 每次前向/反向传播前都需要 all-gather 完整参数.
        这是 ZeRO-3 的额外通信开销来源.
        """
        p = self._find_param(param_name)
        full_flat = p.get_full_from_shards(all_shards)
        return full_flat

    def _find_param(self, name: str) -> ParameterShard:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def step(self, grad_shards: Dict[str, np.ndarray]):
        """ZeRO-3 更新: 所有状态都是分片的.

        流程:
          1. 反向传播后, reduce-scatter 聚合梯度 (外部调用)
          2. 用本地分片的梯度和优化器状态更新本地参数分片
          3. 不需要 broadcast——参数始终以分片形式存储
        """
        self.t += 1
        for name, g_shard in grad_shards.items():
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g_shard
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g_shard ** 2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.param_shards[name] -= update

    def memory_per_param_bytes(self) -> int:
        # 参数 (2B) + 梯度 (2B) + 优化器状态 (8B) = 12B / N_d
        return (2 + 2 + 4 + 4) // self.world_size


# ============================================================
# 显存对比分析
# ============================================================

def compare_zeRO_stages(model_size_B: float, world_size: int) -> Dict:
    """对比 ZeRO 各阶段的显存需求.

    Args:
        model_size_B: 模型参数量 (十亿), 如 7.5 表示 7.5B
        world_size: 数据并行度 N_d
    """
    params = model_size_B * 1e9  # 总参数数

    # FP16 模型参数 (2 字节/参数)
    fp16_params_GB = params * 2 / 1e9

    # FP32 优化器状态 (Adam: momentum + variance, 各 4 字节)
    fp32_opt_GB = params * 8 / 1e9

    # FP16 梯度 (2 字节/参数)
    fp16_grad_GB = params * 2 / 1e9

    total_baseline_GB = fp16_params_GB + fp32_opt_GB + fp16_grad_GB

    # ZeRO 各阶段
    zero1_GB = fp16_params_GB + fp32_opt_GB / world_size + fp16_grad_GB
    zero2_GB = fp16_params_GB + fp32_opt_GB / world_size + fp16_grad_GB / world_size
    zero3_GB = (fp16_params_GB + fp32_opt_GB + fp16_grad_GB) / world_size

    return {
        "model_size_B": model_size_B,
        "world_size": world_size,
        "baseline_DP_GB": total_baseline_GB,
        "ZeRO-1_GB": zero1_GB,
        "ZeRO-2_GB": zero2_GB,
        "ZeRO-3_GB": zero3_GB,
        "ZeRO-1_reduction": total_baseline_GB / zero1_GB,
        "ZeRO-2_reduction": total_baseline_GB / zero2_GB,
        "ZeRO-3_reduction": total_baseline_GB / zero3_GB,
    }


# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("ZeRO (Zero Redundancy Optimizer) 三级分片模拟")
    print("=" * 60)

    # ---- 1. 显存对比 ----
    print("\n1. ZeRO 各阶段显存对比\n")

    configs = [
        (1.5, 4),   # 小模型, 4 GPU
        (7.5, 8),   # 7.5B, 8 GPU
        (7.5, 64),  # 7.5B, 64 GPU
        (175, 64),  # GPT-3 级别, 64 GPU
    ]

    header = f"{'模型':>8} {'Nd':>3} | {'标准DP':>10} | {'ZeRO-1':>10} | {'ZeRO-2':>10} | {'ZeRO-3':>10} | {'节省(3)':>8}"
    print(header)
    print("-" * len(header))

    for size, ws in configs:
        result = compare_zeRO_stages(size, ws)
        print(f"{size:>5.0f}B  {ws:>3d} | "
              f"{result['baseline_DP_GB']:>8.0f}GB | "
              f"{result['ZeRO-1_GB']:>8.0f}GB | "
              f"{result['ZeRO-2_GB']:>8.0f}GB | "
              f"{result['ZeRO-3_GB']:>8.0f}GB | "
              f"{result['ZeRO-3_reduction']:>6.1f}x")

    # ---- 2. 各阶段通信量分析 ----
    print(f"\n{'='*50}")
    print("ZeRO 各阶段通信量分析 (以 Ψ 为单位)")
    print(f"{'阶段':<10} {'梯度':>12} {'参数广播':>12} {'总通信':>12}")
    print("-" * 50)
    print(f"{'标准DP':<10} {'all-reduce 2Ψ':>12} {'0 (全参数持有)':>12} {'2Ψ':>12}")
    print(f"{'ZeRO-1':<10} {'all-reduce 2Ψ':>12} {'(Nd-1)/Nd Ψ':>12} {'~3Ψ':>12}")
    print(f"{'ZeRO-2':<10} {'reduce-scatter 2Ψ':>12} {'(Nd-1)/Nd Ψ':>12} {'~3Ψ':>12}")
    print(f"{'ZeRO-3':<10} {'all-gather+reduce-scatter':>12} {'每次层all-gather':>12} {'~3Ψ':>12}")

    # ---- 3. ZeRO-3 工作流演示 ----
    print(f"\n{'='*50}")
    print("ZeRO-3 工作流演示")

    world_size = 4
    param = ParameterShard("weight", (100,), world_size, rank=0)
    print(f"  参数: {param.name}, 形状: {param.shape}")
    print(f"  总参数: {param.total_params}, 每 rank 分片: ~{param.shard_size}")

    # 模拟 all-gather
    shards = [np.random.randn(param.shard_size).astype(np.float32) for _ in range(world_size)]
    full = param.get_full_from_shards(shards)
    print(f"  All-gather 重建参数形状: {full.shape}")
    print(f"  所有分片并集 == 完整参数: {len(full) == param.total_params}")

    # ---- 4. 显存节省验证 ----
    print(f"\n{'='*50}")
    print("关键数值验证")

    # 7.5B 模型, 64 GPU
    result = compare_zeRO_stages(7.5, 64)
    print(f"  7.5B 模型, 64 GPU:")
    print(f"    标准 DP: {result['baseline_DP_GB']:.0f} GB")
    print(f"    ZeRO-3:  {result['ZeRO-3_GB']:.1f} GB (单块 A100 80GB {'✅ 足够' if result['ZeRO-3_GB'] <= 80 else '❌ 不足'})")

    # 175B 模型, 64 GPU
    result = compare_zeRO_stages(175, 64)
    print(f"  GPT-3 175B, 64 GPU:")
    print(f"    标准 DP: {result['baseline_DP_GB']:.0f} GB (❌ 完全不可行)")
    print(f"    ZeRO-3:  {result['ZeRO-3_GB']:.1f} GB (单块 A100 80GB {'✅ 足够' if result['ZeRO-3_GB'] <= 80 else '❌ 不足'})")

    print("\n✅ ZeRO 三级分片模拟完成")

```

```
