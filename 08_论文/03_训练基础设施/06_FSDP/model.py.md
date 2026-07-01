---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# FSDP (Fully Sharded Data Parallel) 核心流程模拟 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
FSDP (Fully Sharded Data Parallel) 核心流程模拟
================================================
PyTorch 原生的 ZeRO-3 实现, 模拟 flatten + shard + all-gather + reduce-scatter 核心流程。

论文: [[FSDP]] (Zhao et al., VLDB 2023)
核心思想:
  - 将模型参数、梯度和优化器状态分片到所有 GPU
  - 前向: all-gather 收集完整参数 → 计算 → 丢弃其他分片
  - 反向: all-gather 收集完整参数 → 计算梯度 → reduce-scatter 聚合

关键设计:
  1. FSDP Unit: 每个 FSDP 包装的模块是一个分片单元
  2. FlatParameter: 将模块内所有参数展平为一个一维张量, 便于分片
  3. Prefetch: 前向/反向预取, 隐藏通信延迟
  4. auto_wrap_policy: 按 Transformer block 自动包裹

与 [[ZeRO]] 的关系: FSDP 是 ZeRO-3 的 PyTorch 原生等价实现.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ============================================================
# FlatParameter: 扁平化参数 (FSDP 的核心数据结构)
# ============================================================

@dataclass
class FlatParameter:
    """扁平化参数——将模块内所有参数拼接成一维张量.

    为什么需要 flatten:
      - 分片操作在一维张量上最简单高效
      - 减少 all-gather / reduce-scatter 的调用次数
      - 非扁平化参数会导致大量小通信 (overhead > 收益)
    """

    name: str
    total_size: int          # 扁平化后的总元素数
    param_names: List[str]   # 包含的原始参数名
    param_shapes: List[Tuple[int, ...]]  # 各参数的原始形状
    param_offsets: List[Tuple[int, int]]  # 各参数在扁平化数组中的 (start, end)


def flatten_params(params: Dict[str, np.ndarray]) -> Tuple[FlatParameter, np.ndarray]:
    """将多个参数扁平化为一个一维数组.

    Args:
        params: {param_name: param_array} 字典
    Returns:
        flat_param: 扁平化元数据
        flat_data: 一维数组
    """
    total = 0
    offsets = []
    arrays = []
    names = []
    shapes = []

    for name, arr in params.items():
        flat = arr.flatten()
        offsets.append((total, total + len(flat)))
        arrays.append(flat)
        names.append(name)
        shapes.append(arr.shape)
        total += len(flat)

    flat_data = np.concatenate(arrays)

    fp = FlatParameter(
        name="flat_all",
        total_size=total,
        param_names=names,
        param_shapes=shapes,
        param_offsets=offsets,
    )
    return fp, flat_data


def unflatten_params(flat_param: FlatParameter, flat_data: np.ndarray) -> Dict[str, np.ndarray]:
    """从扁平化数据恢复原始参数."""
    result = {}
    for i, name in enumerate(flat_param.param_names):
        start, end = flat_param.param_offsets[i]
        result[name] = flat_data[start:end].reshape(flat_param.param_shapes[i])
    return result


# ============================================================
# FSDP 通信原语
# ============================================================

class FSDPCommunicationGroup:
    """FSDP 通信组——模拟 NCCL all-gather 和 reduce-scatter."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def all_gather(self, local_shard: np.ndarray,
                   all_shards: List[np.ndarray]) -> np.ndarray:
        """模拟 all-gather: 从所有 rank 收集分片, 拼接为完整张量.

        前向传播时: 需要完整参数做计算
        反向传播时: 需要完整参数计算梯度
        """
        return np.concatenate(all_shards)

    def reduce_scatter(self, local_grad: np.ndarray,
                       all_grads: List[np.ndarray], shard_idx: int) -> np.ndarray:
        """模拟 reduce-scatter: 先求和所有 rank 的梯度, 再分发各 rank 的分片.

        反向传播后: 聚合梯度并将结果按分片返回各 rank.
        """
        # 求和
        total_grad = sum(all_grads)
        # 分片
        chunk_size = len(total_grad) // self.world_size
        start = self.rank * chunk_size
        end = start + chunk_size
        return total_grad[start:end].copy()


# ============================================================
# FSDP Unit: 一个分片单元 (包装一个 Transformer block)
# ============================================================

class FSDPUnit:
    """FSDP 单元——包装一个子模块 (如一个 Transformer block).

    FSDP 的核心执行流:
      前向:
        [pre] all_gather → 收集完整参数
        [compute] module.forward(完整参数)
        [post] discard collected shards (释放显存)
      反向:
        [pre] all_gather → 重新收集完整参数 (因为前向后已释放)
        [compute] autograd.grad → 计算梯度
        [post] reduce_scatter → 聚合+分片梯度
        [update] optimizer.step(own_shard) → 仅更新本地分片
    """

    def __init__(self, params: Dict[str, np.ndarray],
                 world_size: int, rank: int):
        """
        Args:
            params: 该模块包含的参数
            world_size: FSDP 进程组大小
            rank: 当前进程 rank
        """
        self.world_size = world_size
        self.rank = rank
        self.comm = FSDPCommunicationGroup(world_size, rank)

        # 扁平化所有的参数
        self.flat_param_meta, flat_data = flatten_params(params)

        # 分片: 每 rank 只持有 total_size / world_size 的参数
        self.shard_size = flat_data.shape[0] // world_size
        start = rank * self.shard_size
        end = start + self.shard_size
        self.local_shard = flat_data[start:end].copy().astype(np.float32)

        # 梯度分片 (用于反向传播后的 reduce-scatter 结果)
        self.grad_shard = np.zeros_like(self.local_shard)

        # 记录前向通过的输入 (用于反向传播)
        self.saved_input: Optional[np.ndarray] = None
        self.saved_full_param: Optional[np.ndarray] = None

    def _get_all_shards(self) -> List[np.ndarray]:
        """模拟收集所有 rank 的参数分片.

        真实场景中, 这通过 NCCL all-gather 实现.
        这里用模拟: 从扁平化数据中提取所有分片.
        """
        # 重建完整参数再分片 (模拟 all-gather 的效果)
        full = np.zeros(self.flat_param_meta.total_size, dtype=np.float32)
        start = self.rank * self.shard_size
        full[start:start + self.shard_size] = self.local_shard
        # 实际中, 从 NCCL 收集
        return [full[i * self.shard_size:(i + 1) * self.shard_size]
                for i in range(self.world_size)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """带 all-gather 的前向传播.

        流程:
          1. all-gather 收集完整参数
          2. 用完整参数做前向计算
          3. 释放其他 rank 的分片 (仅保留自己的)
        """
        # ---- Step 1: all-gather 参数 ----
        all_shards = self._get_all_shards()
        full_param = self.comm.all_gather(self.local_shard, all_shards)
        self.saved_full_param = full_param.copy()  # 保存完整参数供反向使用

        # ---- Step 2: 前向计算 (模拟: y = Wx + b) ----
        # 为什么先 all-gather: 前向需要完整参数才能计算正确结果
        params = unflatten_params(self.flat_param_meta, full_param)
        W = params.get("weight", np.eye(x.shape[-1]))
        b = params.get("bias", np.zeros(x.shape[-1]))
        output = x @ W + b
        self.saved_input = x.copy()

        # ---- Step 3: 释放非本地分片 (模拟) ----
        # 在真实 FSDP 中, 这会 free 掉从其他 rank all-gather 来的参数内存
        # 只保留 self.local_shard

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """带 reduce-scatter 的反向传播.

        流程:
          1. 重新 all-gather 完整参数 (因为前向释放了)
          2. 用完整参数计算梯度
          3. reduce-scatter: 聚合所有 rank 的梯度, 分发分片到各 rank
        """
        # ---- Step 1: all-gather 完整参数 (必须重新收集) ----
        all_shards = self._get_all_shards()
        full_param = self.comm.all_gather(self.local_shard, all_shards)

        # ---- Step 2: 反向计算梯度 ----
        # 模拟: dW = x^T @ grad_output, db = sum(grad_output)
        x = self.saved_input
        # 扁平化 gradient 以便 reduce-scatter
        dW_flat = (x.T @ grad_output).flatten()
        db_flat = grad_output.sum(axis=0).flatten()
        full_grad_flat = np.concatenate([dW_flat, db_flat])

        # ---- Step 3: reduce-scatter 聚合 + 分片 ----
        # 模拟: 所有 rank 的梯度先求和, 再按分片返回
        # (实际中这是 NCCL 原子化完成)
        all_grads = [full_grad_flat for _ in range(self.world_size)]
        self.grad_shard = self.comm.reduce_scatter(full_grad_flat, all_grads, self.rank)

        # 返回对输入的梯度: dL/dx = grad_output @ W^T
        params = unflatten_params(self.flat_param_meta, full_param)
        W = params.get("weight", np.eye(x.shape[-1]))
        grad_input = grad_output @ W.T

        # 释放保存的完整参数
        self.saved_full_param = None
        self.saved_input = None

        return grad_input

    def optimizer_step(self, lr: float = 1e-3):
        """使用本地梯度分片更新本地参数分片.

        每个 rank 只更新自己持有的参数分片——无需求助于其他 rank.
        """
        # Zero-3 风格: 优化器状态也是分片的
        self.local_shard -= lr * self.grad_shard
        self.grad_shard.fill(0.0)


# ============================================================
# Prefetch 策略 (通信-计算重叠)
# ============================================================

class PrefetchManager:
    """管理 FSDP 的预取策略——隐藏通信延迟.

    两种预取:
      - forward_prefetch: 在前向当前层时预取下一层参数
      - backward_prefetch: 在反向当前层时预取上一层参数

    效果: 通信时间几乎完全被计算掩盖.
    """

    def __init__(self):
        self.prefetch_queue = []

    def schedule_forward_prefetch(self, next_unit: FSDPUnit):
        """前向预取: 在计算当前层时预先 all-gather 下一层的参数."""
        self.prefetch_queue.append(("forward", next_unit))

    def schedule_backward_prefetch(self, prev_unit: FSDPUnit):
        """反向预取: 在反向当前层时预先 all-gather 上一层的参数."""
        self.prefetch_queue.append(("backward", prev_unit))


# ============================================================
# 完整的 FSDP 模型
# ============================================================

class FSDPModel:
    """使用 FSDP 包装的多层模型.

    每层是一个独立的 FSDP Unit——分片边界在 Transformer block.

    auto_wrap_policy:
      - 按 Transformer block 自动包装 (每个 block 是一个 FSDP unit)
      - block 间通信可通过 prefetch 隐藏
    """

    def __init__(self, dim: int, num_layers: int, world_size: int, rank: int):
        self.dim = dim
        self.num_layers = num_layers
        self.world_size = world_size
        self.rank = rank

        # 每层是一个 FSDP Unit
        self.fsdp_units = []
        for i in range(num_layers):
            params = {
                "weight": np.random.randn(dim, dim).astype(np.float32) * np.sqrt(2.0 / dim),
                "bias": np.zeros(dim, dtype=np.float32),
            }
            self.fsdp_units.append(FSDPUnit(params, world_size, rank))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播 (每层先 all-gather 参数)."""
        h = x
        for i, unit in enumerate(self.fsdp_units):
            h = unit.forward(h)
        return h

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向传播 (每层先 all-gather 参数, 再 reduce-scatter 梯度)."""
        grad = grad_output
        for unit in reversed(self.fsdp_units):
            grad = unit.backward(grad)
        return grad

    def optimizer_step(self, lr: float = 1e-3):
        """FSDP 优化步骤 (每 rank 只更新自己的分片)."""
        for unit in self.fsdp_units:
            unit.optimizer_step(lr)

    def get_all_param_shard_sizes(self) -> List[int]:
        """获取各 FSDP unit 的参数分片大小."""
        return [unit.shard_size for unit in self.fsdp_units]


# ============================================================
# 显存/通信量分析
# ============================================================

def analyze_fsdp_memory(
    model_params_GB: float,
    optimizer_states_GB: float,
    world_size: int,
) -> Dict:
    """分析 FSDP 与 DDP 的显存对比.

    FSDP (FULL_SHARD) vs DDP:
      - DDP: 每卡 = 参数 + 优化器状态 + 梯度 (完整)
      - FSDP: 每卡 = (参数 + 优化器状态 + 梯度) / 世界大小
    """
    ddp_total = model_params_GB + optimizer_states_GB + model_params_GB

    fsdp_params = model_params_GB / world_size
    fsdp_opt = optimizer_states_GB / world_size
    fsdp_grad = model_params_GB / world_size
    fsdp_total = fsdp_params + fsdp_opt + fsdp_grad

    fsdp_comm = 2 * model_params_GB + 2 * model_params_GB

    return {
        "DDP_total_per_GPU_GB": ddp_total,
        "FSDP_total_per_GPU_GB": fsdp_total,
        "memory_reduction": ddp_total / fsdp_total if fsdp_total > 0 else float('inf'),
        "FSDP_comm_GB": fsdp_comm,
        "DDP_comm_GB": 2 * model_params_GB,
    }


# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("FSDP (Fully Sharded Data Parallel) 核心流程模拟")
    print("=" * 60)

    # ---- 1. FlatParameter 演示 ----
    print("\n1. FlatParameter 扁平化演示")
    params = {
        "weight": np.random.randn(64, 64).astype(np.float32) * 0.02,
        "bias": np.zeros(64, dtype=np.float32),
    }
    fp_meta, flat_data = flatten_params(params)
    print(f"  原始参数: weight={params['weight'].shape}, bias={params['bias'].shape}")
    print(f"  扁平化后: total_size={fp_meta.total_size}, flat={flat_data.shape}")
    recovered = unflatten_params(fp_meta, flat_data)
    print(f"  恢复验证: weight max diff={np.abs(recovered['weight'] - params['weight']).max():.10f}")
    print(f"  恢复验证: bias max diff={np.abs(recovered['bias'] - params['bias']).max():.10f}")

    # ---- 2. FSDP Unit 前向/反向 ----
    print(f"\n{'='*50}")
    print("2. FSDP Unit 单层演示")
    world_size = 4
    dim = 16
    x = np.random.randn(2, dim).astype(np.float32) * 0.1

    shared_params = {
        "weight": np.random.randn(dim, dim).astype(np.float32) * 0.02,
        "bias": np.zeros(dim, dtype=np.float32),
    }

    units = []
    outputs = []
    for rank in range(world_size):
        params_copy = {"weight": shared_params["weight"].copy(),
                       "bias": shared_params["bias"].copy()}
        unit = FSDPUnit(params_copy, world_size, rank)
        units.append(unit)
        output = unit.forward(x)
        outputs.append(output)
        print(f"  Rank {rank}: local_shard={unit.shard_size} params, "
              f"output shape={output.shape}")

    for rank in range(1, world_size):
        diff = np.abs(outputs[0] - outputs[rank]).max()
        print(f"  Rank 0 vs Rank {rank} 输出差异: {diff:.10f}")

    # ---- 3. 反向传播 + optimize ----
    print(f"\n{'='*50}")
    print("3. 反向传播 + reduce-scatter + optimizer.step")
    grad_out = np.random.randn(2, dim).astype(np.float32) * 0.1

    for rank in range(world_size):
        grad_input = units[rank].backward(grad_out)
        units[rank].optimizer_step(lr=1e-3)
        print(f"  Rank {rank}: grad_shard norm={np.linalg.norm(units[rank].grad_shard):.6f}, "
              f"param_shard norm={np.linalg.norm(units[rank].local_shard):.6f}")

    # ---- 4. 显存分析 ----
    print(f"\n{'='*50}")
    print("4. FSDP vs DDP 显存对比")
    configs = [
        ("7B, 4 GPU", 14, 56, 4),
        ("7B, 8 GPU", 14, 56, 8),
        ("13B, 8 GPU", 26, 104, 8),
        ("70B, 64 GPU", 140, 560, 64),
    ]
    for label, params_gb, opt_gb, ws in configs:
        mem = analyze_fsdp_memory(params_gb, opt_gb, ws)
        print(f"  {label}:")
        print(f"    DDP 每卡: {mem['DDP_total_per_GPU_GB']:.0f} GB")
        print(f"    FSDP 每卡: {mem['FSDP_total_per_GPU_GB']:.1f} GB "
              f"({'✅' if mem['FSDP_total_per_GPU_GB'] <= 80 else '❌'} A100 80GB)")

    # ---- 5. 通信开销 ----
    print(f"\n{'='*50}")
    print("5. 通信开销对比 (7B 模型)")
    print(f"  DDP:  all-reduce 梯度 = {2*14:.0f} GB")
    print(f"  FSDP: all-gather(前向) + all-gather(反向) + reduce-scatter = ~{2*14+2*14:.0f} GB")
    print(f"  FSDP 通信 ≈ 1.5× DDP")

    print("\n✅ FSDP 核心流程模拟完成")

```
