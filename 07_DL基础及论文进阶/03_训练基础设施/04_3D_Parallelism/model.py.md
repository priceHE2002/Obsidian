---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# 3D Parallelism (DP + TP + PP) 统一模拟 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# 3D Parallelism (DP + TP + PP) 统一模拟 - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
3D Parallelism (DP + TP + PP) 统一模拟
======================================
将数据并行 (DP)、张量并行 (TP) 和流水线并行 (PP) 统一为三维混合并行策略 (PTD-P),
模拟 GPipe 微批次调度、1F1B 调度和设备网格映射。

论文: [[3D Parallelism]] (Narayanan et al., SC 2021)
核心思想:
  - DP 善跨节点: 梯度 all-reduce 跨 InfiniBand
  - TP 善节点内: 层内参数切分依赖 NVSwitch 低延迟
  - PP 跨节点组内: 层间流水线, 使用 1F1B 减少 bubble

关键设计:
  1. 设备网格映射: (dp_size, pp_size, tp_size) 三维网格
  2. GPipe 调度: 所有 micro-batch 先全部前向再全部反向 (bubble 大)
  3. 1F1B 调度: 前向/反向交替 (one-forward-one-backward), bubble 减半
  4. Bubble ratio: GPipe = (p-1)/(m+p-1), 1F1B bubble 约为 GPipe 的一半

与 [[Megatron-LM]] (TP) 和 [[ZeRO]] (DP 显存优化) 互补.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# ============================================================
# 设备网格映射
# ============================================================

class DeviceMesh:
    """三维设备网格: (dp_size, pp_size, tp_size).

    映射规则:
      - TP = 节点级并行 (DGX 节点内的 8 张 GPU)
      - PP = 跨节点组内并行 (4-8 个 TP 组通过 IB 组成流水线)
      - DP = 跨流水线副本 (多个 PP 流水线之间数据并行)
    """

    def __init__(self, dp_size: int, pp_size: int, tp_size: int):
        """
        Args:
            dp_size: 数据并行度
            pp_size: 流水线并行度 (流水线阶段数)
            tp_size: 张量并行度
        """
        self.dp_size = dp_size
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.total_gpus = dp_size * pp_size * tp_size

        # 构建 rank → (dp, pp, tp) 映射
        self.rank_map = {}
        rank = 0
        for dp in range(dp_size):
            for pp in range(pp_size):
                for tp in range(tp_size):
                    self.rank_map[rank] = (dp, pp, tp)
                    rank += 1

    def get_total_gpus(self) -> int:
        return self.total_gpus

    def get_rank(self, dp: int, pp: int, tp: int) -> int:
        """根据 (dp, pp, tp) 坐标获取全局 rank."""
        return dp * (self.pp_size * self.tp_size) + pp * self.tp_size + tp

    def get_coords(self, rank: int) -> Tuple[int, int, int]:
        return self.rank_map[rank]

    def __repr__(self):
        return (f"DeviceMesh(dp={self.dp_size}, pp={self.pp_size}, "
                f"tp={self.tp_size}, total={self.total_gpus})")


# ============================================================
# 模拟的 "层" 计算
# ============================================================

class SimulatedLayer:
    """模拟一个 Transformer 层的前向/反向计算开销.

    在真实训练中, 这里会是 Megatron-LM 的 TransformerLayer,
    包含 self-attention + FFN, 可能还包含张量并行通信.
    """

    def __init__(self, layer_id: int, fwd_time: float = 1.0, bwd_time: float = 2.0):
        self.layer_id = layer_id
        self.fwd_time = fwd_time  # 模拟前向耗时
        self.bwd_time = bwd_time  # 模拟反向耗时 (通常约为前向的 2 倍)
        # 模拟激活值显存 (MB)
        self.activation_memory = np.random.uniform(50, 200)

    def forward(self, x):
        return x + self.layer_id * 0.1  # 模拟计算

    def backward(self, grad):
        return grad


# ============================================================
# 流水线阶段
# ============================================================

class PipelineStage:
    """一个流水线阶段——管理一组连续的层."""

    def __init__(self, stage_id: int, layers: List[SimulatedLayer]):
        self.stage_id = stage_id
        self.layers = layers
        self.num_layers = len(layers)

    def forward(self, x) -> np.ndarray:
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, grad) -> np.ndarray:
        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g


# ============================================================
# GPipe 调度器
# ============================================================

class GPipeScheduler:
    """GPipe 流水线调度.

    调度模式: 所有 micro-batch 先全部前向, 再全部反向.

    Bubble ratio: (p - 1) / (m + p - 1)
      其中 p = 流水线深度, m = micro-batch 数量

    显存需求高: 所有 micro-batch 的激活值在反向开始前都需要保持.
    """

    def __init__(self, stages: List[PipelineStage], num_microbatches: int):
        """
        Args:
            stages: 流水线阶段列表 (p 个)
            num_microbatches: micro-batch 数量 (m)
        """
        self.stages = stages
        self.p = len(stages)
        self.m = num_microbatches

    def bubble_ratio(self) -> float:
        """计算空闲 bubble 比例.

        公式: bubble = (p - 1) / (m + p - 1)

        为什么有 bubble:
          GPipe 需要等待所有 micro-batch 前向完成才开始反向,
          第一个和最后一个阶段有显著的等待时间.
        """
        return (self.p - 1) / (self.m + self.p - 1)

    def schedule(self) -> List[List[Tuple[str, int]]]:
        """生成 GPipe 调度时间表.

        Returns:
            每阶段的调度: [(操作类型, micro-batch id), ...]
            操作类型: "F"=前向, "B"=反向
        """
        schedule = [[] for _ in range(self.p)]

        # 前向阶段: 按 micro-batch 顺序流过流水线
        for mb in range(self.m):
            for s in range(self.p):
                # 阶段 s 在时间步 (s + mb) 执行 micro-batch mb 的前向
                schedule[s].append(("F", mb))

        # 反向阶段: 从最后一个 micro-batch 开始反向流过流水线
        for mb in range(self.m - 1, -1, -1):
            for s in range(self.p - 1, -1, -1):
                schedule[s].append(("B", mb))

        return schedule

    def print_gantt(self):
        """打印类甘特图的调度可视化."""
        sched = self.schedule()
        max_len = max(len(s) for s in sched)

        print(f"\nGPipe 调度甘特图 (p={self.p}, m={self.m}, bubble={self.bubble_ratio():.2%}):")
        header = "Time  |" + "".join(f" Gpu{i}  |" for i in range(self.p))
        print(header)
        print("-" * len(header))

        for t in range(max_len):
            row = f" t={t:2d} |"
            for s in range(self.p):
                if t < len(sched[s]):
                    op, mb = sched[s][t]
                    row += f" {op}{mb}  |"
                else:
                    row += f"     |"
            print(row)


# ============================================================
# 1F1B 调度器 (One-Forward-One-Backward)
# ============================================================

class OneFOneBScheduler:
    """1F1B 流水线调度.

    这是 3D Parallelism 论文的核心贡献之一.

    调度模式: 每个设备在前向/反向之间交替, 一旦收到结果立即处理.

    优点:
      - Bubble 比例约为 GPipe 的一半
      - 更好的硬件利用率

    缺点:
      - 显存压力更大: 多个 micro-batch 的激活值需要同时保留
      - 调度实现更复杂
    """

    def __init__(self, stages: List[PipelineStage], num_microbatches: int):
        self.stages = stages
        self.p = len(stages)
        self.m = num_microbatches

    def bubble_ratio(self) -> float:
        """1F1B 的 bubble 比例 (近似为 GPipe 的一半)."""
        # 1F1B 的理论 bubble 更小, 近似公式:
        return (self.p - 1) / (self.m + self.p - 1) * 0.5  # 近似

    def schedule(self) -> List[List[Tuple[str, int]]]:
        """生成 1F1B 调度时间表.

        策略:
          1. 暖机阶段 (warm-up): 前 m-p+1 个 micro-batch 只做前向
          2. 稳态阶段 (steady): 每个阶段 1 前向 + 1 反向交替
          3. 冷却阶段 (cool-down): 剩余 micro-batch 只做反向
        """
        sched = [[] for _ in range(self.p)]

        # 简化模拟: 交错前向和反向
        total_ops = self.m * 2  # 每个 micro-batch 有一前向一反向

        for t in range(total_ops + self.p - 1):
            for s in range(self.p):
                # 每个阶段可执行的操作索引范围
                min_op = s  # 最早可执行的操作 (前向到达时间)
                max_op = min(t, total_ops + s - self.p)  # 最晚可执行的操作

                # 在这个范围内找一个未完成的 micro-batch
                # 优先选择前向操作
                mb = (t - s)  # 当前阶段在当前时间步对应的 micro-batch
                if 0 <= mb < self.m:
                    # 决定是前向还是反向
                    ops_before = len(sched[s])
                    if ops_before < self.m:
                        # 还有前向没做 → 做前向
                        sched[s].append(("F", mb))
                    else:
                        # 前向做完了 → 找最早的反向
                        rev_mb = ops_before - self.m  # 从最后一个 micro-batch 开始反向
                        if rev_mb < self.m:
                            sched[s].append(("B", rev_mb))
        return sched

    def schedule_simple(self) -> List[List[Tuple[str, int]]]:
        """简化版 1F1B 调度 (更容易理解)."""
        sched = [[] for _ in range(self.p)]
        warmup_mb = self.p  # 暖机 micro-batch 数

        # 暖机: 阶段 0 做 m 个前向, 阶段 1 做 m-1 个, ...
        for mb in range(self.m):
            for s in range(self.p):
                if s <= mb:
                    sched[s].append(("F", mb - s))

        # 稳态 + 冷却 (从最后一个 micro-batch 开始反向交错)
        for s in range(self.p):
            # 阶段 s 已经有 s+1 个前向
            fwd_count = len(sched[s])
            # 做反向
            for i in range(fwd_count):
                rev_mb = fwd_count - 1 - i
                sched[s].append(("B", rev_mb))

        return sched

    def print_gantt(self):
        """打印 1F1B 甘特图."""
        sched = self.schedule_simple()
        max_len = max(len(s) for s in sched)

        print(f"\n1F1B 调度甘特图 (p={self.p}, m={self.m}, bubble~={self.bubble_ratio():.2%}):")
        header = "Time  |" + "".join(f" Gpu{i}  |" for i in range(self.p))
        print(header)
        print("-" * len(header))

        for t in range(max_len):
            row = f" t={t:2d} |"
            for s in range(self.p):
                if t < len(sched[s]):
                    op, mb = sched[s][t]
                    row += f" {op}{mb}  |"
                else:
                    row += f"     |"
            print(row)


# ============================================================
# PTD-P 统一训练模拟器
# ============================================================

class PTDPTrainer:
    """PTD-P (Pipeline-Tensor-Data Parallelism) 统一训练模拟器.

    组合三种并行策略:
      - DP: 跨流水线副本, 梯度 all-reduce
      - PP: 流水线调度 (GPipe 或 1F1B)
      - TP: 节点内张量并行

    显存分析 (530B 模型, A100 80GB):
      无并行: ~13.5TB
      TP(8) + DP(384): ~118GB
      TP(8) + PP(4) + DP(96): 84GB ✅
      TP(8) + PP(4) + DP(96) + 检查点: 80GB ✅
    """

    def __init__(self, mesh: DeviceMesh, num_layers: int,
                 num_microbatches: int, schedule_type: str = "1f1b"):
        """
        Args:
            mesh: 设备网格
            num_layers: 总层数
            num_microbatches: micro-batch 数量
            schedule_type: 流水线调度类型 ("gpipe" 或 "1f1b")
        """
        self.mesh = mesh
        self.num_layers = num_layers
        self.num_microbatches = num_microbatches
        self.schedule_type = schedule_type

        # 将层分配到流水线阶段
        layers_per_stage = num_layers // mesh.pp_size
        self.stages = []
        for pp in range(mesh.pp_size):
            start = pp * layers_per_stage
            end = start + layers_per_stage
            layers = [SimulatedLayer(i) for i in range(start, end)]
            self.stages.append(PipelineStage(pp, layers))

        # 选择调度器
        if schedule_type == "gpipe":
            self.scheduler = GPipeScheduler(self.stages, num_microbatches)
        else:
            self.scheduler = OneFOneBScheduler(self.stages, num_microbatches)

    def estimate_memory(self, model_params_gb: float,
                        optimizer_states_gb: float) -> Dict[str, float]:
        """估算各维度下的每 GPU 显存.

        公式:
          每GPU显存 = 模型参数/TP + 优化器状态/(DP*TP) + 激活值/(PP*TP)
        """
        dp, pp, tp = self.mesh.dp_size, self.mesh.pp_size, self.mesh.tp_size

        # 模型参数由 TP 分担
        params_per_gpu = model_params_gb / tp

        # 优化器状态由 DP 和 TP 共同分担 (DP 是分片, TP 也是分片)
        opt_per_gpu = optimizer_states_gb / (dp * tp)

        # 激活值由 PP 和 TP 分担 (每个 PP 阶段只存部分层的激活值)
        # 假设激活值总量为模型参数的 1-3 倍 (取决于序列长度)
        activation_total = model_params_gb * 1.5
        activation_per_gpu = activation_total / (pp * tp)

        # 临时缓冲 (通信缓冲区等)
        buffer = model_params_gb * 0.1

        total = params_per_gpu + opt_per_gpu + activation_per_gpu + buffer

        return {
            "params_per_gpu_GB": params_per_gpu,
            "optimizer_per_gpu_GB": opt_per_gpu,
            "activation_per_gpu_GB": activation_per_gpu,
            "buffer_GB": buffer,
            "total_per_gpu_GB": total,
            "gpu_limit_exceeded": total > 80,  # A100 80GB 上限
        }

    def run_simulation(self):
        """运行调度模拟, 输出可视化结果."""
        # 打印调度甘特图
        if self.schedule_type == "gpipe":
            self.scheduler.print_gantt()
        else:
            self.scheduler.print_gantt()

        # Bubble 分析
        bubble = self.scheduler.bubble_ratio()
        print(f"\nBubble 比例: {bubble:.2%}")
        print(f"有效利用率: {1 - bubble:.2%}")


# ============================================================
# Bubble Ratio 对比分析
# ============================================================

def compare_bubble_ratios():
    """对比不同调度策略和配置下的 bubble ratio."""
    print(f"\n{'='*50}")
    print("Bubble Ratio 对比分析")
    print(f"{'p':>3} {'m':>3} | {'GPipe':>8} | {'1F1B(~)':>8} | {'改善':>8}")
    print("-" * 45)

    configs = [
        (2, 4), (2, 8), (4, 8), (4, 16), (8, 16), (8, 32), (16, 64),
    ]
    for p, m in configs:
        gpipe_bubble = (p - 1) / (m + p - 1)
        f1fb_bubble = gpipe_bubble * 0.5  # 近似
        improvement = (gpipe_bubble - f1fb_bubble) / gpipe_bubble * 100
        print(f"{p:3d} {m:3d} | {gpipe_bubble:7.2%} | {f1fb_bubble:7.2%} | {improvement:7.1f}%")


# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("3D Parallelism (DP + TP + PP) 统一模拟")
    print("=" * 60)

    # ---- 1. 设备网格映射 ----
    print("\n1. 设备网格映射示例")
    mesh = DeviceMesh(dp_size=4, pp_size=4, tp_size=8)
    print(f"  {mesh}")
    print(f"  Rank 0 坐标: {mesh.get_coords(0)}")
    print(f"  Rank 127 坐标: {mesh.get_coords(127)}")
    print(f"  (dp=0, pp=0, tp=0) → rank {mesh.get_rank(0, 0, 0)}")
    print(f"  (dp=3, pp=3, tp=7) → rank {mesh.get_rank(3, 3, 7)}")

    # ---- 2. GPipe 调度演示 ----
    print("\n2. GPipe 调度")
    stages_gpipe = [
        PipelineStage(0, [SimulatedLayer(i) for i in range(2)])
        for _ in range(4)
    ]
    gpipe = GPipeScheduler(stages_gpipe, num_microbatches=4)
    gpipe.print_gantt()

    # ---- 3. 1F1B 调度演示 ----
    print("\n3. 1F1B 调度")
    stages_1f1b = [
        PipelineStage(0, [SimulatedLayer(i) for i in range(2)])
        for _ in range(4)
    ]
    f1fb = OneFOneBScheduler(stages_1f1b, num_microbatches=4)
    f1fb.print_gantt()

    # ---- 4. Bubble 对比 ----
    compare_bubble_ratios()

    # ---- 5. 显存估算 (模拟 530B 模型) ----
    print(f"\n{'='*50}")
    print("显存估算 (530B 模型, A100 80GB 模拟)")
    trainer = PTDPTrainer(
        mesh=DeviceMesh(dp_size=96, pp_size=4, tp_size=8),
        num_layers=72,
        num_microbatches=16,
        schedule_type="1f1b",
    )

    # 530B 参数: FP16 模型 ~1TB, 优化器状态 ~8TB (FP32 Adam)
    mem = trainer.estimate_memory(model_params_gb=1060, optimizer_states_gb=8480)
    for k, v in mem.items():
        marker = " ❌" if k == "gpu_limit_exceeded" and v else ""
        print(f"  {k}: {v}{marker}")

    print("\n✅ 3D Parallelism 模拟完成")

```

```
