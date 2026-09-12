---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Megatron-LM 张量并行 (Tensor Parallelism) 模拟实现 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Megatron-LM 张量并行 (Tensor Parallelism) 模拟实现 - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
Megatron-LM 张量并行 (Tensor Parallelism) 模拟实现
==================================================
将 Transformer 层的 Attention 和 FFN 权重沿列/行切分到多个设备,
通过 all-gather 和 all-reduce 通信实现层内并行。

论文: [[Megatron-LM]] (Shoeybi et al., 2019)
核心思想:
  - Transformer 的 MHA (Multi-Head Attention) 和 FFN 天然可张量切分
  - 列切分 (column-parallel): 沿输出维度切分 QKV / FFN W1, 各设备独立计算
  - 行切分 (row-parallel): 沿输入维度切分 Attention Output / FFN W2, 需要 all-reduce

关键设计:
  1. Attention: QKV 列切分 → 独立计算 → W_O 行切分 + all-reduce
  2. FFN: W1 列切分 → GeLU → W2 行切分 + all-reduce
  3. 通信模式: f (forward all-gather) + g (backward all-reduce) = f/g 操作
  4. 每 Transformer 层总通信量 = 4bsh (与模型大小无关, 仅与 隐藏维度+序列长度 相关)

与 [[3D Parallelism]] 中的 TP 维度完全对应.
"""

import numpy as np
from typing import List, Tuple


# ============================================================
# 通信原语模拟 (f 和 g 操作)
# ============================================================

class CommunicationGroup:
    """模拟张量并行组内的通信操作.

    在真实 Megatron-LM 中, 这些操作通过 NCCL all-gather / all-reduce 实现.
    这里用 numpy 模拟, 展示数学等价性.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size  # TP 组内 GPU 数

    # ---- f 操作: 前向传播中的 identity / all-reduce ----
    # 在前向中, f 通常对应 identity (不需要通信)
    # 在反向中, f 对应 all-reduce (梯度同步)

    # ---- g 操作: 前向传播中的 all-reduce ----
    # 在前向中, g 对应 all-reduce (聚合分片结果)
    # 在反向中, g 对应 identity (梯度已经分布在各设备上)

    @staticmethod
    def all_reduce(tensors: List[np.ndarray]) -> np.ndarray:
        """模拟 all-reduce: 所有设备上的张量求和后广播.

        在 Megatron 中, Attention Output 和 FFN W2 输出需要 all-reduce,
        因为每个设备只计算了部分和.
        """
        return sum(tensors)

    @staticmethod
    def all_gather(tensors: List[np.ndarray], dim: int = -1) -> np.ndarray:
        """模拟 all-gather: 收集所有设备上的张量并拼接.

        在前向传播中, 如果 Attention 头被切分, 需要 all-gather 收集各头的输出.
        """
        return np.concatenate(tensors, axis=dim)

    @staticmethod
    def reduce_scatter(tensors: List[np.ndarray], world_size: int,
                       rank: int, dim: int = -1) -> np.ndarray:
        """模拟 reduce-scatter: 先求和再按 rank 分片."""
        summed = sum(tensors)
        total = summed.shape[dim]
        chunk_size = total // world_size
        start = rank * chunk_size
        end = start + chunk_size
        return np.take(summed, range(start, end), axis=dim)


# ============================================================
# 张量切分工具
# ============================================================

def column_split(weight: np.ndarray, world_size: int, rank: int) -> np.ndarray:
    """列切分 (column-parallel): 沿输出维度(axis=1)切分.

    用于:
      - QKV 投影: W_Q, W_K, W_V 沿列切分 (每个设备持有一部分 head)
      - FFN W1: 沿列切分 (每个设备持有部分中间维度)

    数学:
      W ∈ R^{h × d_out} → W_i ∈ R^{h × d_out/t}  其中 t=world_size
    """
    d_out = weight.shape[1]
    chunk_size = d_out // world_size
    start = rank * chunk_size
    end = start + chunk_size
    return weight[:, start:end].copy()


def row_split(weight: np.ndarray, world_size: int, rank: int) -> np.ndarray:
    """行切分 (row-parallel): 沿输入维度(axis=0)切分.

    用于:
      - Attention Output W_O: 沿行切分
      - FFN W2: 沿行切分

    数学:
      W ∈ R^{d_in × h} → W_i ∈ R^{d_in/t × h}  其中 t=world_size
    """
    d_in = weight.shape[0]
    chunk_size = d_in // world_size
    start = rank * chunk_size
    end = start + chunk_size
    return weight[start:end, :].copy()


# ============================================================
# 带张量并行的 Transformer 层
# ============================================================

class TensorParallelAttention:
    """张量并行 Self-Attention 层.

    切分策略:
      - QKV 投影矩阵按列切分: 每个设备持有 h/t 个 attention head
      - W_O 输出投影按行切分: 各设备计算部分输出后 all-reduce 求和

    通信:
      - 前向: QKV计算无需通信(列切分), 输出投影需要 all-reduce(W_O)
      - 反向: QKV梯度需要 all-gather, W_O 梯度无需通信(行切分)
    """

    def __init__(self, hidden_size: int, num_heads: int, tp_size: int, rank: int):
        """
        Args:
            hidden_size: 隐藏维度 h
            num_heads: attention 头数 (必须能被 tp_size 整除)
            tp_size: 张量并行度 t
            rank: 当前设备编号
        """
        assert num_heads % tp_size == 0, \
            f"头数 {num_heads} 必须能被 TP 大小 {tp_size} 整除"
        assert hidden_size % tp_size == 0, \
            f"隐藏维度 {hidden_size} 必须能被 TP 大小 {tp_size} 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.tp_size = tp_size
        self.rank = rank
        self.local_heads = num_heads // tp_size
        self.comm = CommunicationGroup(tp_size)

        # ---- 列切分: QKV 投影 ----
        # 原形状: (hidden_size, hidden_size * 3)
        # 每个设备持有: (hidden_size, hidden_size * 3 / tp_size)
        full_qkv_dim = hidden_size * 3
        # 初始化完整权重, 再切分
        self.W_qkv_full = np.random.randn(hidden_size, full_qkv_dim).astype(np.float32) * 0.02
        self.W_qkv = column_split(self.W_qkv_full, tp_size, rank)

        # ---- 行切分: 输出投影 W_O ----
        # 原形状: (hidden_size, hidden_size)
        self.W_o_full = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_o = row_split(self.W_o_full, tp_size, rank)  # 形状: (h/t, h)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播.

        Args:
            x: 输入, 形状 (batch, seq_len, hidden_size)
        Returns:
            输出, 形状 (batch, seq_len, hidden_size)
        """
        batch, seq_len, hidden = x.shape
        head_dim = self.head_dim
        local_heads = self.local_heads

        # ---- 1. QKV 投影 (列切分, 无通信) ----
        # 每个设备只计算自己持有的 head 的 Q, K, V
        qkv = x @ self.W_qkv  # (batch, seq_len, 3 * hidden / tp)

        # 拆分为 Q, K, V
        qkv = qkv.reshape(batch, seq_len, 3, local_heads, head_dim)
        q = qkv[:, :, 0]  # (batch, seq_len, local_heads, head_dim)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # ---- 2. Scaled Dot-Product Attention (每个设备独立计算) ----
        # 为什么不需要通信: 每个 head 的 attention 完全独立
        scale = np.sqrt(head_dim)
        attn_scores = q @ k.transpose(0, 1, 3, 2) / scale  # (b, s, lh, s)
        # Causal mask for decoder-only
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32) * -1e9, k=1)
        attn_scores = attn_scores + causal_mask[np.newaxis, np.newaxis, :, :]
        attn_weights = self._softmax(attn_scores, axis=-1)
        attn_out = attn_weights @ v  # (b, s, lh, hd)

        # 合并本地 head 输出
        attn_out = attn_out.reshape(batch, seq_len, local_heads * head_dim)

        # ---- 3. 局部输出投影 (行切分 W_O) ----
        # 每个设备计算: local_out_i = attn_out @ W_O_i
        # W_O_i 形状: (local_heads * head_dim, hidden_size) = (h/t, h)
        local_output = attn_out @ self.W_o  # (b, s, h)

        return local_output  # 需要后续 all-reduce 求和得到完整输出

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """数值稳定的 softmax."""
        x_max = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=axis, keepdims=True)

    def post_forward_all_reduce(self, local_outputs: List[np.ndarray]) -> np.ndarray:
        """前向传播后的 all-reduce——聚合各设备的局部输出.

        对应 Megatron 中的 g 操作 (前向 all-reduce).
        """
        return self.comm.all_reduce(local_outputs)


class TensorParallelFFN:
    """张量并行 Feed-Forward Network.

    切分策略:
      - W1 按列切分: 每个设备持有部分中间维度
      - W2 按行切分: 各设备计算局部输出后 all-reduce 求和

    为什么 FFN 的 "列+行" 双切分巧妙:
      中间激活值 (GeLU 输出) 始终在设备本地, 无需跨设备通信,
      仅需在 W2 输出后做一次 all-reduce.
    """

    def __init__(self, hidden_size: int, tp_size: int, rank: int):
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4  # FFN 中间维度通常为 4h
        self.tp_size = tp_size
        self.rank = rank

        # ---- 列切分: W1 (hidden, intermediate/tp) ----
        self.W1_full = np.random.randn(hidden_size, self.intermediate_size).astype(np.float32) * 0.02
        self.W1 = column_split(self.W1_full, tp_size, rank)

        # ---- 行切分: W2 (intermediate/tp, hidden) ----
        self.W2_full = np.random.randn(self.intermediate_size, hidden_size).astype(np.float32) * 0.02
        self.W2 = row_split(self.W2_full, tp_size, rank)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播.

        流程:
          1. h = x @ W1_i  (列切分, 无通信)
          2. h = GeLU(h)   (设备本地)
          3. out_i = h @ W2_i  (行切分, 无通信)
          返回 local_output, 需要后续 all-reduce.
        """
        # Step 1: 列切分 W1 前向
        hidden = x @ self.W1  # (batch, seq_len, inter/tp)

        # Step 2: GeLU 激活 (每个设备完全独立)
        activated = self._gelu(hidden)

        # Step 3: 行切分 W2 前向
        local_output = activated @ self.W2  # (batch, seq_len, hidden)

        return local_output

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """GELU 激活函数 (近似)."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def all_reduce_outputs(local_outputs: List[np.ndarray]) -> np.ndarray:
        """聚合各设备的局部 FFN 输出."""
        return sum(local_outputs)


# ============================================================
# 完整的 Transformer 层 (张量并行版本)
# ============================================================

class TensorParallelTransformerLayer:
    """利用张量并行的单层 Transformer.

    通信模式总览:
      每 Transformer 层:
        Attention:
          - QKV 前向: 无通信 (列切分)
          - Attn 计算: 无通信 (各 head 独立)
          - W_O 后: all-reduce (每个设备的部分和需要聚合)
        FFN:
          - W1 前向: 无通信 (列切分)
          - GeLU: 无通信
          - W2 后: all-reduce (每个设备的部分和需要聚合)
    """

    def __init__(self, hidden_size: int, num_heads: int, tp_size: int, rank: int):
        self.attention = TensorParallelAttention(hidden_size, num_heads, tp_size, rank)
        self.ffn = TensorParallelFFN(hidden_size, tp_size, rank)
        # LayerNorm 权重 (通常不做张量并行, 每个设备持有完整副本)
        self.ln1_weight = np.ones(hidden_size, dtype=np.float32)
        self.ln2_weight = np.ones(hidden_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """单层前向 (返回局部输出, 需要 all-reduce 后才是完整结果).

        Returns:
            attn_local: Attention 的局部输出 (需要 all-reduce)
            ffn_local: FFN 的局部输出 (需要 all-reduce)
        """
        # ---- Attention 块 ----
        residual = x
        x_norm = self._layer_norm(x, self.ln1_weight)
        attn_local = self.attention.forward(x_norm)

        # Attention 输出需要 all-reduce
        # (此处只返回局部结果, 调用者在收集所有 rank 的结果后 all-reduce)
        # 实际 Megatron: attn_local 在 forward 最后做了 all-reduce

        # ---- FFN 块 (为简化, 先对 attention 输出做 LN) ----
        x_norm2 = self._layer_norm(x, self.ln2_weight)
        ffn_local = self.ffn.forward(x_norm2)

        return attn_local, ffn_local

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """LayerNorm (在最后一维)."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return weight * (x - mean) / np.sqrt(var + eps)


# ============================================================
# 通信量估算 (每层)
# ============================================================

def estimate_communication(batch_size: int, seq_len: int, hidden_size: int,
                           tp_size: int) -> dict:
    """估算每 Transformer 层的通信量.

    论文关键结论:
      TP 的通信量 = 4bsh (每层), 与模型参数总量无关.
      而 DP 的通信量 = 2|P| (与参数总数成正比).
      因此当模型很大时, TP 的通信量相对 DP 非常小.
    """
    # 每次 all-reduce 通信量 = 2 * tensor_size (ring algorithm)
    # Tensor size = b * s * h
    tensor_size_fp16 = batch_size * seq_len * hidden_size * 2  # bytes

    # Attention W_O all-reduce (forward + backward, 各 2×)
    attn_comm = 2 * tensor_size_fp16 * 2  # fwd: 1 all-reduce, bwd: 1 all-reduce

    # FFN W2 all-reduce (forward + backward, 各 2×)
    ffn_comm = 2 * tensor_size_fp16 * 2

    total_per_layer = attn_comm + ffn_comm  # = 4 * b * s * h * 2 bytes

    # 对比: DP 通信量 = 2|P| = 2 * 参数量
    return {
        "tensor_size_MB": tensor_size_fp16 / (1024 ** 2),
        "attn_comm_MB": attn_comm / (1024 ** 2),
        "ffn_comm_MB": ffn_comm / (1024 ** 2),
        "total_per_layer_MB": total_per_layer / (1024 ** 2),
    }


# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("Megatron-LM 张量并行 (Tensor Parallelism) 模拟")
    print("=" * 60)

    # 配置
    hidden_size = 64
    num_heads = 8
    tp_size = 4  # 4 个 GPU 做张量并行
    batch_size = 2
    seq_len = 8

    print(f"\n配置: hidden={hidden_size}, heads={num_heads}, "
          f"TP={tp_size}, batch={batch_size}, seq_len={seq_len}")

    # ---- 1. 检查权重切分的正确性 ----
    print(f"\n{'='*40}")
    print("权重切分验证")
    # 完整权重
    full_w = np.arange(12).reshape(3, 4).astype(np.float32)
    print(f"  完整权重形状: {full_w.shape}")

    # 列切分
    w_col_0 = column_split(full_w, 2, 0)
    w_col_1 = column_split(full_w, 2, 1)
    print(f"  列切分 (rank 0): {w_col_0.tolist()}")
    print(f"  列切分 (rank 1): {w_col_1.tolist()}")
    # 验证列切分可被 all-gather 恢复
    recovered = np.concatenate([w_col_0, w_col_1], axis=1)
    assert np.allclose(recovered, full_w), "列切分恢复失败"
    print("  ✅ 列切分可正确恢复")

    # 行切分
    w_row_0 = row_split(full_w, 2, 0)
    w_row_1 = row_split(full_w, 2, 1)
    print(f"  行切分 (rank 0): {w_row_0.shape}")
    print(f"  行切分 (rank 1): {w_row_1.shape}")
    assert np.allclose(np.concatenate([w_row_0, w_row_1], axis=0), full_w)
    print("  ✅ 行切分可正确恢复")

    # ---- 2. 各 rank 的前向传播模拟 ----
    print(f"\n{'='*40}")
    print("各 rank 前向传播")
    x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32) * 0.1

    # 创建所有 rank 的层 (共享相同的完整权重)
    layers = []
    for rank in range(tp_size):
        layer = TensorParallelTransformerLayer(hidden_size, num_heads, tp_size, rank)
        layers.append(layer)

    # 让所有 rank 共享相同的 Attention/FFN 完整权重 (确保一致性)
    ref_attn_full = layers[0].attention.W_qkv_full.copy()
    ref_o_full = layers[0].attention.W_o_full.copy()
    ref_w1_full = layers[0].ffn.W1_full.copy()
    ref_w2_full = layers[0].ffn.W2_full.copy()
    for rank in range(1, tp_size):
        layers[rank].attention.W_qkv_full = ref_attn_full.copy()
        layers[rank].attention.W_o_full = ref_o_full.copy()
        layers[rank].ffn.W1_full = ref_w1_full.copy()
        layers[rank].ffn.W2_full = ref_w2_full.copy()
        # 重新切分
        layers[rank].attention.W_qkv = column_split(ref_attn_full, tp_size, rank)
        layers[rank].attention.W_o = row_split(ref_o_full, tp_size, rank)
        layers[rank].ffn.W1 = column_split(ref_w1_full, tp_size, rank)
        layers[rank].ffn.W2 = row_split(ref_w2_full, tp_size, rank)

    # 前向传播——各 rank 独立计算
    attn_locals = []
    ffn_locals = []
    for rank in range(tp_size):
        attn_l, ffn_l = layers[rank].forward(x)
        attn_locals.append(attn_l)
        ffn_locals.append(ffn_l)
        print(f"  Rank {rank}: attn_local shape={attn_l.shape}, "
              f"ffn_local shape={ffn_l.shape}")

    # All-reduce 聚合 (g 操作)
    attn_full = sum(attn_locals)
    ffn_full = sum(ffn_locals)
    print(f"  All-reduce 后 attn shape: {attn_full.shape}")
    print(f"  All-reduce 后 ffn shape: {ffn_full.shape}")

    # ---- 3. 通信量估算 ----
    print(f"\n{'='*40}")
    print("通信量估算")
    comm = estimate_communication(batch_size, seq_len, hidden_size, tp_size)
    for k, v in comm.items():
        print(f"  {k}: {v:.4f} MB")

    print("\n✅ Megatron-LM 张量并行模拟完成")

```

```
