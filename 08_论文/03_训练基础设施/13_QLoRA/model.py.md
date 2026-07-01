---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# QLoRA 完整实现 - 基于 [[QLoRA]] (Dettmers et al., NeurIPS 2023) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
QLoRA 完整实现 - 基于 [[QLoRA]] (Dettmers et al., NeurIPS 2023)

实现 NF4 (NormalFloat4) 量化格式、双重量化 (Double Quantization)、
以及 QLoRA 训练流程。NF4 将权重从 16-bit 压缩到 4-bit，通过信息论
最优分位数设计在正态分布假设下最小化量化误差；双重量化进一步压缩
尺度因子本身，节省约 3GB 显存（对 65B 模型）。

核心组件:
- NF4Tensor: NormalFloat4 量化/反量化
- DoubleQuantizer: 双重量化（对尺度因子做 fp8 二次量化）
- QLoRALayer: QLoRA 线性层（NF4 冻结权重 + bf16 LoRA 适配器）

参考:
- [[QLoRA]] - 原始论文 (NeurIPS 2023)
- [[LoRA]] - QLoRA 的基础 PEFT 方法
- [[LLM.int8()]] - 同作者的 8-bit 推理量化前驱
- [[GPTQ]] - 后训练量化的对比基线
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


_NF4_QUANTIZATION_LEVELS = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250734090805054,
    -0.3949171304702759,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.2461123025417328,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)


class NF4Tensor:
    """NormalFloat4 量化/反量化。基于 N(0,1) 的等概率分位数设计。"""

    def __init__(self, block_size: int = 64):
        self.block_size = block_size
        self.levels = _NF4_QUANTIZATION_LEVELS.clone()

    def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逐块量化为 NF4。每 block_size 个权重共享一个 fp32 缩放因子。"""
        original_shape = weight.shape
        w_flat = weight.float().reshape(-1)
        num_elements = w_flat.numel

        if num_elements % self.block_size != 0:
            pad_size = self.block_size - (num_elements % self.block_size)
            w_flat = torch.cat([w_flat, torch.zeros(pad_size)])
            num_elements = w_flat.numel

        w_blocks = w_flat.reshape(-1, self.block_size)
        scales = w_blocks.abs().max(dim=-1).values
        scales = torch.clamp(scales, min=1e-12)

        normalized = w_blocks / scales.unsqueeze(-1)
        dist = (normalized.unsqueeze(-1) - self.levels.to(w_flat.device).unsqueeze(0).unsqueeze(0)).abs()
        indices = dist.argmin(dim=-1)

        return indices, scales

    def dequantize(self, indices: torch.Tensor, scales: torch.Tensor,
                   original_shape: torch.Size) -> torch.Tensor:
        """从 NF4 索引反量化回浮点数。"""
        levels = self.levels.to(indices.device)
        values = levels[indices]
        values = values * scales.unsqueeze(-1)
        values = values.reshape(-1)[:math.prod(original_shape)]
        return values.reshape(original_shape)


class DoubleQuantizer:
    """对 NF4 的尺度因子进行二次量化（fp8），进一步压缩显存。"""

    def __init__(self, block_size: int = 256):
        self.block_size = block_size

    def quantize(self, scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = scales.shape[0]
        if num_blocks % self.block_size != 0:
            pad = self.block_size - (num_blocks % self.block_size)
            scales = torch.cat([scales, torch.zeros(pad)])
            num_blocks = scales.shape[0]

        scale_blocks = scales.reshape(-1, self.block_size)
        second_scales = scale_blocks.abs().max(dim=-1).values
        second_scales = torch.clamp(second_scales, min=1e-12)

        normalized = scale_blocks / second_scales.unsqueeze(-1)
        quantized_scales = torch.round(normalized * 127).clamp(-127, 127).to(torch.int8)
        return quantized_scales, second_scales

    def dequantize(self, quantized_scales: torch.Tensor,
                   second_scales: torch.Tensor,
                   original_num_blocks: int) -> torch.Tensor:
        values = quantized_scales.float() / 127.0
        values = values * second_scales.unsqueeze(-1)
        values = values.reshape(-1)[:original_num_blocks]
        return values


class QLoRALinear(nn.Module):
    """QLoRA 线性层：NF4 冻结权重 + LoRA 低秩适配器。"""

    def __init__(
        self, in_features: int, out_features: int,
        r: int = 32, lora_alpha: float = 32.0, block_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.nf4 = NF4Tensor(block_size=block_size)
        self.dq = DoubleQuantizer(block_size=256)
        self.nf4_indices: Optional[torch.Tensor] = None
        self.dq_scales: Optional[torch.Tensor] = None
        self.dq_second_scales: Optional[torch.Tensor] = None
        self.weight_shape: Optional[torch.Size] = None

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self._init_lora_weights()

    def _init_lora_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def prepare_weights(self, original_weight: torch.Tensor):
        self.weight_shape = original_weight.shape
        indices, scales = self.nf4.quantize(original_weight.detach())
        self.nf4_indices = indices
        dq_scales, dq_second_scales = self.dq.quantize(scales)
        self.dq_scales = dq_scales
        self.dq_second_scales = dq_second_scales
        self._num_first_scales = scales.shape[0]

    def _get_dequantized_weight(self) -> torch.Tensor:
        first_scales = self.dq.dequantize(
            self.dq_scales, self.dq_second_scales, self._num_first_scales
        )
        return self.nf4.dequantize(self.nf4_indices, first_scales, self.weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_deq = self._get_dequantized_weight().to(dtype=x.dtype, device=x.device)
        base_out = F.linear(x, W_deq)
        lora_out = F.linear(F.linear(x, self.lora_A) * self.scaling, self.lora_B)
        return base_out + lora_out


if __name__ == "__main__":
    print("=" * 60)
    print("QLoRA 演示: NF4 量化 + 双重量化 + LoRA 训练")
    print("=" * 60)

    torch.manual_seed(42)
    weight = torch.randn(256, 256) * 0.15

    nf4 = NF4Tensor(block_size=64)
    indices, scales = nf4.quantize(weight)
    deq_weight = nf4.dequantize(indices, scales, weight.shape)
    print(f"NF4 量化 MSE: {F.mse_loss(deq_weight, weight):.6f}")

    dq = DoubleQuantizer(block_size=256)
    dq_scales, dq_second = dq.quantize(scales)
    recovered = dq.dequantize(dq_scales, dq_second, scales.shape[0])
    print(f"双重量化尺度因子 MSE: {F.mse_loss(recovered, scales):.8f}")

    qlora_layer = QLoRALinear(256, 256, r=8, lora_alpha=16.0)
    qlora_layer.prepare_weights(weight)
    x = torch.randn(4, 256)
    out = qlora_layer(x)
    loss = out.sum()
    loss.backward()
    print(f"LoRA_A 有梯度: {qlora_layer.lora_A.grad is not None}")
    print(f"LoRA_B 有梯度: {qlora_layer.lora_B.grad is not None}")
    print("=" * 60)
    print("演示完成。")

```
