---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Swin Transformer - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
Swin Transformer
================
论文: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
      (Liu et al., ICCV 2021 Best Paper / Marr Prize)
核心贡献: 用窗口注意力（W-MSA）和移位窗口注意力（SW-MSA）解决 ViT 的
         O(N²) 计算复杂度问题，实现线性复杂度 + 多尺度特征金字塔。
架构: 4 个 Stage 逐步下采样 (4x→8x→16x→32x)，每个 Stage 交替使用
      W-MSA 和 SW-MSA blocks。
代码结构:
  1. WindowPartition / WindowReverse - 窗口分割与还原
  2. WindowAttention - 窗口内多头自注意力（含相对位置偏置）
  3. SwinBlock - 两个连续的 Transformer block (W-MSA → SW-MSA)
  4. PatchMerging - 下采样层（类似 CNN pooling）
  5. SwinTransformer - 完整模型

与 [[../11_ViT/ViT|ViT]] 的关系: 解决 ViT 高分辨率 O(N²) 和高层单尺度问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. 窗口操作 —— 将特征图分割为 M×M 的窗口
# ==============================================================================
def window_partition(x: torch.Tensor, window_size: int):
    """
    将特征图分割为不重叠的窗口。

    输入: (B, H, W, C)
    输出: (B * num_windows, window_size, window_size, C)

    为什么分割窗口？
    ViT 的全局自注意力复杂度为 O(N²·d)，其中 N=H×W。
    窗口注意力将复杂度降为 O(N·M²·d)，M=window_size。
    当 M=7 固定时，随分辨率线性增长而非平方增长。
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
    将窗口还原为特征图（window_partition 的逆操作）。

    输入: (B * num_windows, window_size, window_size, C)
    输出: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# ==============================================================================
# 2. 窗口多头自注意力 —— 核心计算（含相对位置偏置）
# ==============================================================================
class WindowAttention(nn.Module):
    """
    在 M×M 窗口内计算多头自注意力，使用相对位置偏置 B。

    公式: Attention(Q, K, V) = Softmax(QK^T/√d + B) · V

    其中 B ∈ ℝ^{M² × M²} 是相对位置偏置矩阵。

    为什么用相对位置偏置而不是绝对位置编码？
    - 绝对位置编码在密集预测任务（检测/分割）上可能有害
    - 相对位置偏置提供平移等变性：无论目标在图像哪个位置，
      patch 之间的相对位置关系不变
    - 消融实验显示相对位置偏置在 COCO 上高 2.8 box AP
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # 通常是 7
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV 一次性投影（比分开三次更高效）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 相对位置偏置表
        # 相对位置范围: [-(M-1), M-1] → 共 2M-1 个值
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # 生成相对位置索引（高效查表）
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)  # (2, M²)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)  # (M², M², 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (M², M²)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask=None):
        """
        x: (num_windows * B, M², C) — 窗口内展平的 patches
        mask: 可选，用于移位窗口的 masking（阻止跨原始窗口的注意力）
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK^T / √d_k
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        # + 相对位置偏置 B
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # 移位窗口的 masking：阻止跨原始窗口的注意力
        if mask is not None:
            nW = mask.shape[0]  # 窗口数
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==============================================================================
# 3. Swin Transformer Block —— 两个连续的 Transformer 层
# ==============================================================================
class SwinBlock(nn.Module):
    """
    包含两个连续的 Transformer block:
      Block 1: W-MSA  (常规窗口自注意力)
      Block 2: SW-MSA (移位窗口自注意力)

    每个 block 内部: LayerNorm → Attention → 残差 → LayerNorm → MLP → 残差
    （使用 Pre-LN 设计，比原始 Transformer 的 Post-LN 更稳定）

    为什么需要交替 W-MSA 和 SW-MSA？
    纯 W-MSA 中不同窗口之间没有信息交互，每个 patch 只能
    "看到"同窗口内的 patches。SW-MSA 将窗口偏移 ⌊M/2⌋，
    让原本在不同窗口的 patches 现在处于同一窗口，实现跨窗口通信。
    """

    def __init__(self, dim: int, input_resolution: tuple, num_heads: int,
                 window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # 如果窗口大小大于输入分辨率，不使用移位
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads,
                                    attn_drop=attn_drop, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # 生成注意力 mask（用于移位窗口）
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size),
                       slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size),
                       slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征大小与输入分辨率不匹配"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # (B, H, W, C)

        # Cyclic Shift（移位窗口）
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        # 窗口分割
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 窗口注意力（含 masking）
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 窗口还原
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                          dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x  # 残差连接 1

        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x  # 残差连接 2

        return x


# ==============================================================================
# 4. Patch Merging —— 下采样层
# ==============================================================================
class PatchMerging(nn.Module):
    """
    将 2×2 邻域 patches 合并，空间分辨率减半，通道数加倍。

    操作: 取 2×2 区域 (每个 C 维) → 拼接为 4C → 线性投影到 2C。
    这类似于 CNN 中的 pooling + channel doubling。

    为什么不用卷积下采样？
    Swin 刻意保持"纯 Transformer"设计，用线性变换而非卷积。
    但本质上 Patch Merging 和 stride-2 卷积的功能是等价的。
    """

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # 取 2×2 邻域中的 4 个位置
        x0 = x[:, 0::2, 0::2, :]  # 左上
        x1 = x[:, 1::2, 0::2, :]  # 左下
        x2 = x[:, 0::2, 1::2, :]  # 右上
        x3 = x[:, 1::2, 1::2, :]  # 右下

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # 4C → 2C

        return x


# ==============================================================================
# 5. Swin Transformer —— 完整模型
# ==============================================================================
class SwinTransformer(nn.Module):
    """
    Swin-T 配置（基础版本）:
      Stage 1: Patch Embed (4x 下采样) → 2× SwinBlock (dim=96, heads=3)
      Stage 2: Patch Merging (8x)      → 2× SwinBlock (dim=192, heads=6)
      Stage 3: Patch Merging (16x)     → 6× SwinBlock (dim=384, heads=12)
      Stage 4: Patch Merging (32x)     → 2× SwinBlock (dim=768, heads=24)

    与 ViT 的关键区别:
    - 层次化: 4 个阶段逐步下采样，构建特征金字塔
    - 局部注意力: 每个窗口内独立计算，线性复杂度
    - 移位窗口: 跨窗口信息交互
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4,
                 in_chans: int = 3, num_classes: int = 1000,
                 embed_dim: int = 96,  # Swin-T 的 C
                 depths: tuple = (2, 2, 6, 2),  # 各 stage 的 block 数
                 num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # Patch Embedding: 4×4 patch → Linear → embed_dim
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm(embed_dim),
        )

        patches_resolution = img_size // patch_size

        # 4 个 Stage
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            # Patch Merging（除了第一个 stage）
            if i_layer > 0:
                layer.append(PatchMerging(dim=int(embed_dim * 2 ** (i_layer - 1))))
            dim = int(embed_dim * 2 ** i_layer)
            resolution = patches_resolution // (2 ** i_layer)
            # Swin Blocks
            for i_block in range(depths[i_layer]):
                shift_size = 0 if (i_block % 2 == 0) else window_size // 2
                layer.append(SwinBlock(
                    dim=dim, input_resolution=(resolution, resolution),
                    num_heads=num_heads[i_layer], window_size=window_size,
                    shift_size=shift_size, mlp_ratio=mlp_ratio,
                    dropout=dropout, attn_drop=attn_drop
                ))
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        # Patch Embedding
        x = self.patch_embed(x)  # (B, 96, 56, 56)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for layer in self.layers:
            for block in layer:
                if isinstance(block, PatchMerging):
                    x = block(x, H, W)
                    H, W = H // 2, W // 2
                else:
                    x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.head(x)
        return x


# ==============================================================================
# 演示
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Swin Transformer 演示")
    print("=" * 60)

    # Swin-T 配置
    model = SwinTransformer(
        img_size=224, patch_size=4, embed_dim=96,
        depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7
    )

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}  (ImageNet 1000 类)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.1f}M  (Swin-T ≈ 28M)")

    # 复杂度对比
    H, W, C = 56, 56, 96
    N = H * W  # 3136
    M = 7
    d = C

    global_ops = 4 * N * d * d + 2 * N * N * d
    window_ops = 4 * N * d * d + 2 * M * M * N * d
    print(f"\n复杂度对比 (H=W=56, C=96):")
    print(f"  全局 MSA FLOPs:  {global_ops / 1e9:.2f}G")
    print(f"  窗口 MSA FLOPs:  {window_ops / 1e9:.2f}G")
    print(f"  加速比:          {global_ops / window_ops:.1f}x")
    print(f"\n关键: 窗口注意力随分辨率线性增长 O(N), 全局注意力平方增长 O(N²)")

```
