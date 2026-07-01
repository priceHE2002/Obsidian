---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# ViT: Vision Transformer - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
ViT: Vision Transformer
=======================
论文: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
      (Dosovitskiy et al., ICLR 2021)
核心贡献: 将标准 Transformer 直接应用于图像——把图像切成 16×16 patches，
         当作"视觉单词"喂给 Transformer Encoder。证明了在数据足够大时，
         纯 Transformer 可超越 CNN，开启视觉 Transformer 时代。
架构: Patch Embedding + [CLS] token + Position Embedding + Transformer Encoder
代码结构:
  1. PatchEmbedding - 将图像分割为 patches 并线性投影
  2. ViTEncoderBlock - Pre-LN Transformer Encoder Block
  3. ViT - 完整模型（ViT-B/16, ViT-L/16, ViT-H/14）

关键设计:
  - Patch Embedding: 16×16 卷积（步长=patch_size）实现"patchify"
  - [class] token: 借鉴 BERT，最终隐藏状态用于图像级分类
  - 1D 可学习位置编码: 位置信息自动习得（无需 2D 偏置）
  - Pre-LN 设计: LayerNorm 在注意力/MLP 之前

与 [[../01_Attention_Is_All_You_Need/Attention Is All You Need|Transformer]] 的关系:
  ViT 只使用 Transformer Encoder（无 Decoder/Cross-Attention）
与 [[../02_BERT/BERT|BERT]] 的关系:
  ViT 的 [CLS] token 直接借鉴 BERT
"""

import torch
import torch.nn as nn
import math


# ==============================================================================
# 1. Patch Embedding —— 将图像变成 tokens
# ==============================================================================
class PatchEmbedding(nn.Module):
    """
    Patch Embedding（ViT 最核心的设计）

    将图像 x ∈ R^(H×W×C) 分割为 N = HW/P² 个不重叠的 patches，
    每个 patch 大小为 P×P，展平后通过线性投影映射到 d_model 维。

    实现方式: 用 stride=patch_size 的 2D 卷积实现 "patchify"。

    序列长度: N = (224/16)² = 196（对于 ViT-B/16）

    为什么用 16×16 的 patch？
    - 太小: 序列太长（如 8×8 → 784 patches），Self-Attention 的 O(N²) 成本爆炸
    - 太大: 序列太短（如 32×32 → 49 patches），丢失空间细节
    - 16×16 是实验验证的最佳权衡
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 用卷积实现 patchify: stride=patch_size 等效于非重叠分割
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, C, H, W)
        returns: (batch, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        # 卷积 patchify: (B, C, H, W) → (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # 展平: (B, embed_dim, H/P, W/P) → (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


# ==============================================================================
# 2. ViT Encoder Block (Pre-LN)
# ==============================================================================
class ViTEncoderBlock(nn.Module):
    """
    ViT Transformer Encoder Block

    结构（Pre-LN）:
      x → LayerNorm → Multi-Head Self-Attn → + x (残差)
      x → LayerNorm → MLP               → + x (残差)

    MLP: d_model → 4×d_model → d_model, GELU 激活

    与原始 [[../01_Attention_Is_All_You_Need/Attention Is All You Need|Transformer]] 的区别:
    - Pre-LN 而非 Post-LN
    - GELU 激活（而非 ReLU）
    - 无 Cross-Attention、无 causal mask（ViT 只用 Encoder）
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)  # Pre-LN
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)  # Pre-LN
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),                         # ViT 使用 GELU
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# 3. 完整 ViT 模型
# ==============================================================================
class ViT(nn.Module):
    """
    Vision Transformer (ViT)

    | 模型       | Patch | Layers | Dim  | MLP   | Heads | 参数   |
    |-----------|-------|--------|------|-------|-------|--------|
    | ViT-B/16  | 16    | 12     | 768  | 3072  | 12    | 86M    |
    | ViT-L/16  | 16    | 24     | 1024 | 4096  | 16    | 307M   |
    | ViT-H/14  | 14    | 32     | 1280 | 5120  | 16    | 632M   |

    训练策略: 在 JFT-300M/ImageNet-21K 上预训练 → ImageNet 上微调
    关键发现: 数据不足时 ViT 不如 CNN，但数据足够大时超越 CNN
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [class] token —— 借鉴 [[../02_BERT/BERT|BERT]] 的 [CLS] 设计
        # 这个可学习向量的最终输出用于图像级分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码（可学习的 1D 位置编码，而非 2D）
        # 实验发现 1D/2D/相对位置编码差距<0.1%
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder 层
        self.blocks = nn.ModuleList([
            ViTEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 最终 LayerNorm + 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # 添加 [CLS] token（在所有 patch 之前）
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer Encoder
        for block in self.blocks:
            x = block(x)

        # 仅用 [CLS] token 的最终输出做分类
        x = self.norm(x)
        cls_output = x[:, 0]  # 取 [CLS] 位置
        return self.head(cls_output)


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ViT (Vision Transformer) 架构演示")
    print("=" * 60)

    # ---- ViT-B/16 ----
    print("\n--- ViT-B/16 ---")
    model = ViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params/1e6:.1f}M")
    print(f"Patch 数量: {model.patch_embed.num_patches} (={224//16}²)")

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        # 查看 patch embedding 后的形状
        patches = model.patch_embed(x)
        print(f"\n输入图像: {x.shape}")
        print(f"Patch Embedding 后: {patches.shape}")
        print(f"  → 196 个 tokens，每个 768 维")
        print(f"  → 每个 16×16 patch ≈ 一个\"视觉单词\"")

        # 完整前向
        logits = model(x)
        print(f"\n分类输出: {logits.shape}")

    # ---- Patch size 的影响 ----
    print("\n--- Patch size 的影响 ---")
    for ps in [8, 14, 16, 32]:
        n_patches = (224 // ps) ** 2
        print(f"  Patch {ps}×{ps}: {n_patches} tokens (短序列 ← → 细粒度)")

    print("\n--- ViT 的核心设计 ---")
    print("  1. Patch Embedding: 图像 → 视觉 token 序列")
    print("  2. [CLS] token: 借鉴 [[../02_BERT/BERT|BERT]]，用于分类")
    print("  3. 1D 可学习位置编码（无需 2D 偏置）")
    print("  4. 纯 Transformer Encoder（双向注意力）")

    print("\n--- ViT 的关键实验发现 ---")
    print("  在 ImageNet-1K 训练: ViT < ResNet（数据不足以学习视觉结构）")
    print("  在 JFT-300M 训练:    ViT > ResNet（数据足够大，消除偏置更好）")
    print("  → 数据量是关键！这也是为什么 [[../14_MAE/MAE|MAE]] 等自监督方法很重要")

    print("\n--- ViT 的局限性 ---")
    print("  1. 全局注意力 O(N²): 高分辨率下计算爆炸")
    print("  2. 缺少多尺度特征（无特征金字塔）")
    print("  3. 位置编码的插值问题")
    print("  → [[../12_Swin_Transformer/Swin Transformer|Swin Transformer]] 用窗口注意力解决")

```
