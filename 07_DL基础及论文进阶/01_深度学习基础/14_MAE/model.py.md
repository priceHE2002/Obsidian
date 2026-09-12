---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# MAE (Masked Autoencoders) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# MAE (Masked Autoencoders) - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
MAE (Masked Autoencoders)
==========================
论文: "Masked Autoencoders Are Scalable Vision Learners"
      (He et al., Meta AI (FAIR), CVPR 2022)
核心贡献: 将 BERT 的 MLM 思想成功移植到视觉领域。关键洞察——图像空间冗余
         远高于语言，因此需要极端 mask ratio (75% vs BERT 的 15%)，
         迫使模型学习全局语义而非局部像素插值。
架构: 非对称 Encoder-Decoder:
      Encoder (重 ViT) 仅处理 25% 的可见 patches → Decoder (轻量 8 层)
      重建全图像素 → MSE Loss (仅 masked patches)

与 [[../11_ViT/ViT.md|ViT]] 的关系: 沿用 ViT 骨干，非对称训练使其高效扩展自监督
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. Patch Embedding —— 将图像分割并线性投影
# ==============================================================================
class PatchEmbed(nn.Module):
    """图像 → patch 序列的线性投影。

    与 ViT 标准实现相同，区别是 MAE 还需提供 patch 尺寸信息
    给后续的 mask/shuffle 操作。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ==============================================================================
# 2. 随机掩码策略 —— MAE 的灵魂
# ==============================================================================
def random_masking(x, mask_ratio=0.75):
    """
    随机采样 mask_ratio 比例的 patches，只保留可见 patches。

    为什么用 uniform random 而非 block-wise/grid-wise？
    - Block-wise: 在 75% mask ratio 下性能严重下降（遮挡区域过大）
    - Grid-wise: 任务过简单（相当于规则下采样），表征质量低
    - Uniform random: 每个 epoch 不同的 mask = 天然的 data augmentation，
      并且防止了中心偏置 (center bias)

    实现原理（极简且高效）:
    1. 对每个样本独立生成随机噪声
    2. 按噪声排序 → 保留前 (1-mask_ratio) 的 patches
    3. 记录保留的 indices → 后续恢复顺序

    Args:
        x: (B, N, D) — 完整的 patch embeddings
        mask_ratio: mask 比例，默认 0.75
    Returns:
        x_masked: (B, N*(1-mask_ratio), D) — 仅保留的可见 patches
        mask: (B, N) — boolean mask (True = 被 mask)
        ids_restore: (B, N) — 恢复原始顺序的索引（shuffle 的逆映射）
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    # 每个样本独立生成随机噪声用于 shuffle
    noise = torch.rand(B, N, device=x.device)

    # 按噪声排序 → shuffle indices
    ids_shuffle = torch.argsort(noise, dim=1)
    # 恢复映射: ids_restore[i] = 原始位置 i 在 shuffle 后的位置
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # 保留前 len_keep 个 patches
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # 生成 mask（用于计算 loss 时只关注 mask 位置）
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


# ==============================================================================
# 3. Transformer Block —— Encoder 和 Decoder 共用
# ==============================================================================
class TransformerBlock(nn.Module):
    """标准 Pre-LN Transformer block。

    MAE 的 Encoder 使用 deep ViT blocks，
    Decoder 使用浅层、窄维度的 blocks（8 层，hidden_dim=512）。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# 4. MAE Encoder —— 仅处理可见 patches
# ==============================================================================
class MAEEncoder(nn.Module):
    """MAE 的 Encoder：标准 ViT，但只处理可见 patches。

    为什么 Encoder 中不包含 mask token？
    - 如果 Encoder 训练时看到 mask token，部署时却看不到，
      造成 train-test mismatch → linear probing 暴跌 (73.5% → 59.6%)
    - 且训练 FLOPs 增加 3.3×（处理全量 196 tokens vs 仅 49 个）

    只处理可见 patches 还意味着:
    - Self-attention 复杂度从 O(196^2) 降到 O(49^2)，约 16× 加速
    - 每 epoch 训练时间约为对比学习方法的 1/3
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 位置编码（与 ViT 相同：可学习 1D positional embedding）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask_ratio=0.75):
        """
        Returns:
            latent: (B, len_keep, embed_dim) — 可见 patches 的编码
            mask: (B, N) — 用于 loss 计算
            ids_restore: (B, N) — 用于恢复顺序
        """
        x = self.patch_embed(x)  # (B, N, D)
        B, N, D = x.shape

        # 添加位置编码 → shuffle/mask → 保留可见 patches
        x = x + self.pos_embed
        x, mask, ids_restore = random_masking(x, mask_ratio)

        # Transformer blocks（仅处理可见 patches）
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x, mask, ids_restore


# ==============================================================================
# 5. MAE Decoder —— 轻量级重建器
# ==============================================================================
class MAEDecoder(nn.Module):
    """MAE 的轻量 Decoder，从 latent 表示重建全图像素。

    Decoder 设计原则:
    - 浅层: 8 层（vs Encoder 12 层）
    - 窄维度: 512（vs Encoder 768）→ 每 token 计算量为 Encoder 的 9%
    - 只有在 Decoder 中才使用 learnable mask token（Enc 中不使用）

    为什么 Decoder 需要一定深度？
    Decoder 需要"吸收"像素重建的 specialization（专用性），
    让 Encoder 的 latent 更抽象、更具语义。
    消融实验: 1-block decoder linear probing 仅 65.5%，
    8-block 达 73.5%，但 12-block 反而下降到 73.3%（过深无益）。
    """

    def __init__(self, num_patches=196, encoder_dim=768, decoder_dim=512,
                 decoder_depth=8, decoder_heads=16, patch_size=16,
                 in_chans=3):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size

        # Encoder latent → Decoder dim 的投影
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

        # 可学习的 mask token（所有被 mask 的 patch 共享同一向量）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Decoder 位置编码
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim)
        )

        # Decoder Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads)
            for _ in range(decoder_depth)
        ])

        self.norm = nn.LayerNorm(decoder_dim)

        # 像素重建头: decoder_dim → patch_size² × 3 (RGB)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)

        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, ids_restore):
        """
        x: (B, len_keep, encoder_dim) — Encoder 输出的可见 patch 特征
        ids_restore: (B, N) — 恢复顺序的索引映射
        Returns: (B, num_patches, patch_size² * 3) — 重建的所有 pixels
        """
        B = x.shape[0]

        # Encoder dim → Decoder dim
        x = self.enc_to_dec(x)

        # 准备完整序列: 可见 patches + mask tokens
        mask_tokens = self.mask_token.repeat(B, self.num_patches - x.shape[1], 1)
        # 拼接 [visible, mask_tokens]
        x_full = torch.cat([x, mask_tokens], dim=1)
        # 恢复到原始顺序（unshuffle）
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2])
        )

        # 添加 Decoder 位置编码 → Transformer blocks → 预测像素
        x_full = x_full + self.decoder_pos_embed
        for block in self.blocks:
            x_full = block(x_full)

        x_full = self.norm(x_full)
        pred = self.pred(x_full)  # (B, N, patch_size² * 3)
        return pred


# ==============================================================================
# 6. MAE 完整模型
# ==============================================================================
class MAE(nn.Module):
    """Masked Autoencoder 完整模型。

    预训练流程:
    1. Encoder 随机 mask 75% patches，仅处理可见 25%
    2. Decoder 用 mask tokens 重建全图像素
    3. MSE Loss 仅在被 mask 的 patches 上计算

    预训练后: 丢弃 Decoder，仅保留 Encoder 做下游任务。

    为什么 Loss 仅在 masked patches 上计算？
    类似于 BERT 的设计——如果全图计算 loss，
    模型会"偷懒"复制可见 patches 而不真正学习重建。
    消融: 全图 loss 精度下降约 0.5%。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 encoder_dim=768, encoder_depth=12, encoder_heads=12,
                 decoder_dim=512, decoder_depth=8, decoder_heads=16,
                 mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        self.encoder = MAEEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=encoder_dim, depth=encoder_depth, num_heads=encoder_heads
        )

        self.decoder = MAEDecoder(
            num_patches=num_patches, encoder_dim=encoder_dim,
            decoder_dim=decoder_dim, decoder_depth=decoder_depth,
            decoder_heads=decoder_heads, patch_size=patch_size,
            in_chans=in_chans
        )

    def forward(self, x):
        """预训练前向传播，计算重建损失。"""
        # Encoder: 只处理可见 patches
        latent, mask, ids_restore = self.encoder(x, self.mask_ratio)

        # Decoder: 重建全图像素
        pred = self.decoder(latent, ids_restore)  # (B, N, patch_size² * 3)

        # 目标: 原始 patch 的像素值
        target = self._patchify(x)  # (B, N, patch_size² * 3)

        # Loss 仅在 masked patches 上计算
        loss = self._compute_loss(pred, target, mask)
        return loss, pred, mask

    def _patchify(self, x):
        """将图像展平为 patch 序列（像素空间）。

        x: (B, 3, H, W) → (B, N, patch_size² * 3)
        这是后续 loss 计算的 ground truth。
        """
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H/p, W/p, p, p, C)
        x = x.reshape(B, (H // p) * (W // p), p * p * C)
        return x

    def _compute_loss(self, pred, target, mask):
        """MSE Loss（仅在 masked patches 上）。

        MAE 对每个被 mask 的 patch 做 per-patch 归一化
        (计算 patch 内像素的 mean 和 std 进行归一化)，
        增强局部对比度，比未归一化像素好 0.5%。
        这里简化为原始 MSE。
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # 每个 patch 的 mean MSE
        loss = (loss * mask).sum() / mask.sum()  # 仅 mask 区域
        return loss


# ==============================================================================
# 7. MAE 分类器（预训练后使用，丢弃 Decoder）
# ==============================================================================
class MAEForClassification(nn.Module):
    """MAE 用于图像分类的封装——仅保留 Encoder + 分类头。

    预训练后丢弃 Decoder，在 Encoder 输出的全局表示上训练分类器。
    这展示了 MAE 的核心使用模式: 先预训练，再 micro-tuning。
    """

    def __init__(self, encoder: MAEEncoder, num_classes=1000,
                 global_pool=True):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.patch_embed.proj.out_channels
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.global_pool = global_pool

    def forward(self, x):
        # 分类时不用 mask —— 所有 patches 都输入
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed
        for block in self.encoder.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.head(x)


# ==============================================================================
# 演示
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MAE — Masked Autoencoder 演示")
    print("=" * 60)

    # 小型 MAE（演示用）
    model = MAE(
        img_size=224, patch_size=16, in_chans=3,
        encoder_dim=384, encoder_depth=6, encoder_heads=6,
        decoder_dim=256, decoder_depth=4, decoder_heads=8,
        mask_ratio=0.75
    )

    x = torch.randn(2, 3, 224, 224)

    loss, pred, mask = model(x)

    print(f"输入形状: {x.shape}")
    print(f"预测形状: {pred.shape}  (N=196, patch_size²×3=768)")
    print(f"Mask 比例: {mask.sum(dim=-1).float().mean().item() / 196:.2%}")
    print(f"MSE Loss (仅 masked patches): {loss.item():.4f}")

    # Encoder 效率分析
    total = 196  # 14×14
    visible = int(total * 0.25)  # 49
    masked = total - visible     # 147
    print(f"\n效率分析 (14×14=196 patches):")
    print(f"  Encoder 处理: {visible} patches (25%)")
    print(f"  Decoder 处理: {total} patches (全量，含 mask tokens)")
    print(f"  Self-attention 加速: {(total*total)/(visible*visible):.1f}x")
    print(f"  (= {total**2}/{visible**2})")

    # 分类器测试
    cls_model = MAEForClassification(model.encoder, num_classes=10)
    logits = cls_model(x)
    print(f"\n分类输出形状: {logits.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"\n总参数量: {total_params / 1e6:.1f}M")
    print(f"Encoder 参数量: {encoder_params / 1e6:.1f}M")
    print(f"真实 MAE ViT-L: Encoder ~307M, Decoder ~30M")
    print(f"\n关键: 75% mask ratio 迫使模型超越局部插值，学习全局语义理解")

```

```
