---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# CLIP (Contrastive Language-Image Pre-training) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# CLIP (Contrastive Language-Image Pre-training) - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
CLIP (Contrastive Language-Image Pre-training)
===============================================
论文: "Learning Transferable Visual Models From Natural Language Supervision"
      (Radford et al., OpenAI, ICML 2021)
核心贡献: 用对比学习在 400M 图文对上训练双塔架构，将图像和文本映射到
         同一语义嵌入空间，实现零样本图像分类和跨模态检索。
架构: 图像编码器(ViT) + 文本编码器(Transformer) → 共享嵌入空间
      对称 InfoNCE 损失 + 可学习温度参数 τ

与 [[../15_DINOv2/DINOv2.md|DINOv2]] 的关系: CLIP 是语义编码器，DINOv2 是空间编码器，
  二者在 OpenVLA 等 VLA 中互补使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. ViT 图像编码器 —— 将图像编码为嵌入向量
# ==============================================================================
class PatchEmbed(nn.Module):
    """将图像分割为 patches 并做线性投影。

    输入: (B, 3, H, W)
    输出: (B, num_patches, embed_dim)

    为什么用 patch embedding 而非像素级 CNN？
    ViT 的设计原则是将图像视为"视觉词序列"，
    patch embedding = word embedding 在视觉中的对应。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 卷积实现 patch embedding（比 reshape+linear 更高效）
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """CLIP 的 ViT 图像编码器。

    CLIP 使用的 ViT 与原始 ViT 有两个关键区别:
    1. 在 patch+position embedding 后添加额外的 LayerNorm（CLIP 特有）
    2. 最终用 LayerNorm + 线性投影映射到共享嵌入空间（而非分类头）

    CLIP 探索了 ViT-B/32、ViT-B/16、ViT-L/14 三种规模，
    其中 ViT-L/14 是最强版本（零样本 ImageNet 76.2%）。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 output_dim=512):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token（全局图像表示）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # CLIP 特有的额外 LayerNorm（patch+position embedding 后）
        self.pre_ln = nn.LayerNorm(embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 最终投影到共享嵌入空间
        self.post_ln = nn.LayerNorm(embed_dim)
        self.visual_proj = nn.Linear(embed_dim, output_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_transformer_weights)

    def _init_transformer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pre_ln(x)  # CLIP 特有的额外 LayerNorm

        for block in self.blocks:
            x = block(x)

        # 取 CLS token 的输出作为图像表示
        x = x[:, 0]  # (B, embed_dim)
        x = self.post_ln(x)
        x = self.visual_proj(x)  # 投影到共享嵌入空间 (B, output_dim)
        return x


# ==============================================================================
# 2. 文本编码器 —— GPT-2 风格 Transformer
# ==============================================================================
class TransformerBlock(nn.Module):
    """标准的 Transformer block（Pre-LN 设计，更稳定）。

    CLIP 文本编码器使用 GPT-2 风格架构:
    - 12 层, 512 宽, 8 头, 63M 参数
    - Masked Self-Attention（保留语言建模潜力，但 CLIP 训练时不用 causal mask）
    - 这里实现为双向 attention（更适合对比学习）
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        # Self-Attention + 残差
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # MLP + 残差
        x = x + self.mlp(self.ln2(x))
        return x


class TextTransformer(nn.Module):
    """CLIP 的文本编码器。

    CLIP 使用 [SOS] 和 [EOS] token 包裹文本序列，
    [EOS] token 的最后一层激活经 LayerNorm 和线性投影后作为文本特征。

    为什么用 [EOS] 而不是 CLS token？
    GPT-2 风格架构没有 CLS token 的概念，
    使用 [EOS] 作为序列结束符天然适合语言模型设计。
    """

    def __init__(self, vocab_size=49408, max_seq_len=77, embed_dim=512,
                 depth=12, num_heads=8, output_dim=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_proj = nn.Linear(embed_dim, output_dim, bias=False)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, text_tokens):
        """
        text_tokens: (B, L) — tokenized 文本，通常 L=77
        返回: (B, output_dim) — [EOS] 位置的投影特征
        """
        x = self.token_embedding(text_tokens)  # (B, L, D)
        x = x + self.pos_embedding[:x.shape[1]]

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        # 取 [EOS] token（最后一个有效 token）的特征
        # 简化处理: 取最后一个 token（假设 padding 在后面）
        x = x[:, -1, :]  # (B, D)
        x = self.text_proj(x)  # 投影到共享嵌入空间 (B, output_dim)
        return x


# ==============================================================================
# 3. CLIP 完整模型 —— 双塔架构 + 对称 InfoNCE 损失
# ==============================================================================
class CLIP(nn.Module):
    """CLIP 对比学习双塔模型。

    核心训练流程:
    1. 图像编码器 → I_e ∈ ℝ^{B×d}
    2. 文本编码器 → T_e ∈ ℝ^{B×d}
    3. 计算 B×B 余弦相似度矩阵（缩放: logits = I_e @ T_e^T * exp(t)）
    4. 对称 InfoNCE Loss:
       L = 0.5 * (CE(I→T) + CE(T→I))

    为什么用对称损失？
    单向损失只保证图像→文本对齐，对称损失确保了
    嵌入空间中双向的对齐，提升表示质量。

    可学习温度 τ 的作用:
    τ 控制 softmax 分布的"锐度"。
    τ 越小 → 分布越尖锐 → 模型更关注最难区分的负样本。
    可学习 τ 让模型自动在困难的辨别和稳定的梯度之间权衡。
    """

    def __init__(self, image_encoder: VisionTransformer,
                 text_encoder: TextTransformer, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 可学习温度参数 τ
        # 实际优化 log(1/τ) = t，确保 τ > 0
        # CLIP 初始化相当于 τ ≈ 1/exp(0) = 1，即初始温度为 ~14.3
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, image):
        """编码图像为归一化嵌入向量。"""
        features = self.image_encoder(image)
        return F.normalize(features, dim=-1)

    def encode_text(self, text_tokens):
        """编码文本为归一化嵌入向量。"""
        features = self.text_encoder(text_tokens)
        return F.normalize(features, dim=-1)

    def forward(self, image, text_tokens):
        """
        image: (B, 3, H, W)
        text_tokens: (B, L)
        返回: logits_per_image (B, B), logits_per_text (B, B)
              以及对称 InfoNCE 损失
        """
        # 提取归一化特征
        I_e = self.encode_image(image)   # (B, d)
        T_e = self.encode_text(text_tokens)  # (B, d)

        # 温度缩放的余弦相似度
        # exp(t) 确保 scale > 0
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * I_e @ T_e.t()  # (B, B)
        logits_per_text = logits_per_image.t()           # (B, B)

        # 对称 InfoNCE Loss
        batch_size = image.shape[0]
        labels = torch.arange(batch_size, device=image.device)

        loss_i = F.cross_entropy(logits_per_image, labels)  # 图像→文本
        loss_t = F.cross_entropy(logits_per_text, labels)   # 文本→图像
        loss = (loss_i + loss_t) / 2

        return logits_per_image, logits_per_text, loss


# ==============================================================================
# 工具函数
# ==============================================================================
def clip_loss_standalone(image_features, text_features, logit_scale):
    """
    独立的 CLIP Loss 计算（用于微调或自定义训练循环）。

    这里直接展示了对称 InfoNCE 的完整数学过程，
    与 CLIP.forward 中的实现等价。

    为什么显式写出 softmax 和 log？
    为了更好地理解温度参数的作用 —
    logit_scale = exp(t) 在 softmax 之前缩放相似度分。
    """
    # 余弦相似度（未归一化时需要 / (|I|·|T|)）
    logits = logit_scale * image_features @ text_features.t()

    # 对称 softmax CE
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


# ==============================================================================
# 演示
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CLIP 双塔对比学习演示")
    print("=" * 60)

    # 小型 CLIP（演示用，非真实规模）
    embed_dim = 256
    output_dim = 128

    image_encoder = VisionTransformer(
        img_size=224, patch_size=32, embed_dim=embed_dim,
        depth=4, num_heads=4, output_dim=output_dim
    )
    text_encoder = TextTransformer(
        vocab_size=49408, max_seq_len=77, embed_dim=embed_dim,
        depth=4, num_heads=4, output_dim=output_dim
    )

    model = CLIP(image_encoder, text_encoder, embed_dim=output_dim)

    # 模拟输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text_tokens = torch.randint(0, 49408, (batch_size, 77))

    logits_i, logits_t, loss = model(images, text_tokens)

    print(f"图像输入形状: {images.shape}")
    print(f"文本 tokens 形状: {text_tokens.shape}")
    print(f"相似度矩阵 (I×T): shape={logits_i.shape}")
    print(f"温度参数 τ = 1/exp(t) = {1 / model.logit_scale.exp().item():.2f}")
    print(f"对称 InfoNCE Loss: {loss.item():.4f}")

    # 验证特征归一化
    with torch.no_grad():
        img_feat = model.encode_image(images)
        text_feat = model.encode_text(text_tokens)
        print(f"\n图像特征 L2 范数: {img_feat.norm(dim=-1).mean().item():.3f}")
        print(f"文本特征 L2 范数: {text_feat.norm(dim=-1).mean().item():.3f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.1f}M")
    print(f"完整 CLIP ViT-L/14 参数量: ~428M (图像 ~300M + 文本 ~63M + 投影)")
    print(f"\n关键: 对比学习将不同模态映射到同一空间，无需类别标签即可零样本分类")

```

```
