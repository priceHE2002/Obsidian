---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# PaLM-E - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 1. ViT 视觉编码器（Patch Embedding + Transformer）
# ============================================================
# PaLM-E 用 ViT 把图像切成 patch，每个 patch 变成一个 "视觉 token"，
# 然后插入到 LLM 的输入序列中。这比直接用 CNN feature map 更自然——
# 因为 LLM 天然接受 token 序列，patch tokens 正好适配这一范式。

class PatchEmbedding(nn.Module):
    """将图像切分为不重叠的 patch，并通过卷积投影为 embedding。
    WHY 卷积而非全连接：卷积天然处理 2D 结构，一个 kernel_size=stride=patch_size
    的 Conv2d 等价于"滑动窗口切 patch + 线性投影"。"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=1024):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, H, W] → [B, embed_dim, H/P, W/P] → [B, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """标准多头自注意力"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_head, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class ViTEncoderBlock(nn.Module):
    """ViT Transformer block（Pre-LN）"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT 编码器 —— 输出 patch-wise 特征序列。
    WHY 输出全部 patch 而非 CLS token：
    PaLM-E 需要每个 patch 作为独立的视觉 token 插入 LLM，
    这样 LLM 的注意力可以定位到图像的任意局部区域。"""

    def __init__(self, image_size=224, patch_size=16, embed_dim=1024,
                 depth=12, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            ViTEncoderBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 3, H, W] → [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ============================================================
# 2. 多模态 Embedding 投影器
# ============================================================
# ViT 输出的维度（如 1024）和 PaLM 的 embedding 维度（如 4096）
# 不一致。需要一个可学习的投影层来对齐。
# PaLM-E 实验表明简单线性投影已足够——不需要花哨的 MLP。

class MultiModalProjector(nn.Module):
    """将各模态的嵌入投影到 LLM 的统一 embedding 空间"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


# ============================================================
# 3. RoPE 旋转位置编码
# ============================================================
# PaLM 使用 RoPE 而非可学习位置编码。RoPE 通过旋转矩阵编码
# 相对位置信息，让模型天然感知 token 之间的距离。
# 对于多模态序列尤为重要——图像 token 和文本 token
# 可能间距很远，RoPE 能自然表达这种距离。

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def _get_rope_cache(self, seq_len, device):
        """预计算 cos 和 sin 表"""
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        return emb.cos(), emb.sin()

    def apply_rope(self, x, cos, sin):
        """对 query 和 key 应用 RoPE。
        将向量前半和后半交换并取反作为旋转分量。"""
        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        return x * cos + x_rot * sin


# ============================================================
# 4. PaLM Decoder-only LLM（简化版）
# ============================================================
# PaLM 使用 "并行" Transformer block：attention 和 MLP 并行计算
# 而非串行——这在 540B 规模下训练更快且质量不降。

class PaLMDecoderBlock(nn.Module):
    """PaLM 风格 decoder block（并行公式 + RoPE + SwiGLU）"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ln = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # SwiGLU MLP（PaLM 的关键组件）
        hidden_dim = int(dim * mlp_ratio * 2 / 3)  # SwiGLU 有 3 个矩阵，等效容量调整
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, causal_mask=None):
        B, T, D = x.shape
        x_norm = self.ln(x)

        # —— 自注意力（带 RoPE + 因果掩码）——
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim)

        # 应用 RoPE 到 q 和 k
        cos, sin = self.rope._get_rope_cache(T, x.device)
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)
        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # [B, n_head, T, head_dim]

        attn_weights = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if causal_mask is not None:
            attn_weights = attn_weights.masked_fill(causal_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)
        attn_out = self.out_proj(attn_out)

        # —— SwiGLU MLP（并行公式：与 attention 并行计算）——
        gate = self.gate_proj(x_norm)
        up = self.up_proj(x_norm)
        mlp_out = self.down_proj(F.silu(gate) * up)

        # 并行公式：x = x + attn(x_norm) + mlp(x_norm)
        x = x + attn_out + mlp_out
        return x


class PaLMLanguageModel(nn.Module):
    """PaLM Decoder-only LLM。
    WHY tied embedding：输入和输出共享词嵌入权重，
    减少参数量，对大模型尤其重要。"""

    def __init__(self, vocab_size, dim, depth, num_heads,
                 mlp_ratio=4.0, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            PaLMDecoderBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.ln_final = nn.LayerNorm(dim)

        # 输出头与输入嵌入共享权重（tied embedding）
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def _get_causal_mask(self, T, device):
        return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))

    def forward(self, input_embeddings):
        """input_embeddings: [B, T, dim] — 已拼好的多模态序列"""
        B, T, D = input_embeddings.shape
        causal_mask = self._get_causal_mask(T, input_embeddings.device)

        x = input_embeddings
        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits


# ============================================================
# 5. PaLM-E 完整模型 —— "多模态句子"组装器
# ============================================================
# 核心思想：不修改 PaLM 架构，只在输入侧拼接多模态 token。
# 序列结构：[状态] [文本] [图像patches]
# - 状态（3D 物体位置）先放——"世界是什么样"
# - 图像放文本附近——"我现在看到了什么"
# 所有模态的 token 在 causal self-attention 中自然融合。

@dataclass
class PaLMEOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class PaLME(nn.Module):
    """PaLM-E: Embodied Multimodal Language Model"""

    def __init__(self, vocab_size=256000, llm_dim=4096, llm_depth=32,
                 llm_heads=32, vit_embed_dim=1024, vit_depth=12, vit_heads=16,
                 state_dim=128, image_size=224, patch_size=14):
        super().__init__()
        self.llm_dim = llm_dim

        # ViT 视觉编码器
        self.vit = ViTEncoder(
            image_size=image_size, patch_size=patch_size,
            embed_dim=vit_embed_dim, depth=vit_depth, num_heads=vit_heads
        )
        # 投影：ViT dim → LLM dim
        self.image_projector = MultiModalProjector(vit_embed_dim, llm_dim)

        # 状态投影：state_dim → LLM dim（单独 MLP，因为状态是密集向量而非空间序列）
        self.state_projector = nn.Sequential(
            nn.Linear(state_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # PaLM LLM
        self.llm = PaLMLanguageModel(
            vocab_size=vocab_size, dim=llm_dim, depth=llm_depth, num_heads=llm_heads
        )

        # 可学习的特殊标记 embedding（标记多模态 token 的边界）
        # WHY 需要边界标记：帮助 LLM 区分"这是图像 patch"还是"这是文字"，
        # 类似于 Flamingo 的 <image></image> 标记
        self.soi_embed = nn.Parameter(torch.randn(llm_dim))  # start of image
        self.eoi_embed = nn.Parameter(torch.randn(llm_dim))  # end of image
        self.sos_embed = nn.Parameter(torch.randn(llm_dim))  # start of state

    def build_multimodal_sequence(self, text_ids, images=None, states=None):
        """组装"多模态句子"。序列结构（Fig.2）：
        [场景状态] [文本：任务描述] [图像tokens] [文本：输出前缀] [生成...]
        """
        B = text_ids.shape[0]
        device = text_ids.device
        parts = []

        # (1) 场景状态 token
        if states is not None:
            state_emb = self.state_projector(states)  # [B, llm_dim]
            parts.append(self.sos_embed.view(1, 1, -1).expand(B, 1, -1))
            parts.append(state_emb.unsqueeze(1))       # [B, 1, llm_dim]

        # (2) 文本 token embedding
        text_emb = self.llm.token_embed(text_ids)     # [B, T_text, llm_dim]
        parts.append(text_emb)

        # (3) 图像 patch token（可能多张图）
        if images is not None:
            for i in range(images.shape[1]):
                img = images[:, i]  # [B, 3, H, W]
                vit_out = self.vit(img)                    # [B, N_patches, vit_dim]
                img_emb = self.image_projector(vit_out)    # [B, N_patches, llm_dim]
                parts.append(self.soi_embed.view(1, 1, -1).expand(B, 1, -1))
                parts.append(img_emb)
                parts.append(self.eoi_embed.view(1, 1, -1).expand(B, 1, -1))

        return torch.cat(parts, dim=1)  # [B, total_tokens, llm_dim]

    def forward(self, text_ids, images=None, states=None, target_ids=None):
        """
        text_ids: [B, T_text] — 文本 token
        images: [B, num_imgs, 3, H, W] or None
        states: [B, state_dim] or None
        target_ids: [B, T_total] — 完整序列的 target，用于计算 next-token loss
        """
        input_emb = self.build_multimodal_sequence(text_ids, images, states)
        logits = self.llm(input_emb)  # [B, T_total, vocab_size]

        loss = None
        if target_ids is not None:
            # Shift: 位置 t 预测位置 t+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )
        return PaLMEOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(self, text_ids, images=None, states=None,
                 max_new_tokens=128, temperature=1.0):
        """自回归生成（推理时产出规划步骤或动作序列）。
        WHY 不回传图像：图像是静态输入，只在序列中首尾各出现一次。"""
        self.eval()
        input_emb = self.build_multimodal_sequence(text_ids, images, states)

        for _ in range(max_new_tokens):
            logits = self.llm(input_emb)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            next_emb = self.llm.token_embed(next_token)
            input_emb = torch.cat([input_emb, next_emb], dim=1)

        # 返回完整序列（包含 prompt + 生成）
        return input_emb


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PaLM-E 模型演示")
    print("=" * 60)

    # 演示用小模型配置（原版 PaLM-E 是 562B）
    demo_config = dict(
        vocab_size=5000, llm_dim=512, llm_depth=6, llm_heads=8,
        vit_embed_dim=384, vit_depth=6, vit_heads=8, state_dim=128,
    )

    print(f"\n[1] 初始化 PaLM-E（演示配置: llm_dim={demo_config['llm_dim']}, "
          f"depth={demo_config['llm_depth']}）")
    model = PaLME(**demo_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params / 1e6:.2f}M "
          f"（原版 PaLM-E 为 562B，此处仅为教学演示）")

    # 训练前向
    print("\n[2] 训练前向演示")
    B = 2
    text_ids = torch.randint(0, 5000, (B, 40))
    images = torch.randn(B, 2, 3, 224, 224)  # 前后两台相机
    states = torch.randn(B, 128)
    target_ids = torch.randint(0, 5000, (B, 100))

    output = model(text_ids, images=images, states=states, target_ids=target_ids)
    print(f"  Logits: {output.logits.shape}")
    print(f"  Loss: {output.loss.item():.4f}")

    # 生成演示
    print("\n[3] 自回归生成演示")
    prompt_ids = torch.randint(0, 5000, (1, 10))
    result_emb = model.generate(prompt_ids, images=images[:1], states=states[:1],
                                max_new_tokens=30)
    # result_emb 的最后一个维度是 llm_dim（embedding），
    # token 数量 = prompt_len + generated_len
    print(f"  Prompt 长度: {prompt_ids.shape[1]}")
    print(f"  生成后序列长度: {result_emb.shape[1]}")

    # 多模态序列长度
    print("\n[4] 多模态序列结构分析")
    emb_no_img = model.build_multimodal_sequence(
        text_ids[:1, :5], images=None, states=states[:1])
    emb_1img = model.build_multimodal_sequence(
        text_ids[:1, :5], images=images[:1, :1], states=states[:1])
    emb_2img = model.build_multimodal_sequence(
        text_ids[:1, :5], images=images[:1], states=states[:1])
    print(f"  无图像: {emb_no_img.shape[1]} tokens")
    print(f"  1 张图: {emb_1img.shape[1]} tokens "
          f"(+{emb_1img.shape[1] - emb_no_img.shape[1]})")
    print(f"  2 张图: {emb_2img.shape[1]} tokens "
          f"(+{emb_2img.shape[1] - emb_no_img.shape[1]})")
    print(f"  每张图额外开销: {(emb_1img.shape[1] - emb_no_img.shape[1])} tokens "
          f"(= 2 marker + {model.vit.num_patches} patches)")

    print("\nPaLM-E 模型演示完成")
```
