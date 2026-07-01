---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# OpenVLA 模型实现 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
OpenVLA 模型实现
================
双视觉编码器(DINOv2+SigLIP) + MLP 投影器 + Llama2 骨干 + 动作 token 覆写机制

OpenVLA 是第一个真正开源的大规模 VLA 模型（7B），提出了"空间+语义"双视觉编码器
的设计范式，使用 Llama 2 作为语言骨干，通过覆写词汇表末尾 token 实现动作离散化。

参考: [[OpenVLA]] | [[OpenVLA 原文.pdf]]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ═══════════════════════════════════════════════════════════════════
# 动作离散化工具
# ═══════════════════════════════════════════════════════════════════

class ActionTokenizer:
    """
    OpenVLA 的动作离散化模块。

    WHY: 将连续动作（7 维末端位移+夹爪）映射到 256 个离散 bin，
    然后覆写 Llama tokenizer 词汇表末尾 256 个最不常用的 token。
    关键改进：使用 1st-99th 百分位而非 min-max 分 bin，
    这样可以忽略极端异常值，让正常动作的 bin 粒度更均匀。
    """

    def __init__(
        self,
        action_dim: int = 7,
        num_bins: int = 256,
        percentile_low: float = 0.01,
        percentile_high: float = 0.99,
        vocab_size: int = 32000,  # Llama 2 的 vocab size
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins

        # WHY: 覆写 Llama tokenizer 词汇表末尾 256 个 token 作为动作 token
        # Llama 只有 ~100 个保留 token，不够 256 个 → 直接覆写末尾最不常用的
        self.action_token_start = vocab_size - num_bins  # 32000 - 256 = 31744
        self.action_token_end = vocab_size

    def discretize(
        self, actions: torch.Tensor, action_min: torch.Tensor, action_max: torch.Tensor
    ) -> torch.Tensor:
        """
        将连续动作映射到离散 bin 索引。

        WHY: 使用百分位范围而非 min-max 范围。
        - min-max 会被极端异常值拉大 → 正常动作只占少数 bin → 精细度低
        - 百分位忽略异常值 → bin 粒度均匀 → 精细度更高
        """
        # 归一化到 [0, 1]
        normalized = (actions - action_min) / (action_max - action_min + 1e-8)
        normalized = normalized.clamp(0.0, 1.0)
        # 映射到 [0, num_bins-1]
        bin_indices = (normalized * (self.num_bins - 1)).long()
        return bin_indices  # [..., action_dim]

    def bin_to_token_ids(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """将 bin 索引转换为覆写的 token id。"""
        return bin_indices + self.action_token_start

    def token_ids_to_bin(self, token_ids: torch.Tensor) -> torch.Tensor:
        """反向：token id → bin 索引。"""
        return token_ids - self.action_token_start

    def de_discretize(
        self, bin_indices: torch.Tensor, action_min: torch.Tensor, action_max: torch.Tensor
    ) -> torch.Tensor:
        """将 bin 索引还原为连续动作。"""
        normalized = bin_indices.float() / (self.num_bins - 1)
        actions = normalized * (action_max - action_min) + action_min
        return actions


# ═══════════════════════════════════════════════════════════════════
# 双视觉编码器
# ═══════════════════════════════════════════════════════════════════

class DualVisionEncoder(nn.Module):
    """
    DINOv2 + SigLIP 双编码器，按通道拼接后通过 MLP 投影。

    WHY 用双编码器？
    - DINOv2（自监督 ViT）：擅长细粒度空间特征——物体精确位置、形状、姿态
    - SigLIP（语言监督 ViT）：擅长语义理解——"红杯子"中的"红"和"杯子"
    两者互补：空间定位 + 语义理解 = 机器人控制所需的全量视觉信息。

    WHY 224×224 而非 384×384？
    - 384 分辨率在 VLM 基准上更好，但在 VLA 上无差异
    - 384 训练时间是 224 的 3 倍 → 性价比低
    - 机器人控制任务不需要那么细的视觉粒度
    """

    def __init__(
        self,
        dino_embed_dim: int = 768,      # DINOv2 ViT-B 的输出维度
        siglip_embed_dim: int = 768,    # SigLIP ViT-B 的输出维度
        num_dino_tokens: int = 257,     # 1 CLS + 256 patch tokens (224/14=16, 16x16=256)
        num_siglip_tokens: int = 257,
        projector_hidden: int = 2048,   # MLP 隐藏层维度
        output_dim: int = 4096,         # Llama 2 7B 的 hidden_size
    ):
        super().__init__()
        self.dino_embed_dim = dino_embed_dim
        self.siglip_embed_dim = siglip_embed_dim
        self.num_dino_tokens = num_dino_tokens
        self.num_siglip_tokens = num_siglip_tokens

        # WHY: 2 层 MLP 投影器（来自 Prismatic VLM 的设计）
        # 比单层更能建模 cross-encoder 交互，但不过度增加参数
        combined_dim = dino_embed_dim + siglip_embed_dim  # 1536
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, output_dim),
        )

        # WHY: 学习型的 token 位置编码
        # 视觉 token 来自两个编码器的 patch 空间 → 需要位置信息
        total_tokens = num_dino_tokens + num_siglip_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, output_dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        dino_features: torch.Tensor,    # [B, 257, 768]
        siglip_features: torch.Tensor,  # [B, 257, 768]
    ) -> torch.Tensor:
        """
        WHY: VLA 训练时必须微调视觉编码器（反直觉发现）。
        - 冻结视觉编码器在 VLM 训练中更好，但在 VLA 中是反的
        - 预训练视觉特征在全局语义上很强，但缺少精确空间信息
        - 机器人控制需要后者 → 必须在 VLA 训练中继续调整
        """
        B = dino_features.shape[0]

        # 通道拼接：沿特征维度拼接两个编码器的输出
        combined_features = torch.cat([dino_features, siglip_features], dim=-1)  # [B, 514, 1536]

        # MLP 投影到 Llama 2 的 hidden_size
        projected = self.projector(combined_features)  # [B, 514, 4096]

        # 加位置编码
        projected = projected + self.pos_embed

        return projected


# ═══════════════════════════════════════════════════════════════════
# OpenVLA 主模型
# ═══════════════════════════════════════════════════════════════════

class OpenVLA(nn.Module):
    """
    OpenVLA: 开源 VLA 模型。

    完整前向流程：
    1. 图片 (224x224) -> DINOv2 + SigLIP -> 通道拼接 -> MLP 投影
    2. 语言指令 tokenize -> 与视觉 token 拼接
    3. 送入 Llama 2 7B -> 自回归生成
    4. 词汇表末尾 256 个 token 被覆写为动作 bin -> 输出动作 token
    5. 动作 token 解码为连续动作值

    WHY Llama 2 7B 而非更大的模型？
    - 7B 在性能和可用性间有最佳平衡
    - bf16 推理仅需 ~15GB -> 可在单张消费级 GPU 上运行
    - 支持 4-bit 量化 (NF4) 降至 ~7GB 显存，性能损失近乎零

    WHY 需要 27 个 epochs 训练（远超 VLM 的 1-2 epochs）？
    - VLA 需要比 VLM 更多的"过遍历"来充分吸收机器人控制信号
    - 论文发现 action token 准确率在 27 epochs 才超过 95%
    """

    def __init__(
        self,
        dino_embed_dim: int = 768,
        siglip_embed_dim: int = 768,
        llama_hidden_size: int = 4096,
        llama_num_layers: int = 32,
        llama_num_heads: int = 32,
        llama_vocab_size: int = 32000,
        max_seq_len: int = 2048,
        action_dim: int = 7,
        num_action_bins: int = 256,
    ):
        super().__init__()
        self.llama_hidden_size = llama_hidden_size
        self.llama_vocab_size = llama_vocab_size
        self.max_seq_len = max_seq_len
        self.action_dim = action_dim
        self.num_action_bins = num_action_bins

        # 双视觉编码器
        self.vision_encoder = DualVisionEncoder(
            dino_embed_dim=dino_embed_dim,
            siglip_embed_dim=siglip_embed_dim,
            projector_hidden=2048,
            output_dim=llama_hidden_size,
        )

        # 动作分词器
        self.action_tokenizer = ActionTokenizer(
            action_dim=action_dim,
            num_bins=num_action_bins,
            vocab_size=llama_vocab_size,
        )

        # 简化的 Llama 2 Decoder（实际实现用 HuggingFace transformers）
        self.token_embedding = nn.Embedding(llama_vocab_size, llama_hidden_size)
        self.decoder_layers = nn.ModuleList([
            LlamaDecoderLayer(llama_hidden_size, llama_num_heads)
            for _ in range(llama_num_layers)
        ])
        self.final_norm = nn.RMSNorm(llama_hidden_size)
        self.lm_head = nn.Linear(llama_hidden_size, llama_vocab_size)

        # 动作统计（在训练数据上计算 1st-99th 百分位）
        self.register_buffer("action_min", torch.zeros(action_dim))
        self.register_buffer("action_max", torch.ones(action_dim))

    def forward(
        self,
        dino_features: torch.Tensor,     # [B, num_dino_tokens, dino_dim]
        siglip_features: torch.Tensor,   # [B, num_siglip_tokens, siglip_dim]
        text_token_ids: torch.Tensor,    # [B, text_len]
        action_token_ids: Optional[torch.Tensor] = None,  # [B, action_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播。
        训练目标：对每个动作 bin 做交叉熵损失（和 RT-2 一样）。
        """
        B = dino_features.shape[0]

        # Step 1: 视觉编码
        visual_embeds = self.vision_encoder(dino_features, siglip_features)  # [B, V, H]

        # Step 2: 文本 token -> embedding
        text_embeds = self.token_embedding(text_token_ids)  # [B, T, H]

        # Step 3: 拼接视觉 + 文本 token
        input_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # [B, V+T, H]

        # Step 4: 如果训练，在目标位置放置动作 token embedding
        # WHY: 动作 token 的位置是固定的（在序列末尾），和 RT-2 一样
        if action_token_ids is not None:
            # 将动作 token 的 embedding 追加到序列末尾
            action_embeds = self.token_embedding(action_token_ids)  # [B, A, H]
            input_embeds = torch.cat([input_embeds, action_embeds], dim=1)

        # Step 5: Llama 2 Decoder
        hidden_states = input_embeds
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.final_norm(hidden_states)

        # Step 6: LM head 预测下一个 token
        logits = self.lm_head(hidden_states)  # [B, seq_len, vocab_size]

        # Step 7: 提取动作预测
        # WHY: 只需要词汇表末尾 256 个 token 的 logits（被覆写的动作 token）
        action_logits = logits[:, -self.action_dim:, self.action_tokenizer.action_token_start:]  # [B, 7, 256]

        return {"logits": logits, "action_logits": action_logits, "hidden_states": hidden_states}

    def generate_action(
        self,
        dino_features: torch.Tensor,
        siglip_features: torch.Tensor,
        text_token_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        推理：自回归生成连续动作。

        WHY: 和 RT-2 一样逐 token 自回归生成，
        但 256 个 bin x 7 维 = 7 个 token，生成速度可接受。
        """
        B = dino_features.shape[0]
        visual_embeds = self.vision_encoder(dino_features, siglip_features)
        text_embeds = self.token_embedding(text_token_ids)
        input_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        generated_bin_indices = []
        current_embeds = input_embeds

        for _ in range(self.action_dim):
            hidden_states = current_embeds
            for layer in self.decoder_layers:
                hidden_states = layer(hidden_states)
            hidden_states = self.final_norm(hidden_states)

            # 只看最后一个位置的 logits（自回归）
            logits = self.lm_head(hidden_states[:, -1:, :])  # [B, 1, vocab]

            # 限制到动作 token 范围
            action_logits = logits[:, :, self.action_tokenizer.action_token_start:]  # [B, 1, 256]

            # 采样（带温度）
            probs = F.softmax(action_logits / temperature, dim=-1)  # [B, 1, 256]
            bin_idx = torch.multinomial(probs.squeeze(1), 1)  # [B, 1]
            generated_bin_indices.append(bin_idx)

            # 将生成的 token embedding 拼到序列后面（teacher-forcing 式自回归）
            token_id = self.action_tokenizer.bin_to_token_ids(bin_idx.tolist())
            next_embed = self.token_embedding(torch.tensor(token_id, device=current_embeds.device)).unsqueeze(1)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

        bin_indices = torch.stack(generated_bin_indices, dim=-1)  # [B, 7]
        actions = self.action_tokenizer.de_discretize(bin_indices, self.action_min, self.action_max)
        return actions


# ═══════════════════════════════════════════════════════════════════
# Llama 2 Decoder Layer (简化实现)
# ═══════════════════════════════════════════════════════════════════

class LlamaDecoderLayer(nn.Module):
    """Llama 2 风格的 Decoder Layer（RMSNorm + Grouped-Query Attention + SwiGLU FFN）。"""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # RMSNorm（比 LayerNorm 更快）
        self.input_norm = nn.RMSNorm(hidden_size)
        self.post_attn_norm = nn.RMSNorm(hidden_size)

        # Grouped-Query Attention (GQA)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size // 4)  # num_kv_heads = num_heads/4
        self.v_proj = nn.Linear(hidden_size, hidden_size // 4)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # SwiGLU FFN
        self.gate_proj = nn.Linear(hidden_size, int(hidden_size * 8 / 3))
        self.up_proj = nn.Linear(hidden_size, int(hidden_size * 8 / 3))
        self.down_proj = nn.Linear(int(hidden_size * 8 / 3), hidden_size)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-Attention with GQA
        residual = x
        x = self.input_norm(x)
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads // 4, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads // 4, self.head_dim).transpose(1, 2)
        # 简化的 GQA: repeat KV heads
        k = k.repeat_interleave(4, dim=1)
        v = v.repeat_interleave(4, dim=1)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = residual + self.o_proj(attn_out)

        # SwiGLU FFN
        residual = x
        x = self.post_attn_norm(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = residual + self.down_proj(gate * up)
        return x


# ═══════════════════════════════════════════════════════════════════
# LoRA 适配器（消费级 GPU 微调）
# ═══════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation 线性层。

    WHY: OpenVLA 支持 LoRA 微调，rank=32 约等于全量微调性能，
    但可训练参数只有 1.4%（~100M），显存大幅降低。
    这使得在 24GB 消费级 GPU 上微调成为可能。
    """

    def __init__(self, linear: nn.Linear, rank: int = 32, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_features, out_features = linear.in_features, linear.out_features

        # 原始权重冻结
        self.weight = linear.weight
        self.bias = linear.bias
        for p in [self.weight, self.bias]:
            if p is not None:
                p.requires_grad = False

        # LoRA 参数：A (down) 和 B (up)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_A) @ self.lora_B * self.scaling
        return base + lora


# ═══════════════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("OpenVLA 模型演示")
    print("=" * 60)

    # 动作 tokenizer 演示
    tokenizer = ActionTokenizer(action_dim=7, num_bins=256, vocab_size=32000)
    print(f"\n动作 tokenizer: {tokenizer.num_bins} bins x {tokenizer.action_dim} 维")
    print(f"Token 范围: [{tokenizer.action_token_start}, {tokenizer.action_token_end})")

    # 模拟动作离散化
    dummy_actions = torch.randn(4, 7)
    action_min = torch.tensor([-0.1] * 6 + [0.0])  # xyz_rpy_gripper
    action_max = torch.tensor([0.1] * 6 + [1.0])
    bins = tokenizer.discretize(dummy_actions, action_min, action_max)
    recovered = tokenizer.de_discretize(bins, action_min, action_max)
    print(f"离散化-还原误差: {F.mse_loss(dummy_actions, recovered).item():.6f}")

    # 双视觉编码器
    vision_encoder = DualVisionEncoder()
    dino = torch.randn(2, 257, 768)
    siglip = torch.randn(2, 257, 768)
    vis_out = vision_encoder(dino, siglip)
    print(f"\n双视觉编码器输出: {vis_out.shape}  (B, 514 tokens, 4096 dims)")

    # 完整模型前向
    model = OpenVLA()
    text_ids = torch.randint(0, 31744, (2, 64))  # 文本 token（不用最后 256 个）
    action_ids = torch.randint(31744, 32000, (2, 7))  # 动作 token
    output = model(dino, siglip, text_ids, action_ids)
    print(f"\n完整模型前向: logits {output['logits'].shape}")
    print(f"动作 logits: {output['action_logits'].shape}  (B, 7维, 256 bins)")

    # 推理采样
    with torch.no_grad():
        actions = model.generate_action(dino, siglip, text_ids)
    print(f"推理动作: {actions.shape}")

    # LoRA 演示
    linear = nn.Linear(128, 256)
    lora_linear = LoRALinear(linear, rank=32)
    total_params = sum(p.numel() for p in lora_linear.parameters())
    trainable_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    print(f"\nLoRA 层: 总参数 {total_params}, 可训练 {trainable_params} ({100*trainable_params/total_params:.1f}%)")

    print("\nOpenVLA 演示完成")

```
