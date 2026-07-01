---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# RT-2 模型实现 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# RT-2 模型实现 - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
RT-2 模型实现
=============
基于 [[RT-2]] 论文实现：
- 连续动作离散化（256 bins）
- 动作 token 映射（覆写 VLM 词汇表中不常用 token / 使用专用整数 token）
- Co-Fine-Tuning 逻辑（在 VLM 数据和机器人数据上联合训练）

核心设计理念：
- “动作即语言”：把 7 维动作向量离散化为文本 token 序列，VLM 直接输出动作
- Co-Fine-Tuning：每个 batch 混合 ~50% 原始 VLM 数据 + ~50% 机器人数据，
  防止灾难性遗忘同时学到新能力
- 推理时输出约束：限制词汇表只包含 256 个有效动作 token

相关笔记：[[RT-2]] | [[PaLM-E]] | [[Open X-Embodiment & RT-X]] | [[OpenVLA]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


# ==============================================================================
# 第一部分：动作离散化与 token 映射
# ==============================================================================
# WHY 将连续动作离散化：RT-2 的核心洞察——VLM 已经学会了“从图像+文字生成
#      文字”的能力。如果把动作也变成“文字”（token 序列），VLM 无需任何
#      架构修改就能直接输出动作。

class ActionTokenizer:
    """
    连续动作空间 → 离散 token 映射器。

    7 维动作空间 → 每个维度 256 个 bin → 7 个 token 表示一个动作。

    WHY 256 bins: RT-2 发现 256 级离散化对大多数操作任务足够精确，
    且正好对应 1 字节——与 PaLI-X 的词汇表设计天然契合。
    WHY 均匀离散化而非学习型：简单、可解释，且实验证明效果足够好。
    后续工作（如 OpenVLA）发现非均匀分箱可能有微量提升，但差距不大。
    """
    def __init__(self, action_dim=7, num_bins=256,
                 action_min=None, action_max=None):
        self.action_dim = action_dim
        self.num_bins = num_bins

        # 每个维度的取值范围
        # WHY 预设范围而非从数据学习：推理时可能遇到超出训练分布的极端值，
        # 预设范围保证任何动作都能被编码
        if action_min is None:
            action_min = np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, -1.0])
        if action_max is None:
            action_max = np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0])

        self.action_min = action_min.astype(np.float32)
        self.action_max = action_max.astype(np.float32)
        self.action_range = self.action_max - self.action_min

    def discretize(self, action):
        """
        连续动作 → 离散 token。

        步骤：clip → normalize to [0,1] → scale to [0, num_bins-1] → round to int
        WHY 先 clip 再归一化：防止异常值破坏离散化的均匀性。
        """
        action = np.clip(action, self.action_min, self.action_max)
        normalized = (action - self.action_min) / (self.action_range + 1e-8)
        tokens = np.round(normalized * (self.num_bins - 1)).astype(int)
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        return tokens  # [action_dim] of ints in [0, 255]

    def undiscretize(self, tokens):
        """
        离散 token → 连续动作。
        反过程：token int → [0,1] → action range
        WHY 用 bin 中心而非边界：最小化重建误差。
        """
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        normalized = tokens.astype(np.float32) / (self.num_bins - 1)
        actions = normalized * self.action_range + self.action_min
        return actions

    def encode_to_string(self, action):
        """
        动作向量 → 空格分隔的 token 字符串。
        格式："128 91 241 5 101 127 0"

        WHY 字符串格式：PaLI-X 的 tokenizer 能直接理解这种“数字列表”，
        训练时 VLM 看到的就是这种格式。
        """
        tokens = self.discretize(action)
        return " ".join(str(t) for t in tokens)

    def decode_from_string(self, token_str):
        """token 字符串 → 连续动作"""
        tokens = np.array([int(t) for t in token_str.strip().split()])
        return self.undiscretize(tokens)


class TokenMapping:
    """
    动作 token 到 VLM 词汇表的映射策略。

    RT-2 论文描述了两种 mapping 方式：

    策略 1：PaLI-X 骨干（简单）
    PaLI-X 的词表中整数 0-999 都有专用 token，所以 bin_0→token_0, bin_128→token_128。
    直接映射，无需覆写任何 token。

    策略 2：PaLM-E / Llama 骨干（需覆写）
    这类 tokenizer 没有“0-255”这种特殊 token。RT-2 的做法是找词表中最不常用的 256 个
    token，将它们“覆写”为动作 token。
    WHY 覆写最不常用的 token：尽量减少对正常语言能力的影响——最不常用的 token
    在普通文本中几乎不出现，覆写它们对语言质量的影响可忽略不计。

    这里实现策略 2（更通用的方式），同时保留策略 1 的接口。
    """
    def __init__(self, vocab_size, num_bins=256, token_offset=0):
        """
        token_offset: 策略 1 时有意义的起始偏移（如 0），策略 2 时忽略
        """
        self.num_bins = num_bins
        self.token_offset = token_offset

        # 用于策略 2：记录哪些原始 token 被覆写
        self.overwritten_tokens = {}  # {bin_id: original_token_id}

    def get_action_token_ids(self):
        """
        返回所有动作 token 的 ID 列表。
        这些 ID 在推理时将用于 output constraint（限制词汇表）。
        """
        return list(range(self.token_offset, self.token_offset + self.num_bins))

    def map_action_to_token(self, discrete_action, dim_idx):
        """
        将某一维度的离散动作值映射为 token ID。

        discrete_action: int in [0, num_bins-1]
        dim_idx: 0-6（动作的哪个维度）
        Returns: token_id

        WHY 每个维度使用相同的 256 bins：RT-2 发现不需要为不同维度
        分配不同数量的 bin——各维度的重要性在 Co-Fine-Tuning 中自动学习。
        """
        return self.token_offset + int(discrete_action)

    def map_token_to_action(self, token_id):
        """token ID → 离散动作值"""
        bin_val = token_id - self.token_offset
        return max(0, min(self.num_bins - 1, bin_val))


# ==============================================================================
# 第二部分：RT-2 主体（VLM + 动作 token 化）
# ==============================================================================

class RT2VisionEncoder(nn.Module):
    """
    RT-2 的视觉编码器（简化版）。
    基于 ViT 架构，将图像编码为固定数量的“视觉 token”。
    WHY 用 ViT 而非 CNN：VLM 骨干（PaLI-X）的视觉编码器本身就是 ViT，
    在整个模型中以“嵌入序列”的形式参与自注意力计算。
    """
    def __init__(self, image_size=224, patch_size=14, embed_dim=1024,
                 depth=12, num_heads=16):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_proj(x)                      # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)           # [B, num_patches, D]
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x  # [B, num_patches, D]


class ViTBlock(nn.Module):
    """ViT Transformer block（Pre-LN + MLP）"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


class RT2DecoderBlock(nn.Module):
    """
    RT-2 的 decoder block（简化版 GPT/Palm 风格）。

    输入序列 = 视觉 tokens + 文本 tokens + 动作 tokens
    全用 causal self-attention 统一处理。

    WHY causal: 动作生成是自回归的——action_token_2 依赖于 action_token_1，
    就像文本生成中下一个词依赖于之前的词。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          attn_mask=attn_mask)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class RT2(nn.Module):
    """
    RT-2：Vision-Language-Action Model。

    完整前向流程：
    1. 图像 → ViT 编码 → 视觉 tokens
    2. 文本“Q: 该做什么动作？A: ”→ tokenize → 文本 tokens
    3. [视觉 tokens] + [文本 tokens] → Decoder Transformer
    4. 输出端自回归生成 7 个动作 token
    5. 动作 token → 离散化逆映射 → 连续动作向量

    WHY 不做任何架构修改：RT-2 的精髓。如果把视觉嵌入 + 文本嵌入喂给 VLM，
    它已经会输出文本。如果在训练数据中让“正确的动作 token 序列”出现在
    “A:”后面，VLM 学会的就是：给定图像和指令 → 输出动作 token 序列。
    整个过程不需要新的模块、新的损失函数或新的训练技巧。
    """
    def __init__(self, vocab_size=256000, embed_dim=1024, depth=24,
                 num_heads=16, action_dim=7, num_bins=256,
                 token_offset=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_bins = num_bins

        # ---- Token 嵌入表 ----
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # ---- 视觉编码器 ----
        self.vision_encoder = RT2VisionEncoder(
            image_size=224, patch_size=14, embed_dim=embed_dim,
            depth=12, num_heads=num_heads,
        )
        # 视觉位置嵌入（在视觉 tokens 之间加位置信息）
        num_patches = (224 // 14) ** 2
        self.vis_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.vis_pos_embed, std=0.02)

        # ---- Decoder ----
        self.blocks = nn.ModuleList([
            RT2DecoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)

        # ---- 输出头 ----
        # 每个动作维度输出 256 个 logit（对应 256 bins）
        self.action_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_bins) for _ in range(action_dim)
        ])

        # ---- 可学习的“动作开始”嵌入 ----
        # WHY 特殊 token：标记从哪里开始生成动作序列，
        # 相当于告诉模型“以下输出是动作，不是自然语言”
        self.action_start_embed = nn.Parameter(torch.randn(embed_dim))

    def forward(self, images, text_ids, target_action_tokens=None):
        """
        images: [B, C, H, W] 单帧图像
        text_ids: [B, T_text] 文本 token 序列（如 "Q: pick up the apple A:"）
        target_action_tokens: [B, action_dim] of ints in [0, 255] — 训练目标

        Returns:
            action_logits: List of [B, num_bins] × action_dim
            loss: Optional scalar
        """
        B = images.shape[0]
        device = images.device

        # Step 1: 视觉编码
        vis_tokens = self.vision_encoder(images)      # [B, N_patches, D]
        vis_tokens = vis_tokens + self.vis_pos_embed   # 加视觉位置编码

        # Step 2: 文本嵌入
        text_tokens = self.token_embed(text_ids)       # [B, T_text, D]

        # Step 3: 拼接序列 [视觉 | 文本 | 动作前缀]
        # WHY 视觉在文本前面：这样可以减少视觉 token 的“因果掩码”限制——
        # 让文本 token 可以看到所有视觉信息
        action_start = self.action_start_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        sequence = torch.cat([vis_tokens, text_tokens, action_start], dim=1)  # [B, N+T+1, D]

        # Step 4: 自回归生成动作（逐个维度）
        action_logits = []
        current_seq = sequence

        for dim_idx in range(self.action_dim):
            # Decoder forward
            x = current_seq
            # 生成因果掩码
            T = x.shape[1]
            attn_mask = torch.tril(torch.ones(T, T, device=device)).bool()
            # 转为 MultiheadAttention 需要的格式
            attn_mask_float = torch.zeros(T, T, device=device)
            attn_mask_float = attn_mask_float.masked_fill(~attn_mask, float("-inf"))

            for block in self.blocks:
                x = block(x, attn_mask=attn_mask_float)
            x = self.ln_final(x)

            # 取最后一个 token 预测当前维度的动作
            last_hidden = x[:, -1, :]                   # [B, D]
            dim_logits = self.action_heads[dim_idx](last_hidden)  # [B, num_bins]
            action_logits.append(dim_logits)

            # 为下一维准备：把当前预测的 token 嵌入加到序列末尾
            # 训练时用 teacher forcing（用真实目标），推理时采样
            if target_action_tokens is not None:
                next_token_id = target_action_tokens[:, dim_idx]
            else:
                next_token_id = dim_logits.argmax(dim=-1)

            next_emb = self.token_embed(next_token_id).unsqueeze(1)  # [B, 1, D]
            current_seq = torch.cat([current_seq, next_emb], dim=1)

        # Step 5: 计算 loss（如果有 targets）
        loss = None
        if target_action_tokens is not None:
            # 每个维度的交叉熵 loss 之和
            losses = []
            for dim_idx, logits in enumerate(action_logits):
                targets = target_action_tokens[:, dim_idx]  # [B]
                losses.append(F.cross_entropy(logits, targets))
            loss = sum(losses) / len(losses)

        return action_logits, loss

    def get_action(self, images, text_ids, action_tokenizer, temperature=1.0):
        """
        推理接口：给定图像和文本指令，输出连续动作向量。

        WHY 温度采样而非 argmax：动作空间往往是多模态的（同一个场景
        有多种正确动作），温度采样可以让策略产生多样化的动作。
        但 RT-2 原论文主要用 argmax（温度=0）。
        """
        self.eval()
        B = images.shape[0]
        device = images.device

        with torch.no_grad():
            # 视觉编码
            vis_tokens = self.vision_encoder(images)
            vis_tokens = vis_tokens + self.vis_pos_embed

            # 文本嵌入
            text_tokens = self.token_embed(text_ids)

            # 拼接
            action_start = self.action_start_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            current_seq = torch.cat([vis_tokens, text_tokens, action_start], dim=1)

            # 逐个维度生成动作 token
            action_tokens = []
            for dim_idx in range(self.action_dim):
                x = current_seq
                T = x.shape[1]
                attn_mask = torch.zeros(T, T, device=device)
                attn_mask = attn_mask.masked_fill(
                    ~torch.tril(torch.ones(T, T, device=device)).bool(),
                    float("-inf"),
                )

                for block in self.blocks:
                    x = block(x, attn_mask=attn_mask)
                x = self.ln_final(x)

                last_hidden = x[:, -1, :]
                logits = self.action_heads[dim_idx](last_hidden) / temperature
                probs = F.softmax(logits, dim=-1)

                if temperature > 0:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_token = logits.argmax(dim=-1)

                action_tokens.append(next_token)

                next_emb = self.token_embed(next_token).unsqueeze(1)
                current_seq = torch.cat([current_seq, next_emb], dim=1)

            # 离散 token → 连续动作
            action_tokens = torch.stack(action_tokens, dim=1)  # [B, action_dim]
            original_action = action_tokenizer.undiscretize(
                action_tokens[0].cpu().numpy()
            )
            return original_action


# ==============================================================================
# 第三部分：Co-Fine-Tuning 训练器
# ==============================================================================
# WHY Co-Fine-Tuning 而非纯微调：如果只用机器人数据微调 VLM，模型会
# 灾难性遗忘所有互联网学到的语义知识——这正是 RT-2 实验中“无互联网预训练”
# 组 0% 成功率的原因。Co-Fine-Tuning 在每个 batch 中混合 ~50% 原始 VLM
# 数据和 ~50% 机器人数据，让模型同时保持两种能力。

class CoFineTuningTrainer:
    """
    联合微调训练器。

    核心逻辑：
    - 每个 batch 按比例采样两种数据
    - VLM 数据用标准 next-token prediction loss
    - 机器人数据用动作 token 交叉熵 loss
    - 两种 loss 直接加权求和

    WHY 相同的优化器同时优化两种 loss：让模型在学习新动作能力的同时
    不遗忘旧的语言/视觉能力。这与多任务学习的“共享表示”直觉一致。
    """
    def __init__(self, model, vlm_data_loader, robot_data_loader,
                 robot_data_weight=0.5, lr=3e-5):
        self.model = model
        self.vlm_data_loader = vlm_data_loader
        self.robot_data_loader = robot_data_loader
        self.robot_data_weight = robot_data_weight  # ~50% (与论文一致)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(self, vlm_batch, robot_batch):
        """
        单个 Co-Fine-Tuning 训练步。

        vlm_batch: {"images": ..., "text_ids": ..., "target_ids": ...}
        robot_batch: {"images": ..., "text_prompt_ids": ..., "action_tokens": ...}

        Returns: {"total_loss": ..., "vlm_loss": ..., "action_loss": ...}
        """
        self.model.train()
        total_loss = torch.tensor(0.0)

        # -- VLM 数据部分（保持语言/视觉能力）--
        vlm_images = vlm_batch["images"]
        vlm_text = vlm_batch["text_ids"]

        # VLM forward: 用 next-token prediction
        vis_tokens = self.model.vision_encoder(vlm_images)
        vis_tokens = vis_tokens + self.model.vis_pos_embed
        text_tokens = self.model.token_embed(vlm_text)
        full_seq = torch.cat([vis_tokens, text_tokens], dim=1)

        # 简单的语言建模 loss（预测下一个 token）
        x = full_seq
        T = x.shape[1]
        attn_mask = torch.zeros(T, T, device=x.device)
        attn_mask = attn_mask.masked_fill(
            ~torch.tril(torch.ones(T, T, device=x.device)).bool(), float("-inf"),
        )
        for block in self.model.blocks:
            x = block(x, attn_mask=attn_mask)
        # 输出 logits 与 target 做 cross entropy
        # （简化：匹配文本维度的 targets）
        vlm_loss = F.cross_entropy(
            x[:, :-1, :vlm_text.shape[1]].reshape(-1, self.model.embed_dim)
            @ self.model.token_embed.weight[:self.model.token_embed.num_embeddings].T,
            vlm_text[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        # -- 机器人数据部分（学习动作生成）--
        robot_images = robot_batch["images"]
        robot_text = robot_batch["text_prompt_ids"]
        action_targets = robot_batch["action_tokens"]

        _, action_loss = self.model(robot_images, robot_text, action_targets)

        # -- 联合优化 --
        batch_loss = (1 - self.robot_data_weight) * vlm_loss + \
                      self.robot_data_weight * action_loss

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": batch_loss.item(),
            "vlm_loss": vlm_loss.item(),
            "action_loss": action_loss.item(),
        }

    def train_epoch(self):
        """完整 epoch 循环"""
        epoch_losses = []
        vlm_iter = iter(self.vlm_data_loader)
        robot_iter = iter(self.robot_data_loader)

        # 取较长的 data loader 的长度作为 epoch 长度
        num_steps = max(len(self.vlm_data_loader), len(self.robot_data_loader))

        for _ in range(num_steps):
            vlm_batch = next(vlm_iter, None)
            robot_batch = next(robot_iter, None)

            # 如果某个 loader 耗尽就重新循环
            if vlm_batch is None:
                vlm_iter = iter(self.vlm_data_loader)
                vlm_batch = next(vlm_iter)
            if robot_batch is None:
                robot_iter = iter(self.robot_data_loader)
                robot_batch = next(robot_iter)

            result = self.train_step(vlm_batch, robot_batch)
            epoch_losses.append(result)

        # 汇总统计
        avg_total = np.mean([r["total_loss"] for r in epoch_losses])
        avg_vlm = np.mean([r["vlm_loss"] for r in epoch_losses])
        avg_action = np.mean([r["action_loss"] for r in epoch_losses])
        return {"avg_total": avg_total, "avg_vlm": avg_vlm, "avg_action": avg_action}


# ==============================================================================
# 演示
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RT-2 Model Demo")
    print("=" * 60)

    # ---------- 1. 动作离散化 ----------
    print("\n[1] 动作离散化与 token 映射...")
    action_tokenizer = ActionTokenizer(action_dim=7, num_bins=256)
    action = np.array([0.05, -0.12, 0.03, 0.1, -0.05, 0.02, 1.0])

    discrete = action_tokenizer.discretize(action)
    print(f"  原始动作: {action}")
    print(f"  离散化:   {discrete}")
    print(f"  Token 字符串: \"{action_tokenizer.encode_to_string(action)}\"")

    recovered = action_tokenizer.undiscretize(discrete)
    print(f"  重建误差: {np.abs(action - recovered).max():.6f}")

    # ---------- 2. Token 映射策略 ----------
    print("\n[2] Token 映射策略...")
    mapping = TokenMapping(vocab_size=50000, num_bins=256, token_offset=0)
    action_token_ids = mapping.get_action_token_ids()
    print(f"  动作 token ID 范围: [{action_token_ids[0]}, {action_token_ids[-1]}]")
    print(f"  共 {len(action_token_ids)} 个 token 被用于动作表示")

    # ---------- 3. RT-2 模型前向 ----------
    print("\n[3] RT-2 模型前向（演示小模型）...")
    model_config = {
        "vocab_size": 5000,
        "embed_dim": 256,
        "depth": 4,
        "num_heads": 4,
        "action_dim": 7,
        "num_bins": 256,
    }

    model = RT2(**model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params / 1e6:.2f}M （注：RT-2 原版 ~55B）")

    # 模拟 inputs
    B = 2
    images = torch.randn(B, 3, 224, 224)
    text_ids = torch.randint(0, model_config["vocab_size"], (B, 20))
    action_targets = torch.randint(0, 256, (B, 7))

    action_logits, loss = model(images, text_ids, action_targets)
    print(f"  动作维度数: {len(action_logits)}")
    print(f"  每维 logit 形状: {action_logits[0].shape}")
    print(f"  训练 loss: {loss.item():.4f}")

    # ---------- 4. 推理演示 ----------
    print("\n[4] 推理：图像+指令 → 动作...")
    action_pred = model.get_action(images[:1], text_ids[:1], action_tokenizer,
                                   temperature=0.5)
    print(f"  预测动作 (7维): {np.round(action_pred, 3)}")

    # ---------- 5. Co-Fine-Tuning 数据混合 ----------
    print("\n[5] Co-Fine-Tuning 数据混合演示...")
    print(f"  VLM 数据权重:   ~50% （保持互联网知识）")
    print(f"  机器人数据权重: ~50% （学习动作生成）")
    print(f"  与纯微调的关键区别: 混合数据防止灾难性遗忘")
    print(f"  与冻结 VLM + 动作头的区别: 端到端梯度流动，更好的语义融合")

    # ---------- 6. 输出约束 ----------
    print("\n[6] 推理时输出词汇表约束...")
    all_action_ids = mapping.get_action_token_ids()
    print(f"  约束后词汇表大小: {len(all_action_ids)} (仅动作 token)")
    print(f"  约束前词汇表大小: {model_config['vocab_size']}")
    print(f"  为什么需要约束: 推理时防止 VLM 输出自然语言而非动作")

    print("\n✓ RT-2 模型演示完成")

```

```
