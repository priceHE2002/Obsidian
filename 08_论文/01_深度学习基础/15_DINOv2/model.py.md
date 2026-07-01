---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# DINOv2: Learning Robust Visual Features without Supervision - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# DINOv2: Learning Robust Visual Features without Supervision - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
DINOv2: Learning Robust Visual Features without Supervision
===========================================================
论文: "DINOv2: Learning Robust Visual Features without Supervision"
      (Oquab et al., Meta AI, 2023 / TMLR 2024)
核心贡献: 组合 DINO(对比学习) + iBOT(MIM) + KoLeo(多样性正则) 三种损失,
         在 142M 精选图像上训练出 1.1B ViT-g。其 frozen features 同时
         适用于图像级和像素级任务。
架构: Student-Teacher 自蒸馏框架:
      Student 处理不同 view (含 mask) → Teacher 是 Student 的 EMA →
      DINO loss (CLS token 对齐) + iBOT loss (patch token 对齐)
      + KoLeo 正则(防特征坍塌) + Sinkhorn-Knopp 中心化

与 [[../13_CLIP/CLIP.md|CLIP]] 的关系: DINOv2 是空间编码器，CLIP 是语义编码器，
  OpenVLA 用二者互补作为双视觉编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ==============================================================================
# 1. ViT 基础组件
# ==============================================================================
class PatchEmbed(nn.Module):
    """图像分块 + 线性投影。"""
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with SwiGLU FFN (DINOv2 使用 SwiGLU)。"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        # SwiGLU FFN: 两个 Linear → SiLU → 相乘 → Linear
        hidden_dim = int(dim * mlp_ratio * 2 / 3)  # SwiGLU 需调整 hidden dim
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # SwiGLU FFN
        residual = x
        x = self.norm2(x)
        x = self.w3(F.silu(self.w1(x)) * self.w2(x))
        x = residual + self.dropout(x)
        return x


# ==============================================================================
# 2. DINOv2 的 Projection Head（解耦设计）
# ==============================================================================
class DINOHead(nn.Module):
    """DINO Loss 的 projection head。

    将 CLS token 特征映射到 prototype scores，
    然后 softmax 得到 teacher/student 的概率分布。

    关键: DINOv2 将 DINO head 和 iBOT head 的权重**解耦**(untie) ——
    iBOT 原论文认为共享 head 更好，但 DINOv2 发现大规模训练时解耦更好。

    Sinkhorn-Knopp (SK) 中心化的作用:
    Teacher 端用 SK 迭代保证 batch 内各类别被均匀分配，
    防止 teacher 始终输出相同的高概率类别（模式坍塌）。
    """

    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048,
                 num_sk_iterations=3):
        super().__init__()
        self.num_sk_iterations = num_sk_iterations

        # 3 层 MLP: in_dim → hidden → hidden → out_dim
        # 最后一层后接 L2 norm + weight norm (类似 DINO 原始设计)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        """x: (B, D) → (B, out_dim) proto scores"""
        x = self.mlp[:4](x)  # 前 4 层 MLP
        x = F.normalize(x, dim=-1, eps=1e-6)
        x = self.last_layer(x)  # weight-normalized 线性投影
        return x


class iBOTHead(nn.Module):
    """iBOT Loss 的 patch-level projection head（与 DINO head 解耦）。

    将 patch token 特征映射到 prototype scores，
    用于对 masked patches 做 teacher-student 对齐。
    """

    def __init__(self, in_dim, out_dim=8192, hidden_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        """x: (B, N, D) → (B, N, out_dim)"""
        x = self.mlp[:4](x)
        x = F.normalize(x, dim=-1, eps=1e-6)
        x = self.last_layer(x)
        return x


# ==============================================================================
# 3. Sinkhorn-Knopp 中心化
# ==============================================================================
def sinkhorn_knopp_centering(logits, n_iter=3, eps=1e-8):
    """
    Sinkhorn-Knopp 迭代算法，对 teacher 输出做中心化。

    为什么需要 SK 中心化？
    如果没有中心化，teacher 可能坍塌为始终输出某个固定原型——
    所有样本被分到同一类，模型学不到任何有用表示。
    SK 算法通过迭代行列归一化保证:
    - 每个 batch 中各类别(prototype)被均匀分配
    - 约束满足: sum over prototypes for each sample = 1
                sum over samples for each prototype = 1/B (均匀分布)

    DINOv2 与 DINO 的关键区别:
    DINO 原始用 moving average centering (EMA 更新 teacher 输出的均值)，
    DINOv2 改用 SK（来自 SwAV），在大规模训练中效果更好。
    """
    Q = torch.exp(logits).t()  # (K, B)，转置以便行列操作
    K, B = Q.shape

    # SK 迭代: 交替归一化列和行
    sum_Q = Q.sum(dim=0, keepdim=True) + eps  # 防除零
    Q /= sum_Q

    for _ in range(n_iter):
        # 列归一化: 每个样本的概率和 = 1
        Q /= (Q.sum(dim=0, keepdim=True) + eps)
        # 行归一化: 每个 prototype 被分配的总概率 = 1/K
        Q /= (Q.sum(dim=1, keepdim=True) + eps)

    # 重新缩放: Q * (B/K) → 每个 prototype 期望被分配 B/K 个样本
    Q *= B
    return Q.t()  # (B, K)


# ==============================================================================
# 4. DINOv2 核心损失函数
# ==============================================================================
class DINOLoss(nn.Module):
    """DINO Loss: 图像级 self-distillation。

    公式: L_DINO = -sum(p_t * log(p_s))
    p_s = softmax(student_cls_head / τ_s)
    p_t = softmax(teacher_cls_head / τ_t) (teacher 端温度更低)

    为什么 teacher 温度更低？
    让 teacher 的预测更"自信"（更尖锐的分布），
    将知识"蒸馏"给 student。标准的自蒸馏技巧。

    Teacher 是 Student 的 EMA:
    θ_teacher ← m * θ_teacher + (1-m) * θ_student
    其中 m = 0.994 (cosine schedule from 0.996 → 1.0)
    """

    def __init__(self, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output):
        """
        student_output: (B, K) — student head 输出的 prototype scores
        teacher_output: (B, K) — teacher head 输出的 prototype scores
        """
        # Teacher: softmax + SK centering（注意: centering 在 teacher 端）
        teacher_out = teacher_output / self.teacher_temp
        teacher_out = sinkhorn_knopp_centering(teacher_out.detach())

        # Student: 标准 softmax（温度更高 → 更平滑）
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)

        # Cross-entropy: -sum(p_t * log(p_s))
        loss = - (teacher_out * student_out).sum(dim=-1).mean()
        return loss


class iBOTLoss(nn.Module):
    """iBOT Loss: patch-level masked image modeling。

    公式: L_iBOT = -sum_i(p_t^i * log(p_s^i))
    其中 i 遍历被 mask 的 patch 索引。

    Student 输入被 mask 的图像，Teacher 输入完整图像，
    Teacher 对应位置的 patch 输出作为 Student 的 target。

    为什么需要 iBOT（MIM 组件）？
    DINO 只关注 CLS token（全局特征），在像素级任务（分割、深度）
    上表现不足。iBOT 补充了 patch-level 理解能力。
    """

    def __init__(self, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_patch, teacher_patch, mask=None):
        """
        student_patch: (B, N, K) — student 的 patch scores
        teacher_patch: (B, N, K) — teacher 的 patch scores
        mask: (B, N) — True 表示被 mask（仅计算这些位置）
        """
        teacher_out = teacher_patch / self.teacher_temp
        teacher_out = sinkhorn_knopp_centering(
            teacher_out.reshape(-1, teacher_out.shape[-1]).detach()
        ).reshape(teacher_out.shape)

        student_out = F.log_softmax(student_patch / self.student_temp, dim=-1)

        loss_per_patch = - (teacher_out * student_out).sum(dim=-1)  # (B, N)

        if mask is not None:
            loss_per_patch = loss_per_patch * mask.float()

        return loss_per_patch.sum() / (mask.sum() + 1e-8 if mask is not None
                                       else student_patch.shape[1])


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko 微分熵正则化。

    公式: L_KoLeo = -(1/n) * sum_i log(d_{n,i})
    其中 d_{n,i} = min_{j≠i} ||x_i - x_j|| 是 x_i 与最近邻的距离。

    为什么需要 KoLeo？
    - 防止特征坍塌: 强制 batch 内特征在空间中均匀分布
    - 本质是最大化微分熵，鼓励特征覆盖更多"空间"
    - 在实例检索上提升 8% mAP，对其他任务无负面影响

    计算前对特征做 L2 归一化（所有点在单位球面上）。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: (B, D) — CLS token 特征（L2 normalized）
        返回标量 loss
        """
        # 计算成对距离 ||x_i - x_j||²
        x = F.normalize(x, dim=-1, eps=1e-6)
        # x@x^T + x@x^T - 2*x@x^T = (x_i-x_j)^2 的展开
        dot = x @ x.t()
        dist_sq = 2 - 2 * dot  # 对 L2 归一化向量: ||x_i-x_j||² = 2-2<x_i,x_j>

        # 排除对角线 (i=j)
        diag_mask = torch.eye(len(x), device=x.device, dtype=torch.bool)
        # 设为极大值，确保最近邻不是自己
        dist_sq = dist_sq.masked_fill(diag_mask, float('inf'))

        # 每个点的最近邻距离
        min_dist = torch.sqrt(dist_sq.min(dim=1).values + 1e-8)
        # KoLeo: -log(min_dist) 的均值
        return -torch.log(min_dist).mean()


# ==============================================================================
# 5. EMA 动量更新
# ==============================================================================
@torch.no_grad()
def ema_update(student, teacher, momentum=0.996):
    """
    Teacher 参数是 Student 的指数移动平均(EMA)。

    为什么用 EMA 而不是反向传播更新 Teacher？
    - EMA 使 Teacher 成为 Student 的"时间集成"，输出更稳定
    - 防止 Teacher 和 Student 同步崩溃 (mode collapse)
    - 动量值: DINOv2 用 cosine schedule，从 0.996 → 1.0

    注意: Teacher 只更新参数，不需要梯度。
    """
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)


# ==============================================================================
# 6. DINOv2 完整模型
# ==============================================================================
class DINOv2(nn.Module):
    """DINOv2: Student-Teacher 自蒸馏 + DINO/iBOT/KoLeo 多损失训练。

    训练流程（每个 step）:
    1. 对同一图像生成 views: global (224) + local (98)
    2. Teacher 接收 global views → 输出 CLS scores 和 patch scores
    3. Student 接收 local views（可选 mask）→ 输出 CLS 和 patch scores
    4. 计算 L = L_DINO + L_iBOT + L_KoLeo
    5. Student 梯度下降更新，Teacher EMA 更新

    三个损失的协同作用:
    - DINO: 全局语义对齐（分类、检索）
    - iBOT: 局部细节建模（分割、深度估计）
    - KoLeo: 特征均匀分布（防坍塌 + 提升检索）
    """

    def __init__(self, img_size=224, patch_size=14, in_chans=3,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 dino_out_dim=65536, ibot_out_dim=8192,
                 ema_momentum=0.996):
        super().__init__()
        self.ema_momentum = ema_momentum

        # Student 网络（需要梯度）
        self.student = self._build_vit(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio
        )
        # Teacher 网络（EMA 更新，无梯度）
        self.teacher = self._build_vit(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio
        )
        # 初始化 Teacher 与 Student 相同
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 投影头（DINO + iBOT，解耦权重）
        self.student_dino_head = DINOHead(embed_dim, out_dim=dino_out_dim)
        self.teacher_dino_head = DINOHead(embed_dim, out_dim=dino_out_dim)
        self.teacher_dino_head.load_state_dict(self.student_dino_head.state_dict())
        for p in self.teacher_dino_head.parameters():
            p.requires_grad = False

        self.student_ibot_head = iBOTHead(embed_dim, out_dim=ibot_out_dim)
        self.teacher_ibot_head = iBOTHead(embed_dim, out_dim=ibot_out_dim)
        self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())
        for p in self.teacher_ibot_head.parameters():
            p.requires_grad = False

        # 损失函数
        self.dino_loss_fn = DINOLoss()
        self.ibot_loss_fn = iBOTLoss()
        self.koleo_loss_fn = KoLeoLoss()

    def _build_vit(self, img_size, patch_size, in_chans, embed_dim,
                   depth, num_heads, mlp_ratio):
        """构建 ViT backbone。"""
        patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos_embed = nn.Parameter(
            torch.zeros(1, patch_embed.num_patches + 1, embed_dim)
        )
        blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(pos_embed, std=0.02)
        nn.init.trunc_normal_(cls_token, std=0.02)
        for block in blocks:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

        model = nn.Module()
        model.patch_embed = patch_embed
        model.cls_token = cls_token
        model.pos_embed = pos_embed
        model.blocks = blocks
        model.norm = norm
        model.embed_dim = embed_dim
        model.num_patches = patch_embed.num_patches
        return model

    def _forward_vit(self, vit, x, mask_ratio=0.0):
        """ViT 前向传播，返回 CLS token 和 patch tokens。"""
        B = x.shape[0]
        x = vit.patch_embed(x)
        N = x.shape[1]

        # 添加 CLS token + 位置编码
        cls_tokens = vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, D)
        x = x + vit.pos_embed

        # 可选: mask 部分 patches（iBOT 组件需要）
        mask = None
        if mask_ratio > 0:
            patch_tokens = x[:, 1:, :]  # 仅 patches, 不含 CLS
            N_patches = patch_tokens.shape[1]
            len_keep = int(N_patches * (1 - mask_ratio))
            noise = torch.rand(B, N_patches, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # 保留可见 patches
            patch_kept = torch.gather(
                patch_tokens, dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, patch_tokens.shape[2])
            )
            x = torch.cat([x[:, :1, :], patch_kept], dim=1)
            # 记录 mask
            mask = torch.ones([B, N_patches], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

        # Transformer blocks
        for block in vit.blocks:
            x = block(x)

        x = vit.norm(x)
        cls_out = x[:, 0]       # (B, D)
        patch_out = x[:, 1:, :]  # (B, N, D)

        return cls_out, patch_out, mask

    def forward(self, x_student, x_teacher):
        """
        x_student: (B, 3, H, W) — student 输入（可含 mask）
        x_teacher: (B, 3, H, W) — teacher 输入（完整图）
        """
        # Teacher 前向 (no mask, no grad)
        with torch.no_grad():
            t_cls, t_patch, _ = self._forward_vit(self.teacher, x_teacher, 0.0)
            t_dino = self.teacher_dino_head(t_cls)
            t_ibot = self.teacher_ibot_head(t_patch)

        # Student 前向（含 mask, 需要 grad）
        s_cls, s_patch, mask = self._forward_vit(self.student, x_student, 0.3)
        s_dino = self.student_dino_head(s_cls)
        s_ibot = self.student_ibot_head(s_patch)

        # 三个损失
        loss_dino = self.dino_loss_fn(s_dino, t_dino)
        loss_ibot = self.ibot_loss_fn(s_ibot, t_ibot, mask)
        loss_koleo = self.koleo_loss_fn(s_cls)

        total_loss = loss_dino + loss_ibot + loss_koleo

        return {
            'loss': total_loss,
            'loss_dino': loss_dino.item(),
            'loss_ibot': loss_ibot.item(),
            'loss_koleo': loss_koleo.item(),
        }

    def update_teacher(self):
        """EMA 更新 Teacher 参数。"""
        with torch.no_grad():
            ema_update(self.student, self.teacher, self.ema_momentum)
            ema_update(self.student_dino_head, self.teacher_dino_head,
                       self.ema_momentum)
            ema_update(self.student_ibot_head, self.teacher_ibot_head,
                       self.ema_momentum)


# ==============================================================================
# 演示
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DINOv2 — Student-Teacher 自蒸馏演示")
    print("=" * 60)

    # 小型 DINOv2（演示用，真实 ViT-g 有 1.1B 参数）
    model = DINOv2(
        img_size=224, patch_size=14, in_chans=3,
        embed_dim=384, depth=6, num_heads=6,
        dino_out_dim=2048, ibot_out_dim=1024
    )

    batch_size = 4
    x_student = torch.randn(batch_size, 3, 224, 224)
    x_teacher = torch.randn(batch_size, 3, 224, 224)

    # 训练步骤
    losses = model(x_student, x_teacher)
    print(f"Total Loss: {losses['loss'].item():.4f}")
    print(f"  DINO Loss:  {losses['loss_dino']:.4f}  (图像级对比)")
    print(f"  iBOT Loss:  {losses['loss_ibot']:.4f}  (patch级 MIM)")
    print(f"  KoLeo Loss: {losses['loss_koleo']:.4f}  (特征多样性正则)")

    # EMA 更新 Teacher
    model.update_teacher()

    # 验证 Teacher 是 Student 的 EMA
    s_cls = model.student.cls_token
    t_cls = model.teacher.cls_token
    print(f"\nStudent CLS token mean: {s_cls.mean().item():.6f}")
    print(f"Teacher CLS token mean: {t_cls.mean().item():.6f}")
    print(f"(Teacher 是 Student 的 EMA, 动量={model.ema_momentum})")

    # 仅推理: 提取 frozen features
    with torch.no_grad():
        cls_feat, patch_feat, _ = model._forward_vit(model.teacher, x_teacher, 0.0)
        print(f"\n推理模式 (Teacher, frozen):")
        print(f"  CLS 特征形状:  {cls_feat.shape}  (图像级)")
        print(f"  Patch 特征形状: {patch_feat.shape}  (像素级)")

    total_params = sum(p.numel() for p in model.student.parameters())
    print(f"\nStudent 参数量: {total_params / 1e6:.1f}M")
    print(f"真实 DINOv2 ViT-g: ~1.1B params, 128k prototypes")
    print(f"\n关键: DINO(全局) + iBOT(局部) + KoLeo(防坍塌) 多损失协同")

```

```
