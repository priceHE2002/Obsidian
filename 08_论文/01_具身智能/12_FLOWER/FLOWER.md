---
tags:
  - 论文
  - VLA
  - 轻量模型
  - CoRL2025
  - Rectified-Flow
created: 2026-06-30
paper_title: "FLOWER: Democratizing Generalist Robot Policies"
paper_authors: "Moritz Reuss et al. (KIT + Microsoft Research)"
paper_year: 2025
paper_venue: "CoRL 2025"
paper_url: "https://arxiv.org/abs/2509.04996"
github: "https://github.com/intuitive-robots/flower_vla_pret"
---

# FLOWER

**FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies**
*KIT Intuitive Robots Lab + Microsoft Research | CoRL 2025 | arXiv 2509.04996*

> **"VLA 太贵了——不仅参数大，预训练成本更高。能不能用 1% 的算力做差不多的事？"** FLOWER 的答案是：能。950M 参数 + 200 H100 小时（OpenVLA 的 1/100）+ 190 个任务上达到 SOTA。更重要的是，它揭示了一个反直觉的设计规律：**VLM 的中间层特征比最终层更适合机器人控制。**

---

## 一、核心洞察：VLA 的"预算分配"问题

### 1.1 VLA 的构成

一个典型的 Flow-based VLA 由两块组成：

| 组件 | 角色 | 典型大小 |
|------|------|---------|
| **VLM 骨干** | 编码感知信息和语言指令 | 3-55B 参数（占据了绝大部分）|
| **Flow/Diffusion Transformer** | 从噪声生成动作 | 通常很小（被 VLM 挤占了预算）|

### 1.2 FLOWER 发现的"预算陷阱"

现有 VLA 默认为"VLM 越大越好"。但 FLOWER 发现这造成了严重的不平衡：

- VLM 的大部分算力花在**最后一层**——这一层针对"下一个文本 token 预测"优化
- 但机器人控制**不需要生成文本**——它需要的是语义丰富的中间表征
- 同时，Flow Transformer 因为参数太少，**无法充分建模复杂的多模态动作分布**

这就像一个翻译员的口语能力极强，但给他的写作用纸只有便签大小。

### 1.3 FLOWER 的"再分配"方案

FLOWER 的解决方案极其直接：

1. **砍掉 VLM 的 30-50% 层**（不需要文本生成能力）
2. **把省下来的参数给 Flow Transformer**（需要更大的动作建模能力）
3. **从 VLM 的中间层获取特征**（语义最丰富的位置）

---

## 二、四大技术创新

### 2.1 Intermediate-Modality Fusion（中间模态融合）

这是 FLOWER 最核心的贡献。论文引用了 LLM 可解释性研究的发现：**Transformer 的倒数第二 quarter 层捕获了最广泛的语义信息，而最后一层过度专业化于 next-token 预测。**

具体实现：
- 对 **Encoder-Decoder VLM**（如 Florence-2）：直接砍掉整个 Decoder，只保留 Encoder。层数减 50%。
- 对 **Decoder-Only VLM**（如 SmolFlow2-Video）：移除最后 30% 的 Transformer 层。
- 从保留下来的最后几层中提取 hidden states → 投影到 Flow Transformer 的输入空间

这种做法的额外好处：**显存和推理延迟大幅降低**（每步少算 30-50% 的 VLM 层）。

### 2.2 Global Action-Specific AdaLN Conditioning

Flow Transformer 中的 AdaLN (Adaptive Layer Normalization) 是现代 DiT 架构的标准组件。FLOWER 的改进是：

**问题**：不同的机器人有不同的动作空间（不同维度、不同控制模式）。一个统一的 AdaLN 难以同时适配所有形态。

**解决**：为每种动作空间分配**独立的 AdaLN 参数**（scale & shift 向量），同时共享 Flow Transformer 的其他所有参数。

- 参数减少 **20%**（相比于为每种动作空间设计单独的动作头）
- 精度**无损**（因为 AdaLN 足够表达动作空间的差异）

### 2.3 Rectified Flow（整流流匹配）

相比于标准 Diffusion（~100 步训练/~10 步推理），FLOWER 使用 **Rectified Flow**：

$$\mathcal{L}(\theta) = \mathbb{E}\left[\|z_1 - \bar{a} - v_\theta(z_t, t, \bar{s}, g, e)\|^2\right]$$

其中 $z_t = (1-t) \cdot z_0 + t \cdot z_1$ 是噪声和数据的线性插值，$v_\theta$ 学习的是从噪声到数据的**直线**速度场。

优势：
- 单臂设置：仅需 **4 步**去噪
- 双臂设置：仅需 **8 步**去噪
- 训练收敛更快（因为路径是直的）

### 2.4 完整的 VLA 设计空间消融

论文做了大量消融实验（这是对研究者最有价值的部分）：

| 消融维度 | 发现 |
|---------|------|
| 中间层选择 | 倒数第二 quarter 最优 |
| 剪枝比例 | 30% 最优（Decoder-Only），太多会丢失语义 |
| 融合方式 | 交叉注意力 > 拼接 > FiLM |
| VLM 架构 | Encoder-Decoder (Florence-2) > Decoder-Only |
| VLM 预训练目标 | 视觉定位 + 描述 > 仅描述 |

---

## 三、FLOWER 的参数与效率

| | OpenVLA | RDT-1B | FLOWER |
|---|---|---|---|
| 总参数 | 7.7B | 1.2B + 11.4B语言 | **947M** |
| ViT | 600M | — | 360M |
| VLM (LLM部分) | 7B | — | **205M**（剪枝后）|
| Flow Transformer | N/A（无） | — | **339M** |
| 预训练 | 21,500 A100h | 35,000 A100h | **~200 H100h** |
| 推理 VRAM | ~15GB | >16GB | **~1.85GB** |
| 微调 VRAM | ~62GB | >40GB | **~10GB** |

训练使用 **4×H100，48 小时**（总计 ~200 GPU 小时），在 8 个公共数据集的"OXE-soup"（约 250k 轨迹）上预训练。

---

## 四、实验结果

### 4.1 190 个任务覆盖 10 个基准

| 基准 | FLOWER 性能 |
|------|------------|
| CALVIN ABC | **4.53 (SOTA)** |
| SIMPLER (WidowX) | ~0.94 |
| SIMPLER (Google Robot) | ~0.93 |
| LIBERO Spatial | ~0.97 |
| LIBERO Long | ~0.61 |
| ALOHA Sim | ~0.66 |
| Kitchen Single Task | ~0.91 |
| Kitchen Generalization | ~0.51 |
| 真实世界 (OK-Robot) | ~0.40 |

### 4.2 与更大模型的对比

在所有基准上，FLOWER（950M）的性能与 OpenVLA（7B）、RDT-1B（~12.4B）、π0（3B）处于同一水平，有时甚至更优。而且预训练成本只有它们的 1% ~ 5%。

---

## 五、失败的尝试（同样有价值）

论文记录了开发过程中遇到的挫折：

- **可变长度动作块**：收敛慢，性能差。最终采用了固定长度 20（用 1D RoPE 位置编码适配不同频率）
- **多图像 + 自定义掩码**：训练太慢、显存太高。预训练阶段最终只用单张静态图
- **MoE Flow Transformer**：NaN loss，收敛慢。最终共享所有 Flow Transformer 参数，只通过 action-specific AdaLN 做区分

这些"失败经验"对后来者是极其宝贵的设计参考。

---

## 六、对你的硬件适配评估

✅ **4070 Ti Super 16GB 完美适配 FLOWER**
- 推理：~1.85GB（极低）
- 微调：~10GB（低于你的 16GB 上限）
- 预训练：200 H100h → 在 RTX 4070 Ti Super 上约 ~600-800 小时（~1个月）——预训练仍然不便宜，但**在单卡上从零预训练一个 950M 的 VLA 是可行的**

这是目前最适合你硬件的开源 VLA 预训练方案。

## PDF

[[FLOWER.pdf]]
