---
tags:
  - 论文
  - VLA
  - 参数高效微调
  - PEFT
  - LoRA
created: 2026-06-30
paper_title: "VLA-Adapter: An Effective Paradigm for Tiny-Scale VLA Model"
paper_authors: "Yihao Wang et al. (OpenHelix Team)"
paper_year: 2025
paper_venue: "arXiv 2025.9"
paper_url: "https://arxiv.org/abs/2509.09372"
github: "https://github.com/OpenHelix-Team/VLA-Adapter"
---

# VLA-Adapter

**VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model**
*OpenHelix Team | 2025.9 | arXiv 2509.09372*

> **为什么要让 VLM 内部去学怎么输出动作？把 VLM 冻住，在外面加一个精巧的 Bridge Attention + 轻量 Policy Network，效果反而更好。** 0.5B 参数的模型在 LIBERO 上达到 98.5%——超越了 7B 的 OpenVLA-OFT 和 3B 的 π0。**这是目前最适合你 16GB 显卡的 SOTA VLA 训练方案。**

---

## 一、核心洞察：VLM 不应该被"污染"

### 1.1 当前 VLA 微调的问题

当前 VLA 微调（如 OpenVLA 的 LoRA）在 VLM 内部训练额外的参数来输出动作。但这有一个微妙的问题：

**VLM 在互联网数据上学到的丰富语义表征，在被"微调为动作预测器"的过程中可能被"稀释"。** 预训练阶段学到的"苹果是红色的、圆形的、可以吃"等知识，可能被覆盖为"当像素分布像 X 时往 Y 方向移动"。

### 1.2 VLA-Adapter 的解决思路

> **冻住 VLM，不要碰它。在它外面加一个聪明的"翻译器"，把 VLM 的语义表征翻译成动作。**

这类似于 NLP 中的 Adapter 微调——不修改预训练模型，只在小模块中学习下游任务。

---

## 二、核心架构

### 2.1 整体设计

```
┌──────────────────────────────────────────┐
│           冻住的 VLM (Qwen2.5-0.5B)       │
│                                           │
│  视觉 tokens → [VLM layers] → hidden states│
│  文本 tokens →                 → hidden states│
└─────────────┬───────────────┬────────────┘
              │               │
     Raw Features      ActionQuery Features
     (中间层)           (深层)
              │               │
              └───────┬───────┘
                      ↓
              Bridge Attention
                      ↓
              Policy Network
              (轻量 Transformer)
                      ↓
                 连续动作
```

### 2.2 Bridge Attention —— 核心创新

这是 VLA-Adapter 最精妙的设计。Bridge Attention 解决了"从哪层取特征、怎么融合"的问题。

**三个子模块：**

1. **Raw-to-Action 交叉注意力**：VLM 中间层的原始 hidden states → 动作空间
2. **ActionQuery-to-Action 交叉注意力**：深层可学习的 ActionQuery tokens → 动作空间
3. **自注意力融合**：将上述两种特征融合

**关键设计——可学习的门控参数 $\tanh(g)$**：

$$\text{fused} = \text{SelfAttn}(\text{RawFeat} \cdot \tanh(g) + \text{ActionQueryFeat})$$

$g$ 控制了注入多少"原始 VLM 特征"。当 $g$ 小 → 更多依赖 ActionQuery（任务特定）；当 $g$ 大 → 更多依赖原始特征（更通用）。

### 2.3 Condition Exploration（特征层选择）

论文对不同层的特征做了系统分析：

| 特征来源 | 适用性 |
|---------|--------|
| **中间层 Raw Features** | 最适合动作生成（语义丰富而不过度专业化）|
| **深层 ActionQuery Features** | 蕴含更丰富的多模态细节 |
| **单层特征** | 不如多层特征 |
| **多层特征组合** | **最优**（论文的最终选择）|

这个发现与 [[FLOWER]] 的"中间层融合"高度一致——两篇独立的论文都得出"中间层比最终层更适合机器人控制"。

---

## 三、训练效率

### 3.1 惊人的参数效率

| | OpenVLA-OFT | VLA-Adapter |
|---|---|---|
| VLM 骨干 | 7B | **0.5B** (14×小) |
| 可训练参数 | ~100M (LoRA) | **~4.7M** (仅 Bridge + Policy) |
| 微调 GPU 小时 | 304 | **8** (38×快) |
| 训练 VRAM (batch=1) | ~62GB | **~9.6GB** |
| 训练 VRAM (batch=16) | 不可能 | **~24.7GB** |
| 推理吞吐 | 71.4 Hz | **219.2 Hz** (3×快) |
| 训练时间 (LIBERO-Object) | ~12h (A100) | **~8h (单 GPU)** |

### 3.2 "8 小时从零到 SOTA"

在单张消费级 GPU 上（RTX 3090/4090），你可以在 8 小时内微调出一个在 LIBERO 上达到 97%+ 成功率的 VLA 策略。这是 VLA 研究中前所未有的可及性。

---

## 四、核心结果

### 4.1 LIBERO 基准（最主要的测试平台）

| 模型 | 参数量 | Spatial | Object | Goal | Long | **Average** |
|------|--------|---------|--------|------|------|------------|
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | 3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| GR00T N1 | 2B | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| SmolVLA | 2.2B | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| VLA-OS (0.5B 基线) | 0.5B | 87.0 | 96.5 | 92.7 | 66.0 | 85.6 |
| **VLA-Adapter** | **0.5B** | 97.8 | 99.2 | 97.2 | 95.0 | **97.3** |
| **VLA-Adapter-Pro** | **0.5B** | **99.6** | **99.6** | **98.2** | **96.4** | **98.5** |

### 4.2 关键洞察：Bridge > Scale

VLA-Adapter (0.5B) 打败 OpenVLA-OFT (7B) 的核心原因不是参数多，而是 **Bridge Attention 有效地将 VLM 的语义表征"翻译"成了动作**。

这揭示了一个重要原则：**如何连接 VLM 表征和动作空间，比 VLM 有多大更重要。**

---

## 五、与你的硬件的适配评估

✅ **VLA-Adapter 是目前最适合 4070 Ti Super 16GB 的 SOTA VLA 训练方案**

| 配置 | VRAM 需求 | 你的 16GB 适配 |
|------|----------|:---:|
| 标准版 (lora_rank=64) | ~9.6GB | ✅ **完美** |
| Pro 版 | ~17.6GB | ⚠️ batch_size=1 可能勉强 |
| 训练时间 | ~8h/任务 | ✅ 合理 |

**建议路线**：用 VLA-Adapter 的标准配置（Qwen2.5-0.5B + Bridge Attention + LoRA rank=64）在 LeRobot 的 LIBERO 环境上开始你的第一个 VLA 训练实验。

---

## 六、与其他小模型方案的对比

| | VLA-Adapter | FLOWER | SmolVLA |
|---|---|---|---|
| 核心策略 | 冻 VLM + Bridge + Policy | 砍 VLM 层 + 扩 FlowT | 全量微调小 VLM |
| 参数 | 0.5B | 950M | 450M |
| LIBERO | **98.5%** | ~97% | 88.8% |
| 训练 VRAM | **~10GB** | ~10GB | ~12GB |
| 推理 VRAM | ~5GB | ~2GB | ~5GB |
| 开源 | ✅ | ✅ | ✅ |

VLA-Adapter 在 LIBERO 上是最优的，但 FLOWER 在跨形态泛化上更好（因为它做了大规模预训练）。

---

## 七、启示

1. **"冻住 VLM + 外部策略网络"可能比"微调 VLM 输出动作"更好**——这对架构设计有深远影响
2. **Bridge Attention 的核心思想是"特征桥接"**——不只是取 VLM 的输出，而是设计精巧的注意力机制来连接两个不同的表示空间
3. **0.5B 能做到 98.5%，说明数据效率和架构设计比规模重要得多**
4. **ActionQuery tokens** 是一个值得尝试的设计——它们充当"中间语言"，在 VLM 表征和动作之间做翻译

## PDF

[[VLA-Adapter.pdf]]
