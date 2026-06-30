---
tags:
  - 论文
  - 扩散模型
  - Transformer
created: 2026-06-30
paper_title: "Scalable Diffusion Models with Transformers"
paper_authors: "William Peebles, Saining Xie"
paper_year: 2022
paper_venue: "ICCV 2023 (Oral)"
paper_citations: "~4,000+"
paper_url: "https://arxiv.org/abs/2212.09748"
---

# DiT

**Scalable Diffusion Models with Transformers**
*UC Berkeley → NYU | ICCV 2023 (Oral) | arXiv: 2212.09748*

> 证明了"U-Net 不是扩散模型的必需品——Transformer 更好"。将扩散模型的骨干从 U-Net 替换为 ViT，通过 AdaLN 注入条件信息。DiT 是 $\pi_0$ Action Expert、GR00T N1 System 1、FLOWER Flow Transformer 的架构蓝本。

---

## 一、研究背景与动机

扩散模型领域几乎被 U-Net 架构统治——从 DDPM 到 Improved DDPM、Stable Diffusion，U-Net 一直是去噪网络的标准选择。然而，在 NLP 和视觉领域，Transformer 已经在规模化（scaling）上展现了远超 CNN 的能力。一个自然的问题是：**U-Net 对扩散模型是否不可或缺？**

DiT 的核心洞见：扩散的去噪过程在本质上不需要 U-Net 的归纳偏置（下采样-上采样对称结构），只需要一个对输入分布建模足够灵活的 Transformer。并且 Transformer 的 **scaling law**——参数越多性能越好——在扩散模型中同样成立。

## 二、核心方法

**整体架构：** 输入 noise latent + 时间步 t + 条件 c → 预测噪声 / v-prediction

**AdaLN (Adaptive Layer Normalization)：** 不使用 cross-attention，而是通过调制 LayerNorm 参数注入条件信息：

$$\text{AdaLN}(h, c) = \gamma(c) \cdot \text{LN}(h) + \beta(c)$$

其中 $\gamma$ 和 $\beta$ 由小的 MLP 从时间步 t 和类别标签 c 回归得到。

**四种条件注入方案对比：**

| 方案 | 机制 | 参数效率 | FID-计算量平衡 |
|------|------|---------|--------------|
| In-context | 在输入序列中拼接条件 token | 低 | 一般 |
| Cross-attention | 标准 Transformer cross-attention | 中等 | 灵活但计算高 |
| **AdaLN** | 调制 LayerNorm 的 $\gamma, \beta$ | **高** | **最佳** |
| AdaLN-Zero | AdaLN + 初始化为零 | 高 | 训练最稳定 |

**缩放实验 (Scaling Study)：**

| 模型 | 参数 | Gflops | ImageNet 256x256 FID |
|------|------|--------|---------------------|
| DiT-S | ~33M | ~6 | 5.5 |
| DiT-B | ~130M | ~19 | 4.2 |
| DiT-L | ~458M | ~62 | 3.0 |
| DiT-XL/2 | ~675M | ~119 | **2.27** |

Gflops 越多 → FID 越低——证明了扩散 Transformer 可规模化。

## 三、关键实验与发现

- **DiT-XL/2 在 ImageNet 256x256 上 FID 2.27**：超越当时所有扩散模型（包括 U-Net 架构）
- **Scaling law 成立**：计算量/Gflops 和 FID 存在清晰的倒数关系
- **AdaLN-Zero 是最优方案**：初始化为零让模型从"无条件"开始训练，逐步学习条件调制，训练最稳定
- **patch size 的影响**：patch size 越小（ViT-S/2 vs ViT-S/8），计算量越大但性能越好

## 四、局限性与后续影响

**局限：**
- 原始论文仅在 ImageNet class-conditional 生成上验证，未涉及文本条件或多模态条件
- 与 U-Net 相比，在低分辨率任务上没有明显优势
- 计算量比 U-Net 更大（但也更可扩展）

**后续影响：**
- Stable Diffusion 3 —— 核心架构从 U-Net 切换为 DiT（MMDiT）
- Sora —— OpenAI 视频生成模型，基于 DiT 扩展到时空维度
- $\pi_0$, GR00T N1, FLOWER —— 全部采用 DiT 作为动作生成骨干

## 五、VLA/机器人研究中的角色

DiT 是 VLA 动作生成的标准骨干架构：

- **$\pi_0$ 的 Action Expert (300M) = DiT** —— 输入 VLM 语义 embedding（作为条件，通过 AdaLN 注入）+ 本体感觉 + 噪声动作 → Flow Matching 去噪输出动作
- **GR00T N1 的 System 1 = DiT** —— Flow Matching Transformer，~50Hz 动作生成
- **FLOWER 的 Flow Transformer = DiT 的变体** —— AdaLN 条件化 + Rectified Flow
- **Cosmos Policy 的骨干 = Video DiT** —— DiT 扩展到视频生成 + 策略预测
- AdaLN 是理解 $\pi_0$ Action Expert 内部机制的前提——VLM 输出的语义 embedding 通过 AdaLN 调制 DiT 的每一层

## 六、对你的启示

- **DiT 替代了 U-Net 作为扩散（和 Flow Matching）骨干的标准架构**——阅读 VLA 动作头代码时，看到 DiT + AdaLN 是标配
- **AdaLN 是关键设计**：理解 $\gamma, \beta$ 如何调制 LayerNorm，能帮你理解 $\pi_0$ 的 Action Expert 内部机制
- **可扩展性验证**：DiT 证明了动作头的性能也受 scaling law 支配
- **条件注入方式的选择**：对于机器人动作生成，AdaLN 是最优选（参数高效、计算轻量）
- DiT-L 推理约需 8-12GB VRAM，16GB GPU 可运行

## PDF

[[DiT.pdf]]
