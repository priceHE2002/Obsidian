---
tags:
  - 论文
  - 扩散模型
  - Transformer
  - DiT
  - 可扩展生成模型
created: 2026-06-30
paper_title: "Scalable Diffusion Models with Transformers"
paper_authors: "William Peebles, Saining Xie"
paper_year: 2022
paper_venue: "ICCV 2023 (Oral)"
paper_citations: "~4,000+"
paper_url: "https://arxiv.org/abs/2212.09748"
github: "https://github.com/facebookresearch/DiT"
---

# DiT

**Scalable Diffusion Models with Transformers**
*William Peebles, Saining Xie / UC Berkeley → NYU + Meta AI | ICCV 2023 (Oral) | arXiv: 2212.09748*

> **Pitch**: 证明了 "U-Net 不是扩散模型的必需品——Transformer 更好"。将扩散模型的骨干从 U-Net 替换为 ViT，通过 AdaLN（Adaptive Layer Normalization）注入条件信息。关键发现：**DiT 遵循清晰的 scaling law**——Gflops 越高（通过堆叠层或减小 patch size），FID 越低，且未见饱和。ImageNet 256×256 下 DiT-XL/2 FID **2.27**（当时 SOTA）。DiT 是 [[π0]] Action Expert、[[GR00T N1]] System 1、[[FLOWER]] Flow Transformer、[[Stable Diffusion 3]] 的架构蓝本。

---

## 一、Background / Core Idea

### 1.1 U-Net 的支配地位

在 DiT 之前，扩散模型的骨干几乎被 U-Net 垄断——从 [[DDPM]]（2020）到 Improved DDPM、ADM（Diffusion Models Beat GANs, 2021），U-Net 一直是去噪网络的标准选择。尽管 Dhariwal & Nichol（2021）对 U-Net 架构做了消融研究和改进（如自适应归一化层用于注入条件信息、通道数调整），但 U-Net 的高层设计基本保持不变。

U-Net 的好处来自其归纳偏置：下采样-上采样的对称结构天然保留局部-全局信息。

### 1.2 核心问题

**U-Net 对扩散模型是否不可或缺？**

在 NLP 和视觉领域，Transformer 已经在规模化（scaling law）上展现了远超 CNN 的能力——参数越多性能越好，且未见饱和。DiT 的工作假设是：**扩散的去噪过程在本质上不需要 U-Net 的卷积归纳偏置，只需要一个对输入分布建模足够灵活的神经网络**。

### 1.3 DiT 的洞见

DiT 的核心结论：**U-Net 不是必备品，Transformer 更好。**

- Transformer 的 scaling law 在扩散模型中同样成立
- ViT 的 patchify 操作天然适合处理 latent space 的 2D 特征图
- AdaLN 比 cross-attention 更适合扩散模型的条件注入

---

## 二、Method / Architecture / Technical Contribution

### 2.1 整体架构

DiT 基于 **Latent Diffusion Model (LDM)** 框架：

```
RGB Image (256×256×3) → VAE Encoder → Latent z (32×32×4)
→ Patchify → Token Sequence → DiT Blocks → Linear Decoder → Noise/Covariance Prediction
→ VAE Decoder → Image
```

VAE 来自 Stable Diffusion（下采样因子 8，把 256×256×3 图像压缩为 32×32×4 latent）。**DiT 在 latent space 中做扩散**。

### 2.2 Patchify：输入处理

DiT 的第一层是"patchify"——将空间 latent 表示 $z$（32×32×4）转换为 token 序列：

$$T = (I/p)^2$$

其中 $I$ 是 latent 的空间宽高（32），$p$ 是 patch size 超参数。

| Patch Size | Token 数 | 相对 Gflops |
|------------|---------|-------------|
| **p=2** | 256 | 最高（细节最多） |
| p=4 | 64 | 中等 |
| p=8 | 16 | 最低 |

Patch size 改变 token 数但**几乎不改变参数量**，它是 DiT 设计空间中控制计算量的独立维度。DiT 的实验包含了 p=2, 4, 8。

### 2.3 DiT Block 设计：四种条件注入方案

**（一）In-context conditioning（输入上下文条件化）：**
将 $t$ 和 $c$（类别标签）的 embedding 作为两个额外 token 拼接到输入序列中。不修改 ViT Block。最简单但效果最差。

**（二）Cross-attention block（交叉注意力）：**
在自注意力层之后增加一个多头的交叉注意力层，用条件 embedding 做 query。增加约 15% Gflops。

**（三）AdaLN（Adaptive Layer Normalization）——论文推荐方案：**
替换标准 LayerNorm 为 adaLN：

$$\text{AdaLN}(h, c) = \gamma(c) \cdot \text{LN}(h) + \beta(c)$$

其中 $\gamma$ 和 $\beta$ 由一个小 MLP 从时间步 $t$ 和类别标签 $c$ 的 embedding 之和回归得到。

优点：
- 计算量最小（几乎不增加 Gflops）
- 参数高效（无需额外 cross-attention 权重）
- 效果优——与 cross-attention 相当

缺点：
- 所有 token 共享同样的 $\gamma, \beta$——无法对 token 分别做条件调制

**（四）AdaLN-Zero（论文推荐中的最优方案）：**
在 AdaLN 的基础上，额外回归 $\alpha$ 参数，初始化为零，用于残差连接前缩放：

$$\text{输出} = \alpha(c) \cdot \text{DiTBlock}_{\text{adaln}}(\text{输入}) + \text{输入}$$

- 初始时将 $\alpha$ 设为零（MLP 输出零向量），整个 DiT Block 行为为**恒等函数**
- 训练从"无条件"开始，逐步学习条件调制
- **训练最稳定，效果最好**——FID 约是 In-context 的一半

### 2.4 四种方案对比

| 方案 | Gflops 开销 | 效果（FID） | 训练稳定性 |
|------|-----------|------------|-----------|
| In-context | ~0% | 差（~8） | 一般 |
| Cross-attention | +15% | 中等（~4.5） | 好 |
| **AdaLN** | **~0%** | **好（~3.5）** | **好** |
| **AdaLN-Zero** | **~0%** | **最佳（~3.0）** | **最佳** |

### 2.5 模型规模配置

| 模型 | Layers | Hidden dim | Heads | Gflops (p=4) | 参数量 |
|------|--------|-----------|-------|-------------|--------|
| DiT-S | 12 | 384 | 6 | 1.4 | ~33M |
| DiT-B | 12 | 768 | 12 | 5.6 | ~130M |
| DiT-L | 24 | 1024 | 16 | 19.7 | ~458M |
| DiT-XL | 28 | 1152 | 16 | 29.1 | ~675M |

配置跟随 ViT 的设计（S/B/L 与标准 ViT 一致，XL 是新增的最大配置）。

### 2.6 AdaLN 的数学细节

AdaLN-Zero Block 的详细前向流程：

```
1. 输入 x (T×d)
2. t_embed + c_embed → MLP(3×2d) → [γ₁, β₁, α₁, γ₂, β₂, α₂]
   （每个参数都是 d 维向量）
3. x = x + α₁ · Attn(LN_γ₁,β₁(x))    # 自适应 Attention
4. x = x + α₂ · MLP(LN_γ₂,β₂(x))     # 自适应 MLP
```

其中 $\alpha_1, \alpha_2$ 初始化为零，确保 Block 初始为恒等函数。

### 2.7 Classifier-free Guidance（CFG）

DiT 使用标准的 classifier-free guidance：

$$\hat{\varepsilon}_\theta(x_t, c) = \varepsilon_\theta(x_t, \varnothing) + s \cdot (\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \varnothing))$$

其中 $s > 1$ 控制 guidance 强度。训练时随机丢弃 $c$（替换为 learned null embedding）。

DiT-XL/2 使用 CFG 后 FID 从 9.62 降至 **2.27**（s=1.50）。

### 2.8 训练配置

| 超参数 | 值 |
|--------|-----|
| Optimizer | AdamW |
| Learning rate | 1e-4（恒定） |
| Weight decay | 0 |
| Batch size | 256 |
| EMA decay | 0.9999 |
| Data augmentation | 仅水平翻转 |
| No warmup | 不需要（训练高度稳定） |
| No regularization | 不需要（dropout、weight decay 等） |

DiT 的训练稳定性值得注意：没有学习率 warmup、没有正则化、所有配置共用相同的超参数——这在 Transformer 训练中非常罕见。

---

## 三、Experiments and Key Findings

### 3.1 Scaling Law 的核心发现

**DiT 的 Gflops 与 FID 存在强负相关（correlation = -0.93）：**

Gflops 越高（更多层、更大隐藏维度、更小的 patch size），FID 越低。这一关系在 12 个不同配置上都成立。

关键发现：
- **参数量不是最佳预测指标**——DiT-S/2 和 DiT-B/4 的参数量不同但 Gflops 相似 → FID 相似
- **patch size 是独立于参数量的 Gflops 控制维度**——减小 patch size 增加 token 数，提升模型容量
- **未见饱和**——即使到 675M 参数的 DiT-XL/2，scaling 收益仍持续

### 3.2 ImageNet 256×256 结果

| 模型 | FID↓ | sFID↓ | IS↑ | Precision | Recall |
|------|------|-------|-----|-----------|--------|
| BigGAN-deep | 6.95 | 7.36 | 171.4 | 0.87 | 0.28 |
| StyleGAN-XL | 2.30 | 4.02 | 265.12 | 0.78 | 0.53 |
| ADM-G | 4.59 | 5.25 | 186.70 | 0.82 | 0.52 |
| LDM-4-G (cfg=1.50) | 3.60 | - | 247.67 | 0.87 | 0.48 |
| **DiT-XL/2** | 9.62 | 6.85 | 121.50 | 0.67 | 0.67 |
| **DiT-XL/2-G (cfg=1.25)** | 3.22 | 5.28 | 201.77 | 0.76 | 0.62 |
| **DiT-XL/2-G (cfg=1.50)** | **2.27** | **4.60** | **278.24** | **0.83** | **0.57** |

DiT-XL/2 在 FID 2.27 下超越所有先前扩散模型和 StyleGAN-XL。使用 CFG 后 recall 高于 LDM（0.57 vs 0.48），表示更好的模式覆盖。

**FID 2.27 在当时的含义**：这是 SOTA，超越之前所有的扩散模型和 GAN 模型。DiT 证明 Transformer 可以在扩散图像生成中达到甚至超越精心设计的 CNN 架构。

### 3.3 ImageNet 512×512 结果

| 模型 | FID↓ | IS↑ |
|------|------|------|
| ADM-G, ADM-U | 3.85 | 221.72 |
| **DiT-XL/2-G (cfg=1.50)** | **3.04** | **240.82** |

在更高分辨率（512×512）上，DiT-XL/2 的 latent 大小为 64×64，patch size 2 → 1024 tokens（524.6 Gflops），仍比 ADM（1983 Gflops）更高效。

### 3.4 Gflops vs 采样计算

DiT 研究了模型计算量（训练后固定）与采样计算量（可调整，通过增加采样步数）的关系：

- 小模型（如 DiT-S/8）即使使用 1000 步采样，也**无法达到大模型（如 DiT-XL/2）128 步采样的表现**
- DiT-L/2 + 1000 步（80.7 Tflops）的 FID 仍不如 DiT-XL/2 + 128 步（15.2 Tflops）
- 结论：**增加采样计算无法弥补模型计算量的不足**

### 3.5 计算效率（图 9）

DiT-XL/2 使用相同的总训练计算量比小模型更高效：
- 在总训练 Gflops 固定时，更大的 DiT 模型总能达到更低的 FID
- 即使在总计算量低的时候（<10^10 Gflops），大模型的表现也不输给小模型

---

## 四、Limitations and Challenges

1. **仅验证 class-conditional ImageNet**：原始论文仅在 class-conditional ImageNet 生成上验证，未涉及文本条件生成、多模态条件等更复杂的 setting。

2. **LDM 框架的依赖**：DiT 仍然依赖 LDM 的预训练 VAE（来自 Stable Diffusion）。VQ-VAE 的质量可能成为瓶颈。

3. **低分辨率任务无优势**：与 U-Net 相比，DiT 在低分辨率任务（如 CIFAR-10）上没有明显优势——Transformer 的 scaling law 在"小"问题时无法展现。

4. **计算量高于 U-Net**：在参数数相同时，DiT 的 Gflops 一般高于 U-Net（因为自注意力的 O(n²) 复杂度）。但 DiT 的可扩展性远超 U-Net。

5. **Patch size 的边界**：patch size 越小 token 越多，自注意力计算量是 O(T²)。当 T 很大时（如 512×512 下 p=2 的 1024 tokens），计算开销仍然显著。

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 关联 | 时间 |
|---------|------|------|
| **Stable Diffusion 3 (MMDiT)** | 核心架构从 U-Net 切换为 DiT（双模态 DiT） | 2024 |
| **Sora (OpenAI)** | 视频生成模型，基于 DiT 扩展到时空维度（Video DiT） | 2024 |
| **[[π0]] Action Expert (300M)** | **Action Expert = DiT**——输入 VLM 语义 embedding + 本体感觉 + 噪声动作，通过 AdaLN 注入条件，[[Flow Matching]] 去噪输出动作 | 2024 |
| **[[GR00T N1]] System 1** | **DiT + Flow Matching 动作头**——~50Hz 动作生成，被称为"Flow Matching Transformer" | 2025 |
| **[[FLOWER]] Flow Transformer** | **DiT 变体**——AdaLN 条件化 + Rectified Flow | 2024 |
| **[[Cosmos Policy]]** | **Video DiT**——从视频生成扩展到策略预测 | 2025 |

### AdaLN 为什么是 VLA 动作生成的标准条件机制

在 VLA 系统中，VLM（如 [[OpenVLA]]）输出视觉-语言理解的语义 embedding，这个 embedding 需要"注入"到动作生成头中。AdaLN 的参数效率（几乎不增加 Gflops）和效果（FID 上优于 cross-attention）使其成为自然选择。

具体机制：
- VLM 输出的 embedding → 小 MLP → 回归 $\gamma, \beta, \alpha$（每层均不同）
- $\gamma, \beta$ 调制 LayerNorm（调整特征分布的 scale 和 shift）
- $\alpha$ 缩放残差（初始为零，确保训练稳定性）
- 所有 token 共享相同的调制——在动作生成中可能不如 cross-attention 灵活，但计算开销小

### DiT 的 AdaLN vs LayerNorm vs RMSNorm 演进

| 归一化方式 | 条件注入 | 参数效率 | 在扩散/流匹配中的使用 |
|-----------|---------|---------|-------------------|
| LayerNorm | 无（标准 LN） | - | 不适合作 backbone |
| AdaLN (DiT) | $\gamma, \beta$ 从条件回归 | ✅ 高 | [[π0]], [[GR00T N1]] |
| AdaLN-Zero (DiT) | AdaLN + 零初始化残差 | ✅ 最高 | DiT-XL/2, [[FLOWER]] |
| RMSNorm | 无缩放偏移 | 极简 | 某些后期工作（如 Llama 风格） |

---

## 六、Implications for You / Hardware Compatibility

| 维度 | 评价 |
|------|------|
| 训练硬件 | ⚠️ DiT-XL/2（675M）需 TPU v3-256 pod（约 8×A100-80GB）。DiT-S/B 可在 16GB GPU 上训练 |
| 推理硬件 | ✅ DiT-L (~458M) 推理约 8-12GB VRAM。DiT-B (~130M) 约 4-6GB。16GB GPU 可运行除 XL 外的所有配置 |
| 条件注入机制 | ✅ AdaLN 是理解 [[π0]] Action Expert、[[GR00T N1]] 内部机制的前提 |
| 对 VLA 的意义 | ✅ **核心架构**——VLA 动作头的主流 backbone 选择 |
| 代码复杂度 | ✅ 相对简洁：在标准 ViT 基础上增加了 AdaLN 调制层。开源实现丰富 |

**核心启示：**
1. **DiT 替代了 U-Net 作为扩散（和 [[Flow Matching]]）骨干的标准架构**——阅读 VLA 动作头代码时，看到 DiT + AdaLN 是标配
2. **AdaLN 是关键设计**：理解 $\gamma, \beta, \alpha$ 如何调制 LayerNorm，能帮你理解 [[π0]] 的 Action Expert 内部机制
3. **可扩展性验证**：DiT 证明了动作头的性能也受 scaling law 支配。更大的动作头 = 更好的动作质量
4. **条件注入方式的选择**：对机器人动作生成，AdaLN 是最优选（参数高效、计算轻量、效果优秀）

---

## PDF

[[DiT 原文.pdf]]
