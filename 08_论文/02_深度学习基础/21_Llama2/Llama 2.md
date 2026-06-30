---
tags:
  - 论文
  - LLM
  - 开源
  - VLA骨干
  - Transformer架构
created: 2026-06-30
paper_title: "Llama 2: Open Foundation and Fine-Tuned Chat Models"
paper_authors: "Hugo Touvron, Louis Martin, Kevin Stone et al. (Meta AI)"
paper_year: 2023
paper_venue: "arXiv 2307.09288"
paper_citations: "~20,000+"
paper_url: "https://arxiv.org/abs/2307.09288"
github: "https://github.com/facebookresearch/llama"
---

# Llama 2

**Llama 2: Open Foundation and Fine-Tuned Chat Models**
*Hugo Touvron, Louis Martin, Kevin Stone et al. | Meta AI | arXiv 2307.09288*

> OpenVLA 的骨干就是 Llama 2 7B——它的每一个架构选择（RMSNorm、SwiGLU、RoPE、GQA）都直接决定了 VLA 的推理行为和效率。Llama 2 的开源（商业友好协议）是整个开源 VLA 生态的根基，没有 Llama 2 就没有 OpenVLA。

---

## 一、Background / Core Idea

### 1.1 背景：闭源大模型 vs 开源大模型的鸿沟

2023 年初的情况：
- GPT-3.5 / GPT-4 / Claude 等闭源模型在对话能力上远超开源模型
- **Llama 1**（2023 年 2 月）首次证明开源 LLM 可以达到接近 SOTA 的性能，但受限于：
  - **研究限制许可**（非商业友好）
  - **缺乏对话微调版**（只有基础模型）
  - **仅 1T tokens 预训练**、2048 上下文长度、无 GQA

### 1.2 Llama 2 的目标

1. **开源高性能基础模型系列**：7B、13B、34B（未发布）、70B 四个规模
2. **开源经过 RLHF 微调的对话模型**（Llama 2-Chat）：在多数基准上接近 GPT-3.5
3. **商业友好许可**：允许商业使用——直接催生了 OpenVLA 等下游应用的商业化可能
4. **安全对齐**：提供全面的红队测试和安全微调方法论

### 1.3 训练数据与规模

| 配置 | Llama 1 | Llama 2 |
|------|---------|---------|
| 训练 tokens | 1.0T | **2.0T** (+100%) |
| 上下文长度 | 2048 | **4096** (2x) |
| 数据来源 | 公开数据 | 更新版公开数据，增采样事实性来源 |
| 分词器 | SentencePiece BPE (32K) | 同 Llama 1 |
| 模型规模 | 7B/13B/33B/65B | 7B/13B/34B(未发布)/70B |
| GQA | 无 | 34B/70B 版本使用 |
| 全局 batch size | — | 4M tokens |
| 优化器 | AdamW | AdamW ($\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-5}$) |
| 学习率调度 | cosine decay | cosine decay, warmup 2000 steps, 最终 LR = 峰值 10% |
| 权重衰减 | — | 0.1 |
| 梯度裁剪 | — | 1.0 |

### 1.4 预训练碳足迹

| 模型 | GPU 小时数 | 功耗 (W) | CO₂ 排放 (tCO₂ eq) |
|:-:|:-:|:-:|:-:|
| 7B | 184,320 | 400 | 31.22 |
| 13B | 368,640 | 400 | 62.44 |
| 34B | 1,038,336 | 350 | 153.90 |
| 70B | 1,720,320 | 400 | 291.42 |
| **总计** | **3,311,616** | — | **539.00** |

所有排放由 Meta 的可持续发展计划直接抵消。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 整体架构概述

Llama 2 采用 **Decoder-only Transformer**，核心组件链为：

```
Input → Embedding → [Decoder Block × N] → RMSNorm → Linear → Softmax
```

每个 Decoder Block：
```
x → RMSNorm → Self-Attention (RoPE + GQA) → Residual + x
  → RMSNorm → SwiGLU FFN → Residual + x
```

### 2.2 RMSNorm (Pre-Norm)

**公式：**

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

其中 $\gamma \in \mathbb{R}^d$ 是可学习的缩放参数，$\epsilon$ 是数值稳定项。

**为什么用 Pre-Norm 而非 Post-Norm：**

| 方面 | Post-Norm（原始 Transformer） | Pre-Norm（Llama 2） |
|------|:-:|:-:|
| 归一化位置 | 每个子层之后 | 每个子层**之前** |
| 梯度流动 | 容易梯度消失（深层次） | **梯度恒等路径**（输出 = $x+\text{SubLayer}(\text{Norm}(x))$） |
| 训练稳定性 | 需要 warmup + 小心调参 | 更稳定，允许更高学习率 |
| 训练速度 | 较慢 | 约快 1.5-2x |
| 最终性能 | 理论上界更高（有人论证） | 实践中更好 |

**$x + \text{SubLayer}(\text{RMSNorm}(x))$** 的残差连接确保梯度可以通过恒等路径畅通传播，这是训练 70B 模型的关键设计。

**为什么 RMSNorm 而非 LayerNorm**：RMSNorm 去除了 LayerNorm 的均值减法操作（只做均方根缩放），计算量减少约 15-20%，且在实验中对 LLM 性能没有明显影响。详见 [[RMSNorm]].

### 2.3 SwiGLU 激活函数

SwiGLU（Swish-Gated Linear Unit）在 Shazeer (2020) 中提出：

$$\text{SwiGLU}(x) = \underbrace{(xW_g \odot \sigma(xW_g + b_g))}_{\text{门控路径}} \otimes \underbrace{(xW_u + b_u)}_{\text{上投影路径}}$$

其中 $\sigma$ 是 SiLU/Swish 激活：$\sigma(x) = x \cdot \text{sigmoid}(\beta x)$，$\odot$ 为逐元素乘法。

**实际实现**：FFN 包含**三个**权重矩阵：
- $W_g$（门控，$d \to \frac{8}{3}d$）
- $W_u$（上投影，$d \to \frac{8}{3}d$）
- $W_d$（下投影，$\frac{8}{3}d \to d$）

**为什么 SwiGLU 优于其他激活函数：**

| 激活函数 | 公式 | 参数量 | K 步后困惑度 | 特点 |
|---------|------|:-:|:-:|------|
| ReLU | $\max(0,x)$ | 2 个权重矩阵 | 基线 | 简单但可能有死神经元 |
| GELU | $x\Phi(x)$ | 2 个权重矩阵 | 略好 | 平滑近似 ReLU |
| Swish | $x\sigma(x)$ | 2 个权重矩阵 | 与 GELU 接近 | 非单调、无上界 |
| **SwiGLU** | $(xW_g \odot \sigma(xW_g)) \cdot xW_u$ | **3** 个权重矩阵 | **显著更好** | 门控机制提供额外表达能力 |

**对 VLA 的影响**：SwiGLU 的梯度结构复杂（涉及门控路径的乘积），使得 **AdamW** 优化器成为必需——SGD 无法有效优化这类参数化。

### 2.4 RoPE (Rotary Position Embedding)

RoPE 的核心思想是将位置信息编码为**旋转矩阵**作用于 query 和 key：

对于位置 $m$ 处的 token，其 query 向量 $q$ 的变换为：

$$f_q(x_m, m) = R_{\Theta,m} \cdot W_q \cdot x_m$$

其中 $R_{\Theta,m}$ 是块对角旋转矩阵：

$$R_{\Theta,m} = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & \cdots & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & \cdots & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

其中 $\theta_i = \text{base}^{-2i/d}$（通常 base = 10000，即 $\theta_i = 10000^{-2i/d}$）。

**核心性质**：注意力分数 $q_m^\top k_n$ 只依赖于 $(m-n)$ 即相对位置——因为旋转矩阵满足 $R_{\Theta,m}^\top R_{\Theta,n} = R_{\Theta,m-n}$。

**为什么 RoPE 优于传统位置编码：**

| 方法 | 绝对/相对 | 外推能力 | 理论基础 | 可学习性 |
|------|:-:|:-:|:-:|:-:|
| 绝对位置编码 | 绝对 | **差**（无法处理超长序列） | 浅层 | 可学习 |
| T5 偏置 | 相对 | 中等 | 手工设计 | 不可学习 |
| ALiBi | 相对 | **好** | 线性偏置 | 不可学习 |
| **RoPE** | **相对** | **好** | **旋转矩阵** | **编码在 attention 中** |

**对 VLA 的影响**：OpenVLA 生成 7 个动作步 × 256 tokens = 1792 tokens 的动作序列。这些动作 token 之间的相对位置关系通过 RoPE 编码。如果修改动作序列长度（如 7→10 步），可能需要调整 RoPE 的 $\theta$ 参数（增大 base 值以改进外推）。

### 2.5 GQA (Grouped-Query Attention)

**三种注意力机制对比：**

| 机制 | Query Heads | Key/Value Heads | KV Cache 大小 | 推理内存 | 质量 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| **MHA** (Multi-Head Attention) | $h$ | $h$ | $h \times L \times d$ | 最大 | 最高 |
| **MQA** (Multi-Query Attention) | $h$ | 1 | $1 \times L \times d$ | 最小 | 有损失 |
| **GQA** (Grouped-Query Attention) | $h$ | $g$ | $g \times L \times d$ | 折中 | ≈ MHA |

Llama 2 中：7B/13B 使用 MHA（无 GQA），34B/70B 使用 GQA。

**为什么 GQA 对推理效果关键**：

KV Cache 在推理时存储所有层的 Key 和 Value 矩阵。对于 70B 模型：
- MHA: $h=64$, $d=128$ → 每层 KV Cache = $64 \times L \times 128$
- GQA (g=8): 每层 KV Cache = $8 \times L \times 128$ → **减少 8x**

在 VLA 场景（动作 token 序列 1792 tokens）下，KV Cache 的显存占用可能超过模型参数本身。

### 2.6 预训练 Base Model 评估

与开源模型的全面对比：

| 模型 | 参数量 | 代码 | 常识推理 | 世界知识 | 阅读理解 | 数学 | MMLU | BBH | AGI Eval |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| MPT | 7B | 20.5 | 57.4 | 41.0 | 57.5 | 4.9 | 26.8 | 31.0 | 23.5 |
| Falcon | 40B | 15.2 | 69.2 | 56.7 | 65.7 | 12.6 | 55.4 | 37.1 | 37.0 |
| Llama 1 | 65B | 30.7 | 70.7 | 60.5 | 68.6 | 30.8 | 63.4 | 43.5 | 47.6 |
| **Llama 2** | **70B** | **37.5** | **71.9** | **63.6** | **69.4** | **35.2** | **68.9** | **51.2** | **54.2** |

**与闭源模型比较**（Llama 2 70B）：

| 基准 | GPT-3.5 | GPT-4 | PaLM | PaLM-2-L | Llama 2 70B |
|:-:|:-:|:-:|:-:|:-:|:-:|
| MMLU (5-shot) | 70.0 | 86.4 | 69.3 | 78.3 | 68.9 |
| GSM8K (8-shot) | 57.1 | 92.0 | 56.5 | 80.7 | 56.8 |
| HumanEval (0-shot) | 48.1 | 67.0 | 26.2 | — | 29.9 |

Llama 2 70B 接近 GPT-3.5 但远落后于 GPT-4，差距主要在代码任务上。

### 2.7 SFT (Supervised Fine-Tuning)

**关键原则："Quality Is All You Need"**

- 收集 **27,540** 条高质量 SFT 数据（而非数百万条）
- 发现不同数据标注平台产生的质量差异显著影响下游性能
- SFT 输出质量在人工评估中与标注员书写的答案相当
- SFT 超参：cosine LR，峰值 $2\times10^{-5}$，权重衰减 0.1，batch size 64，序列长度 4096

### 2.8 RLHF 全流程

#### 2.8.1 奖励模型

**分离训练**：有用性（Helpfulness）和安全（Safety）的各一个 RM。
- 训练数据：**1,418,091** 对二元偏好比较（Meta 内部标注）
- 通过公式 $L_{\text{ranking}} = -\log(\sigma(r_\theta(x,y_c) - r_\theta(x,y_r) - m(r)))$ 训练
- $m(r)$ 是偏好等级的函数（significantly better / better / slightly better / negligibly better）

**关键发现：**
- 有用性和安全性存在张力：过于有用的回答可能不安全
- 分离训练解决了这一矛盾
- 在 70B 上，RM 在"显著更好"样本上的准确率 > 80%，"略微更好"样本上 ~55%

#### 2.8.2 PPO + Rejection Sampling

Llama 2-Chat 经过 5 轮迭代 RLHF（V1-V5）：
1. **Rejection Sampling**（V1-V4）：每个 prompt 采样 K 个输出，用奖励模型选最佳
2. **PPO**（V5+）：在 Rejection Sampling 检查点之上，使用标准 PPO 优化

最终奖励函数：

$$R(g|p) = \tilde{R}_c(g|p) - \beta D_{\text{KL}}(\pi_\theta(g|p)\|\pi_0(g|p))$$

KL 散度惩罚项 $\beta$ 防止奖励黑客（reward hacking）。

#### 2.8.3 Ghost Attention (GAtt)

解决多轮对话中系统指令被遗忘的问题：
1. 将系统指令（如"以拿破仑的身份回答"）拼接到每一轮的用户消息中
2. 训练时对历史轮次 token 设置 loss=0（只对最新回答计算梯度）
3. 推理时保持 20+ 轮的一致性

---

## 三、Experiments and Key Findings

### 3.1 人类评估结果

**有用性评估**（约 4,000 prompts，单轮 + 多轮，95% CI ±1-2%）：
- Llama 2-Chat 70B 优于所有开源对话模型
- 与 ChatGPT (GPT-3.5) 在某些基准上接近或持平
- 与 GPT-4 仍有显著差距

**安全性评估**（约 2,000 对抗性 prompts）：
- 安全 RLHF 显著提升了拒绝有害请求的能力
- 安全 RM 在识别不安全输出上准确率最高（64.5% 在 Meta Safety 测试集）

### 3.2 奖励模型缩放趋势

关键发现：**RM 尚未饱和**——更大的模型和更多数据持续提升 RM 准确率。70B RM 显著优于 7B/13B 版本，数据量增加持续带来收益。

### 3.3 训练损失

Llama 2 在 2T tokens 后训练损失仍未饱和（没有平台期），说明更多训练数据会持续带来收益（这一点被 Llama 3 的 15T+ tokens 验证）。

---

## 四、Limitations and Challenges

1. **上下文长度仅 4096**：显著限制了需要长上下文的 VLA 应用（如长轨迹推理）。对比 GPT-4 (128K)、Claude (200K)
2. **缺乏 MoE 架构**：密模型在相同 FLOPs 下容量不如混合专家模型。对比 Mixtral 8x7B (47B 参数，但推理效率约 12B)
3. **训练数据量相对有限**：2T tokens 已被后续模型大幅超越（Llama 3: 15T+, DeepSeek: 14T+）
4. **34B 模型未发布**：原因是"缺乏足够时间进行红队测试"——说明安全对齐的成本极高
5. **SentencePiece 32K 词表效率低**：非英语语言分词效率差，中文平均每字约 1.5 tokens
6. **70B 推理成本高**：需多 GPU 推理，在消费级硬件上不可行
7. **RLHF 的遗忘问题**：Rejection Sampling 迭代中 RLHF V3 丢失了作诗能力，说明 RLHF 可能导致灾难性遗忘

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 关系 |
|---------|------|
| **Llama 3** (Meta, 2024) | 直接继承架构，15T+ tokens 预训练，8B/70B/405B，GQA 拓展到全系列 |
| **CodeLlama** (Meta, 2023) | 基于 Llama 2 代码继续预训练 + LoRA |
| **OpenVLA** (2024) | **使用 Llama 2 7B 作为骨干**，替换 LM head 为动作 token 投影层 |
| **SmolVLA / BitVLA** | 继承 Llama decoder-only 架构但缩小模型维度 |
| **Mistral 7B** (2023) | 使用相似架构但引入 Sliding Window Attention |
| **Qwen 系列** (Alibaba, 2023) | 使用类似的 RMSNorm + RoPE 配方 |

### 架构影响总结

Llama 2 的架构选择（RMSNorm + SwiGLU + RoPE + GQA）已成为开源 LLM 的**事实标准"Transformer 配方"**。Llama 2 的发布也被广泛视为"开源 AI 的分水岭时刻"。

---

## 六、Implications for You / Hardware Compatibility

### 各个 Llama 2 模型的推理显存需求

| 模型 | bf16 参数显存 | bf16 + KV Cache | 4-bit 量化 | 最小推荐 GPU |
|------|:-:|:-:|:-:|:--|
| 7B | ~14 GB | ~16 GB | ~4-5 GB | ✅ RTX 3060 (12GB) |
| 13B | ~26 GB | ~29 GB | ~7-8 GB | ✅ RTX 3090 (24GB) |
| 70B | ~140 GB | ~150+ GB | ~35-40 GB | ❌ 仅 A100-80GB |

### 对 VLA 的具体指导

1. **OpenVLA 的骨干是 Llama 2 7B**：需理解其 Pre-Norm 设计（不影响 LoRA 微调），SwiGLU 要求 AdamW 优化器
2. **Token 嵌入改造**：OpenVLA 移除 256 个最不常用的词表 token，替换为 256 个离散化动作 token。总词表仍为 32K，但动作 token 使用额外投影层
3. **KV Cache 是关键瓶颈**：OpenVLA 生成 7 步动作 × 256 tokens = 1792 tokens。GQA 设计在 7B 中未使用（仅 MHA），因此推理时动作序列的 KV Cache 增长显著
4. **RoPE 外推**：若需生成长动作序列（>4096 tokens），需调整 RoPE 的 base 值（如从 10000→500000）
5. **商业化可行性**：Llama 2 的商业友好协议使基于 OpenVLA 的机器人产品可以直接部署

### 硬件兼容性总结
- ✅ 7B 模型推理：RTX 3060 (12GB, 4-bit) / RTX 3090 (24GB, bf16+KV cache)
- ✅ 7B model LoRA 微调：RTX 3090/4090 (24GB) 或 RTX 4060 (16GB, QLoRA)
- ⚠️ 7B 全量微调：仅 A100 (80GB) 或 H100
- ❌ 70B 模型消费级推理：仅 A100/H100

## PDF

[[Llama 2 原文.pdf]]
