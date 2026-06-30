---
tags:
  - 论文
  - LLM
  - 开源
  - VLA骨干
created: 2026-06-30
paper_title: "Llama 2: Open Foundation and Fine-Tuned Chat Models"
paper_authors: "Hugo Touvron, Louis Martin, Kevin Stone et al."
paper_year: 2023
paper_venue: "arXiv 2307.09288"
paper_citations: "~20,000+"
paper_url: "https://arxiv.org/abs/2307.09288"
---

# Llama 2

**Llama 2: Open Foundation and Fine-Tuned Chat Models**
*Meta AI | arXiv 2307.09288*

> OpenVLA 的骨干就是 Llama 2 7B。理解 Llama 2 的架构细节——RMSNorm、SwiGLU、RoPE、GQA——等于理解了 OpenVLA 的内部机制。Llama 2 的开源（商业友好协议）使得整个 VLA 社区可以基于它做研究和部署。

---

## 一、研究背景与动机

在 Llama 2 之前，开源 LLM 在质量上远远落后于闭源模型（GPT-3.5/GPT-4）。Meta 的 Llama 1（2023年2月）首次展示了开源 LLM 可以达到接近 SOTA 的性能，但仅限于研究用途（非商业友好协议），且缺乏对话微调版本。

Llama 2 的目标是：

1. **开源高性能基础模型**：7B、13B、70B 三个规模，在 2T tokens 上预训练
2. **开源经过 RLHF 微调的对话模型**：Llama 2-Chat 在多数基准上接近 GPT-3.5
3. **商业友好协议**：允许商业使用，激活了整个开源 LLM 生态

这次发布被广泛视为"开源 AI 的分水岭时刻"——它使得基于 LLM 的应用开发（包括 VLA）不再受制于闭源 API。

## 二、核心方法

### 架构细节

Llama 2 的架构是一个 Decoder-only Transformer，包含以下关键设计：

#### 1. RMSNorm (Pre-Norm)

每个 Transformer 子层（Multi-Head Attention 和 Feed-Forward Network）之前进行归一化，而非之后：

$$ \text{Output} = x + \text{SubLayer}(\text{RMSNorm}(x)) $$

其中 $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2}} \cdot \gamma$

#### 2. SwiGLU 激活函数

SwiGLU (Swish-Gated Linear Unit) 替代了传统的 ReLU 或 GELU：

$$ \text{SwiGLU}(x) = (x \cdot \sigma(\beta x)) \odot (W_u \cdot x) $$

其中 $\sigma(\beta x)$ 是 Swish 激活函数，$\odot$ 是逐元素乘法。实际实现中，FFN 包含三个权重矩阵 $W_g$（用于门控）、$W_u$（用于上投影）、$W_d$（用于下投影），输入维度从 $d$ 投影到 $\frac{8}{3}d$ 再压回 $d$。

SwiGLU 相比 ReLU 的优势在于：（1）允许负梯度通过门控机制；（2）在相同参数量下性能更好；（3）梯度分布更平滑。

#### 3. RoPE (Rotary Position Embedding)

RoPE 将相对位置信息直接编码到 query 和 key 中：

$$ \text{RoPE}(q_m, k_n) = f(q, m)^T \cdot f(k, n) = g(q, k, m-n) $$

通过旋转矩阵 $R_{\Theta,m}$ 对 $q$ 和 $k$ 进行变换：

$$ f_q(x_m, m) = R_{\Theta,m} \cdot W_q \cdot x_m $$

其中 $R_{\Theta,m}$ 是稀疏旋转矩阵，使得注意力的结果只依赖于 $q$ 和 $k$ 的内容以及它们的相对位置差 $m-n$。

#### 4. Grouped Query Attention (GQA, 仅 70B 版本)

GQA 介于 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 之间：

| 方法 | Query Heads | Key/Value Heads | KV Cache 大小 |
|------|:-:|:-:|:-:|
| MHA | $h$ | $h$ | $h \times L \times \text{dim}$ |
| MQA | $h$ | 1 | $L \times \text{dim}$ |
| GQA | $h$ | $g$ (1 < g < h) | $g \times L \times \text{dim}$ |

GQA 在推理时显著减少 KV cache 的显存占用，同时保持接近 MHA 的模型质量。

### 训练数据

| 配置 | Llama 1 | Llama 2 |
|------|---------|---------|
| 训练 tokens | 1.0T | 2.0T |
| 上下文长度 | 2048 | 4096 |
| 数据来源 | 公开数据 | 公开数据（更新版） |
| 分词器 | SentencePiece (32K) | SentencePiece (32K) |

### RLHF 微调

Llama 2-Chat 的 RLHF 流程包含多个阶段：

1. **SFT**：监督微调（人工标注的高质量对话数据）
2. **Reward Model**：训练两个独立的奖励模型（有用性 + 安全性）
3. **PPO**：近端策略优化进行 RLHF
4. **GAtt (Ghost Attention)**：一种新的多轮对话一致性增强方法——通过修改 attention mask 让系统提示在对话前期保持权重

## 三、关键实验与发现

### 基准测试

| 模型 | MMLU (5-shot) | HellaSwag | HumanEval | GSM8K |
|------|:-:|:-:|:-:|:-:|
| Llama 2 7B | 45.3 | 77.2 | 12.8 | 14.6 |
| Llama 2 13B | 54.8 | 80.7 | 18.3 | 28.7 |
| Llama 2 70B | 68.9 | 83.9 | 29.9 | 56.8 |
| GPT-3.5 (text-davinci-003) | 70.0 | 85.5 | 48.1 | 57.1 |

最关键的结论：**Llama 2 70B 在多数基准上接近 GPT-3.5，7B 和 13B 版本优于同规模所有开源模型。**

### RLHF 的分析

- 有用性和安全性奖励模型存在冲突：提升有用性有时会降低安全性
- 使用安全-specific RLHF 可以在不损失有用性的情况下提升安全性
- GAtt 显著提升了多轮对话中系统指令的遵循能力

## 四、局限性与后续影响

### 局限性

1. **上下文长度仅 4096**：限制了一些需要长上下文的应用
2. **缺乏 MoE 架构**：相比 Mixtral 8×7B，密度模型在相同 FLOPs 下容量较低
3. **训练数据量有限**：2T tokens 已被后续模型大幅超越（Llama 3: 15T+）
4. **分词器效率较低**：SentencePiece 32K 词表在非英语语言上效率差

### 后续影响

Llama 2 的发布开启了开源 LLM 的黄金时代：Llama 3（8B/70B/405B）、Qwen 系列、Mistral、Gemma 等竞相出现。其架构设计（RMSNorm + SwiGLU + RoPE）成为事实上的"标准 Transformer 配方"。

## 五、VLA/机器人研究中的角色

Llama 2 对 VLA 研究的影响无可替代：

- **OpenVLA 直接使用 Llama 2 7B 作为骨干**：移除原 Llama 2 的 LM head，替换为 256 个机器人动作 token 的投影层。Llama 2 的隐藏层（4096 维）直接作为视觉-语言-动作融合的特征空间
- **架构影响**：
  - SwiGLU → 解释为何 VLA 训练用 AdamW 而非 SGD（SwiGLU 的梯度结构需要自适应学习率）
  - RoPE → VLA 中的动作 token 序列（如 7 个动作 × 256 tokens = 1792 tokens）依赖 RoPE 来保持相对位置关系
  - GQA → VLA 推理时最大的瓶颈是 KV cache 增长，GQA 的设计直接决定了 VLA 的推理效率
- **SmolVLA、BitVLA 等轻量 VLA**：直接继承 Llama 架构的 decoder-only 设计，但缩小模型维度
- **商业友好协议**：使得基于 Llama 2 的 VLA 系统可以部署到实际机器人产品中
- **社区生态**：HuggingFace、vLLM、llama.cpp 等工具链围绕 Llama 2 成熟，VLA 研究者可以直接复用这些工具

## 六、对你的启示

1. **OpenVLA 微调需要理解 Llama 2 的 Pre-Norm 设计**：在 LoRA 微调时，通常不对 RMSNorm 层添加 adapter，因为它们已经足够稳定
2. **7B 模型（bf16）约占 14GB，4-bit 量化后降至 4-5GB**：这意味着在 16GB GPU 上可以运行 fully-finetuned OpenVLA（需要梯度 checkpointing），或运行 4-bit 版本的完整 VLA
3. **KV cache 管理是关键**：推理 OpenVLA 时，动作 token 序列很长（7 步 × 256 tokens = 1792 tokens），KV cache 的显存占用可能超过模型参数本身
4. **RoPE 的 `theta` 参数影响长序列泛化**：如果修改动作序列长度（例如从 7 步到 10 步），可能需要调整 RoPE 的频率
5. **SwiGLU 的 FFN 有 3 个权重矩阵**：LoRA 微调时，通常对所有 QKV + FFN 投影矩阵都加 adapter

## PDF

[[Llama 2 原文.pdf]]
