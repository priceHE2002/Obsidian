---
tags:
  - 论文
  - 归一化
  - 高效架构
  - Transformer组件
created: 2026-06-30
paper_title: "Root Mean Square Layer Normalization"
paper_authors: "Biao Zhang, Rico Sennrich"
paper_year: 2019
paper_venue: "NeurIPS 2019"
paper_citations: "~2,500+"
paper_url: "https://arxiv.org/abs/1910.07467"
---

# RMSNorm

**Root Mean Square Layer Normalization**
*University of Edinburgh | NeurIPS 2019 | arXiv: 1910.07467*

> LayerNorm 的简化版——只保留 re-scaling（除以 RMS），去掉 re-centering（减均值）。RMSNorm 在保持甚至略优于 LayerNorm 性能的同时，计算速度提升约 7-15%。Llama 系列（包括 OpenVLA 的骨干 Llama 2）全部使用 RMSNorm 而非原始 LayerNorm。

---

## 一、研究背景与动机

Layer Normalization (LayerNorm) 是 Transformer 架构的核心组件之一，它在每个 token 的特征维度上计算均值和方差，然后进行归一化和仿射变换：

$$ \text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta $$

其中 $\mu = \frac{1}{d}\sum_{i=1}^d x_i$, $\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d(x_i - \mu)^2}$。

LayerNorm 包含两个操作：**re-centering**（减均值 $\mu$）和 **re-scaling**（除以标准差 $\sigma$）。作者观察到：（1）减均值操作在 Transformer 的残差连接架构中贡献有限；（2）Re-centering 和 re-scaling 对训练的稳定性贡献是不对称的。这启发他们思考：是否可以只保留 re-scaling，从而减少计算量？

## 二、核心方法

### RMSNorm 的定义

RMSNorm 完全去掉 re-centering 操作，只通过 Root Mean Square (RMS) 进行归一化：

$$ \overline{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i $$

其中 RMS 统计量定义为：

$$ \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2} $$

### 与 LayerNorm 的对比

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算均值 $\mu$ | 需要 | 不需要 |
| 计算标准差 $\sigma$ | 需要 | 不需要 |
| 计算 RMS | 不需要 | 需要 |
| Re-centering (减均值) | 有 | 无 |
| Re-scaling (除标准差/RMS) | 有 | 有 |
| 可学习参数 | $\gamma$ (scale) + $\beta$ (shift) | 仅 $\gamma$ (scale) |
| 计算开销 | 高（约 2× RMSNorm） | 低 |

### 梯度分析

论文提供了 RMSNorm 的梯度推导。对于输出 $\overline{x} = \frac{x}{\text{RMS}(x)} \cdot \gamma$，其关于输入 $x_i$ 的梯度为：

$$ \frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \overline{x}_i} \cdot \frac{\gamma_i}{\text{RMS}(x)} - \frac{\gamma_i \cdot x_i}{d \cdot \text{RMS}(x)^3} \sum_{j} \frac{\partial \mathcal{L}}{\partial \overline{x}_j} \cdot \overline{x}_j $$

这个形式与 LayerNorm 的梯度结构类似，**只是缺少了均值相关的项**——作者正是通过这一差异论证了 re-centering 的可省略性。

### 理论论证：为何 Re-centering 是不必要的

论文的核心理论贡献是证明了在残差网络中，re-centering 的作用可以被残差连接自身替代：

1. 残差连接 $x^{(l+1)} = x^{(l)} + F(x^{(l)})$ 已经提供了某种形式的中心化
2. 权重衰减正则化鼓励权重趋近于零，间接控制了输出的偏移
3. LayerNorm 的 `$\beta$` 参数对模型容量的贡献极小，大多数情况下其梯度接近零

## 三、关键实验与发现

### 机器翻译任务

| 模型 | WMT En-De (BLEU) | WMT En-Fr (BLEU) | 训练速度 |
|------|:-:|:-:|:-:|
| Transformer + LayerNorm | 27.33 | 38.95 | 1.0× (baseline) |
| Transformer + RMSNorm | 27.43 | 38.98 | 1.07-1.15× |

- RMSNorm 在多个 MT 数据集上与 LayerNorm 性能持平或略优
- **训练速度提升 7-15%**，主要来自省去的均值/方差计算和 $\beta$ 参数更新
- 收敛速度更快——同等训练步数下，RMSNorm 的验证 loss 更低

### 语言建模任务

在 WikiText-2 上的语言建模实验中，RMSNorm 同样展现了与 LayerNorm 相当的困惑度（perplexity），且在大 batch size 情况下训练更稳定。

## 四、局限性与后续影响

### 局限性

1. **实验规模有限**：原始论文的实验主要集中在机器翻译和语言建模上，在 CNN、RL 等领域的验证不足
2. **理论解释不够深入**：为什么 re-centering 在 Transformer 中不是必需的？论文的论证更多是实验导向而非理论完备
3. **大模型验证缺失**：论文实验模型规模在 6 层 Transformer 左右，未验证在数十/数百亿参数模型上的行为
4. **非 Transformer 架构不明确**：RMSNorm 在 CNN、RNN 等架构上与 LayerNorm 的性能比较没有广泛结论

### 后续影响

RMSNorm 已成为现代 LLM 的事实标准归一化方法。Llama 1/2/3、Qwen 系列、Mistral、Gemma 等几乎全部使用 RMSNorm。

## 五、VLA/机器人研究中的角色

RMSNorm 在 VLA 领域是一个"隐形的基础设施"——几乎不被讨论，但无处不在：

- **OpenVLA**：骨干 Llama 2 7B 的每一层（每个 Transformer block）都包含两个 RMSNorm：一个在 Attention 子层前，一个在 FFN 子层前（Pre-Norm 设计）
- **Llama 3 / Qwen2.5**：这两个模型被大量 VLA 系统用作骨干，全部继承 RMSNorm
- **π0**：基于 PaliGemma，Gemma 使用 RMSNorm
- **RT-2**：基于 PaLM-E，PaLM 使用 RMSNorm 变体

理解 RMSNorm 的公式 $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}$ 是理解"Pre-Norm vs Post-Norm"设计选择的基础——**为什么现代 VLA 的 Transformer block 结构都是 `x → RMSNorm → Attention → 残差 → RMSNorm → FFN → 残差`**。

## 六、对你的启示

1. **计算效率直接转化为训练速度**：如果你的 VLA 训练每次 epoch 要几天，去掉不必要的计算（如 $\mu$）可以省下大量时间
2. **RMSNorm 几乎零成本替代 LayerNorm**：在开发 VLA 原型时，永远选择 RMSNorm 而非 LayerNorm
3. **16GB GPU 上，RMSNorm 的计算开销可以忽略**：相比 KV cache 和 FFN 参数，RMSNorm 在推理/微调中的耗时占比 < 0.1%
4. **微调时 RMSNorm 可以冻结**：在 LoRA 微调中，冻结 RMSNorm 的参数（$\gamma$）不影响最终性能，减少可训练参数量
5. **理解 RMSNorm 方能理解 VLA 骨干**：当你看到 VLA 论文中的 architecture diagram 画着 `LN` 时，90% 的情况下实际使用的是 RMSNorm

## PDF

[[RMSNorm 原文.pdf]]
