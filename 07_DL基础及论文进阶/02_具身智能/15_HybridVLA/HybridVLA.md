---
tags:
  - 论文
  - VLA
  - ICLR2026
  - 扩散+自回归
created: 2026-06-30
paper_title: "HybridVLA: Collaborative Diffusion and Autoregression in a Unified VLA"
paper_authors: "Jiaming Liu et al. (北大 + 智源BAAI + 港中文CUHK)"
paper_year: 2025
paper_venue: "ICLR 2026 Poster"
paper_url: "https://arxiv.org/abs/2503.10631"
project: "https://hybrid-vla.github.io"
---

# HybridVLA

**HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model**
*北大 + 智源BAAI + 港中文CUHK | ICLR 2026 Poster | arXiv 2503.10631*

> **自回归 VLA 擅长"理解"（语义推理），扩散 VLA 擅长"执行"（精细控制）。谁说必须二选一？** HybridVLA 把两种模式统一到同一个 LLM 里——不额外加扩散头，而是让 LLM 自己学会何时用扩散、何时用自回归。这是 VLA 架构从"选边站"走向"融合"的代表作。

---

## 一、核心洞察：两种范式各有所长

### 1.1 自回归 VLA（RT-2, OpenVLA）

**做法**：连续动作 → 离散成 256 bins → LLM 自回归预测下一个 action token

**优势**：
- 充分利用 LLM 从互联网数据学到的推理能力
- 语义泛化强——"把红瓶子放到绿杯子上"中的"红"和"绿"能被 LLM 理解
- 简单——不需要额外的动作模块

**劣势**：
- 离散化损失连续信息（256 bins 对高精度操作不够）
- 自回归生成慢（串行逐个 token）
- 无法表达复杂的多模态动作分布

### 1.2 扩散 VLA（π0, GR00T）

**做法**：LLM 的输出作为条件（conditioning），独立的 Diffusion/Flow Transformer 去噪生成连续动作

**优势**：
- 连续动作建模（没有离散化损失）
- 天然多模态（可以去噪到不同模式）
- 并行去噪快

**劣势**：
- LLM 只是"特征提取器"，推理能力没有用于动作生成
- Diffusion Transformer 需要大量额外参数

### 1.3 HybridVLA 的假设

**如果在同一个 LLM 内部同时进行扩散和自回归，让两种模式互相增强，会不会比两者都好？**

---

## 二、方法：在 LLM 内部注入扩散

### 2.1 架构

HybridVLA 使用标准的 LLaMA-2 7B（或 Phi-2 2.7B）作为骨干，**不加任何外部扩散头**。关键修改是在 token 序列中引入特殊标记：

```
标准自回归模式: [obs_tokens] → [action_token_1] → [action_token_2] → ...

扩散模式: [obs_tokens] → <BOD> → [noise_1, ..., noise_N] → <EOD> → ...
                                                        ↓
                                                    去噪 N 步
                                                        ↓
                                                   连续动作
```

`<BOD>` (Begin of Diffusion) 和 `<EOD>` (End of Diffusion) 是特殊 token，告诉模型"接下来要进入扩散模式"。

### 2.2 混合损失

$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{ce}}$$

- $\mathcal{L}_{\text{diff}}$：扩散去噪的 MSE 损失（只在 `<BOD>...<EOD>` 范围内计算）
- $\mathcal{L}_{\text{ce}}$：自回归的交叉熵损失（在所有 token 上计算）

两种损失在**同一个优化过程**中联合优化共享的 LLM 参数。

### 2.3 协作推理：自适应集成

推理时，两种模式**同时**生成候选动作，然后根据**置信度**做自适应融合：

$$\hat{a} = \alpha \cdot a_{\text{diff}} + (1-\alpha) \cdot a_{\text{ar}}, \quad \alpha = f(\text{confidence}_{\text{ar}})$$

- 当自回归的 token 置信度高时（LLM "很确定"），权重偏向自回归
- 当自回归的 token 置信度低时（LLM "不确定"），权重偏向扩散

直觉：**需要语义理解时信自回归，需要精确控制时信扩散。**

---

## 三、实验与结果

### 3.1 仿真基准

| 基准 | HybridVLA vs 之前 SOTA |
|------|----------------------|
| 仿真平均 | **+17%** 成功率 |
| 未见物体 | 鲁棒 |
| 未见背景 | 鲁棒 |
| 空间变化 | 鲁棒 |
| 光照变化 | 鲁棒 |

### 3.2 真实世界

| 场景 | HybridVLA vs 之前 SOTA |
|------|----------------------|
| 单臂操作 (平均) | **+19%** |
| 双臂操作 (平均) | **+19%** |
| 泛化场景 | 一致更优 |

### 3.3 效率

HybridVLA-dif 变体（推理时只用扩散）可达 **9.4 Hz**——快于将两种模式混合的版本，但仍然保留训练时的联合优化收益。

---

## 四、两种模式的"分工"

通过置信度分析，论文发现了一个有趣的模式：

- **语义密集的任务**（如"把红色的那个放进标有'3'的抽屉"）→ 自回归模式置信度高
- **精度密集的任务**（如"把针插入针孔"）→ 扩散模式置信度高
- **混合任务** → 两种模式的置信度交替变化

这个发现为未来的"动态架构选择"提供了基础——根据任务自动切换或融合模式。

---

## 五、与你的研究的关系

"统一的扩散 + 自回归"是一个肥沃的研究方向。你可以在以下方面探索：

1. **在小模型上复现**：用 SmolVLA (450M) 或 FLOWER (950M) 做 Hybrid 架构实验
2. **探索更优的融合策略**：置信度加权只是一个起点——可以探索基于不确定性估计、基于任务类型检测等
3. **速度 vs 质量权衡**：HybridVLA-dif 表明可以牺牲一点自回归的语义增强来换取几倍的速度提升

## 六、硬件适配

⚠️ **4070 Ti Super 16GB**：HybridVLA 基于 LLaMA 7B / Phi-2.7B，16GB 微调很紧。建议使用 Phi-2 版本或量化版本。

## PDF

[[HybridVLA.pdf]]
