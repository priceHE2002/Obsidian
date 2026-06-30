---
tags:
  - 论文
  - 训练基础设施
  - 优化器
  - 符号发现
  - 进化搜索
created: 2026-06-30
paper_title: "Symbolic Discovery of Optimization Algorithms"
paper_authors: "Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le"
paper_year: 2024
paper_venue: "NeurIPS 2024 (Spotlight)"
paper_citations: "~800+"
paper_url: "https://arxiv.org/abs/2302.06675"
github: "https://github.com/google/automl/tree/master/lion"
---

# Lion

**Symbolic Discovery of Optimization Algorithms**
*Xiangning Chen, Chen Liang, Da Huang et al. | Google Research | NeurIPS 2024 | arXiv: 2302.06675*

> 通过进化搜索（evolutionary search）在程序空间中自动发现的新优化器。Lion（Evo**L**ved **S**ign **S**gd**O**ptimizer）仅需追踪动量（momentum），用 sign 操作替代 Adam 的一阶/二阶矩自适应步长，显存减半且训练速度更快。

---

## 一、Background / Core Idea

### 1.1 问题：优化器的"手工设计天花板"

从 SGD (1950s) → Adam (2015) → AdamW (2019)，优化器的发展经历了数十年的手动改进：

| 时期 | 优化器 | 创新 | 设计方式 |
|:----|:------|:-----|:--------|
| 1950s-2010s | SGD, SGD+Momentum | 基础梯度下降 | 数学推导 |
| 2012-2015 | AdaGrad, RMSProp, Adam | 自适应学习率 | 启发式 + 经验 |
| 2017-2019 | AdamW, RAdam, LAMB | 解耦、修正 | 理论 + 经验 |
| 2023-2024 | **Lion, Sophia** | 符号搜索、Hessian | **自动化搜索** |

**核心问题**：优化器的设计空间远超人类的直觉探索能力。不同架构、损失函数、数据集的"最优"优化器可能完全不同。

### 1.2 核心洞察：程序搜索 + 功能校验

论文提出使用**进化搜索**（population-based evolution）在数学表达式的程序空间中寻找优化器：

$$\text{update}_t = f(\text{gradient}_t, \text{momentum}_t, \text{params}_t, \text{lr}_t, ...)$$

搜索空间的定义：
- **原生操作符（Primitives）**：$+, -, \times, \div, \text{sign}, \text{clip}, \text{abs}, \text{exp}, \text{log}, \text{min}, \text{max}$
- **输入变量**：梯度 $g_t$、动量 $m_t$、参数 $w_t$、学习率 $\eta$
- **常数**：$\epsilon, \beta_1, \beta_2$ 等可学习参数

搜索目标：在某组任务上最小化验证 loss。

### 1.3 Lion 的发现结果

经过进化搜索，Lion 的更新规则**出乎意料地简单**：

$$\boxed{m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t}$$
$$\boxed{w_t = w_{t-1} - \eta \cdot (\text{sign}(\beta_2 \cdot m_t + (1 - \beta_2) \cdot g_t) + \lambda \cdot w_{t-1})}$$

核心创新：**仅使用 sign 操作**替代 Adam 的 $\frac{m_t}{\sqrt{v_t + \epsilon}}$。这意味着 Lion 不需要二阶矩估计 $v_t$，完全消除了一组 $O(n)$ 的状态存储。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 优化器搜索的演进评估流程

```
进化搜索流程:
┌──────────────┐
│  初始化种群    │  →  随机构造 100 个优化器表达式树
└──────┬───────┘
       ↓
┌──────────────┐
│  变异/交叉     │  →  子树替换、插入、删除
└──────┬───────┘
       ↓
┌──────────────┐
│  训练评估     │  →  在代理任务（小模型+小数据）上完整训练
└──────┬───────┘
       ↓
┌──────────────┐
│  选择清洗     │  →  保留 top-50%，填充新一代
└──────┬───────┘
       ↓
      ↺ 重复直到收敛
```

**评估设置**：
- **代理任务**：ImageNet 上的 ResNet-34（小规模训练 10 epoch）
- **验证方式**：前 20% 的验证 loss 作为 fitness
- **计算成本**：约 10,000 个 GPU 小时的总搜索成本
- **搜索结果筛选**：从 top 种群中手动挑选简洁、可解释的方案

### 2.2 Lion 的完整算法

```
Lion 优化器（EvoLved Sign SGD）：

输入: 参数 w₀, 学习率 η, 权重衰减 λ, β₁, β₂
初始化: 动量 m₀ = 0

for t = 1 to T:
  g_t = ∇L(w_{t-1})                      # 计算梯度
  
  # 更新动量（与 Adam 完全相同）
  m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
  
  # 核心差异: 使用 sign() 替代自适应步长
  # update = sign(β₂ · m_t + (1 - β₂) · g_t)
  # sign(x) = 1 if x > 0, -1 if x < 0, 0 if x = 0
  
  # 参数更新（权重衰减解耦）
  w_t = w_{t-1} - η · (sign(β₂ · m_t + (1 - β₂) · g_t) + λ · w_{t-1})
```

### 2.3 Lion vs AdamW 数学对比

| 步骤 | AdamW | **Lion** |
|:----|:-----:|:--------:|
| 一阶矩 | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ |
| 二阶矩 | $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ | **无** |
| 偏差校正 | $\hat{m}_t, \hat{v}_t$ | **无** |
| 步长缩放 | $m_t / (\sqrt{v_t + \epsilon})$ | $\text{sign}(\beta_2 m_t + (1-\beta_2)g_t)$ |
| 每参数状态 | 2×（m + v） | **1×（m 仅动量）** |
| 权重衰减 | $\eta\lambda w_t$ | $\eta\lambda w_t$（与 AdamW 相同架构） |

### 2.4 搜索发现的关键洞察

1. **Sign 操作的优越性**：进化搜索独立发现 $\text{sign}(\cdot)$ 比 $\tanh$、$\text{clip}$ 表现更好，sign 输出的 L-infinity 范数（每个维度更新幅度相同）在宽泛任务中更鲁棒
2. **二阶矩被淘汰**：没有任何搜索出的优秀方案使用 $v_t$ 或梯度平方——二阶矩估计在进化空间中被"自然淘汰"
3. **动量对权重衰减的抵抗**：Lion 使用 $\text{sign}(m_t)$ 时，$\lambda$ 需比 AdamW 小约 **3-10 倍**（因为 sign 输出的梯度噪声大，更强的权重衰减会抑制模型学习）

### 2.5 为什么 Lion 比 AdamW 快？

| 因素 | 解释 | 量化估计 |
|:----|:-----|:--------:|
| 显存减半 | 不需存储 $v_t$（二阶矩） | 7B 模型省 ~14GB（bf16） |
| 计算简化 | sign + 线性组合 vs 除 + 开方 | 约 30-50% 更快（每步） |
| 通信减少（FSDP） | 优化器状态分片减半 | 分布式环境下约 20% 通信节省 |
| 更大的有效步长 | sign 输出恒为 ±1，不受梯度幅度影响 | 训练的损失下降更快 |

---

## 三、Experiments and Key Findings

### 3.1 图像分类

| 数据集 | 模型 | AdamW | **Lion** | 加速 |
|:------|:----|:-----:|:--------:|:----:|
| ImageNet | ResNet-50 | 76.6% | **77.2%** | +0.6% (top-1) |
| ImageNet | ViT-B/16 | 80.5% | **81.3%** | +0.8% |
| ImageNet | ViT-L/16 | 82.6% | **83.1%** | +0.5% |
| ImageNet | ViT-L/32 (384) | 79.8% | **80.7%** | +0.9% |
| JFT-300M | ViT-B/16 | 66.2% | **66.9%** | +0.7% |

### 3.2 语言建模与文本任务

| 数据集 | 模型 | AdamW | **Lion** | **训练速度** |
|:------|:----|:-----:|:--------:|:-----------:|
| C4 (PPL) | 1.5B Transformer | 15.4 | **14.9** | **快 1.5×** |
| C4 (PPL) | 6.1B Transformer | 13.8 | **13.0** | **快 2.0×** |
| C4 (PPL) | 12.9B Transformer | 12.6 | **12.2** | **快 2.5×** |
| Wiki-40B | mT5 (13B+13B) | 78.9 | **79.8** (Accuracy) | — |

### 3.3 大规模预训练对比（Google 内部生产级）

| 模型 | 规模 | AdamW Loss | **Lion Loss** | 训练时间节省 |
|:----|:----:|:----------:|:------------:|:-----------:|
| PaLM (Decoder-Only) | ~540B | 1.80 | **1.74** | **~20%**（同 loss 更快） |
| T5-XXL (Encoder-Decoder) | 11B | 1.62 | **1.58** | **~15%** |
| ViT-G/14 (Vision) | 2B | — | **更好** | **~20%** |

**关键**：在大规模上（540B PaLM），Lion 达到相同 loss 所需的训练步数比 AdamW 少约 **20%**。换算到实际训练时间，相当于节省数百万美元的计算成本。

### 3.4 超参数鲁棒性

| 优化器 | 学习率范围 | 最佳 LR | 鲁棒性 |
|:------|:---------:|:-------:|:------:|
| AdamW | $10^{-4}$ - $3\times10^{-3}$ | $3\times10^{-4}$ 到 $10^{-3}$ | 中等 |
| **Lion** | $10^{-5}$ - $3\times10^{-4}$ | **$3\times10^{-5}$ 到 $10^{-4}$** | **高** |
| SGD+Momentum | $10^{-2}$ - $3\times10^{-1}$ | $10^{-1}$ | 低 |

**注意**：Lion 的典型学习率比 AdamW 低 **10×**（因为 sign 操作导致有效更新幅度为常数 ±$\eta$，而不是 AdamW 的小幅度自适应更新）。

---

## 四、Limitations and Challenges

1. **理论理解不足**：Lion 的 $\text{sign}(\cdot)$ 操作为何有效缺乏严格的数学解释。sign 函数的非连续性（在 0 处不可微）在优化理论上令人困惑
2. **超参数迁移困难**：AdamW → Lion 的学习率约为 10× 缩小，权重衰减约为 3-10× 缩小。对于新的模型架构，寻找合适的 LR 和 WD 需要额外调优成本
3. **对梯度噪声的放大**：sign 操作只保留符号信息，丢失了梯度幅度。在高噪声场景（如批量大小很小）下，sign 会加剧参数抖动
4. **未见在 GPT-4 级别模型上的公开报告**：目前已知的最大规模验证是 PaLM 540B（Google 内部），在 GPT-4 级别（>1T 参数）上的行为未知
5. **与特定架构的冲突**：LayerNorm 的 γ 参数在 Lion 下可能不稳定（sign 更新过于激进）。LLaMA 系列的实际训练中 AdamW 仍占主导
6. **缺乏与 Scheduled LR 的交互研究**：Lion 与 cosine schedule、warmup、cooldown 的组合效应未系统探索
7. **社区采纳度低**：截至 2024 年底，HuggingFace 的 transformers 生态默认优化器仍为 AdamW，Lion 的集成程度远不及 AdamW

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 Lion 的关系 |
|---------|:----:|---------------|
| **AdamW** (Loshchilov & Hutter) | 2019 | Lion 的权重衰减直接沿用 AdamW 的解耦策略 |
| **Sophia** (Liu et al.) | 2023 | 另一种新优化器，使用 Hessian 对角线估计步长，与 Lion 采用不同方向 |
| **Muon** (Jordan et al.) | 2024 | 基于 Newton-Schulz 迭代，在 Llama 3 训练中表现优于 AdamW |
| **Adan** (Xie et al.) | 2023 | 将 Nesterov 动量引入 AdamW，与 Lion 竞争 |
| **Schedule-Free AdamW** (Defazio et al.) | 2024 | 移除调度器简化优化器使用 |
| **SignSGD** (Bernstein et al., 2018) | 2018 | Lion 的 sign 操作的理论基础——压缩通信的 SignSGD |
| **EvoGrad / AutoML-Zero** (Google) | 2021 | 同一研究线的进化搜索框架前身 |

**影响评估**：Lion 证明了"自动化发现 > 手工设计"的方向。作为符号搜索优化器的代表，Lion 在显存效率上优于 AdamW（少存一组 $v_t$），在训练速度和最终质量上也有改善。但其在社区中的采纳度远不及 AdamW——这可能是因为 AdamW 的"经验担保"和更宽的 LR 调参范围。Lion 在大规模生产环境（如 PaLM）上的验证表明它是 AdamW 的最强竞争者。

---

## 六、Implications for You / Hardware Compatibility

### 显存对比：Lion vs AdamW（训练 7B 模型）

| 优化器 | 每参数额外状态 | 7B 额外显存 | 训练每步加速 | 适用场景 |
|:------|:-------------:|:-----------:|:-----------:|:---------|
| AdamW (bf16) | m + v = 2 × 2B | ~28GB | 1× baseline | 通用首选 |
| **Lion (bf16)** | **m = 2B** | **~14GB** | **~1.5×** | **显存敏感场景** |
| 8-bit AdamW | 1B × 2 | ~16GB | 0.9× | 现成集成 |
| **Lion (8-bit)** | **1B** | **~7GB** | **~1.5×** | **极致显存优化** |

### Lion 的调参指南（从 AdamW 迁移）

| 超参数 | AdamW 典型值 | **Lion 推荐值** | 调整幅度 |
|:------|:-----------:|:--------------:|:--------:|
| 学习率 $\eta$ | 3e-4 | **3e-5** | **降低 10×** |
| $\beta_1$ | 0.9 | 0.9 | 相同 |
| $\beta_2$ | 0.95 | **0.99** | **增大** |
| 权重衰减 $\lambda$ | 0.1 | **0.01-0.03** | **降低 3-10×** |
| Warmup steps | 2000 | 相同 | 相同 |
| Batch size | 4M tokens | 相同 | 相同 |

### 实用建议

- **迁移 AdamW → Lion 的陷阱**：最常犯的错误是保持 LR 不变——Lion 的 sign 操作使得有效步长比 AdamW 大得多，必须降低 LR
- **小批量场景慎用**：批量 < 64 时，梯度噪声会被 sign 放大，导致 loss 抖动
- **用于 LoRA 微调**：Lion 的显存优势在 LoRA 微调中不那么突出（LoRA 的可训练参数很少），但训练速度提升仍有益
- **蒸馏/微调优于预训练**：Lion 在大规模预训练中被验证有效（PaLM），但在很多社区实践中，Lion 的微调结果优于 AdamW，预训练则接近

### 硬件兼容性总结
- ✅ Lion (bf16)：所有支持 bf16 的 GPU（A100, H100, RTX 3090/4090, RTX 4060）
- ✅ Lion (fp32)：所有 GPU，包括 V100、T4、RTX 3060
- ✅ Lion 8-bit：bitsandbytes 集成支持下，消费级 GPU 上 7B 训练更省显存
- ✅ Lion 配合 FSDP：优化器状态减半，分布式训练通信压力更小
- ⚠️ Lion（小批量场景）：梯度噪声放大可能导致训练不稳定，建议保持批量 >= 128

---

## PDF

[[Lion 原文.pdf]]
