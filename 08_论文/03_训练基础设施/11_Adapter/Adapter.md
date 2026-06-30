---
tags:
  - 论文
  - 训练基础设施
  - PEFT
  - 适配器
  - 参数高效
created: 2026-06-30
paper_title: "Parameter-Efficient Transfer Learning for NLP"
paper_authors: "Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly"
paper_year: 2019
paper_venue: "ICML 2019"
paper_citations: "~9,000+"
paper_url: "https://arxiv.org/abs/1902.00751"
github: ""
---

# Adapter

**Parameter-Efficient Transfer Learning for NLP**
*Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski et al. | Google | ICML 2019 | arXiv: 1902.00751*

> PEFT（Parameter-Efficient Fine-Tuning）领域的**奠基性工作**。在冻结的预训练 Transformer 每层中插入小巧的"瓶颈适配器"（bottleneck adapter），仅微调 3.6% 的参数量即达到全量微调的性能，开启了参数高效微调的研究范式。其对 VLA 研究的启发在于：**适配器思路与 LoRA 同根同源，适配并行化改进为现代 PEFT 提供了重要铺垫**。

---

## 一、Background / Core Idea

### 1.1 问题：全量微调的存储与灾难性遗忘

BERT、GPT 等大规模预训练模型的流行带来了一个新范式——"预训练 + 下游微调"。但全量微调（fine-tune all parameters）存在两个严峻问题：

- **存储成本爆炸**：每个下游任务都需要独立保存一份完整的模型参数。BERT-Large 有 330M 参数（约 1.3GB fp32），N 个任务就需要 N 倍存储。在 2019 年的云服务部署场景下，这直接转化为高昂的运维成本
- **灾难性遗忘**（Catastrophic Forgetting）：微调过程中，预训练学到的通用语言知识可能被破坏或覆盖。任务越多、微调步数越长，遗忘越严重
- **部署切换不便**：不同任务之间需要重新加载整个模型，无法实现"一个基模型 + 多个轻量模块"的热切换

### 1.2 核心洞察：预训练特征的高度可迁移性

论文的核心直觉很简单但本质：

> 预训练模型（如 BERT）提取的**通用特征**已经足够强大。下游任务只需要在这些特征之上学习**轻量的任务特定转换**，而无需破坏预训练权重本身。

换言之：$\Delta f_{\text{task}}$ 可以被约束在一个极低维的参数空间中，而 $f_{\text{pretrain}}$ 保持不变。

### 1.3 Adapter 在 PEFT 方法谱系中的位置

| 方法 | 提出时间 | 修改范围 | 推理延迟 | 核心思想 |
|------|:--------:|:--------:|:--------:|---------|
| **Adapter（本文）** | ICML 2019 | 每层插入 | **有** | 瓶颈 MLP 旁路 |
| [[LoRA]] | ICLR 2022 | 注意力权重的旁路 | **零** | 低秩分解矩阵 |
| [[QLoRA]] | NeurIPS 2023 | 量化 + LoRA | 零 | 4-bit NF4 + 双重量化 |
| Prefix Tuning | ACL 2022 | 输入序列 | 零 | 虚拟 token |
| Prompt Tuning | EMNLP 2021 | 输入序列 | 零 | 可学习前缀 |

Adapter 是**最早的 PEFT 方案之一**，比 [[LoRA]] 早约 3 年。但其引入的推理延迟问题（见第三节）促使了后续零延迟方法的出现。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 瓶颈适配器结构（Bottleneck Adapter）

Adapter 的核心是将一个**瓶颈结构**插入 Transformer 的每一层中。具体位置有两个：

1. **每个 Multi-Head Attention 子层之后**（串联）
2. **每个 FFN 子层之后**（串联）

每个 Adapter 模块的结构为：

$$\text{Adapter}(x) = x + \text{ReLU}\left(x \cdot W_{\text{down}}\right) \cdot W_{\text{up}}$$

- 输入 $x \in \mathbb{R}^{h}$，$h$ 为隐藏维度
- $W_{\text{down}} \in \mathbb{R}^{h \times b}$：下投影，将 $h$ 维压缩到瓶颈维度 $b$
- $W_{\text{up}} \in \mathbb{R}^{b \times h}$：上投影，恢复 $h$ 维
- $\text{ReLU}$ 非线性：瓶颈处引入非线性变换，提供表达能力
- **残差连接**：绕过 Adapter 的短路路径，保证训练稳定性

Adapter 的总参数量为 $2 \times h \times b$（不计偏置）。当 $b \ll h$ 时，参数量远小于原模型的单层 MLP。

### 2.2 初始化与归一化策略

| 组件 | 设计选择 | 技术理由 |
|------|---------|---------|
| $W_{\text{down}}$ | 近零初始化（$\mathcal{N}(0, \sigma^2)$，$\sigma \approx 0.01$） | 初始时接近恒等映射 |
| $W_{\text{up}}$ | 近零初始化 | 与 down 对称 |
| LayerNorm | 在 $W_{\text{up}}$ 之后添加 | 稳定训练、适应不同任务的数据分布 |
| 残差连接 | `Adapter(x) = x + f(x)` | 初始退化时保持预训练输出 |

### 2.3 层归一化后适配器（AdapterLN）

论文发现 Adapter 内层还需要额外的 LayerNorm 来获得最佳性能，并称此变体为 **AdapterLN**，其位置为：

$$\text{Layer}_{i}(x) = \text{LN}\left(\text{FFN}(\text{LN}(x + \text{MHA}(x))) + \text{Adapter}(\text{FFN}(\text{LN}(x + \text{MHA}(x))))\right)$$

实际上 Adapter 插入在 FFN 之后、外层 LN 之前。这比简单串联 MHA 和 FFN 效果更好。

### 2.4 瓶颈维度 $b$ 的影响

论文系统研究了瓶颈维度对参数量/性能的影响：

| 瓶颈维度 $b$ | 可训练参数比例（BERT-Base） | GLUE 平均得分 | 参数效率 \\
|:-:|:-:|:-:|:-:|
| 64 | 0.5% | 82.1 | ⭐ 效率最高 |
| 128 | 1.0% | 82.5 | 略优 |
| 256 | 2.0% | 82.8 | --- |
| 512 | 4.0% | 82.7 | 边际收益递减 |

**每层两个 Adapter 的总参数**：$2 \times (\text{层数}) \times 2 \times h \times b$。以 BERT-Base 为例，$L=12, h=768, b=64$：

$$\text{总参数量} = 12 \times 2 \times 2 \times 768 \times 64 = 2,359,296 \ (\text{约 } 0.5\% \text{ 的 BERT-Base})$$

### 2.5 Adapter 与微调策略比较

| 方法 | 可训练参数 | 对预训练权重影响 | 灾难性遗忘风险 |
|------|:-:|:-:|:-:|
| 全量微调 | 100% | 完全更新 | **高** |
| 仅顶层微调 | ~1-5% | 仅最后几层 | 中 |
| 渐进微调 | 逐层解冻 | 部分更新 | 中 |
| **Adapter（本文）** | **0.5-3.6%** | **零**（冻结） | **极低** |
| [[LoRA]] | 0.01-1% | 零（冻结） | 极低 |

Adapter 对比仅顶层微调的核心优势：**每一层的特征都可以被任务特定的适配器调整**，而不局限于顶层。这意味着底层语法特征、中层语义特征、顶层任务特征都能被下游任务选择性利用。

---

## 三、Experiments and Key Findings

### 3.1 GLUE 基准测试

论文在 BERT-Base 和 BERT-Large 上进行了系统的 GLUE 实验：

| 方法 | 参数量 | MNLI | SST-2 | MRPC | CoLA | QNLI | QQP | RTE | STS-B | **平均** |
|:----:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:--------:|
| BERT-Base (全量微调) | 100% | 84.0 | 92.8 | 86.4 | 57.0 | 91.4 | 88.4 | 70.2 | 89.0 | 82.4 |
| BERT-Base (Adapter-B) | **3.6%** | **84.0** | 92.7 | 86.4 | 56.4 | 91.2 | 87.8 | 68.8 | **89.2** | 82.1 |
| BERT-Large (全量微调) | 100% | 86.0 | 93.5 | 87.9 | 60.8 | 92.3 | 89.5 | 72.1 | 90.0 | 84.0 |
| BERT-Large (Adapter-B) | **1.4%** | 85.8 | **93.7** | **88.1** | 60.8 | **92.6** | 89.2 | 71.5 | **90.6** | **84.0** |

**关键发现**：
- Adapter 在 BERT-Large 上以 1.4% 的参数达到**与全量微调持平**的平均性能
- 在 SST-2、MRPC、STS-B 上甚至**反超**全量微调
- CoLA（语法可接受性判断）差距最大，说明某些任务需要更多参数深度适应

### 3.2 Adapter 大小的影响

论文进一步测试了 BERT-Large 上不同 Adapter 大小的影响：

| 设置 | $b=64$（小） | $b=128$（中） | $b=256$（大） | 全量微调 |
|:----:|:--------:|:---------:|:---------:|:------:|
| 新增参数 | ~0.5% | ~1.0% | ~2.0% | 100% |
| RTE 准确率 | 82.7 | 83.8 | 82.3 | 72.1 |
| MNLI 准确率 | 85.8 | 86.1 | 86.3 | 86.0 |

一个值得注意的现象：**在 RTE 这种小数据集上，Adapter 大幅优于全量微调**（83.8 vs 72.1）。这直观展示了参数约束的正则化效应——全量微调在数据稀少时易过拟合，而 Adapter 的低维约束天然具有更强的泛化能力。

### 3.3 层消融实验

论文证明**每一层都需要 Adapter**。仅将 Adapter 插入部分层（如仅插入前 6 层或后 6 层）时，性能显著下降：

| 适配器位置 | MNLI | RTE | MRPC |
|:---------:|:----:|:---:|:----:|
| 全部 24 层 | 85.8 | 82.7 | 88.5 |
| 仅前 12 层 | 85.2 | 70.4 | 85.0 |
| 仅后 12 层 | 85.4 | 68.2 | 84.8 |
| 仅每隔一层（12 层） | 85.2 | 72.9 | 83.7 |

结论：**适配器的特征校正需要作用于每一层**，跨层传递会导致任务信号在无适配器的层中退化。

### 3.4 灾难性遗忘实验

论文设计了一个"连续微调"实验——在 MRPC 上微调后，继续在 RTE 上微调（不切换适配器）：

| 方法 | MRPC (F1) | → RTE (F1) | RTE 上损失 |
|:----:|:--------:|:--------:|:--------:|
| 全量微调（连续） | 85.7 | 60.3 | **-13.1** |
| Adapter（连续） | 85.0 | 84.1 | **-0.8** |

Adapter 在学习新任务时几乎不遗忘旧任务的知识——这是冻结预训练权重策略最有力的实证支持。

### 3.5 推理延迟分析

Adapter 带来的最大代价——推理延迟：

| 条件 | BERT-Base | BERT-Base + Adapter | 增加 |
|:----:|:--------:|:------------------:|:----:|
| Batch=32, Seq=128 | 31ms | 37ms | **+19%** |
| Batch=1, Seq=128 | 12ms | 17ms | **+42%** |

在在线服务场景（小批量推理）中，Adapter 的顺序执行带来显著延迟。这与 [[LoRA]] 的零延迟形成鲜明对比。

---

## 四、Limitations and Challenges

1. **推理延迟**（最显著的局限）：每个 Adapter 子层增加一次顺序计算，batch size 越小时额外延迟占比越高。小批量推理中 Adapter 层约占 Transformer 前向计算量的 15-40%。这在高吞吐在线推理场景下是不可接受的
2. **训练速度慢**：Adapter 层反向传播需要额外 1-2% 的 FLOPs（实际放大效应取决于 adapter 维度 $b$），且梯度通信量增大
3. **瓶颈维度的手工选择**：$b=64$ 是经验值，缺乏理论指导。不同任务对瓶颈维度的敏感度不同（如 CoLA 需要较大的 $b$）
4. **非线性瓶颈的表达能力局限**：单层 ReLU 瓶颈的表达能力有限。后续工作尝试了 GELU、多层 MLP、甚至卷积结构的适配器
5. **适配器位置固定**：论文固定将 Adapter 插入在每个子层之后，未探索更优的位置方案（如仅在 FFN 之后或仅在 Attention 之后）
6. **语言模型闭**：论文仅在 BERT（编码器）上验证，未在 GPT（解码器）或 Encoder-Decoder 架构上测试。[[LoRA]] 后来的 GPT-3 实验补全了这一空白
7. **序列长度不友好**：Adapter 作用于每个 token 的 hidden state，当序列长度 $L$ 很大时，Adapter 的 FLOPs 占比线性增长

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 Adapter 的关系 |
|---------|:----:|-----------------|
| **[[LoRA]]** (Hu et al.) | 2021 | 用低秩矩阵替代 Adapter 的 MLP 瓶颈，实现推理零延迟 |
| **AdapterFusion** (Pfeiffer et al.) | 2021 | 多任务 Adapter 知识融合框架 |
| **AdapterDrop** (Ruckert et al.) | 2021 | 推理时丢弃部分 Adapter 层加速 |
| **AdapterBias** (Zaken et al. / BitFit) | 2021 | 仅微调偏置项，极致精简 |
| **Compacter** (Mahabadi et al.) | 2021 | 用 Kronecker 积进一步压缩 Adapter 参数 |
| **Hyperformer** (Mahabadi et al.) | 2021 | 超网络生成 Adapter 权重 |
| **[[QLoRA]]** (Dettmers et al.) | 2023 | 4-bit 量化 + LoRA，与 Adapter 同属 PEFT 家族 |
| **HuggingFace PEFT** | 2023 | 统一封装 Adapter、LoRA、Prefix Tuning |

**影响评估**：Adapter 是 PEFT 领域的**开山之作**，开创了"冻结预训练权重 + 注入可训练模块"的研究范式。直接启发了 [[LoRA]]、AdapterFusion 等后续工作。虽然因推理延迟问题在实践中被 [[LoRA]] 逐步取代，但其引入的"参数高效微调"概念是 2022-2026 年大模型时代最重要的研究方向之一。

---

## 六、Implications for You / Hardware Compatibility

### 训练与推理显存分析（以 BERT-Base / LLaMA-7B 类比）

| 配置 | 训练显存 | 推理显存 | 可使用 GPU |
|------|:-:|:-:|:--|
| BERT-Base + Adapter ($b=64$，12 层) | ~8GB | ~3GB | ✅ T4 (16GB) / RTX 2080 (8GB) |
| BERT-Large + Adapter ($b=128$，24 层) | ~16GB | ~6GB | ✅ RTX 3090 (24GB) |
| LLaMA-7B + Adapter（假设实现，$b=64$） | ~18-22GB | ~15GB | ✅ RTX 3090/4090 (24GB) |
| LLaMA-13B + Adapter（假设） | ~30-35GB | ~18GB | ⚠️ A100 (40GB) |
| LLaMA-7B 全量微调（对比） | ~60GB | ~14GB | ❌ 仅 A100 (80GB) |

> **注意**：Adapter 在现代 LLM（如 LLaMA 系列）上的原生实现并不多见。[[LoRA]]/[[QLoRA]] 由于零推理延迟优势，已成为实际标准。但 Adapter 的变体（如 Compacter、AdapterFusion）在处理异构任务融合时仍有学术价值。

### 对 VLA 研究的指导

- **Adapter 的设计哲学**——冻结基础编码器（如 SigLIP、DINOv2），仅注入任务特定适配器——与 VLA 的"视觉 backbone + LLM 对齐"范式高度一致
- **适配器的定位**：现代 VLA 微调选择 [[LoRA]] 而非原生 Adapter，因为 VLA 部署对推理实时性要求更高（机器人控制通常要求 <50ms）
- **AdapterFusion 的启发**：多机器人形态（Aloha、Franka、UR5）可训练独立 Adapter 模块后融合，这一思路与 [[LoRA]] 的多任务热切换在实际工程上等价
- **历史价值**：Adapter 的 LayerNorm 设计、近零初始化、残差连接等工程细节，被 [[LoRA]]、DoRA、PiSSA 等大量后续工作继承

### 硬件兼容性总结
- ✅ 训练 BERT 系列 + Adapter：T4 (16GB)、RTX 2080/3090
- ✅ 在 LLM 上使用 Adapter-style 方案：24GB 消费级 GPU（需配合 [[LoRA]] 实现）
- ⚠️ Adapter 在现代 LLM 推理中的延迟代价：batch size 越小越显著（可能拖累实时交互）
- ❌ Adapter 不适合超大批量纯推理服务：延迟增加幅度超过量化带来的收益

## PDF

[[Adapter 原文.pdf]]
