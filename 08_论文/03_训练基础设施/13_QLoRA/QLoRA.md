---
tags:
  - 论文
  - 训练基础设施
  - PEFT
  - 量化
  - LoRA
  - QLoRA
created: 2026-06-30
paper_title: "QLoRA: Efficient Finetuning of Quantized LLMs"
paper_authors: "Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer"
paper_year: 2023
paper_venue: "NeurIPS 2023"
paper_citations: "~3,500+"
paper_url: "https://arxiv.org/abs/2305.14314"
github: "https://github.com/artidoro/qlora"
---

# QLoRA

**QLoRA: Efficient Finetuning of Quantized LLMs**
*Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer | University of Washington | NeurIPS 2023 | arXiv: 2305.14314*

> 将预训练模型量化为 **4-bit NormalFloat (NF4)**，配合**双重量化**（Double Quantization）和**分页优化器**（Paged Optimizers），在仅使用一张 48GB GPU 的情况下成功微调 **65B LLaMA** 模型，且性能损失几乎为零。QLoRA 使得消费级 GPU 微调大规模 LLM 成为可能——这是 2023 年开源 LLM 生态爆发的关键推动力。对 VLA 研究而言，QLoRA 是让 7B-13B VLA 模型在 12-24GB GPU 上可微调的核心技术。

---

## 一、Background / Core Idea

### 1.1 问题：大模型微调的显存瓶颈

[[LoRA]] 虽然大幅减少了可训练参数（<1%），但存储和计算 LoRA 之外的基模型权重仍需要大量显存：

| 模型规模 | fp16 基模型 | LoRA 梯度 | 优化器状态 | **总训练显存** |
|:--------:|:----------:|:---------:|:---------:|:------------:|
| LLaMA-7B | ~14GB | ~140MB | ~280MB | ~18GB |
| LLaMA-13B | ~26GB | ~260MB | ~520MB | ~34GB |
| LLaMA-33B | ~66GB | ~660MB | ~1.3GB | ~80GB |
| LLaMA-65B | ~130GB | ~1.3GB | ~2.6GB | **~160GB** |

其中基模型权重是显存消耗的绝对主体。量化可以大幅压缩这部分，但**标准量化方法无法在微调中保持精度**——反向传播的梯度噪声会放大量化误差。

### 1.2 核心洞察：信息理论上最优的 NF4 量化

QLoRA 的核心创新来源于对**信息论量化**的深刻理解：

> 神经网络权重的分布通常呈现**零均值、正态形状**（训练良好时）。标准 INT4 量化假设均匀分布，导致信息浪费——量化层级中大量 bin 落入低概率区间，而高概率中心区域的 bin 不足。

NormalFloat (NF4) 设计思路：找到一个量化级别 $\{q_i\}_{i=1}^{16}$，使得标准正态分布 $N(0,1)$ 的 16 个分位数被**等概率**覆盖：

$$\int_{-\infty}^{q_{i+1}} p(x)dx = \frac{i+1}{16}, \quad p(x) \sim \mathcal{N}(0,1)$$

这样，每个量化级别对应输入分布中**同等信息量**的区域。

### 1.3 QLoRA 的整体架构

```
┌─────────────────────────────────────────┐
│           预训练权重 (NF4)               │  ← 冻结，4-bit 存储
├─────────────────────────────────────────┤
│        双重量化 (fp8 尺度因子)            │  ← 冻结，进一步压缩
├─────────────────────────────────────────┤
│                LoRA B A (bf16)           │  ← 唯一可训练模块
├───┬───┬───┬───┬───┬───┬───┬───┬───┬───┤
│         分页优化器 (CPU offload)          │  ← 应对梯度检查点 OOM
└─────────────────────────────────────────┘
```

每个组件解决显存消耗的一个环节，是**系统级优化**的典范。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 NormalFloat (NF4) 量化

**信息论最优的 4-bit 数据类型**。

给定剪裁因子 $\alpha$（通常取权重的最大绝对值），归一化权重 $w/\alpha \sim N(0,1)$ 被映射到 16 个量化区间：

量化过程分为两步：

1. **分位数计算**：将 $N(0,1)$ 的 CDF 等分为 16 个区间，得到分裂点
2. **对称扩展**：考虑到神经网络权重通常有对称分布，NF4 使用**对称分位数**（省略 $q_0$ 和 $q_{16}$ 两个极端）：

$$q_i = \Phi^{-1}\left(\frac{i}{16} + \frac{1}{32}\right), \quad i \in \{0,...,15\}$$

其中 $\Phi^{-1}$ 是标准正态 CDF 的逆函数（probit 函数）。

| 对比维度 | NF4 (QLoRA) | INT4 | INT8 | fp16 |
|:-------:|:-----------:|:----:|:----:|:----:|
| 存储容量 | 4-bit | 4-bit | 8-bit | 16-bit |
| 分布假设 | **正态**（匹配真实权重） | 均匀（不匹配） | 均匀 | —— |
| 表示精度 | **最优分位数** | 线性等距 | 线性等距 | —— |
| 量化误差 | **最小**（信息论意义上） | 较大 | 可忽略 | 0 |

相比于 [[GPTQ]]（后训练层压缩，需要校准集），NF4 无需校准数据，直接对权重进行逐张量量化。

### 2.2 双重量化（Double Quantization, DQ）

NF4 虽然高效，但尺度因子（scaling factor）本身也是一个存储成本：

对于 block size $B=64$，每 64 个权重共享一个 fp32 尺度因子（4 字节）：

$$\text{尺度因子开销} = \frac{4\text{ bytes}}{64\text{ weights}} = 0.5 \text{ bit per weight}$$

对于一个 65B 模型，这相当于增加了 $65B \times 0.5 = 32.5$ bits ≈ **4.07GB** 的额外开销。

**双重量化的解决方案**：对尺度因子本身进行二次量化。

- **第一步**：逐块量化权重到 NF4，产生第一级尺度因子 $c_1$（fp32）
- **第二步**：将 $c_1$ 按更大 block size（$B_2=256$）分组，每组的尺度因子用 fp8 量化，产生第二级尺度因子 $c_2$（fp32）

尺度因子开销变化：

| 配置 | 每权重额外 bits | 65B 模型的开销 |
|:---:|:--------------:|:------------:|
| 无 DQ（fp32 尺度因子） | 0.5 bits | 4.07 GB |
| 有 DQ（fp8 二次量化） | $\frac{4}{256} + \frac{1}{64} \approx 0.127$ bits | **~1.03 GB** |

DQ 使得 **NF4 的实际存储从 4.5 bits/weight 降至约 4.125 bits/weight**。

### 2.3 分页优化器（Paged Optimizers）

当 GPU 显存不足时（如单卡 48GB 微调 65B 模型），优化器状态（momentum + variance）可能导致 OOM。

QLoRA 借助 NVIDIA **统一内存**（Unified Memory）技术：优化器状态存储在 CPU 内存中，当 GPU 需要时通过页面错误机制**按需迁移**到 GPU。这与操作系统虚拟内存的分页机制完全一致——数据按 4KB 页面粒度在 CPU-GPU 之间交换。

### 2.4 与标准 LoRA 的整合

QLoRA 并非简单地在量化模型上加 LoRA：

| 组件 | 数据类型 | 存储位置 | 是否可训练 |
|:----:|:-------:|:--------:|:---------:|
| 预训练权重 | **NF4** (4-bit) | GPU | ❌ 冻结 |
| 双重量化尺度因子 | **fp8** | GPU | ❌ |
| LoRA 矩阵 $A,B$ | **bf16** | GPU | ✅ |
| 优化器状态 | — | **CPU** (分页) | ✅ |
| 梯度（检查点后） | bf16 | **CPU** (分页) | — |

**计算过程**：前向传播时，NF4 权重**反量化**（dequantize）到 bf16，与 LoRA 的 bf16 输出相加，计算输出。反向传播时，梯度通过 LoRA 权重更新 $A,B$，但**不更新预训练权重**。

### 2.5 与 LLM.int8() 的技术渊源

QLoRA 的作者 Tim Dettmers 也是 [[LLM.int8()]]（2022）的第一作者。LLM.int8() 首次实现了 8-bit 量化下不损失性能的 Transformer 推理。QLoRA 继承并发展了其核心理念：

- 混合精度分解（LLM.int8() 使用 outlier-aware 通道分离）
- 分块量化（block-wise quantization）
- 反量化计算（dequantize-and-compute）范式

但 QLoRA 首次将量化从"推理加速"扩展到"训练兼容"——NF4 对于梯度的噪声容忍度远优于 INT4。

---

## 三、Experiments and Key Findings

### 3.1 核心结果：性能与全精度的差距

论文在 LLaMA 7B 到 65B 的模型上进行了全面评估：

| 模型 | 方法 | MMLU (5-shot) | 与 fp16 差距 |
|:----:|:----:|:------------:|:-----------:|
| LLaMA-7B | fp16 + LoRA | 38.7 | — |
| LLaMA-7B | **NF4 + DQ + LoRA** | **38.8** | **+0.1** ✨ |
| LLaMA-7B | INT4 + LoRA | 37.9 | -0.8 |
| LLaMA-13B | fp16 + LoRA | 45.8 | — |
| LLaMA-13B | **NF4 + DQ + LoRA** | **46.0** | **+0.2** ✨ |
| LLaMA-33B | NF4 + DQ + LoRA | **56.8** | — |
| LLaMA-65B | NF4 + DQ + LoRA | **63.9** | — |
| LLaMA-65B | fp16 + 全量微调 | 63.4 | — |

> **NF4 + QLoRA 在性能上超过了 fp16 全量微调**（63.9 vs 63.4）。这一违反直觉的结果被归因于 NF4 量化引入的**正则化效应**以及对 LoRA + 量化组合的超参数调优优势。

### 3.2 组件消融实验

| 配置 | MMLU | 显存节省 | 说明 |
|:----:|:----:|:--------:|:----|
| fp16 + LoRA（基线） | 47.3 | — | 13B 模型 |
| **NF4 + DQ + LoRA** | **47.3** | **-74%** | 零退化，最大压缩 |
| NF4 + LoRA（无 DQ） | 47.3 | -69% | 略多 5% 显存 |
| fp4 + DQ + LoRA | 46.8 | -74% | fp4 < NF4 |
| INT4 + DQ + LoRA | 46.5 | -74% | INT4 < NF4 |

**三个结论**：
1. DQ 节省 5% 显存且无损性能
2. NF4 优于 fp4 和 INT4 约 0.3-0.8 个点
3. NF4 + DQ 实现了 fp16 LoRA 的**完全相同**的性能

### 3.3 不同量化级别的显存对比

| 配置 | 7B 训练显存 | 13B 训练显存 | 65B 训练显存 |
|:----:|:----------:|:-----------:|:-----------:|
| fp16 LoRA | ~18GB | ~34GB | ~160GB |
| INT8 LoRA | ~12GB | ~22GB | ~100GB |
| **NF4 + DQ + LoRA** | **~10GB** | **~17GB** | **~48GB** |
| NF4 + DQ（仅推理，无 LoRA 训练）| ~5GB | ~9GB | ~38GB |

**65B 微调仅需一张 48GB GPU**（如 A6000 或 RTX 8000），这是 QLoRA 最具冲击力的成果。

### 3.4 Guanaco 对话模型评估

论文使用 QLoRA 微调了 **Guanaco** 对话模型系列，在 Vicuna benchmark 上与 ChatGPT 对比：

| 模型 | 参数量 | Vicuna 评分 | 与 ChatGPT 对比 |
|:----:|:-----:|:----------:|:--------------:|
| **Guanaco-65B** | 65B | **71.5** | **相当 (ChatGPT: 70.5)** |
| Guanaco-33B | 33B | 68.3 | — |
| Guanaco-13B | 13B | 65.7 | — |
| Guanaco-7B | 7B | 63.2 | — |
| Vicuna-13B（全量微调） | 13B | 65.4 | — |

**65B Guanaco 在对话质量上首次以开源模型追平 ChatGPT**，而训练仅用一张 48GB GPU 和 24 小时。

### 3.5 基准测试全面评估

| 基准 | Guanaco-65B | ChatGPT | GPT-4 |
|:---:|:-----------:|:-------:|:-----:|
| MMLU | 63.9 | 70.0 | 86.4 |
| Vicuna (GPT-4 评估) | 71.5 | 70.5 | 84.2 |
| OASST | 73.5 | 74.3 | — |

---

## 四、Limitations and Challenges

1. **训练速度降低**：尽管显存需求大幅减少，但 NF4 权重的反量化（dequantize）步骤增加了前向传播的计算时间。QLoRA 训练速度通常比 fp16 LoRA 慢 20-40%
2. **NF4 的分布假设局限**：NF4 假设权重服从 $N(0,1)$，但训练过程中权重分布可能漂移。如果 LoRA 更新幅度很大，原权重的量化误差开始显著
3. **量化噪声的累积**：每个训练 step 都执行"NF4 量化 → bf16 反量化 → 计算 → 量化"的循环，多次循环可能累积量化误差
4. **分页优化器的性能开销**：CPU-GPU 页交换的延迟不可忽略（PCIe 带宽限制），在梯度检查点场景下可能成为瓶颈
5. **LoRA rank 受限**：量化权重的更新能力有限，QLoRA 通常需要更大的 rank（$r \geq 32$）来补偿量化损失，但这增加了训练显存
6. **对 [[GPTQ]] 等后训练量化的适配问题**：QLoRA 的 NF4 量化在训练中需要频繁反量化，与 [[GPTQ]] 的静态量化设计哲学冲突，二者不能简单叠加
7. **Guanaco 数据集与评测局限**：论文使用 Open Assistant 数据集（含 ChatGPT 生成数据），存在数据污染风险

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 QLoRA 的关系 |
|---------|:----:|----------------|
| **[[LoRA]]** | 2021 | QLoRA 的基础——低秩适应在量化权重上的应用 |
| **[[LLM.int8()]]** | 2022 | 同作者的 8-bit 推理量化，QLoRA 的 NF4 思路源于此 |
| **[[GPTQ]]** (Frantar et al.) | 2022 | 后训练量化对比基线，不支持训练 |
| **Bitsandbytes** | 2023 | Dettmers 开发的量化库，QLoRA 的 NF4 实现基础 |
| **AWQ** (Lin et al.) | 2023 | 激活感知权重量化，与 QLoRA 正交 |
| **AutoGPTQ** | 2023 | GPTQ 的工程实现，作为 QLoRA 的对比基线 |
| **Unsloth** (2023-) | 2023 | QLoRA 的加速优化版本，减少反量化开销 2x |
| **HuggingFace TRL + PEFT** | 2023 | QLoRA 已成为 PEFT + TRL 的标配训练方案 |
| **DoRA** (Liu et al.) | 2024 | 权重分解 LoRA，与 QLoRA 兼容 |
| **LISA** (Pan et al.) | 2024 | 层重要性采样，与 QLoRA 正交 |

**影响评估**：QLoRA 是 **2023 年开源 LLM 生态爆发的核心技术推手**。它使独立研究者、中小团队能够在消费级硬件上微调 65B+ 规模的模型。2024 年几乎所有面向社区的开源微调方案（从 Alpaca-LoRA 到各种 Llama 3 微调教程）都默认采用 QLoRA 或其变体。

---

## 六、Implications for You / Hardware Compatibility

### GPU 显存需求（QLoRA 微调各规模模型）

| 模型规模 | 配置 | 训练显存 | 推理显存 | 可使用 GPU |
|:--------:|:----:|:--------:|:--------:|:----------|
| LLaMA-7B / Mistral-7B | NF4 + LoRA r=32 | **~10-12GB** | ~5-6GB | ✅ RTX 3060 (12GB) / RTX 4060 (16GB) |
| LLaMA-7B | NF4 + LoRA r=64 | ~12-14GB | ~6GB | ✅ RTX 3090/4090 (24GB) |
| LLaMA-13B | NF4 + LoRA r=32 | ~17-19GB | ~9GB | ✅ RTX 3090/4090 (24GB) |
| LLaMA-33B | NF4 + LoRA r=32 | ~28-30GB | ~19GB | ✅ RTX 4090 (24GB, 需梯度检查点) / A5000 (32GB) |
| LLaMA-65B / Llama-2-70B | NF4 + LoRA r=16 | **~46-48GB** | ~35GB | ⚠️ A6000 (48GB) / A100 (80GB) |
| LLaMA-65B | NF4 + LoRA r=32 | ~52GB | ~36GB | ❌ 需 A100 (80GB) |

### 对 VLA 研究的指导

- **7B VLA 模型 QLoRA 微调已完全可用**（12-16GB GPU）：推荐 NF4 + r=32 + gradient checkpointing，可在 RTX 3060 (12GB) 上以 batch_size=1 运行
- **OpenVLA + QLoRA 是消费级 VLA 微调的标准方案**：同时利用 QLoRA 的显存效率和 LoRA 的推理零延迟
- **NF4 的正则化效应**在 VLA 数据量少时可能带来额外优势——这与论文中 MMLU 上 NF4 超过 fp16 的现象一致
- **Gradient checkpointing + Paged Optimizers** 的组合：对于序列长度较长的 VLA 任务（如 224×224 图像 patch 序列），梯度检查点节省的显存尤为显著
- **多模态适配建议**：同时量化视觉编码器（SigLIP/DINOv2）和 LLM backbone，但视觉编码器的分布不一定符合正态假设（可能不适合 NF4）

### 硬件兼容性总结
- ✅ QLoRA 微调 7B 模型：RTX 3060 (12GB) 起即可运行
- ✅ QLoRA 微调 13B 模型：RTX 3090/4090 (24GB) 流畅运行
- ✅ QLoRA 微调 33B 模型：RTX 4090 (24GB，需梯度检查点)
- ⚠️ QLoRA 微调 65B-70B 模型：需要 A6000 (48GB) 或 A100 (80GB)
- ❌ QLoRA 不适合需要快速训练的迭代实验场景（训练速度比 fp16 慢 20-40%）
- ❌ QLoRA 不适合推理 > 65B 参数规模的模型（量化无法将 130B 以下压入 48GB 训练）

## PDF

[[QLoRA 原文.pdf]]
