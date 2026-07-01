---
tags:
  - 论文
  - 训练基础设施
  - 混合精度
  - 显存优化
created: 2026-06-30
paper_title: "Mixed Precision Training"
paper_authors: "Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksi Kuchaiev, Ganesh Venkatesh, Hao Wu"
paper_year: 2018
paper_venue: "ICLR 2018"
paper_citations: "~6,000+"
paper_url: "https://arxiv.org/abs/1710.03740"
github: ""
---

# Mixed Precision Training

**Mixed Precision Training**
*Paulius Micikevicius, Sharan Narang, Jonah Alben et al. | NVIDIA & Baidu | ICLR 2018 | arXiv: 1710.03740*

> 以 FP16 存储激活值和权重、以 FP32 维护一份主权重副本并累加梯度，在 Volta V100 GPU 上实现 2-5 倍训练加速且几乎不损失精度。AMP（Automatic Mixed Precision）的奠基性工作——当前所有大规模训练框架（PyTorch AMP、TensorFlow AMP、Megatron、DeepSpeed）的底层依赖。

---

## 一、Background / Core Idea

### 1.1 问题：FP32 训练的瓶颈与 FP16 的精度困境

深度学习模型的训练精度需求与计算效率之间存在根本矛盾：

- **FP32（全精度）**：提供 8 位指数 + 23 位尾数 ≈ 7 位十进制精度，训练稳定但吞吐受限。V100 的 FP32 峰值算力为 15.7 TFLOPS，FP16 则为 125 TFLOPS（Tensor Cores，相差约 8 倍）
- **FP16（半精度）**：IEEE 标准 5 位指数 + 10 位尾数 ≈ 3.3 位十进制精度，可表示的数值范围仅为 $[2^{-24}, 2^{15}] \approx [6\times 10^{-8}, 65504]$。梯度在训练中极易下溢为 0（当梯度值 < $2^{-24} \approx 6\times 10^{-8}$）

| 数据类型 | 指数位 | 尾数位 | 最大可表示值 | 最小有效值 | 相对精度 |
|:--------:|:------:|:------:|:-----------:|:----------:|:--------:|
| FP32 | 8 | 23 | $\approx 3.4 \times 10^{38}$ | $\approx 1.4 \times 10^{-45}$ | $\sim 1.2 \times 10^{-7}$ |
| FP16 | 5 | 10 | 65504 | $6 \times 10^{-8}$ | $\sim 9.8 \times 10^{-4}$ |
| FP16 的问题 | — | — | 足够 | **梯度下溢风险** | **精度不足** |

### 1.2 核心洞察：三分法 + 主权重副本

论文提出一个简单但极其有效的思想：**不需要所有操作都使用同一精度**。将训练过程拆分为三个角色：

1. **主权重（Master Copy, FP32）**：权重更新必须在 FP32 中进行，因为 FP16 的精度不足以累加大量微小梯度更新
2. **前向/反向传播（FP16）**：使用 Tensor Cores 以 FP16 执行矩阵乘法和卷积，获得 8 倍加速
3. **梯度累加（FP32）**：FP16 梯度转换为 FP32 后再更新权重

$$\text{FP32 主权重 } W_{32} \xrightarrow{\text{cast}} W_{16} \xrightarrow{\text{FP16 前向}} \text{loss} \xrightarrow{\text{FP16 反向}} \nabla W_{16} \xrightarrow{\text{cast}} \nabla W_{32} \xrightarrow{\text{FP32 update}} W_{32} \leftarrow \text{更新完成}$$

### 1.3 Loss Scaling：对抗梯度下溢的关键

即使使用 FP16 存储梯度，许多梯度仍会下溢至零。Loss Scaling 的解决方案极其简洁：**在前向计算后放大 loss**，使反向传播的梯度相应放大后落入 FP16 的可表示范围：

$$\mathcal{L}' = \mathcal{L} \cdot S$$

其中 $S$ 为缩放因子（一般 $S=8$ 到 $S=32768$）。训练完成后，梯度在更新权重前除以 $S$ 还原：

$$W_{t+1} = W_t - \eta \cdot \frac{\nabla \mathcal{L}'}{S}$$

论文进一步提出**动态 loss scaling**：以 $2^{15} = 32768$ 为初始值，若连续 N 次迭代未发生溢出则乘以 2，发生溢出则除以 2。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 完整的混合精度训练流程

| 步骤 | 精度 | 存储位置 | 计算量占比 |
|:----:|:----:|:--------:|:----------:|
| 权重初始化 | FP32 | 显存 | 一次性 |
| **前向传播** | **FP16** | FP16 Tensor | $\sim 50\%$ |
| Loss 计算 | FP32 | Scalar | $\ll 1\%$ |
| Loss Scaling | FP32 | Scalar | $\ll 1\%$ |
| **反向传播** | **FP16** | FP16 Tensor | $\sim 50\%$ |
| 梯度反缩放 | FP32 | 临时 | $\ll 1\%$ |
| 权重更新 | FP32 | FP32 主权重 | $2N$（N 为参数数） |
| FP16 副本更新 | FP32→FP16 | 权重指针 | 每次迭代 |

**关键约束**：所有需要累加的操作（梯度、权重更新、优化器状态）必须在 FP32 中完成。仅矩阵乘法和卷积在前向/反向时可以使用 FP16。

### 2.2 累加为 FP32 的数学必要性

考虑一个典型的梯度更新场景：学习率 $\eta = 3\times 10^{-5}$，Adam $\beta_2 = 0.999$，梯度的二阶动量积累以 $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ 进行。

在 FP16 中，$v_t$ 从第 $\approx 1000$ 步开始就会下溢，因为 $(0.999)^{1000} \approx 2.7 \times 10^{-5}$ 已逼近 FP16 最小表示范围。FP32 累加是保证训练不崩溃的必要条件。

论文使用 **FP32 权重副本**（master weights）而非 FP16 权重直接更新：

$$W_{32} \leftarrow W_{32} - \eta \cdot \frac{\partial\mathcal{L}}{\partial W_{16}}$$

然后截断拷贝回 FP16 供下次前向使用：$W_{16} = \text{cast}(W_{32}, \text{FP16})$。

### 2.3 整数算术指令的利用（V100 Tensor Cores 早期版本）

Volta V100 的 Tensor Cores 在 FP16 输入上执行 $4\times 4$ 矩阵乘加，累加至 FP32 再写回 FP16：

$$D_{16} = \text{cast}(A_{16} \times B_{16} + C_{16}, \text{FP16})$$

论文的混合精度策略与 Tensor Cores 的硬件行为完美对齐：**FP16 输入 + FP32 内部累加 + FP16 输出**。这解释了为何混合精度训练在 V100 上首次落地。

### 2.4 各网络架构的特殊处理

| 网络类型 | Loss Scale 需求 | 特殊处理 |
|:--------:|:--------------:|:---------|
| CNN（ResNet-50/152） | 低（$S=8$） | BatchNorm 在 FP32 计算 |
| RNN/LSTM（GNMT） | 中（$S=128$） | LSTMCell 内部矩阵在 FP32 |
| Transformer（BERT） | 高（$S=1024$） | LayerNorm + Softmax 在 FP32 |
| SSD/Faster R-CNN | 极高（$S=2048$） | 定位损失梯度极小 |

**规律**：越深的网络、越小的 mini-batch 对 loss scale 的需求越高。

---

## 三、Experiments and Key Findings

### 3.1 单卡加速比（V100 vs P100）

| 模型 | FP32 基准 (P100) | FP32 基准 (V100 TF32) | 混合精度 (V100) | 加速比 | 精度差异 |
|:---:|:-:|:-:|:-:|:-:|:-:|
| ResNet-50 (BS=128) | 410 img/s | 770 img/s | **1330 img/s** | **3.2×** | $<0.1\%$ |
| Inception v3 (BS=128) | 310 img/s | 580 img/s | **1060 img/s** | **3.4×** | $<0.1\%$ |
| GNMT (BS=128) | — | 3700 tok/s | **6870 tok/s** | **1.9×** | BLEU $<0.1$ |
| SSD300 (BS=32) | — | 140 img/s | **390 img/s** | **2.8×** | mAP $+0.3$ |

**混合精度训练的加速主要来自三方面**：Tensor Cores 的 8 倍峰值算力提高、FP16 存储的显存减半（允许更大 batch size）、减少 PCIe 传输带宽需求。

### 3.2 多卡扩展性

| GPU 数量 | ResNet-50 FP32 (img/s) | ResNet-50 Mixed (img/s) | 扩展效率 |
|:-------:|:---------------------:|:----------------------:|:--------:|
| 1 | 770 | 1330 | — |
| 4 | 2930 | 5040 | 94.7% |
| 8 | 5470 | 9860 | 92.6% |
| 32 | 19200 | 31000 | 72.8% |

大 scale 下混合精度的扩展效率略低于 FP32，主要因为 FP16 下更低的算术强度限制了通信隐藏能力。

### 3.3 Loss Scale 的敏感性

| Loss Scale S | ResNet-50 (Top-1) | GNMT (BLEU) | SSD300 (mAP) |
|:-----------:|:-----------------:|:-----------:|:------------:|
| 1（无 scaling） | 75.5 | 24.0 | 不可训练 |
| 8 | 76.2 | 24.2 | 不可训练 |
| 128 | **76.3** | **24.6** | 65.1 |
| 1024 | 76.2 | 24.5 | 73.5 |
| 32768 (动态) | 76.2 | 24.6 | **75.3** |

**损失缩放极其敏感——太小无法恢复梯度，太大导致溢出（inf/NaN）。动态 Loss Scaling 是最稳健的选择。**

### 3.4 残差梯度分析

论文首次分析了残差网络中梯度的数值分布：

$$\text{梯度分布中位数} \approx 10^{-4},\; \text{最小值} \approx 10^{-7}$$

在 FP16 中可表示的最小正数为 $6 \times 10^{-8}$，这意味着约有 $0.1\%$ 的梯度在 FP16 中会下溢。Loss scaling 将这些梯度"推"入可表示范围。

---

## 四、Limitations and Challenges

1. **硬件绑定**：论文方法依赖于 NVIDIA V100 的 Tensor Cores 特性。在 AMD GPU、Google TPU 或 CPU 上无法直接使用相同策略
2. **Loss Scale 的超参数负担**：虽然动态 loss scaling 减少了调参，但它引入了额外的溢出检测逻辑，对 batch size 特别敏感（大 batch 下梯度更集中，loss scale 需求改变）
3. **Tensor Cores 精度非确定性**：由于 Tensor Cores 内部累加顺序的差异，混合精度训练的结果在硬件层面不是严格确定性的——这对调试和可复现性造成困扰
4. **特定层的 FP32 需求**：LayerNorm、Softmax 和 BatchNorm 仍需要 FP32 计算，这些操作在 Transformer 架构中频繁出现，限制了加速比
5. **主权重副本的显存消耗**：FP32 主权重 + FP16 工作权重 = 1.5 倍模型参数显存，在超大模型（>10B）下仍然构成显著压力（[[ZeRO]] 通过分片进一步优化）
6. **小模型加速有限**：当模型算力密度较低时（如小型 RNN），FP16 的加速收益被计算/访存比降低抵消

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 Mixed Precision Training 的关系 |
|:--------|:----:|:---------------------------------|
| [[Megatron-LM]] (Shoeybi et al.) | 2019 | 在张量并行框架中原生集成混合精度训练，扩展到 8.3B 参数 |
| [[ZeRO]] (Rajbhandari et al.) | 2020 | FP16 主权重的显存分片，进一步降低混合精度存储需求 |
| [[3D Parallelism]] (Narayanan et al.) | 2021 | 混合精度作为 PTD-P 三维并行基础设施的核心底层 |
| **Apex O2/O3** (NVIDIA) | 2019 | 自动混合精度库，论文的直接工程产出 |
| **PyTorch AMP** | 2020 | 原生 `torch.cuda.amp` 支持，GPUs 自动选择精度 |
| **FP8 Training** (NVIDIA) | 2023 | 混合精度的下一代演进，H100 Transformer Engine 原生支持 FP8 |
| **BF16 Training** | 2020 | Google 的 Brain Float 16，消除 loss scaling 需求（8位指数） |

**影响评估**：混合精度训练是**整个大规模深度学习的基础设施**。没有它，V100 之后的每代 GPU 都无法发挥峰值算力。从 BERT（2018）到 GPT-4（2023），所有超过 1B 参数的模型训练完全依赖此技术。PyTorch AMP 的自动封装使其成为透明基础设施。

---

## 六、Implications for You / Hardware Compatibility

### GPU 训练性能对比

| 配置 | FP32 相对速度 | 混合精度相对速度 | 显存需求（7B 模型） |
|:----|:------------:|:---------------:|:-----------------:|
| RTX 3090 (24GB) | 1.0× | **2.5-3.5×** | ~14GB (AMP) |
| RTX 4090 (24GB) | 1.3× | **4.0-5.0×** | ~14GB (AMP) |
| A100 (80GB) | 1.8× | **6.0-7.5×** | ~28GB (AMP + 大 batch) |
| RTX 4060 (12GB) | 0.5× | **1.8-2.2×** | ❌ AMP 模式可运行 7B（bs=1） |
| H100 (80GB) | 2.0× (TF32) | **10-12×** (FP8) | ~14GB (FP8) |

### 对大规模训练的指导

- **AMP 已成为默认选项**：PyTorch `torch.cuda.amp.autocast()` + `GradScaler()` 是所有 >1B 模型训练的标配
- **Loss Scale 策略选择**：推荐动态 loss scaling（`GradScaler` 默认）。若发现 mask 中出现大量 inf/NaN，先检查学习率是否过大，而非调整 loss scale
- **关键算子的精度需求**：LayerNorm、Softmax、CrossEntropyLoss 必须在 FP32；MatMul 和 Conv 在 FP16/BF16；Embedding 建议 FP32
- **与 gradient checkpointing 配合**：[[Gradient Checkpointing]] 的激活值检查点在 FP16 下可节省更多显存，两者正交叠加

### 硬件兼容性总结

- ✅ 混合精度训练（AMP）：RTX 3060 及以上全系支持（需 Turing+ 架构）
- ✅ Tensor Cores FP16：Turing (RTX 20xx) + Ampere (RTX 30xx / A100) + Hopper (H100)
- ✅ BF16 混合精度：Ampere+ 架构原生支持（消除 loss scaling 需求）
- ⚠️ FP16 AMP 在双向 LSTM 等低算术强度模型上加速有限
- ❌ 纯 FP16 训练（无主权重副本）：几乎所有架构不可用

## PDF

[[Mixed Precision Training 原文.pdf]]
