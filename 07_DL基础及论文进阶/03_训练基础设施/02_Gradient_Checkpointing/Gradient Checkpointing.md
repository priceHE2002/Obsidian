---
tags:
  - 论文
  - 训练基础设施
  - 显存优化
  - 激活值检查点
created: 2026-06-30
paper_title: "Training Deep Nets with Sublinear Memory Cost"
paper_authors: "Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin"
paper_year: 2016
paper_venue: "NeurIPS 2016"
paper_citations: "~2,000+"
paper_url: "https://arxiv.org/abs/1604.06174"
github: ""
---

# Gradient Checkpointing

**Training Deep Nets with Sublinear Memory Cost**
*Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin | CMU & Microsoft | NeurIPS 2016 | arXiv: 1604.06174*

> 用 $O(\sqrt{n})$ 激活值显存训练 $n$ 层深度网络——每 $\sqrt{n}$ 层保存一个检查点，反向传播时从检查点重算中间激活。这是"以计算换显存"范式的开创性工作，让 $n>100$ 层网络在 $O(1)$ 显存中可训练成为可能。

---

## 一、Background / Core Idea

### 1.1 问题：激活值显存随深度线性增长

深层神经网络训练的一个隐性瓶颈是**激活值（activations）的显存消耗**：

$$\text{训练显存} = \text{模型参数} + \text{优化器状态} + \text{激活值} + \text{临时缓冲区}$$

对于 $n$ 层网络，每层前向需要存储激活值供反向传播计算梯度时使用。设批次大小为 $b$，每层激活体积为 $f(l)$，则激活值总显存为 $\sum_{l=1}^n f(l) \propto O(n)$。

| 模型 | 层数 | 激活值显存（FP32, bs=32） | 总训练显存 |
|:---|:----:|:------------------------:|:----------:|
| ResNet-152 | 152 | ~8.7GB | ~10.5GB |
| DenseNet-264 | 264 | ~18.2GB | ~21.0GB |
| GPT-2 (1.5B) | 48 | ~45GB | ~70GB |
| GPT-3 175B | 96 | ~2.8TB | ~3.5TB |

**激活值最终成为比参数更大的显存占用者。** 参数可以通过模型并行分片，但激活值在同一设备上逐层依赖。

### 1.2 核心洞察：反向传播的计算拓扑

传统的自动微分使用**前向缓存（forward caching）**：前向时存储每一层的输出 $a^{(i)} = f_i(a^{(i-1)})$，反向时依次使用：

$$\text{前向: } a^{(1)}, a^{(2)}, ..., a^{(n)} \quad \text{全部存储}$$

$$\text{反向: } \frac{\partial \mathcal{L}}{\partial a^{(n)}}, \frac{\partial \mathcal{L}}{\partial a^{(n-1)}},..., \frac{\partial \mathcal{L}}{\partial a^{(1)}} \quad \text{需要 } a^{(i)}$$

论文的关键洞察：**反向传播只需要特定位置的激活值，其余可以通过"重算"恢复**。这等价于在计算图中选择一组"检查点"（checkpoints），每次反向传播到两个检查点之间时，从上游检查点开始重新前向计算。

### 1.3 计算—存储权衡的形式化

设网络为 $n$ 层，检查点间隔为 $k$。存储代价与计算代价的 trade-off 为：

| 策略 | 存储复杂度 | 计算额外开销 | 典型适用场景 |
|:----|:---------:|:-----------:|:-----------:|
| 无检查点（全存） | $O(n)$ | 0× | 显存充足 |
| 每 $k$ 层检查点 | $O(n/k + k)$ | $O(k)$ × 重算 | 平衡模式 |
| 最优检查点 | $O(\sqrt{n})$ | $O(\sqrt{n})$ × 重算 | **论文核心贡献** |
| 极简（仅存输入） | $O(1)$ | $O(n)$ × 重算 | 调试模式 |

当 $k = \sqrt{n}$ 时，存储复杂度 $n/k + k = 2\sqrt{n} = O(\sqrt{n})$，计算开销约为 $O(\sqrt{n})$ 倍额外前向。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 最优检查点策略算法

论文提出了一种**确定最优检查点位置**的贪心策略。设 $F(N, M)$ 为在 $M$ 块显存限制下训练 $N$ 层网络的最小计算代价：

$$F(N, M) = \min_{1 \le k \le N} \left\{ \underbrace{T(k, N)}_{\text{重算代价}} + \underbrace{F(N - k, M - 1)}_{\text{子问题}} \right\}$$

其中 $T(k, N)$ 是在第 $N-k$ 层检查点、从该检查点重算至第 $N$ 层的计算代价。

**实际实现**采用更简单的策略：

1. 将 $n$ 层网络均匀划分为 $\sqrt{n}$ 个段（segment）
2. 每个段保存第一层的输入作为检查点
3. 反向传播到段 $i$ 时，从段 $i$ 的检查点开始重新前向计算段内所有激活值
4. 计算段内各层的梯度

### 2.2 激活值 vs 梯度存储的不同处理

| 数据类型 | 检查点策略 | 存储时机 | 释放时机 |
|:--------|:----------|:--------|:--------|
| 激活值 $a^{(i)}$ | **分段检查点** | 仅保存每段首层 | 重算后立即释放 |
| 权重 $W^{(i)}$ | **全程保留** | 前向初始化 | 训练结束 |
| 梯度 $\nabla W^{(i)}$ | **随用随丢** | 反向计算 | 权重更新后释放 |
| 中间临时变量 | **不保存** | 仅反求时存在 | 反求后释放 |

### 2.3 与自动微分的集成

论文在 MXNet 框架中实现了这一策略，关键代码逻辑为：

```
函数 train_network(f, n, M):
    checkpoints = select_checkpoints(n, M)  // 选择检查点位置
    前向时:
        for i = 1 to n:
            if i in checkpoints: 保存 a^{(i)} 到检查点列表
            a^{(i+1)} = f_i(a^{(i)})
    反向时:
        for i = n down to 1:
            if i in checkpoints: 从上一检查点加载
            else: 从最近检查点重算到第 i 层
            ∇W^{(i)} = a^{(i)} · δ^{(i+1)}  // 使用重算后的激活
```

**关键**：微妙的实现细节是检查点保存的时机——必须在计算图构建阶段标记，而非运行时，因为计算图的拓扑决定了重算链的起点。

### 2.4 数学上的计算代价分析

设一个深度网络的前向计算代价为 $C_{\text{fwd}}$，反向计算代价通常为 $C_{\text{bwd}} \approx 2 \cdot C_{\text{fwd}}$（具体取决于自动微分的实现）。

| 策略 | 总计算时间 | 相对于无检查点的开销比例 |
|:----|:----------:|:----------------------:|
| 无检查点 | $C_{\text{fwd}} + C_{\text{bwd}} \approx 3C_{\text{fwd}}$ | 1.0×（基准） |
| $\sqrt{n}$ 均匀分割 | $C_{\text{bwd}} + \sqrt{n} \cdot C_{\text{fwd}}$ | $\frac{2+\sqrt{n}}{3}$ |
| $n=100, \sqrt{n}=10$ | $C_{\text{bwd}} + 10 \cdot C_{\text{fwd}} = 12C_{\text{fwd}}$ | **4.0×** |
| $n=256, \sqrt{n}=16$ | $C_{\text{bwd}} + 16 \cdot C_{\text{fwd}} = 18C_{\text{fwd}}$ | **6.0×** |

实践中，均匀分割策略的计算开销是对角线增长的。论文指出：**更优的检查点选择可以降低到约 1.5-2.0×**（通过利用网络不同阶段的不同激活体积和计算开销）。

---

## 三、Experiments and Key Findings

### 3.1 实际显存节省效果

| 模型 | 无检查点 (MB) | 检查点策略 (MB) | 节省比例 |
|:---|:------------:|:--------------:|:--------:|
| ResNet-110 (CIFAR-10) | 519 | **47** | **91.0%** |
| ResNet-152 (ImageNet) | 1850 | **147** | **92.1%** |
| DenseNet-100 (CIFAR-10) | 722 | **65** | **91.0%** |
| DenseNet-264 (ImageNet) | 2530 | **201** | **92.1%** |
| Inception-v4 | 2480 | **196** | **92.1%** |

**显存从 GB 级降至 MB 级**，变化幅度与 $O(n) \to O(\sqrt{n})$ 的理论预测吻合。

### 3.2 训练时间开销

| 模型 | 无检查点 (ms/iter) | 检查点策略 (ms/iter) | 开销比例 |
|:---|:-----------------:|:-------------------:|:--------:|
| ResNet-110 | 210 | 279 | **1.33×** |
| ResNet-152 | 310 | 409 | **1.32×** |
| DenseNet-100 | 288 | 370 | **1.28×** |
| DenseNet-264 | 456 | 571 | **1.25×** |

**实际开销远低于理论 worst-case（$4\times$）**，原因有三：
1. 重算部分（前向）计算效率高于反向传播（需要大量写操作）
2. 显存约束放松后可以增加 batch size，抵消计算开销
3. CUDA kernel launch 延迟被分摊

### 3.3 精度验证

| 模型 | 无检查点 (Top-1) | 检查点策略 (Top-1) | 差异 |
|:---|:---------------:|:-----------------:|:----:|
| ResNet-110 (CIFAR-10) | 93.57% | 93.68% | $+0.11\%$ |
| ResNet-152 (ImageNet) | 77.80% | 77.84% | $+0.04\%$ |
| DenseNet-264 (ImageNet) | 81.90% | 81.80% | $-0.10\%$ |

**检查点策略不改变数值计算精度**——重算的结果与原始前向的结果在浮点误差范围内一致。

---

## 四、Limitations and Challenges

1. **计算开销在大模型上不可忽略**：对于 GPT-3 175B 规模，每个迭代的额外重算成本可达 30-50%。虽然显存节省了 90%+，但训练时间显著延长
2. **均匀分割非最优**：论文的 $\sqrt{n}$ 分割假设每层的激活体积均匀，但实际上 Transformer 中自注意力的激活体积随序列长度平方增长，需要非均匀检查点策略
3. **与数据并行的交互**：梯度检查点减少的是每设备的显存占用，但数据并行的梯度通信不受影响——计算开销叠加在通信之上
4. **batch size 极小时无收益**：当 batch size = 1 时，激活值显存已接近参数的显存，检查点的收益被参数显存稀释
5. **框架集成复杂度**：检查点需要框架级别的计算图标注支持（如 PyTorch 的 `torch.utils.checkpoint`），手动实现极易出错（容易产生计算图泄漏）
6. **激活值重算的数值一致性问题**：若模型包含 dropout、随机深度等随机操作，重算会得到不同结果——需要实现 seed 回溯

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 Gradient Checkpointing 的关系 |
|:--------|:----:|:-------------------------------|
| **Mixed Precision Training** (Micikevicius et al.) | 2018 | 两者正交叠加：FP16 激活值 + FP32 检查点 + 重算，显存进一步减半 |
| **Megatron-LM** (Shoeybi et al.) | 2019 | 张量并行中集成激活值检查点，解决大序列的激活值显存爆炸 |
| **ZeRO** (Rajbhandari et al.) | 2020 | 三阶段显存优化中检查点用于减少激活值，与参数/优化器状态的分片互补 |
| **3D Parallelism** (Narayanan et al.) | 2021 | PTD-P 中检查点+张量并行+流水线并行共同管理激活值显存 |
| **Rematerialization (Checkpointing)** (Griewank & Walther) | 2000 | 更早的先驱工作——自动微分领域的最优重算点选择理论 |
| **Selective Checkpointing** (Rhu et al.) | 2020 | 选择性的逐层检查点策略，利用每层激活体积的异质性 |
| **SwapOut** (Wang et al.) | 2018 | 将激活值换出到 CPU 内存而非重算，减少计算开销 |

**影响评估**：Gradient Checkpointing 是**大模型训练的事实标准组件**。在 PyTorch 中一行 `torch.utils.checkpoint.checkpoint(fn, *args)` 即可启用，是所有 GPU 显存捉襟见肘场景的默认配置。它不属于可选项，而是在有限显存下训练超大规模模型的前提条件。

---

## 六、Implications for You / Hardware Compatibility

### 梯度检查点在不同 GPU 上的影响

| GPU 型号 | 显存 (GB) | 无检查点可训模型 | 有检查点可训模型 | 实际开销比例 |
|:--------|:--------:|:---------------:|:---------------:|:----------:|
| RTX 3060 | 12 | ~4B (bs=1, AMP) | ~7B (bs=1, AMP+QLoRA) | ~1.25× |
| RTX 4060 | 12 | ~4B | ~7B | ~1.25× |
| RTX 3090 | 24 | ~7B (bs=1, AMP) | ~13B (bs=1, AMP+QLoRA) | ~1.30× |
| RTX 4090 | 24 | ~7B (bs=2) | ~13B (bs=1+梯度累积) | ~1.25× |
| A100 80GB | 80 | ~13B (bs=4) | ~30B (bs=2, AMP+ZeRO-3) | ~1.35× |
| H100 80GB | 80 | ~30B (FP8, bs=2) | ~70B (FP8+检查点+ZeRO-3) | ~1.20× |

### 对大规模训练的指导

- **检查点 + 混合精度是显存优化的最低配置**：二者组合可训练纯 FP32 模式下 4-6 倍规模的模型
- **与 QLoRA 配合**：4-bit QLoRA + gradient checkpointing 是消费级 GPU 训练 7B 模型的现实路径
- **Transformer 中特殊处理**：不要在 self-attention 内部使用检查点（QKV 计算已极轻量），重点在 FFN 层
- **批量大小与检查点间隔**：当 $n$ 很大时（>100），检查点间隔应取 $k = \sqrt{n}$ 的经验值；当显存极有限时，调整为每层检查点（$k=n$，退化到 $O(1)$ 模式）
- **训练脚本实践**：始终启用 `torch.utils.checkpoint`，除非已确认显存充足

### 硬件兼容性总结

- ✅ 梯度检查点：全平台兼容（与 GPU 架构无关，纯计算策略）
- ✅ 与 AMP（FP16/BF16）正交兼容，可叠加使用
- ✅ 与 ZeRO-3 分片配合，使 7B 模型在 12GB GPU 上训练成为可能
- ⚠️ 计算开销在大规模集群上累计显著（1000+ GPU 集群中约 20-30% 的总训练时间用于重算）
- ❌ 不兼容某些量化训练策略（如纯 INT8 训练，需要存储量化参数）

## PDF

[[Gradient Checkpointing 原文.pdf]]
