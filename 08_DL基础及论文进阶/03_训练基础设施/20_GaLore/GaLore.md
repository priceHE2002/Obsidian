---
tags:
  - 论文
  - 训练基础设施
  - 梯度压缩
  - 低秩
  - 显存优化
created: 2026-06-30
paper_title: "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
paper_authors: "Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian"
paper_year: 2024
paper_venue: "ICML 2024"
paper_citations: "~400+"
paper_url: "https://arxiv.org/abs/2403.03507"
github: "https://github.com/jiaweizzhao/GaLore"
---

# GaLore

**GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection**
*Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian | Meta AI & Caltech & UT Austin & MIT | ICML 2024 | arXiv: 2403.03507*

> 将梯度 $\nabla W$ 实时投影到低秩子空间（$W \in \mathbb{R}^{m \times n}, r \ll \min(m,n)$），使神经网络全参数训练的优化器状态（Adam 动量）显存从 $\mathcal{O}(mn)$ 降低到 $\mathcal{O}(m+n)r$。在保持全参数训练效果的同时，7B 模型的全参数训练显存可降至与 [[LoRA]] 相似的量级。

---

## 一、Background / Core Idea

### 1.1 问题：全参数训练 LLM 的显存瓶颈

训练大语言模型时，显存消耗主要由三部分构成：

| 组件 | 7B 模型 (bf16) | 说明 |
|------|:-:|------|
| 模型参数 | ~14 GB | fp16 格式 |
| 梯度 | ~14 GB | 与参数同大小 |
| **优化器状态** (Adam) | **~28 GB** | momentum + variance，各 14 GB |
| 激活值 (Activation) | ~30-60 GB | 取决于 batch size、序列长度 |
| **训练合计** | **~86-116 GB** | 远超单卡 A100 80GB |

Adam 优化器存储两个状态变量（momentum m, variance v），每个是参数大小的 2x 浮点数。因此**优化器状态是显存中的最大单一开销**——占全参数训练总显存的 ~30-40%。

现有的显存节省方法各有局限：
- **[[LoRA]]**（参数高效微调）：只训练低秩适配器，**不是全参数训练**，对于需要全参数适应的任务（如从零预训练、全参数 SFT）不适用
- **ZeRO (FSDP)**（分片优化器）：将优化器状态分片到多个 GPU，但**总显存消耗不变**，只是分布到更多设备
- **Gradient Checkpointing**：以计算换显存，减少激活值而非优化器状态

### 1.2 核心洞察：梯度流本身的低秩结构

GaLore 的核心洞察来自对 LLM 训练过程中梯度矩阵的实证观察：

> 在整个训练过程中，梯度矩阵 $\nabla W \in \mathbb{R}^{m \times n}$ 的奇异值衰减缓慢但不是均匀的——前 10% 的奇异值贡献了 90% 的梯度"能量"。

更具体地说，论文发现：

$$
\frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2} \approx 0.9, \quad \text{当 } r \approx 0.1 \cdot \min(m,n)
$$

这意味着**梯度的有效秩约为满秩的 10%**。通过将梯度投影到其主导低秩子空间，可以在保留绝大部分梯度信息的同时，大幅降低优化器状态的存储需求。

### 1.3 与 LoRA 的根本区别

| 方面 | [[LoRA]] | **GaLore** |
|------|:-:|:-:|
| 参数更新 | 仅训练低秩适配器 $BA$ | 全参数更新，但梯度在低秩空间优化 |
| 优化器状态 | 适配器的大小（极小） | 低秩投影空间（中等） |
| 表达能力 | 受 $r$ 限制的秩约束更新 | 全秩更新（通过低秩梯度累积实现） |
| 训练模式 | PEFT（微调） | 全参数训练（预训练、全参数微调） |
| 学习方式 | $\Delta W$ 低秩 | $\nabla W$ 低秩投影 |

GaLore 不限制参数更新的秩——所有权重都完整更新，只是优化器状态在低秩空间中管理。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 梯度低秩投影

GaLore 的核心数学公式是优化器状态的**投影-反转投影**机制：

$$
\begin{aligned}
\text{传统 Adam:} \quad & M_t, V_t = \text{Adam}(\nabla W_t, M_{t-1}, V_{t-1}) \\
& W_{t+1} = W_t - \eta \cdot \frac{M_t}{\sqrt{V_t} + \epsilon} \\
\\
\text{GaLore Adam:} \quad & \rho_t = P^\top \nabla W_t Q \quad \text{(投影到低秩空间)} \\
& \tilde{M}_t, \tilde{V}_t = \text{Adam}(\rho_t, \tilde{M}_{t-1}, \tilde{V}_{t-1}) \\
& W_{t+1} = W_t - \eta P \cdot \frac{\tilde{M}_t}{\sqrt{\tilde{V}_t} + \epsilon} \cdot Q^\top \quad \text{(反投影回原始空间)}
\end{aligned}
$$

其中：
- $P \in \mathbb{R}^{m \times r}$ 和 $Q \in \mathbb{R}^{n \times r}$ 是投影矩阵
- $\rho_t \in \mathbb{R}^{r \times r}$ 是低秩投影梯度
- $\tilde{M}_t, \tilde{V}_t \in \mathbb{R}^{r \times r}$ 是低秩空间的优化器状态

### 2.2 投影矩阵的更新策略

GaLore 的一个关键实践技巧是**周期性更新投影矩阵**：

$$
\begin{aligned}
&\text{每 } T \text{ 步进行一次:} \\
&U, S, V^\top = \text{SVD}(\nabla W_{\text{accum}}) \\
&P = [u_1, u_2, ..., u_r], \quad Q = [v_1, v_2, ..., v_r]
\end{aligned}
$$

其中 $u_i, v_i$ 是 $U$ 和 $V$ 的前 $r$ 个奇异向量。

**更新频率**：实践表明每 200-1000 步更新一次投影矩阵就足够了。SVD 计算开销约为正常前向一步的 0.1-0.5%，对整体训练速度几乎无影响。

### 2.3 与现有的"梯度压缩"工作的区别

GaLore 与之前梯度压缩方法的不同之处在于：

| 方法 | 压缩目标 | 是否影响收敛 |
|------|:-:|:-:|
| 梯度量化 (QSGD) | 梯度的位宽 | 是，引入量化误差 |
| 梯度稀疏化 | 梯度的元素数量 | 是，引入稀疏误差 |
| 梯度低秩近似 | 优化器状态的存储 | **否**，参数更新是全秩的 |

GaLore 的核心创新在于：**优化器状态在低秩空间中管理，但参数更新仍然发生在全秩空间中**。梯度先被投影到低秩空间进行 Adam 动量计算，然后由全秩的原始梯度驱动实际参数更新。这与直接低秩近似梯度（会丢失信息）有本质区别。

### 2.4 与权重低秩分解的比较

| 方面 | 低秩权重矩阵 | GaLore 梯度低秩投影 |
|------|:-:|:-:|
| 参数空间 | $W \in \mathbb{R}^{r \times r}$ | $W \in \mathbb{R}^{m \times n}$ (全秩) |
| 优化空间 | $\Delta W \in \mathbb{R}^{r \times r}$ | $\tilde{M}, \tilde{V} \in \mathbb{R}^{r \times r}$ |
| 表达能力 | $rank(\Delta W) \leq r$ | $rank(\Delta W) \leq min(m,n)$ (无秩约束) |

这是 GaLore 和 [[LoRA]] 的另一个关键区别：GaLore 的最终参数更新 $-\eta P \cdot \text{Adam}(\rho) \cdot Q^\top$ 可以是满秩的。

### 2.5 显存节省分析

对于权重矩阵 $W \in \mathbb{R}^{m \times n}$，Adam 优化器的标准状态大小为 $2 \times m \times n$。GaLore 将其降低到 $2 \times r \times r + (m + n) \times r$（投影矩阵存储）。

以 LLaMA-7B 的注意力投影为例：

| 权重 | 形状 (m, n) | 标准 Adam 状态 | GaLore (r=128) | 节省比例 |
|------|:-:|:-:|:-:|:-:|
| QKV 投影 | (4096, 4096) | 32 MB | 2 × 128² + (4096+4096)×128 ≈ 1.05 MB | **~30x** |
| 输出投影 | (4096, 4096) | 32 MB | 同上 ≈ 1.05 MB | **~30x** |
| Up 投影 | (4096, 11008) | 90 MB | 同 (r=256) ≈ 2.1 MB | **~43x** |
| Down 投影 | (11008, 4096) | 90 MB | 同 (r=256) ≈ 2.1 MB | **~43x** |

---

## 三、Experiments and Key Findings

### 3.1 LLaMA-7B 从头预训练

GaLore 在 LLaMA-7B 规模从零开始的预训练中表现出色：

| 配置 | 训练显存 | 每步时间 | 最终 PPL (C4) | 最终 PPL (Wiki) |
|------|:-:|:-:|:-:|:-:|
| 全参数 Adam (32-bit) | 82 GB（需 2×A100） | ~1.0x | 18.7 | 15.8 |
| 全参数 Adam (bf16) | 58 GB | ~1.0x | 18.7 | 15.8 |
| **GaLore (r=256)** | **28 GB** | **~1.1x** | **18.9** | **16.1** |
| [[LoRA]] (r=8) | 22 GB | ~1.0x | >30 | >35 |

**关键发现**：
1. GaLore 将全参数预训练的显存从 58GB 降至 28GB（**降低 52%**），与 LoRA 的显存需求相近
2. 最终困惑度损失 < 0.2 PPL（几乎可以忽略）
3. 训练速度仅下降 10%（主要是 SVD 更新的开销）
4. LoRA 由于秩约束无法有效预训练，PPL 远高于 GaLore

### 3.2 LLaMA-1B 和 LLaMA-7B 的完整预训练

| 模型 | 优化器 | 显存峰值 | 训练样本 | 验证 PPL (Wiki-103) |
|------|:-:|:-:|:-:|:-:|
| LLaMA-1B | Adam (bf16) | 21.1 GB | 15B tokens | 16.1 |
| LLaMA-1B | **GaLore** | **12.0 GB** | 15B tokens | **16.4** |
| LLaMA-7B | Adam (bf16) | 58.4 GB | 25B tokens | 12.1 |
| LLaMA-7B | **GaLore** | **28.3 GB** | 25B tokens | **12.4** |

GaLore 的显存节省在 7B 模型上最为显著（节省 ~30GB），而在 1B 模型上也可将全参数训练放入 12GB 消费级 GPU——这在以前是不可能的。

### 3.3 LLaMA-2-7B 的全参数微调

| 方法 | 显存 | 训练时间 | GSM8K | 说明 |
|------|:-:|:-:|:-:|------|
| Full FT (Adam bf16) | 56 GB | ~8h | 55.3 | 基线 |
| **GaLore (r=256)** | **21 GB** | ~8.5h | **55.8** | 显存降低 62% |
| [[LoRA]] (r=32) | 18 GB | ~8h | 47.5 | 质量损失明显 |
| QLoRA (4-bit) | 12 GB | ~9h | 46.1 | 质量损失更明显 |

在需要全参数适应的任务（如数学推理 GSM8K）上，GaLore 保持了与全参数微调同等的准确率（55.8% vs 55.3%），而 LoRA 和 QLoRA 分别下降了 7.8 和 9.2 个百分点。

### 3.4 投影秩 $r$ 的选择

| $r$ | 显存节省比例 | LLaMA-1B PPL |
|:-:|:-:|:-:|
| 64 | 85% | 16.7 |
| 128 | 75% | 16.5 |
| 256 | 60% | 16.4 |
| 全秩 | 0% | 16.1 |

**建议**：对于注意力层，$r \in [64, 256]$ 是合理范围；对于 FFN 层（输入输出维度差异大），$r$ 应适当增大（因为 FFN 的梯度秩往往更高）。

### 3.5 SVD 更新频率的影响

| 更新间隔 T | LLaMA-1B PPL | SVD 开销 |
|:-:|:-:|:-:|
| 10 | 16.3 | 5% |
| 200 | 16.4 | 0.5% |
| 1000 | 16.4 | 0.1% |
| 5000 | 16.8 | <0.1% |

当 $T \leq 1000$ 时，SVD 成本不到训练的 0.5%。$T$ 增加到 5000 后，由于投影子空间无法及时跟踪梯度流的变化，质量开始下降。

---

## 四、Limitations and Challenges

1. **子空间跟踪的局限性**：GaLore 假定梯度流在相对长的时间窗口（200-1000 步）内保持低秩子空间。对于训练初期或学习率预热阶段等梯度方向快速变化的时期，固定的投影矩阵更新间隔可能不是最优的
2. **SVD 计算的累积开销**：虽然单次 SVD 开销很小（~0.1%），但全训练周期内乘积的累积仍然是可感知的。在 7B 模型上 GaLore 的训练速度约为正常全参数训练的 85-95%
3. **对超参数敏感**：投影秩 $r$、SVD 更新间隔、SVD 累积步数等新增超参数需要调优。不同的模型大小和架构可能需要不同的配置
4. **梯度流低秩假设的"经验性"**：论文的核心理念"梯度在低秩子空间中流"主要基于经验观察和可视化，缺乏严格的理论证明。对于某些架构（如 MoE、多模态模型），梯度低秩性可能不成立
5. **投影矩阵的额外显存**：虽然优化器状态显著减少，但 $P, Q$ 投影矩阵本身需要 $(m+n)r$ 的存储。对于极宽或极深的层（如 embedding 层 $m \gg n$），这一开销不可忽略
6. **仅针对优化器状态**：GaLore 不解决激活值显存的问题。对于长序列训练（如 8k+ token），Activation Checkpointing 仍然必不可少

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 相关/后续工作 | 年份 | 与 GaLore 的关系 |
|-------------|:----:|------------------|
| **[[LoRA]]** (Hu et al.) | 2021 | 参数更新的低秩分解 vs GaLore 的梯度低秩投影；两者互补 |
| **ReLoRA** (Lialin et al.) | 2023 | 在训练中周期性合并低秩更新，与 GaLore 的周期性 SVD 类似 |
| **ByteGrad** | 2024 | 将 GaLore 扩展到 byte 级预训练 |
| **GaLore v2 / LoRETTA** | 2024 | 对 GaLore 的改进，使用随机投影而非 SVD 来计算子空间 |
| **Flora** (Hao et al.) | 2024 | 将 GaLore 的梯度低秩投影与权重低秩更新结合 |
| **Q-Galore** | 2024 | 在 GaLore 基础上增加量化，进一步减少显存 |
| **Tensor Train GaLore** | 2025 | 将 GaLore 扩展到张量列分解，进一步降低低秩容量 |

**影响评估**：GaLore 是第一个让全参数 LLM 训练在消费级 GPU 上成为可能的训练压缩方法。它打破了"全参数训练必须依赖多卡 A100/H100"的默认假设，为个人研究者和小型团队打开了一个全新的可能性空间。ICML 2024 录用后迅速被社区接受，多位用户验证了在单张 RTX 4090 (24GB) 使用 GaLore 训练 7B 级模型。对 VLA 研究而言，GaLore 为需要全参数适应的多模态基础模型训练提供了经济高效的替代方案。

---

## 六、Implications for You / Hardware Compatibility

### 显存对比

| 场景 | 传统全参数 | GaLore | [[LoRA]] | [[QLoRA]] |
|------|:-:|:-:|:-:|:-:|
| LLaMA-7B 全参数微调 | 56 GB | **21 GB** | 18 GB | 12 GB |
| LLaMA-7B 从头预训练 | 82 GB (bf16 Adam) | **28 GB** | 不适用 | 不适用 |
| LLaMA-13B 全参数微调 | ~98 GB | **~40 GB** | ~32 GB | ~20 GB |
| LLaMA-1B 全参数微调 | 21 GB | **12 GB** | 10 GB | 8 GB |

GaLore 使 7B 全参数训练在单张 24GB GPU 上**首次成为可能**。

### 实际使用建议

- **最佳使用场景**：从头预训练中小模型（1B-7B），或需要全参数适应的任务（如数学推理 SFT）。在这些场景下，GaLore 是唯一能在消费级 GPU 上运行的方法
- **与 [[LoRA]] 的选择**：若下游任务质量对"全参数更新"敏感（如代码生成、数学推理），优先选择 GaLore；若仅为标准指令微调，LoRA 通常已足够
- **与 FSDP/DeepSpeed 的组合**：GaLore 可以与 FSDP 联合使用——FSDP 将模型参数分片到多个 GPU，GaLore 在每个分片上独立减少优化器状态，实现近乎线性的显存缩减
- **对 VLA 研究的价值**：多模态基础模型（如 OpenVLA 的微调）可以从 GaLore 中受益，特别是当需要全参数适应时。但 GaLore 对视觉编码器的梯度低秩性尚未充分验证
- **实践配置建议**：$r=256$ 对注意力层是安全默认值；FFN 层的 $r$ 建议按比例增大（如 $r=512$）；SVD 更新频率设为 200 步/次，预热阶段可适当增加
- **激活显存仍是大问题**：GaLore 不解决激活值问题。当使用长序列（如 VLA 的 4096 token）时，Gradient Checkpointing 仍然是必要的

### 硬件兼容性总结
- ✅ 单卡 RTX 4090 (24GB) 全参数训练 LLaMA-7B：首度可行
- ✅ 单卡 RTX 3090 (24GB) 全参数训练 LLaMA-7B：支持（需 Gradient Checkpointing）
- ⚠️ 单卡 RTX 4060 (16GB) 全参数训练 LLaMA-7B：可能勉强（需 GaLore + 量化 + Activation Checkpointing）
- ✅ 单卡 RTX 4090 全参数微调 LLaMA-13B：GaLore 使此场景从"不可能"变为"可能"
- ❌ 单卡消费级 GPU 从头预训练 LLaMA-13B：即使有 GaLore，模型的 26GB 参数 + 激活值也超出 24GB 限制

## PDF

[[GaLore.pdf]]
