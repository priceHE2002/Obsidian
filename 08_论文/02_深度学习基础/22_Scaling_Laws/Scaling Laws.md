---
tags:
  - 论文
  - Scaling Law
  - 模型规模
  - 涌现
  - 计算最优
created: 2026-06-30
paper_title: "Scaling Laws for Neural Language Models"
paper_authors: "Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei"
paper_year: 2020
paper_venue: "arXiv 2001.08361"
paper_citations: "~8,000+"
paper_url: "https://arxiv.org/abs/2001.08361"
---

# Scaling Laws

**Scaling Laws for Neural Language Models**
*Jared Kaplan, Sam McCandlish et al. | OpenAI | arXiv 2001.08361*

> 建立了神经语言模型性能与模型规模 ($N$)、数据量 ($D$)、计算量 ($C$) 之间的精确幂律关系。给定计算预算，**最优策略是优先扩大模型规模而非数据量**（该结论后被 Chinchilla 部分修正）。对 VLA 研究而言，这是 RT-2 55B > 5B 能力涌现的理论基础，也是 OpenVLA 选择 7B 骨干而非更小模型的理论依据。

---

## 一、Background / Core Idea

### 1.1 2020 年前的困惑

在 Scaling Laws 论文之前，深度学习领域对"规模与性能的关系"只有模糊的直觉知识：
- 更大的模型通常更好，但没有精确的数学模型
- 不知道固定计算预算下应该优先扩大模型还是数据
- 不知道模型性能是否有天花板

### 1.2 核心问题

论文回答了三个核心问题：
1. **给定固定计算预算 $C$，最优的参数分配（模型大小 vs 训练步数 vs batch size）是什么？**
2. **模型性能是否有上限？幂律关系能否无限外推？**
3. **跨架构（Transformer, LSTM, Universal Transformer）的缩放规律是否统一？**

### 1.3 实验设置

- 训练 **Decoder-only Transformer** 语言模型，从 768 参数到 1.5B 参数（跨 7 个数量级）
- 数据集：**WebText2**（Reddit 出站链接 ≥3 karma），20.3M 文档，96GB 文本，22.9B tokens
- 主要指标：交叉熵损失（averaged over 1024-token 上下文）
- 优化器：Adam（小模型）/ Adafactor（大模型）
- 默认 250K 步，batch size 512 sequences × 1024 tokens

---

## 二、Method / Architecture / Technical Contribution

### 2.1 三个核心幂律

论文发现语言模型性能与 $N$, $D$, $C$ 分别满足精确的幂律关系（当不受其他两个因素瓶颈时）：

#### (1) 模型规模 Scaling Law

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$

- $N$ = 非嵌入参数量（排除 embedding 矩阵和位置编码，因为它们在 $N$ 中包含会混淆趋势）
- **含义**：模型参数翻倍时，损失减少 $1 - 2^{-0.076} \approx 5.1\%$
- 随着 $N$ 增加，收益递减（幂律指数小于 1）

#### (2) 数据量 Scaling Law

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13}$$

- **含义**：数据量翻倍时，损失减少 $1 - 2^{-0.095} \approx 6.4\%$
- 注意 $\alpha_D > \alpha_N$，意味着在不受瓶颈时，增加数据略优于增加参数

#### (3) 计算量 Scaling Law（最优分配）

$$L(C_{\min}) = \left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_{C_{\min}}}, \quad \alpha_{C_{\min}} \approx 0.050, \quad C_c^{\min} \approx 3.1 \times 10^8 \text{ PF-days}$$

- $C_{\min}$ 是调整到最优 batch size 后的计算量（见 2.3）
- **含义**：计算量翻倍，损失减少 $1 - 2^{-0.050} \approx 3.4\%$
- $\alpha_{C_{\min}}$ 最小，说明计算量的边际收益递减最快

### 2.2 联合模型 $L(N,D)$ 和 $L(N,S)$

#### $L(N,D)$：模型 + 数据的联合依赖

$$L(N,D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$

这个形式满足三个设计原则：
1. $N \to \infty$ 时 $L \to L(D)$（仅受数据限制）
2. $D \to \infty$ 时 $L \to L(N)$（仅受模型限制）
3. 在 $D=\infty$ 处有 1/D 解析展开

**过拟合的衡量**：过拟合程度仅依赖于 $\frac{N^{0.74}}{D}$ 的组合值：

$$\delta L \approx \left[1 + \left(\frac{N}{N_c}\right)^{\frac{\alpha_N}{\alpha_D}} \cdot \frac{D_c}{D}\right]^{\alpha_D} - 1$$

**实际含义**：要避免过拟合，需要 $D \gtrsim 5 \times 10^3 \times N^{0.74}$。即模型每增大 8 倍，数据只需增大约 5 倍。

#### $L(N,S)$：模型 + 训练步数的联合依赖

$$L(N,S) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}(S)}\right)^{\alpha_S}$$

其中 $\alpha_S \approx 0.76$, $S_c \approx 2.1 \times 10^3$。

### 2.3 临界 Batch Size 与 $C_{\min}$

核心概念来自 **McCandlish et al. 2018（An Empirical Model of Large-Batch Training）**：

临界 batch size $B_{\text{crit}}$ 使得训练在时间和计算效率上达到最优均衡：

$$B_{\text{crit}}(L) \approx \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2 \times 10^8, \quad \alpha_B \approx 0.21$$

**梯度噪声尺度（Gradient Noise Scale）** 预测 $B_{\text{crit}}$ $\propto$ 梯度方差的迹 / 梯度平方的范数。

实际含义：$B_{\text{crit}}$ 只依赖于损失值 $L$，**不直接依赖于模型大小**。

$S \to S_{\min}$ 的关系式（训练在 $B \gg B_{\text{crit}}$）：

$$S_{\min}(S) = \frac{S}{1 + B_{\text{crit}}(L)/B}$$

$C \to C_{\min}$ 同理（训练在 $B \ll B_{\text{crit}}$）：

$$C_{\min}(C) = \frac{C}{1 + B/B_{\text{crit}}(L)}$$

### 2.4 计算最优分配（Compute-Efficient Frontier）

给定计算预算 $C$，最优模型大小、batch size、步数和数据量分别为：

| 参数 | 幂律关系 | 指数 | 含义 |
|:-:|:-:|:-:|------|
| $N_{\text{opt}}$ | $\propto C_{\min}^{0.73}$ | 大 | 计算增加主要用来增大模型 |
| $B_{\text{crit}}$ | $\propto C_{\min}^{0.24}$ | 中 | batch size 适度增大 |
| $S_{\min}$ | $\propto C_{\min}^{0.03}$ | **几乎为零** | 串行步数几乎不变 |
| $D_{\text{opt}}$ | $\propto C_{\min}^{0.27}$ | 小 | 数据量增长缓慢 |

**最关键结论**：计算预算增加时，**绝大部分资源应用于扩大模型规模**，而非继续训练更长时间。

### 2.5 幂律矛盾的预测

论文发现一个**内在矛盾**：

- 从 $L(C_{\min})$: $L \propto C_{\min}^{-0.050}$
- 从 $L(D(C_{\min}))$: 数据增长 $D \propto C_{\min}^{0.27}$, $L \propto D^{-0.095} \propto C_{\min}^{-0.026}$

两条曲线在以下点相交：
- $C^* \sim 10^4$ PF-days
- $N^* \sim 10^{12}$ 参数
- $D^* \sim 10^{12}$ tokens
- $L^* \sim 1.7$ nats/token

**这意味着 scaling laws 不能无限外推**——超过该临界点后，仅有数据增长无法支撑计算高效训练。论文推测 $L^*$ 可能对应自然语言熵的下界。

### 2.6 其他关键发现

#### 对模型形状的弱依赖性

当固定非嵌入参数 $N$ 时，模型形状（深度 vs 宽度、head 数量、FFN 比率）对性能的影响极小（<3%）：

| 超参数 | 变化范围 | 性能变化 |
|:-:|:-:|:-:|
| 纵横比 $d_{\text{model}} / n_{\text{layer}}$ | 6 → 4288（40倍） | <3% |
| 注意力 head 维度 | 32 → 128 | <2% |
| FFN 比率 $d_{\text{ff}} / d_{\text{model}}$ | 2 → 8 | <2% |

#### 通用性（Transfer Learning 也服从 Scaling）

模型在非训练分布（Wikipedia, Books, Common Crawl）上的性能与训练分布性能呈**线性偏移**关系。跨域泛化主要取决于分布内验证损失，几乎与训练时长或收敛程度无关。

#### LSTM vs Transformer

LSTM 在短上下文中的性能与 Transformer 相当，但无法利用长上下文。
- Transformer 对所有 token 位置一致提升
- LSTM 在位置 >100 后性能不再提升

#### 学习率调度的无关性

只要总"学习的量"（即 LR 曲线下积分面积）足够大，具体调度策略对最终损失影响不大。

---

## 三、Experiments and Key Findings

### 3.1 核心数值验证

**模型规模实验**（768 参数 → 1.5B 参数，WebText2 全量训练）：

| 非嵌入参数量 | 层数 | $d_{\text{model}}$ | 最终损失 (nats) | 拟合损失 (nats) |
|:-:|:-:|:-:|:-:|:-:|
| 393K | 2 | 128 | ~5.2 | ~5.1 |
| 3M | 2 | 256 | ~4.2 | ~4.2 |
| 25M | 4 | 512 | ~3.4 | ~3.4 |
| 85M | 8 | 768 | ~3.1 | ~3.1 |
| 302M | 16 | 1024 | ~2.8 | ~2.8 |
| 708M | 24 | 1280 | ~2.6 | ~2.6 |
| 1.5B | 48 | 1600 | ~2.5 | ~2.5 |

### 3.2 数据量实验

300M 参数模型在不同数据子集上的表现：

| 数据量 (tokens) | 损失 | 是否过拟合 |
|:-:|:-:|:-:|
| 22M | ~3.9 | 严重 |
| 86M | ~3.6 | 显著 |
| 344M | ~3.3 | 轻微 |
| 1.4B | ~3.1 | 几乎无 |
| 22B (全量) | ~2.9 | 无 |

### 3.3 计算高效 vs 传统训练的对比

| 训练策略 | 参数量 | 步数 | 计算量 | 达到相同 loss |
|:-:|:-:|:-:|:-:|:-:|
| 计算高效 (f=10%) | 2.7x **更大** | **7.7x 更少** | **65% 更少** | 同 loss |
| 传统训练至收敛 (f=2%) | 参考 | 参考 | 参考 | 同 loss |

论文的核心信息是：**传统上训练模型至收敛的做法浪费了大量计算资源**。

### 3.4 样本效率

大模型比小模型更**样本高效**——达到相同损失需要的样本更少。例如，1.5B 参数模型达到损失 3.0 所需的样本量比 3M 参数模型少 100 倍。

---

## 四、Limitations and Challenges

1. **$\alpha_N = 0.076$, $\alpha_D = 0.095$ 等数值特定于 WebText2 数据集**：词汇大小和分词方式会影响 $N_c$, $D_c$ 等缩放常数，跨数据集不通用
2. **"优先扩大模型"的结论被 Chinchilla (2022) 部分修正**：论文建议 $D \propto N^{0.74}$，Chinchilla 的更大规模实验发现 $D \propto N^{1.0}$（同比例增长）才是真正最优。这意味着论文推荐了**过大的模型和过少的数据**
3. **仅研究自回归语言建模**：多模态、指令微调、RLHF 等更复杂训练目标的缩放行为不同
4. **未考虑数据质量**：使用统一的 WebText 数据，实际中数据质量对性能影响可能大于规模
5. **模型形状的弱依赖性**：虽然论文说形状影响小，但现代架构（MoE、GQA、MQA）可能从根本上改变了缩放规律
6. **仅到 1.5B 参数**：论文最大模型仅 1.5B 参数，外推到 175B+ 的 GPT-3 / Llama 规模时不确定性大
7. **Caveats**：论文自身承认没有坚实的理论基础——scaling laws 是现象学而非第一性原理推导

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 关系 |
|---------|:----:|------|
| **Chinchilla** (Hoffmann et al., DeepMind) | 2022 | 修正数据-模型比例：$N \propto D$ 而非 $N^{0.74}$，提出 Compute-Optimal 大模型概念 |
| **GPT-3** (Brown et al., OpenAI) | 2020 | 175B 参数验证了 $L(N)$ 的幂律趋势 |
| **PaLM** (Chowdhery et al., Google) | 2022 | 540B 参数验证可预测性能提升 |
| **Llama 1/2/3** (Meta) | 2023-2024 | Llama 3 的 15T+ tokens 实践了 Chinchilla 的计算最优比例 |
| **RT-2** (Brohan et al., Google) | 2023 | 5B vs 55B 的能力涌现 → VLA 验证 scaling law |
| **Open X-Embodiment** (Padalkar et al.) | 2023 | "模型容量大于数据多样性"的发现与 scaling law 一致 |
| **Scaling Laws for Robot Learning** (DeepMind) | 2024 | 扩展到机器人领域：模型大小、数据量、环境多样性间存在幂律 |
| **Emergent Abilities** (Wei et al.) | 2022 | 用 scaling law 框架解释涌现能力出现的阈值现象 |
| **Thermodynamic AI** | 探索中 | 论文类比"理想气体定律"→ 希望建立"统计力学"层面的理论 |

### VLA 领域的 Scaling 证据

| 模型 | 骨干大小 | 泛化能力 | 涌现现象 |
|------|:-:|:-:|:-:|
| RT-2 5B | 5B | 有限，只能处理训练分布内任务 | 无 |
| RT-2 55B | 55B | **组合泛化**：能推理从未见过的物体组合 | ✅ 符号理解、关系推理 |
| OpenVLA 7B | 7B | OOD 泛化显著优于 1B 模型 | ✅ 多步推理、空间理解 |
| 小模型 (<3B) | <3B | 线性的行为克隆 | ❌ |

---

## 六、Implications for You / Hardware Compatibility

### 对 VLA 研究的五条指导

1. **别在小于 3B 的模型上期待涌现能力**：如果实验目标包括"模型能否学会从未见过的物体操作"，7B 起步。Scaling law 明确指出 $L(N) \propto N^{-0.076}$——1B 到 7B 的理论损失改进约 $7^{-0.076} \approx 0.86$ 倍（14% 的 loss 减少）

2. **数据质量比模型修改更重要**：给定固定计算预算，scaling law 建议把资源投入数据清洗和增强（$\alpha_D = 0.095$ 略高于 $\alpha_N = 0.076$）。对个人研究者而言，不要在改架构上花太多时间

3. **用小模型做快速实验，用大模型做最终验证**：Scaling law 的核心方法学贡献是用小规模预测大规模——在小规模（如 300M）上验证的方法论可以外推到 7B

4. **理解超参缩放**：$\text{LR}(N) \approx 0.003239 - 0.0001395 \log N$，更大模型需要更小学习率。"1B 模型跑通了但 7B 跑不通"往往是学习率未缩放导致的

5. **计算最优训练 ≈ 只训练到接近收敛的 90%**：$\frac{L(N,S) - L(N,\infty)}{L(N,\infty)} \approx \frac{\alpha_N}{\alpha_S} \approx 10\%$ 时即应停止

### 硬件兼容性总结

- ✅ 在 7B VLA 上做 LoRA 微调：Scaling law 告诉你——模型大小和效果的关系是幂律而非线性，LoRA 微调能利用 7B 骨干的大部分能力
- ⚠️ 数据量是关键瓶颈：Scaling law 推断 $D \propto N^{0.74}$，7B 模型可能需要约 100B+ tokens 的训练数据才能充分发挥能力——个人研究者很难获取
- ❌ 全量微调 7B+ VLA：不符合计算最优原则，且硬件需求过高

### 关键记忆

> **Scaling Law 的核心启示**：如果你只能做一个选择来提升模型性能，**增大参数量的效果（$\alpha_N=0.076$）与增大数据量（$\alpha_D=0.095$）相似，但增大模型通常更易操作**。对 VLA 来说，从 5B 到 55B 不是 11 倍的线性提升，而是触发涌现能力的指数级差异。

## PDF

[[Scaling Laws 原文.pdf]]
