---
tags:
  - 论文
  - VLA
  - RL微调
  - ICLR2026
  - GRPO
created: 2026-06-30
paper_title: "SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning"
paper_authors: "PRIME-RL团队 (清华/上海AI Lab/上交/北大/港大, 21位作者)"
paper_year: 2025
paper_venue: "ICLR 2026"
paper_url: "https://arxiv.org/abs/2509.09674"
github: "https://github.com/PRIME-RL/SimpleVLA-RL"
---

# SimpleVLA-RL

**SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning**
*清华 + 上海AI Lab + 上交 + 北大 + 港大 | ICLR 2026 | arXiv 2509.09674*

> **DeepSeek-R1 的思路为什么不能用在机器人上？** SimpleVLA-RL 回答了这个问题：能！而且只需要 1 条演示 + 简单的 0/1 成功奖励 + GRPO，就能让 VLA 模型的成功率从 17.3% 飙升到 91.7%。这是 VLA 领域"从 BC 到 RL"范式转变的标志性工作。

---

## 一、研究背景：VLA 训练的瓶颈

### 1.1 VLA 的两阶段训练范式

当前 VLA 模型（OpenVLA、π0、GR00T）通常采用两阶段训练：

1. **大规模预训练**：在多模态数据（图文对、人类视频、多机器人轨迹）上做自监督/监督预训练
2. **监督微调 (SFT)**：在"高质量"遥操作轨迹上做行为克隆 (Behavior Cloning)

这个范式取得了巨大成功，但随着规模扩大，出现了两个根本问题：

**问题 1：数据稀缺。** 高质量的机器人遥操作轨迹极其昂贵。每条演示需要精心设计的实验场景、多样化的操作对象和熟练的操作员。这与 NLP 可以抓取互联网上的万亿级 token 形成了鲜明对比。

**问题 2：泛化瓶颈。** SFT 本质上是"模仿"——模型只会复制它在训练数据中看到的行为。一旦遇到分布外的新场景（新物体、新背景、新任务组合），性能急剧下降。这尤其严重地体现在组合任务、长周期任务和真实世界部署中。

### 1.2 来自 LLM 的灵感：DeepSeek-R1

2025 年初，DeepSeek-R1 横空出世。它的核心发现是：**只需要最简单的规则奖励（答案对不对），强化学习就能让 LLM 涌现出复杂的推理能力（Chain-of-Thought）。**

SimpleVLA-RL 的核心赌注：**同样的逻辑——纯结果驱动的 RL + 不需要奖励塑形——能不能在 VLA 上复现奇迹？**

---

## 二、方法：从 LLM RL 到 VLA RL

### 2.1 问题形式化

论文首先仔细对比了 LLM 和 VLA 在 RL 框架下的差异：

| | LLM RL (DeepSeek-R1) | VLA RL (SimpleVLA-RL) |
|---|---|---|
| **状态 $s_t$** | prompt + 已生成的 token | 多模态观测（RGB/深度/点云 + 本体感觉 + 语言指令）|
| **动作 $a_t$** | 从词汇表中选择下一个 token | $a_t \in \mathbb{R}^d$（如 7 维末端执行器位移）|
| **环境** | 序列完成后给奖励（离线）| 与环境持续交互 → 动力学转移 → 新状态 → 循环 |
| **Rollout** | 自回归生成直到停止 | 迭代交互——策略输出 action chunk → 执行 → 新状态 → 下一轮 |

VLA RL 比 LLM RL 难得多的地方在于：**每个 rollout 步骤都需要与环境交互**（仿真或真实），而不是仅仅采样下一个 token。这使 rollout 更慢、更昂贵。

### 2.2 GRPO 算法

SimpleVLA-RL 使用 **GRPO (Group Relative Policy Optimization)** 而非标准 PPO。

**为什么不用 PPO？** PPO 需要一个独立的价值网络 (Value Network) 来估计优势函数。对于 7B+ 的 VLA，这等于再训练一个同样大的模型——昂贵且不稳定。

**GRPO 为什么更好？** GRPO 不需要价值网络。它通过**组间相对归一化**来计算优势：

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

具体来说：对于同一个初始状态，策略采样 $G$ 条不同的 rollout 轨迹（通过不同的随机采样）。每条轨迹得到一个奖励 $R_i$（0 或 1）。通过比较这些轨迹之间的相对好坏来计算优势——比平均值好的就是"好"轨迹，差的就是"差"轨迹。

**GRPO 目标函数：**

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{s_0 \sim \mathcal{D}, \{\tau_i\} \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|\tau_i|} \sum_{t=1}^{|\tau_i|} \min\left(r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

和 PPO 一样有 clipping 和 KL 正则化，但不需要价值网络。

### 2.3 VLA 特有的工程挑战

论文在 veRL (Volcano Engine Reinforcement Learning for LLMs) 框架基础上做了大量 VLA 专属的适配：

**交互式 VLA Rollout：**
- VLA 模型必须在动作 token 分布上进行**随机采样**才能产生多样化轨迹（而不是确定性解码）
- 幸运的是，OpenVLA 的 256-bin 离散动作 token 天然支持随机采样
- 如果用 Diffusion/Flow Matching 的动作生成（π0 的风格），采样多样性会更复杂

**并行多环境渲染：**
- 为了加速采样，论文扩展了 veRL 以支持并行多环境渲染
- 将训练-推理-渲染整合为统一框架

**奖励设计：**
- 纯结果奖励：轨迹成功 = 1，失败 = 0
- 不需要任何过程奖励（如"离目标多远"）
- 这大大简化了设计，但需要大量探索来"偶然"成功

### 2.4 三项探索增强策略

GRPO 本身对 exploration 不友好（它偏向保守更新），论文提出三项增强：

**1. 动态采样 (Dynamic Sampling)：**
- 根据当前策略的"自信度"动态调整每个 prompt 的采样数量
- 对不自信的 prompt 多采几个，对已经学会的少采

**2. 自适应 Clipping 扩展：**
- PPO 的标准 clipping 范围是 $[0.8, 1.2]$
- SimpleVLA-RL 扩展到 $[0.8, 1.28]$——允许更大的策略更新步长，鼓励更多探索
- 这在训练初期尤其重要（策略还很不确定时）

**3. 温度退火：**
- 采样温度 $T$ 随训练进度从高到低
- 早期高温 → 多探索，后期低温 → 精细化
- 类似于"模拟退火"在 RL 中的应用

---

## 三、实验与核心结果

### 3.1 数据效率的惊人发现

| 设置 | LIBERO-Long 成功率 | 相对提升 |
|------|------------------|---------|
| SFT only（全量演示） | 17.3% | — |
| 仅 1 条演示做 SFT → SimpleVLA-RL | **91.7%** | +430% |
| 全量演示做 SFT → SimpleVLA-RL | **99.1%** | — |

1 条演示 + RL 的效果几乎追平了全量演示的 SFT + RL。这是在说：**RL 可以从极少的数据中提取远超过 BC 的价值。**

### 3.2 基准对比

| 基准 | SFT | SimpleVLA-RL | π0 |
|------|-----|-------------|-----|
| LIBERO (全量) | 91.0% | **99.1%** | 94.2% |
| RoboTwin 1.0 | 48.9% | **70.4%** | 94.2%* |
| RoboTwin 2.0 | 38.3% | **68.8%** | 39.8% |

*注：RoboTwin 1.0 上 π0 因为使用了更多预训练数据而领先，但在 RoboTwin 2.0 上 SimpleVLA-RL 大幅超越 π0。

### 3.3 "Pushcut" 现象（论文的核心发现）

> **RL 训练出的策略自发发现了一种演示数据中从未出现的行为模式。**

在 LIBERO-Long 的"将罐头移动到锅里"任务中：
- **演示数据教的是**：抓取 (grasp) → 移动 → 放置。标准的 pick-and-place 流程。
- **RL 训练的策略学会了**：不抓取，直接把罐头**推向**锅里。跳过 grasp 步骤。

论文称这个现象为 **Pushcut**（push + shortcut）。这不是"学到更好的 grasp"，而是"发现了不需要 grasp 就能完成任务的方法"。

**Pushcut 的深刻含义：**
1. RL 不只是"优化已有行为"——它能发现**全新的策略**
2. 人类演示隐含了"应该这样做"的偏见（我们觉得需要先抓取），但 RL 只关心"成功了没有"
3. 这是 BC 永远做不到的——BC 只能复制演示，RL 可以超越演示

### 3.4 泛化能力

论文用四个泛化轴测试：
- **空间泛化**（物体在不同位置）：RL >> SFT（+74.4%）
- **物体泛化**（未见过的物体）：RL >> SFT
- **目标泛化**（新任务目标）：RL >> SFT
- **组合泛化**（多个泛化轴叠加）：RL 的优势最大

### 3.5 真实世界实验

在 4 个真实世界任务上（涉及长周期灵巧操作）：
- SFT 基线只有中度成功率
- SFT → SimpleVLA-RL 取得 **~300% 相对提升**
- 甚至在仿真中训练（sim-only）的策略也能在真实世界中泛化（+21% vs SFT）

---

## 四、失败模式与局限

论文诚实地分析了 RL 微调的失败模式：

1. **奖励稀疏问题**：在非常长的周期任务中，策略可能需要数千步才能偶然成功一次——纯结果奖励无法提供中间信号
2. **探索-利用平衡**：三项增强策略有帮助，但在极端复杂的任务中仍然不够
3. **分布外崩溃**：虽然 RL 改善了泛化，但某些极端分布偏移仍会导致崩溃
4. **基础设施要求高**：8×A800 GPU + 并行仿真环境 → 对大多数实验室不友好

---

## 五、与你的研究的关系

❌ **硬件门槛太高**（8×A800），不适合你的 4070 Ti Super 16GB。

但两个思想层面的启示极其重要：

1. **1 条演示 + RL > 100 条演示的 BC**——改变你对数据策略的思考。与其花时间收集更多演示，不如想想怎么加 RL
2. **Pushcut 现象表明 RL 可以发现超越人类的策略**——这是 BC 永远做不到的。在任何 self-improving 系统中，RL 都是必选项而不是可选项

## PDF

[[SimpleVLA-RL.pdf]]
