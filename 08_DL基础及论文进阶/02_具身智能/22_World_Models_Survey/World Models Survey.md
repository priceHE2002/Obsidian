---
tags:
  - 论文
  - 具身智能
  - 世界模型综述
  - 强化学习
  - 视频生成
  - 自动驾驶
  - arXiv2026
created: 2026-07-01
paper_title: "World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications"
paper_authors: "Arif Hassan Zidan, Yi Pan, Hanqi Jiang, Ruiyu Yan, Wei Ruan, Zihao Wu, Lifeng Chen, Weihang You, Xinliang Li, Bowen Chen, Huawen Hu, Peilong Wang, Sizhuang Liu, Jing Zhang, Siyuan Li, Zhengliang Liu, Yu Bao, Lin Zhao, Lichao Sun, Dajiang Zhu, Xiang Li, Jinglei Lv, Quanzheng Li, Wei Liu, Tianming Liu, Wei Zhang 等 26 人"
paper_year: 2026
paper_venue: "arXiv"
paper_citations: "新发表"
paper_url: "https://arxiv.org/abs/2606.00133"
github: ""
---

# World Models: A Comprehensive Survey

> **当前最全面的世界模型综述，147 页，提出四轴分类法（架构 × 方法论 × 推理策略 × 应用领域），从 PlaNet/Dreamer/MuZero 讲到 Sora/Cosmos/Genie，覆盖 RL、机器人、自动驾驶、视频生成、医疗、教育、金融等全部应用场景。对于 [[Motus]]、[[FastWAM]]、[[Cosmos Policy]]、[[Orca]] 等 WAM 论文的理解，这篇综述提供了完整的背景框架。**

---

## 一、背景与核心思想

### 1.1 什么是世界模型

**世界模型（World Model）** 是学习环境结构和动态的内部模拟器，使智能体能够在学到的表征中进行预测、规划和推理。其形式化定义为参数化预测系统：

$$p_\theta(s_{t+1}, o_{t+1}, r_t \mid s_t, a_t)$$

其中 $s_t$ 为潜状态，$o_t$ 为观测，$a_t$ 为动作，$r_t$ 为奖励。

三个核心属性将世界模型与一般的预测模型区分开来：
1. **动作条件化**：预测环境如何响应特定动作（"如果我左转会怎样？"）
2. **多步 rollout**：自回归生成任意长度的轨迹
3. **决策效用**：预测服务于策略优化、规划、数据增强或安全验证

### 1.2 历史脉络

思想根源可追溯到认知科学的心理模型理论（Johnson-Laird）和 Minsky 的框架表征（1970s）。现代深度学习版本由 Ha & Schmidhuber（2018）复兴，证明生成式神经网络可以学习压缩的时空表征。LeCun 随后将世界模型确立为自主智能架构的核心组件，提出 JEPA（Joint-Embedding Predictive Architecture）。

里程碑系统演化：
```
PlaNet (2019) → DreamerV1-V3 (2020-2023) → MuZero (2020)
  → Sora (2024) → V-JEPA 2 / Genie / Cosmos (2024-2025)
```

### 1.3 为什么世界模型现在成为 AGI 的中心范式

大语言模型（LLM）尽管在语言理解和代码生成上取得了显著成功，但存在根本性局限：缺少对连续高维物理世界的 grounded 理解、缺乏持久的全局状态表征、因果推理能力有限、长周期规划困难——这正是 Moravec 悖论的核心。世界模型通过预测动作的物理后果来直接解决这些问题。

一个特别值得关注的新趋势是 **Chain-of-Thought 推理与世界模型想象的融合**：Coconut 引入连续思维表征在潜空间做广度优先推理；LCDrive 将 CoT 推理与动作规划交错进行；FutureX 提出 auto-think 机制在场景复杂度需要时才激活潜世界模型——暗示世界模型可能取代语言化的思维链，成为"时空想象链（Chain of Imagination, CoI）"。

---

## 二、四轴分类体系

这是该综述最核心的贡献——一个统一的四维分类框架：

### 2.1 架构分类（Section 3）

**按表征格式分类：**
- **观测空间（像素级）表征**：直接预测未来帧（如 GameGAN、DIAMOND）
- **连续潜表征**：RSSM 的确定性+随机性分解（Dreamer 系列）
- **离散 Token 表征**：VQ-VAE → Transformer 自回归（IRIS、Genie）
- **联合嵌入预测**：JEPA 风格，在表征空间做预测而非像素空间
- **结构化/对象中心表征**：Slot Attention、图网络
- **3D/Occupancy 表征**：OccWorld、Copilot4D

**按动力学分类：**
- 确定性 vs 随机性 vs 隐式生成 vs 表征空间预测 vs 记忆增强

**按学习范式分类：**
- 自监督/无监督学习 → 在线模型-based RL → 离线/批学习 → 基础模型范式 → 模仿学习 → 混合多阶段

**按下游用途分类：**
- RL 与规划 → 自动驾驶 → 机器人与具身智能 → 医疗影像 → 视频生成 → 语言推理

### 2.2 方法论家族分类（Section 4）

| 家族 | 核心思想 | 代表系统 |
|------|---------|---------|
| **状态空间/RNN** | RSSM：确定性 RNN + 随机潜变量 | PlaNet, DreamerV1-V3, DayDreamer |
| **Transformer-based** | 注意力替代 RNN，更好长程记忆 | IRIS, STORM, TransDreamer, Genie |
| **扩散-based** | 扩散模型做像素级/潜空间预测 | DIAMOND, GameNGen, World4RL |
| **物理知情** | 嵌入守恒律、对称性、哈密顿/拉格朗日力学 | HNN, LNN, GNN 物理模拟器 |
| **语言增强多模态** | LLM/VLM 提供语义先验，文本做任务描述 | Dynalang, RoboDreamer, Cosmos |

**关键洞察：** 每种方法论有各自的精度-泛化权衡。物理知情模型在已知物理系统上精度最高但泛化差；扩散模型在视觉保真度上最强但计算成本高；Transformer 擅长长程依赖但训练不稳定。未来方向是融合——如扩散 Transformer（DiT）、物理知情 Transformer。

### 2.3 推理策略分类（Section 5）

1. **基于想象的规划**：在世界模型中 rollout 候选动作序列
   - 背景规划（Dyna 风格）：训练时用想象数据增强
   - 决策时前向搜索（MCTS/MPC）：MuZero 的 MCTS、TD-MPC2 的 MPC
2. **潜策略学习**：Dreamer 系列——直接在潜空间想象中优化策略
3. **反事实推理**：A-A-P pipeline（Abduction-Action-Prediction），do-calculus 隔离因果效应
4. **不确定性下的规划**：贝叶斯集成、分布 RL、乐观探索（OWM）

### 2.4 应用领域分类（Section 6）

这是该综述最独特的贡献——将世界模型的应用扩展到传统 MBRL 之外：

- **机器人（6.1）**：DayDreamer 在真实机器人上从零学走路；RoboDreamer 组合式世界模型做机器人想象
- **自动驾驶（6.2）**：GAIA-1/2、DriveDreamer 系列、OccWorld、Vista、Copilot4D
- **视频预测（6.3）**：Sora、Genie、UniSim、GameNGen——视频生成即世界模拟
- **多模态 Agent（6.4）**：LLM 作为文本世界模拟器、VLA 模型、Web Agent
- **RL 与游戏（6.5）**：DreamerV3 Minecraft 钻石挑战、DIAMOND/IRIS Atari 100K、MuZero 超人类表现
- **科学建模（6.6）**：天气预报（Pangu-Weather, GraphCast, GenCast）、分子模拟（NequIP, MACE）、宇宙学（D3M）
- **医疗（6.7）**：脑疾病进展预测、肿瘤演化模拟、手术视频世界模型、EHR 疾病进展
- **教育（6.8）**：学习者认知状态的时间演化建模，知识追踪即潜状态动力学
- **商业与金融（6.9）**：信念建模范式——POMDP 框架处理市场部分可观性、反身性

---

## 三、关键实验与基准发现

综述本身不做新实验，但系统梳理了评估全景：

### 3.1 基准环境

- **RL 基准**：Atari 100K（数据效率极限测试）、DMC（连续控制）、Crafter/Minecraft（稀疏奖励探索）、Memory Maze（长程记忆）
- **自动驾驶基准**：nuScenes、CARLA Leaderboard、Bench2Drive
- **机器人基准**：CALVIN、VIMA-Bench、Language-Table、OXE cross-embodiment
- **视频基准**：VBench 2.0、WorldBench（物理真实性 45% mIoU）、WorldModelBench

### 3.2 关键里程碑数字

| 系统 | 成绩 | 意义 |
|------|------|------|
| DreamerV3 | 首个在 Minecraft 从零收集钻石 | 无人类演示，12 步科技树，稀疏奖励 |
| DIAMOND | Atari 100K 1.46 HNS | 纯世界模型训练的 SOTA |
| EfficientZero | Atari 100K 194% 均值 HNS | 2 小时游戏时间，500× 少于 DQN |
| MuZero | 57 Atari 游戏 + 围棋 + 象棋 + 将棋超人类 | 统一架构跨离散/连续域 |
| GenCast | 97.2% 目标优于 ECMWF 操作集合预报 | 8 分钟单 TPU 生成 15 天集合预报 |

---

## 四、局限性（综述视角）

### 4.1 该综述自身局限

- **跨度极大但深度受限**：147 页覆盖 9 个应用领域，每个领域的讨论相对简略
- **四轴分类可能过于复杂**：实际使用中难以判断一个系统在每条轴上的位置
- **缺少对最新 2026 年工作的覆盖**：Orca（BAAI 世界基础模型）等 2026 年中的工作未包含

### 4.2 世界模型领域的核心挑战（综述总结）

1. **组合预测误差**：多步 rollout 中每步小误差指数级放大——这是世界模型最根本的瓶颈
2. **Sim-to-Real 迁移**：在模拟中学到的动力学转移到真实世界时出现灾难性偏差
3. **碎片化评估**：不同子领域使用完全不同的基准和指标，无法横向比较
4. **计算效率**：扩散-based 世界模型推理极慢——GameNGen 20fps 模拟 DOOM 仍需高端 GPU
5. **幻觉控制**：生成式世界模型在未见场景上可能产生物理上不合理的预测
6. **反事实推理的有效性**：非可识别性（non-identifiability）意味着多个潜状态配置都能同等好地解释历史数据，但给出截然不同的反事实预测
7. **安全性**：医疗、自动驾驶等安全关键领域的部署需求远超当前能力

---

## 五、对领域的影响

### 5.1 作为"领域统一框架"的价值

这可能是世界模型领域第一本真正跨学科的系统综述。它将 RL、视频生成、自动驾驶、机器人、科学计算、医疗 AI、教育测量和金融建模统一在世界模型的框架下——此前这些社区很少对话。对于具身智能领域的研究者（尤其是 WAM 方向），这篇综述提供了理解世界模型全局图景的必备地图。

### 5.2 与 WAM 路线的直接关联

本知识库中的 Motus、FastWAM、Cosmos Policy、LDA-1B、Orca 都可以在综述的分类体系中精确锚定：

- **Motus** → 方法论：Transformer-based + 语言增强多模态；推理：潜策略学习；应用：机器人
- **FastWAM** → 关键洞察验证：综述指出"组合预测误差是根本瓶颈"——FastWAM 正是通过去掉推理时想象来规避此问题
- **Orca** → 学习范式：基础模型范式；表征：联合嵌入预测（Next-State-Prediction 而非 Next-Action）

### 5.3 前瞻性洞察

综述提出的若干方向已被后续工作验证：

- **CoT + 世界模型融合**：综述在引言中重点讨论，Orca 和后续工作正在验证"世界隐空间推理"
- **统一多模态世界模型**：综述指出的方向与 Orca 的"世界基础模型"愿景一致
- **科学世界模型**：天气预报和分子模拟的进展出人意料地成熟——可能为具身智能提供物理先验

---

## 六、硬件启示

综述本身不涉及具体硬件需求，但从覆盖的系统可以总结出硬件格局：

| 系统类型 | 典型硬件需求 | 示例 |
|---------|------------|------|
| RSSM 世界模型（DreamerV3） | 单 GPU（V100/A100）| Atari/DMC 训练 |
| Transformer 世界模型（IRIS/STORM） | 1-4×A100 | Atari 100K |
| 扩散世界模型（DIAMOND/GameNGen） | 4-8×A100+ | Atari/DOOM 模拟 |
| 视频基础模型（Sora/Genie） | 大规模 GPU 集群 | 通用世界模拟 |
| 自动驾驶世界模型（GAIA-2） | 多卡训练 + 实时推理 | 驾驶场景生成 |
| 科学世界模型（GenCast） | 单 TPU 推理 | 15 天全球天气集合预报 |

**关键趋势：** 推理效率正在快速提升——GenCast 只需 8 分钟单 TPU 完成传统超算数小时的计算；但训练仍然昂贵。对于个人研究者，DreamerV3 级别的 RSSM 世界模型在单 GPU 上完全可训，是入门的最佳实践选择。

---

## PDF

[[World Models Survey 原文.pdf]]
