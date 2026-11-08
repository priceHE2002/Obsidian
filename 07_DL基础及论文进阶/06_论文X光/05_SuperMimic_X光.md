---
tags:
  - 论文
  - X光
  - 具身智能
  - 人形控制
  - 世界模型
created: 2026-11-08
paper_title: "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model"
paper_authors: "Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath"
paper_year: 2026
paper_venue: "CoRL 2026 Oral · arXiv 2610.28480"
paper_url: "https://arxiv.org/abs/2610.28480"
arxiv_id: "2610.28480"
method: "SuperMimic"
code_url: "https://supermimic.github.io"
---

# SuperMimic_X光

**SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**
*Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath（UC Berkeley 系） | CoRL 2026 Oral | arXiv 2610.28480*

## 论文说了什么

**问题**：人形机器人要做通用的全身控制，现在主流路线是「跟踪大规模人体动作数据集」。但把这件事 scale 上去非常贵：model-free RL 要么给每个动作训一个专家、再蒸馏合并（有损），要么用一个策略硬怼，靠集群级算力（摘要的基线是 7 天 × 128 GPU）。算力成了 scaling 的瓶颈，也就没法系统研究「动作数据量 → 跟踪能力」的 scaling law。

**视角**：不再让每个动作各自为战，而是先学**一个可微的世界模型**，把「所有动作共享的动力学」amortize 掉。世界模型从标准刚体仿真器记录的 transition 上拟合，和跟踪策略在同一个循环里联合训练；对短段 imagined rollout 做 BPTT，策略就拿到解析梯度——相当于用「可微仿真」替代「采样估计梯度」，把样本效率拉满。

**结果**（摘要给的数字）：

| 指标 | SuperMimic | 基线 |
|---|---|---|
| 达到基线 return | 3 小时 / 单 GPU | 7 天 / 128 GPU |
| GPU-hours | 降低 >1000× | — |
| 129,514 个基准动作的失败数 | 25 | 179 |
| 零失败 | 9.8 小时 / 2 GPU | 未达到 |
| 动作数据 1→264 小时 | 世界模型预测误差稳定，达到目标 return 的 GPU-hours 增长缓慢 | — |
| 真机 | Unitree G1，含杂技动作 | — |

## 方法拆解

- **世界模型**：拟合刚体仿真器的 logged transitions，输入是状态+动作，输出下一状态；可微，因此策略梯度可以对它反传。
- **联合训练**：世界模型与 tracking policy 一个循环训练；策略的梯度来自短 horizon 的 imagined rollout 上的 BPTT，不是环境采样。
- **为什么快**：imagined rollout 在 GPU 上批量跑，比真实仿真器采样快几个量级；解析梯度又比 model-free 的方差估计省样本。两者叠加才有 1000× 的 GPU-hour 差距。
- **规模行为**：动作从 1 小时加到 264 小时，世界模型误差不退化、算力需求缓慢增长——这正是能谈 scaling law 的前提。

## 锐评

- **范式对比**：一类是 per-motion 专家 + 蒸馏（如 ExBody / ASAP 这一系「先专后合」的做法），另一类是单策略 + 大规模并行仿真硬训。SuperMimic 走第三条：先学共享动力学，再用可微模型反向教策略。前两类的问题分别是「蒸馏有损」和「算力不可持续」，它正好各打一个。注意：摘要只写了「state-of-the-art motion tracker baseline」，没点名具体方法，上面的归类是我的领域对照，不是论文实验设置，正文基线 `[摘要未给]`。
- **真实局限**：
  1. 世界模型是从**刚体仿真器**的 transition 拟合的，sim-to-real 的接触误差会直接进模型；真机只在 Unitree G1 上验证，跨本体（不同质量分布、不同执行器）是否还成立没给。
  2. 可微世界模型对接触突变（脚落地、手抓取）天然难拟合，误差在长 horizon 上会累积；摘要只说短 rollout，长程稳定性 `[摘要未给]`。
  3. 宣传的「3 小时」不含世界模型预训练/数据采集成本，端到端账要正文才对得平。
- **一句话**：它把「人形动作跟踪」从算力竞赛变成了「世界模型质量」竞赛——这比多快好省更有价值，因为后者可复现、可比较。

## 借鉴意义

- **世界模型做训练加速器**：不一定非要拿世界模型去做规划/想象控制，像本文这样把它当「可微代理环境」给策略提供梯度，是一条被低估的路线，可以直接迁移到 VLA 的动作头训练。
- **scaling law 工具**：数据量 1→264 小时误差稳定的结果，说明可以拿它做「动作数据 scaling」的受控实验——这正是 [[01_具身智能论文精讲|具身智能精讲]] 里各家堆数据时缺的那把尺子。
- **对人形/双足**：如果世界模型能跨本体微调，per-robot 的 RL 管线可以退化成「一个共享模型 + 少量适配」。

## 关联

- [[01_具身智能论文精讲|具身智能精讲]]：人形全身控制与动作跟踪主线。
- [[2026-11-08-论文推荐]]：当日日报的补录条目。
- [[04_ReactWorld_X光|ReactWorld]]：同属「世界模型」，但 ReactWorld 用来生成/预测，SuperMimic 用来给策略提供梯度，可对照看世界模型的两种用法。
