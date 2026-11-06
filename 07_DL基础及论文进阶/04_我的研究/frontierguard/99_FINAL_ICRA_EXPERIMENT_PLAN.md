# FrontierGuard-RL · ICRA 2028 实验方案（Lead 整合终稿）

> 版本：Lead 整合版 v1.1（同步 v3.1 修复与 judge 06 终审）｜ 日期：2026-11-02 ｜ 目标会议：**ICRA 2028**（预计 2027-09 中旬截稿；ICRA 2027 已于 2026-09 截稿）
> 主方案文件：`05_proposal_v3.md`（v3.1，1,124 行）｜ 评审判决：`04_acceptance_judgment.md` ｜ 终审：`06_acceptance_final.md`
> 发布纪律：本文所有"预计/估计"均标注；未核实文献**不引用**；未产生的结果**不假装存在**。

---

## 0. 一页速览（TL;DR）

**一句话结论**：方案可以投 ICRA 2028，且应在 **2027-08-09（W41）** 投稿；**中稿概率 40%（80% CI 29–51%；v3.1 修复 fixed-K 计价后 ≈43%）**。judge 06 的诚实结论是：在 45 周/¥5,000/单人/单一 embodiment 约束下，**>50% 不可承诺**——单发现实上限 ≈45%（仅当 fixed-K 预算修正 + H4a 成立）；H4/H8 同时成立才有 48–52%，但该组合概率仅 ≈13%，不能当作设计目标。若只完成仿真、真机不显著，应主动降到 RA-L（38–44%），不硬投。

| 项目 | 结论 |
|---|---|
| 投哪儿 | **ICRA 2028 主投**；RA-L 备胎（38–44%）；CoRL 2027 仅"W24 超前 ≥3 周"才启动的期权；IROS 2027 结构上不可行 |
| 概率 | v2 36%（judge 04）→ **v3 40%（80% CI 29–51%，judge 06）→ v3.1 修复后 ≈43%**；>50% 未达；单发现实上限 ≈45%（fixed-K 修复 + H4a），H4+H8 同时成立才 48–52%（概率 ≈13%） |
| 科学问题 | 不确定性感知的 **learning-progress 采样**能否在"固定更新数"下改善大规模并行 RL 的经验分配，并给出 success-band 启发式的**可证伪适用边界** |
| 主实验 | Tier-1 4 族 × N∈{256,1024,4096,16384}（本地 MuJoCo Playground）；Tier-2 4 个 SO-101 任务：**fixed-K 主格 N∈{256,1024}**（T2-Bmain，192 runs）+ 协议 A 部署区制 N=1024/2048（本地 + 租卡）；真机 2 任务 540 trials（+90 pilot，条件 +270） |
| 基线 | uniform、SER、**SGS-band 重实现**、PLR/RED、DRATS 配置级适配、FG；扩展：UPER、max-return、SGS+FG、oracle-LP |
| 统计 | Tier-2 8 seeds（Tier-1 4 seeds）、fixed-K 主终点、TOST 等价（Δ=2.5 pp）、分层 Holm、预注册 W8 冻结 |
| 预算 | **基准 ¥2,997**；单因素 ¥4,353；A100 情景 ¥4,988（极紧，余 ¥12）；双因素未削减 ¥6,386（超限）→ **L1+L2+L3 后 ¥4,905**；FLOOR ¥1,406；全程 ≤¥5,000 |
| 算力 | 本地 **1,256.1 GPU-h**（27.9 h/周；峰值 54% @W7–W8）+ 租卡 **962.0 GPU-h** ≈ **2,218.1 GPU-h** |
| 排期 | 45 周：W2 G0、W5 G1、**W8 预注册冻结**、W16 G3、W20 G4、W24 决策、W30 实验冻结、W31–38 写作、**W41 投稿**、W42–45 缓冲（5.2 周）；本地峰值 54%（W7–W8） |
| 人力 | 1 名研究生 × 45 周 ≈ 11 人月；真机 12–14 人日 + 写作 8 周 |
| 硬件 | 本地 4070 Ti Super 16 GB 可跑 ManiSkill3 RGB N=256（官方 ppo_rgb 默认 256、<15 GB）与 MuJoCo Playground；N≥1024 视觉训练租 4090；2×SO-101 已具备 |
| 与 SGS 的关系 | SGS（2610.12465, CoRL 2026）是**最近邻**，不是"可忽略的 baseline"；FG 的差异化 = LP 导数 × 估计量不确定性 + 可证伪判据 + fixed-K 同预算对照 + 真机证据；**不声称 SGS 是 FG 特例** |

---

## 1. 用户要求与三代理流程的闭环

用户（m00002）要求：
1. 以 ICRA 为目标撰写 FrontierGuard-RL 的实验方案；
2. 创建 3 个 subagent，分别写方案、批改方案、判断中稿概率并给优化建议；
3. 尽可能让中稿率 >50%；
4. 硬约束：租卡 ≤5,000 元、本地 RTX 4070 Ti Super 16GB/32GB/2TB/7800X3D、2 套 LeRobot SO-101、1 人全职 45 周。

三个 subagent 的交付（全部落盘）：

| 角色 | 交付物 | 关键产出 |
|---|---|---|
| proposal-writer | `01_proposal_v1.md`（96 KB）、`05_proposal_v3.md`（v3.1，108.5 KB/1,124 行） | v1 初稿 → 按两轮反馈改成 v3 → 按 judge 06 修成 v3.1（fixed-K 计价） |
| proposal-critic | `02a_critic_prep.md`、`02_critique_v1.md`（52 KB）、`03_proposal_v2.md`（67 KB） | C1–C9 CRITICAL、M0–M18 MAJOR、m1–m12 MINOR；预算复算、真机功效重算、v2 修订版 |
| acceptance-judge | `04a_judge_prep.md`、`04_acceptance_judgment.md`（44 KB）、`06_acceptance_final.md` | v2 判决 36%（26–46%）；top-10 ROI；M1–M10 必改；v3 终审 40%（29–51%）并抓到 fixed-K 计价致命不一致；改会建议 |
| Lead | `00_BRIEF.md`、本文件 `99_FINAL_ICRA_EXPERIMENT_PLAN.md` | 前期调研、事实核查、预算/引用审计、最终整合与投稿决策 |

**关于 >50%**：judge 的结论是诚实且可校准的——**在 45 周、¥5,000、单人、1 个真实 embodiment 的约束下，50% 不是靠"再写漂亮一点"能翻过去的**。judge 06 给 v3 的点估计是 40%（29–51%），并明确说 writer 的 42–48% 高估约 5pp；v3.1 修复 fixed-K 计价后 ≈43%，单发现实上限 ≈45%。H4 与 H8 同时成立才有 48–52%，但 judge 估计该组合概率仅 ≈13%，只能作为 limitation/条件叙事，**不能当作中稿率论证的设计目标**。这比虚报 55% 更有用：它告诉你 **W24 该看哪两个数、什么情况下该改投**。

---

## 2. 课题定位与贡献（v3 定稿）

### 2.1 标题与摘要

- **标题（短版，6 页友好）**：*When Does Learning-Progress Sampling Beat Success Bands? Uncertainty-Gated Allocation for Massively Parallel RL*
- **原标题 A（审计用）**：FrontierGuard-RL: Uncertainty-Gated Learning-Progress Allocation for Massively Parallel Reinforcement Learning
- **摘要要点（~190 词）**：大规模并行 RL 的"更多 env"并不自动等于"更好策略"；uniform 采样把环境花在已掌握或不可学的配置上。SGS（CoRL 2026）用 success-band 三档加权 reset 部分解决了这一点。本文问一个更尖锐的问题：**success-band 何时已经够用、何时必须用学习进度与不确定性？** 我们给出两个可证伪判据（噪声-信号比、桶间可分辨性），提出 FG：归一化 LP 导数 × 估计量不确定性门控 + 三个 guard 的配置级分配器，插在 PPO 外循环。fixed-K 同更新数协议下，4 个 MuJoCo 族 + 4 个 SO-101 视觉任务 + 2 任务真机实验；并公开全部日志、SGS 重实现与预注册。

### 2.2 三条贡献（按 ICRA 审稿人可验证性排序）

1. **机制 + 边界**：给出 success-band 足够/失效的**可证伪判据**（不是又一个 "ours is better" 的故事）；FG 是"带不确定性门控的 LP 导数分配器"，与 SGS 同层可 head-to-head。
2. **方法可插拔、开销可忽略**：FG 不改 PPO 目标，只改每轮 env 配额；M=8 桶、O(M) 更新；在 fixed-K 下与 uniform 同更新数比较。
3. **真机 + 全开源**：2 任务、540 trials、配对设计、盲评；代码/配置/日志/预注册全公开；SGS 无代码 → 我们提供忠实重实现（附保真度判据）。

明确**不声称**：跨任务分配（无 task ID 的共享策略不合法，C9 已修）、reward-agnostic、收敛速度证明、"比 SGS 普适更好"。这些边界写进 Limitations，反而降低被"过度承诺"击中的风险。

---

## 3. 方法（FG 的一句话与七个部件）

**一句话**：FG 是一个外挂在任意 on-policy 并行 RL 上的**配置分配器**——每轮 rollout 后，用"归一化 LP 的导数 × 估计量不确定性"给 M 个配置桶打分，用 Hedge 决定下一轮每个桶分几个 env；三个 guard 防止噪声追逐、饿死与抖动。

| # | 部件 | 不加会怎样 | 消融 |
|---|---|---|---|
| P1 | 归一化 LP（双 EMA + 组内 z-score） | 不同桶成功率尺度不可比，简单但没学好的桶被误判为前沿 | A1 |
| P2 | LP 导数项 Δℓ | 只追 LP 水平 = 贪心，在"高原但仍在进步"的桶上停太早 | A2 |
| P3 | 估计量不确定性门控（bootstrap EMA 离散度；**不声称 model epistemic**） | 小样本高方差被误当"在进步"，噪声追逐 | A3 |
| P4 | guard-1 noise（σ̂_est 大 → 冻结权重） | 同上，硬保护 | A4 |
| P5 | guard-2 starvation（w_min + n_min warm-up） | 某桶被永久饿死，min-group 成功率塌陷 | A5/A5b |
| P6 | guard-3 信任域（ρ 平滑 + 单轮变化上限） | 分配抖动打乱 PPO advantage 归一化 | A6 |
| P7 | Hedge（full-information）而非 EXP3-IX | 每组每轮都有 env、都能观测 reward → full-information；EXP3 的 importance weight 是错配 | A7 |

**统计与公平性**：主终点 fixed-K（Tier-2 K=1,172；Tier-1 K≈2,930），所有方法同 K、同 N、同 L；次终点 fixed-B（部署区制）；协议 C 只做描述性墙钟。8 seeds、混合效应模型、分层 Holm、TOST 等价界 2.5 pp。预注册 W8 冻结 + SHA-256。

**理论**：§4 给出可学性模型、两个判据（噪声-信号比 (i)、桶间可分辨性 (ii)）与 P1–P8 预测；不写收敛证明，只写"何时 success-band 的 SNR 掉到阈值以下"——这是 ICRA 审稿人能接受的"分析"体量。

---

## 4. 实验设计（三层 + 三协议 + 真机）

### 4.1 平台与任务

| 层 | 平台 | 任务 | N | 硬件 |
|---|---|---|---|---|
| Tier-1 | MuJoCo Playground（JAX，状态） | PandaPickCube（A，真 LP 已知）、CheetahRun（B，初速度+噪声）、HumanoidRun（C，姿态+地形）、Go1JoystickFlatTerrain（D，8 速度档） | 256–16,384 | 本地 RTX 4070 Ti Super |
| Tier-2 | ManiSkill3 3.0.1 + Squint（SO-101，RGB） | `SO101ReachCan-v1`、`SO101LiftCube-v1`、`SO101PlaceCan-v1`、`SO101StackCube-v1`（精确注册名；Lift 无目标 site → **spawn 区域桶**；Reach 用 `goal_pos`；Place 用 `bin_pose`；Stack 用 `itemB_pose`） | 256（本地）→1024/2048（租卡） | 本地 + 4090/A100 |
| 真机 | 2× LeRobot SO-101 | LiftCube、PlaceCan（StackCube 为条件项） | — | 本地 |

**Tier-2 工程事实（Lead 已核实源码）**：8 个任务统一 `max_episode_steps=50`、控制模式 `pd_joint_target_delta_pos`、obs 模式 `rgb`/`rgb+state`（wrist camera + 黑背景 overlay）；`train_squint.py` 是 SAC+RLPD，**没有 PPO-RGB** → 必须用 ManiSkill3 官方 `examples/baselines/ppo/ppo_rgb.py` 改造 FG 钩子（默认 `num_envs=256`、显存 <15 GB）；bucket 注入需在 env `_reset` 加 per-env spawn/target 配置，**预留 ~1 周工程**，不是零成本。

### 4.2 基线（6 主 + 4 扩展）

- 主：uniform、SER（2511.21011）、**SGS-band 重实现**（含保真度判据）、PLR/RED、DRATS 配置级适配、FG。
- 扩展：UPER、max-return、SGS+FG、oracle-LP（族 A 真 LP 已知 → 验证估计器偏差）。
- 明确不把 PLR 当"稻草人"：它的 level 生成与本文固定桶不同，只做"分配思想"对照。

### 4.3 消融（A1–A10）

去归一化 / 去导数 / 去不确定性 / 去 w_min / 去 warm-up / 去信任域 / Hedge→EXP3 / M=4 与 16 / E=1 与 5 / oracle-LP。每个部件都对应"失效模式 → 预期变化"，避免"加了就是好"。

### 4.4 真机协议

- 主实验：**45 trials × 3 方法 × 2 任务 × 2 轮 = 540 trials**；同一组 45 个场景布局在三方法间复用 → 配对 McNemar；每任务合并 90 对。
- 主终点：**单侧 FG−uniform > 0**（α=0.05）+ 效应量 95% CI；**20 pp 只是期望量级，不是门槛**。
- 前置 pilot：W22–W23 做 15 trials/条件 ×3 方法 ×2 任务 = **90 trials**，**8 pp 门控**：<8 pp → 降为 270 trials 的"非劣/定性"证据，主叙事回仿真。
- 评分：双人盲评、Cohen's κ≥0.8、第三人裁定；成功率定义写进预注册（Lift：抬起 ≥5 cm；Place：落入目标区；60 s 内）。
- 条件项：第 3 任务 StackCube（+270 trials/+2 人日）默认**不启用**，仅当 pilot ≥20 pp 且 W24 排期超前才做。

### 4.5 统计与功效

- fixed-K 主终点、8 seeds（Tier-1 4 seeds，M10 削减）、混合效应 `J ~ method + log2N + method×log2N + (1|task)+(1|seed)`。
- TOST 等价界 ±2.5 pp（H2/P3 的"FG 不是越大 N 越占便宜"主张）；分层 Holm（family 1 = Tier-2 5 个主比较；family 2 = Tier-1 5 个）。
- 真机：15 pp 差 power≈0.8 **需假设 discordant ≤0.3**（judge 06 提醒），10 pp≈0.45–0.5（**不主张 10 pp**）；pilot 锁定效应量后再开主实验。
- 预注册负结果分支 N1–N3 / D1–D3 全部先写好（见 §8）。

### 4.6 运行清单（v3 核对后）

| 层 | runs | GPU-h | 平台 |
|---|---|---|---|
| Tier-1（T1-A 384 + T1-B 36 + T1-C 72） | **492** | 384.6 | 本地（¥0） |
| Tier-2 本地（T2-L2 192 + **T2-Bmain-L 96** + T2-00a 36 + T2-L1 5） | **329** | 871.5 | 本地（¥0） |
| Tier-2 租卡（**T2-Bmain-R 96** + R2 60 + R3 48 + R4 24 + R5 36 + R6 18 + T2-K 24 + T2-N 15 + T2-00b 3 h + T2-S 60 h） | **321** | 962.0 | 4090 / A100 |
| **合计** | **1,142 runs** | **2,218.1 GPU-h** | 本地 1,256.1 + 租卡 962.0 |

> Lead 独立复算：`附录 D` 脚本（v3.1）可执行，输出 **租卡 ¥2,421 → +12% 缓冲 ¥2,712 → +电费/杂项 ¥285 → ¥2,997**；与 v3.1 §6.2/§6.4/§12.4 闭合（逐行取整差 ¥16，偏保守）。

---

## 5. 预算（硬上限 ¥5,000）

### 5.1 基准表

| 项 | 金额（元） | 说明 |
|---|---|---|
| 租卡（Tier-2 大 N，含 12% 重跑缓冲） | 2,712 | 4090 ¥2.5/h 假设上沿；962 GPU-h + 115 h 缓冲（¥291） |
| 本机电费 | 220 | 0.3 kW × ¥0.6/kWh × 1,256 h 等效满载（偏保守） |
| 数据/杂项 | 65 | 对象存储 + 视频渲染 |
| **合计** | **2,997** | **余 2,003** |
| 耗材（不占 5,000） | 300 | 3D 打印耗材 150 + 标签/背景 50 + 线材 100 |

### 5.2 敏感性

| 情景 | 合计 | 是否 ≤5,000 |
|---|---|---|
| 基准 | ¥2,997 | ✓ |
| 单价 −50%（¥1.25/h） | ¥1,641 | ✓ |
| 单因素 +50%（单价 ¥3.75 或吞吐 ≤2/3） | ¥4,353 | ✓（余 647） |
| A100 情景（N=2048 必须 A100 + 单因素） | ¥4,988 | ✓（紧，余 12；已含 12% 缓冲） |
| 双因素 2.25× 原样 | ¥6,386 | ✗ 超 1,386 |
| 双因素 + L1（砍 R3/R5/R6） | ¥5,418 | ✗ 仍超 418 |
| 双因素 + L1 + L2（R2 5→3 seeds） | ¥5,263 | ✗ 仍超 263 |
| **双因素 + L1 + L2 + L3（砍 T2-K）** | **¥4,905** | ✓（余 95） |
| FLOOR（最小可发表集） | ¥1,406 | ✓（安全系数 3.55） |

**再平衡规则（预注册）**：触发条件 = 实测租价 ≥¥3.2/GPU-h 或吞吐 ≤2/3；L1 砍 T2-R3/R5/R6；L2 砍 N=2048 seeds（R2 5→3）；L3 砍 T2-K；**新增实验必须重算最坏情景，>5,000 先削减再谈加**。诚实结论：**5,000 只在执行三级削减时是安全上限**，不能声称"最坏也安全"；A100 情景余量仅 ¥12，须在 W2 实测后锁定。

---

## 6. 时间线（45 周，2026-11-02 → 2027-09-15）

| 周 | 任务 | Gate |
|---|---|---|
| W1 | 环境落地（Playground / ManiSkill3+Squint / MetaWorld）；精确 env_id；**bucket 注入开工** | |
| W2 | G0：吞吐/显存/5M 步 @N=256 ≥25% success；`ppo_rgb` 可行性 | **G0** |
| W3–W4 | Tier-1 四族实现；FG 超参 pilot；bucket 注入完成 | |
| W5 | 族 A 真 LP 异质性 ≥3× | **G1** |
| W5–W9 | Tier-1 协议 A 主格（384 runs / 319 h，63.8 h/周） | |
| W6–W9 | Tier-1 协议 B（36 runs）+ 全消融（72 runs），16.4 h/周 | |
| W7–W8 | **Tier-2 pilot**（4 任务 @N=256，36 runs / 21.8 h）→ **本地峰值 91.1 h/周（54%）** | |
| **W8** | **预注册冻结**（H1–H9、主终点 T2-Bmain、TOST、分层 Holm、sanity、负结果分支） | **G2** |
| **W9** | **T2-K K-sweep**（24 runs / 63.7 h 租卡，3 天） | **G-S3** |
| W9–W15 | T2-L1 sanity（5 runs / 35 h，≈5 h/周） | |
| **W10–W19** | **T2-L2 协议 A 主对照（192 runs / 543 h）+ T2-Bmain-L fixed-K（96 runs / 271.7 h）** = 814.7 h/10 周（81.5 h/周，49%） | **G3（W16）** |
| W11–W19 | **T2-Bmain-R fixed-K（96 runs / 538.6 h 租卡）**，分 3 批 | |
| W16–W20 | 租卡：N=2048（60）+ breadth（48）+ 消融（24）+ 扩展基线（36）+ 2B（18）+ T2-N（15）+ T2-S（60 h） | |
| W20 | 主结果方向、负结果分支判定；SGS 保真度检查 | **G4** |
| W20–W21 | 真机准备：3D 打印、`deploy.py`→FG 推理接口、相机/关节标定、50 trials 空跑 | |
| W22–W23 | **真机 pilot 90 trials**（15×3 方法×2 任务；8 pp 门控） | **P-GATE（W23）** |
| W24 | 决策：≥20 pp→全量 ICRA；8–20 pp→全量单侧；<8 pp→D2；超前≥3 周→CoRL 期权 | **G5** |
| W24–W27 | **真机主实验 540 trials**（45×3 方法×2 任务×2 轮；StackCube 条件项 +270） | |
| W28–W29 | retry buffer（故障重跑、第 3 任务、标定漂移） | |
| **W30** | **实验冻结**（不改代码/超参/分组；冻结 tag + 数据快照） | **G6** |
| W31–W38 | 写作（6 页重排：Method 1.2 / Theory 0.4 / Exp 2.3；附录/artifact） | **G7（W38）** |
| W37 | 内部评审（导师 + 同组 2 人） | |
| W38 | arXiv 预印 + Anonymous.4open 匿名包；W30 竞品复检 | |
| W39–W40 | 图表精修、artifact、投稿日历核对、90 s/3 min 视频 | |
| **W41** | **投稿（≈2027-08-09）** | |
| W42–W45 | 缓冲：补实验/格式/开源；CoRL 期权（若启动） | |

**W24 决策规则（预注册，2027-04-13）**：真机 pilot 点估计 ≥20 pp → 按 ICRA 计划；8–20 pp → 缩到 2 任务×2 轮但保留单侧主终点；<8 pp → D2 降级 + 优先 RA-L；整体超前 ≥3 周 → 启动 CoRL 2027（需 W28 前初稿，且不能与 ICRA 双投）；IROS 2027 排除。

---

## 7. 风险与 Plan B/C

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | SGS 压制 novelty | 高 | 致命 | §1.6 三句话差异 + 同层 head-to-head + 判据 + 负结果分支 |
| R2 | PPO-RGB 不收敛 | 中 | 高 | G0 门控；D1 改 SAC+FG（+1.5 周） |
| R3 | N=2048 显存不足 | 高 | 中 | 4090→A100（+¥378）；或砍该行 |
| R4 | 视觉吞吐低于假设 | 中 | 中 | W1 实测；单因素情景仍闭合 |
| R5 | bucket 注入不可行 / 桶间 a_g 不可分辨 | 中 | 高 | 已核实 goal 观测；Lift 用 spawn 区域；兜底纯 DR 桶 M=4（预注册） |
| R6 | 真机零样本迁移差 | 中高 | 高 | pilot 门控；降级定性；或 1 天 sim-to-real 微调（标注） |
| R7 | 真机硬件故障 | 中 | 中 | 备用舵机 + 第二台 SO-101 |
| R8 | 本地不能 24/7 | 中 | 中 | 砍 T1-A/N=256 与 T2-L1 各 30%；或临时租 4090（+¥300） |
| R9 | 排期滑期 | 高 | 高 | W26 主格 <60% → 6 页降级版 D3 |
| R10 | 负结果（FG≈SGS-band） | 中 | 中 | 转分析型论文（N1）；不追加拿卡 |
| R11 | 统计功效不足 | 低 | 中 | 主终点混合模型 + 配对；报效应量 |
| R12 | 真机评分主观 | 中 | 中 | 双盲 + 第三人裁定 + 预注册 rubric |
| R13 | MetaWorld CPU 并行度不足 | 中 | 低 | N=128→64；附录只做 MT10 |
| R14 | SGS 状态变化（代码发布/被接收） | 中 | 高 | W1 起月度监控；代码发布→W20 前移植官方实现替换重实现（T2-S ¥150 已留）；被接收→Table 1 明写"同层不同信号" |
| R15 | K-sweep 显示更新饥饿主导 | 中 | 中 | 预期结果之一：主终点用 K=1,172，并报告"更新饥饿是 N 增大的主瓶颈"（叙事转向，非失败） |
| R16 | 噪声审计未发现自然噪声 | 中 | 中 | N2 分支：H4a 不成立 → "success-band 足够"的边界结论，仍是贡献 |
| R17 | 预算最坏情景超支 | 中 | 高 | 双因素原样 ¥6,386 → L1 ≈5,418 → L2 ≈5,263 → L3（砍 T2-K）≈4,905；A100 余量仅 ¥12，W2 锁定；仍超则 D3 |
| R18 | 数据/日志丢失 | 低 | 高 | 每周 rsync + SHA-256；预注册与数据快照双备份 |

**Plan C（最坏）**：W26 时 Tier-2 与真机都未成型 → 降为"Tier-1 4 族 × N≤16k + 理论判据 + 2 任务仿真/真机视频"，投 ICRA short paper/workshop，完整版留 CoRL 2027（若时间允许）或 RA-L。

---

## 8. 近邻工作与差异（全部真实引用，Lead 逐条核实）

**最近邻（必读、必须 head-to-head）**
- **SGS**：Zhang, Guaman Castro, Yin, Dagnino, Gupta, Scalise, Boots. *A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control.* arXiv:2610.12465, CoRL 2026. 项目页 sgs-rl.github.io；代码 coming soon（未开源）。
- **Squint**：Almuzairee, Christensen. *Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics.* IEEE RA-L 2026, doi:10.1109/LRA.2026.3730387；arXiv:2602.21203. SO-101 训练/部署平台。

**其他近邻**
- SER 2511.21011 = Bharthulwar, Tao, Su, *Staggered Environment Resets Improve Massively Parallel On-Policy RL*.
- DRATS 2605.14350 = Corrado, Huang, Hanna, *Distributionally Robust MTRL via Adaptive Task Sampling*.
- PGB 2610.25704 = Xu, Shen, Hou, Bi, *Predictive Bandit Curriculum RL via Gradient-Based Sample Group Discovery*.
- UPER 2506.09270 = Carrasco-Davis, Lee, Clopath, Dabney, *Uncertainty Prioritized Experience Replay*.
- GMC 2605.05856 = Blad, Längkvist, Loutfi, *Gradient-Momentum Coupling: A Parameter-Space Proxy for Learning Progress*.
- LPM 2509.25438 = Hou, An, Du, *Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring*.
- PLR 2010.03934 = Jiang, Grefenstette, Rocktäschel；RED 2110.02439 = Jiang et al.（v1 误标 RPLR，已修正；**不再引用 RPLR**）。
- MTBench 2507.23172 = Joshi, Xu, Liu, Stone, Zhang, *Benchmarking Massively Parallelized MTRL for Robotics Tasks*（基于 IsaacGym Preview 4，仅引用不训练）。
- LeRobot 2602.22818 = Cadene, Aliberts, Capuano, Aractingi, Zouitine, Kooijmans et al.
- SO-101 Bench 2610.36420 = Hickok, *SO-101 Bench: Accessible Spatial Manipulation*（VLA 评测基准，不作训练环境）。
- 2506.03404 = Mayor, Obando-Ceron, Courville, Castro, *The Impact of On-Policy Parallelized Data Collection on DRL Networks*。
- 2610.33420 = Zheng, Lyle, Khetarpal, Dabney, *An Empirical Study of RL Scaling Laws*。
- 2603.01741 = Shitanda, Omura, Harada, Osa, *Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale RL*。

**差异一句话**：SGS 用 success-band 三档"结果信号"加权；FG 用"学习进度导数 × 估计量不确定性"并给出 success-band 何时足够的**判据**；两者同层、同预算、fixed-K 下 head-to-head。**不声称 SGS 是特例、不覆盖其阈值/调度/重采样比。**

---

## 9. 论文、图表、视频、开源

- **6 页分配**：Intro 1.0 / Related 0.5 / Method 1.5 / Theory 0.5 / Experiments 2.0 / Conclusion 0.25（v3 重排；Method 1.2、Theory 0.4、Exp 2.3 是 judge 给的压缩版，二者取一，以 6 页实际排版为准）。
- **图表**：Fig.1 概念（uniform 浪费）；Fig.2 方法框图；Fig.3 scaling J vs N（4 族×4 N×6 方法×4 seeds）；Fig.4a fixed-K；Fig.4b 噪声退化；Fig.5 真机；Table 1 近邻；Table 2 Tier-2 主结果；Table 3 消融；附录 MT10/超参/SGS 重实现/预算审计。
- **视频**：90 s 主版（问题 15 s / 方法 30 s / 结果 30 s / 真机 15 s）；3 min 备用；20 段真机成功/失败对比。
- **开源**：`fg/`、`configs/`、`scripts/`、`results/`、`sgs_reimpl/`、`robot/`；投稿同日 Anonymous.4open；中稿后完整版。
- **写作纪律**：所有数字来自预注册实验或公开日志；不编造引用、不把 hypothesis 写成 result；Limitations 写"单一 embodiment、2 任务真机、SGS 无官方代码"。

---

## 10. 中稿率策略（judge 判决 + v3 修订后的结论）

### 10.1 概率链条

| 阶段 | 点估计 | 依据 |
|---|---|---|
| v1 原案 | 25–35% | novelty 被 SGS 压制 + Tier-2 方法硬伤 + 真机功效不足 |
| v2（critic 修订） | 36%（26–46%） | judge 04：rubric S=3.235；三审 [WA/Borderline/Reject] |
| **v3（judge 06 终审）** | **40%（80% CI 29–51%）** | M1–M10 已落地（S=3.40→p_cond 51%），但 fixed-K 计价不一致按"报告值"扣分；执行成功 0.75 |
| **v3.1（本方案，修复后）** | **≈43%** | 修复 fixed-K 计价（T2-Bmain 替换旧 R1/T2-B，预算 ¥2,997，1,142 runs）；judge 06 给修复一致性 ≈+3 pp |
| **条件上限（不作主论证）** | **≈48–52%** | H4a 成立 P≈0.45、H8 成立 P≈0.50、两者同时 ≈0.13；judge 明确：这是 13% 的彩票，不能进主估计 |

> 概率是**估计值**，不是事实。ICRA 近年录取率约 35%–42%（估计，无官方统计口径），不同年份/赛道波动大；本表点估计误差按 ±3–4 pp 理解。**本表以 judge 06 为准；writer 的 42–48%（点估计 45%）已被 judge 判定高估约 5pp，作废。**

### 10.2 judge 的 top-10 ROI（v3 已落地）

1. Tier-2 fixed-K 一等公民 + K-sweep（+3~5 pp）✓
2. SGS 重实现保真度 + 官方代码跟踪（+3~5 pp，防 −15 pp）✓
3. 域内自然噪声审计（+2~3 pp）✓
4. 真机前置 pilot + 效应量预注册 + 第 3 任务条件项（+2~4 pp）✓（第 3 任务默认关）
5. 修 TQ 三硬伤（σ̂ 术语、SGS=特例、A1）（+2~3 pp）✓
6. H2/P3 改 TOST + 分层（+1~2 pp）✓
7. ICRA framing / 6 页 / 视频（+1.5~2.5 pp）✓
8. "可及性"第一类贡献（+1~1.5 pp）✓（SO-101/LeRobot 低成本栈）
9. arXiv 预印 + artifact + 竞品监控（+0.5~1 pp）✓
10. 预算再平衡 + 重跑余量（间接 +2~3 pp）✓

**明确不做**（避免过度承诺）：HIL-SERL 微调、MT50 扩正文、收敛证明、自建 task-conditioned 统一环境、第 3 真机任务默认开启。

### 10.3 审稿人预案（rebuttal bank 8 条，v3 已改）

1. "Incremental over SGS" → 三档单调**近似**（不是特例）+ 判据 + 同预算 head-to-head + 噪声扫描；若 SGS 代码发布则直接在 camera-ready 前对齐。
2. "Only one real robot / two tasks" → 承认；540 trials 配对设计；仿真 4+8 任务补广度；Limitations 明写。
3. "Why not task-level?" → 无 task ID 的共享策略不合法（C9）；MT10 附录给任务级证据；自建统一 env 成本 3–4 周，不划算。
4. "Guards look ad-hoc" → 每个 guard 对应失效模式 + 消融；n_min=30 由二项方差推出。
5. "Hyperparameters tuned on test tasks" → W8 冻结；只在族 A pilot 调参；主格从未用于调参；预注册公开。
6. "Fixed-B saturates by update starvation" → 这正是加协议 B 的原因；fixed-K 单独报告。
7. "Reward-agnostic claim?" → 不声称；outcome-based；稀疏奖励需 demo/curriculum 启动（Limitations）。
8. "SGS comparison reproducibility" → SGS 无代码；我们公开重实现/超参/日志；官方代码发布后对齐。

---

## 11. 改会策略（judge 复核）

| 会议 | 时间 | 可行性 | 概率 | 结论 |
|---|---|---|---|---|
| **ICRA 2028** | ~2027-09 截稿 | 45 周排期闭合（W41 投，5.2 周缓冲） | **40%（29–51%）→ 修复后 ≈43%** | **主投** |
| RA-L | 滚动 | 可（W34 起准备） | 38–44% | 最佳备胎；N1/D2 时优先 |
| CoRL 2027 | ~2027-06 | 仅当 W24 超前 ≥3 周 | 25–30% | 条件期权，默认关闭；不能与 ICRA 双投 |
| IROS 2027 | ~2027-03 | 实验 W30 才冻结 → 结构不可行 | — | 排除（可投 workshop） |
| ICRA workshop | 与主会同期 | 可行 | — | 兜底 |

**结论**：主投 ICRA 2028；把 RA-L 作为"结果不如预期"的默认备胎；CoRL 期权只在 W24 检查明确超前时启动；IROS 不投。

---

## 12. Lead 审计记录（可追责）

1. **引用真实性**：本文件与 v3.1 正文引用的全部 arXiv 条目，Lead 已用 arXiv abs/API 逐条核实标题、作者、年份；v1 的 RPLR→RED 错标已修正；SGS 2610.12465 由 judge 独立发现并核实（CoRL 2026）。**无编造引用**。
2. **预算算术（v3.1 重算）**：用 v3.1 附录 D 脚本实际执行，输出 ¥2,421 → ¥2,712 → ¥2,997，与正文闭合；1,142 runs / 2,218.1 GPU-h（本地 1,256.1 + 租卡 962.0）与 §6.2/§6.3/§12.4 一致（逐行取整差 ¥16，偏保守）。
3. **内部一致性**：writer 收尾前 Lead 发现并修掉 11 处 v2 残留；judge 06 又抓到 **fixed-K 主格按 fixed-B 计价的致命不一致**（若按 fixed-K 真跑，租卡 ¥2,018→≈¥6,450、总价≈¥7,380 破 5,000）；Lead 建 task-6 让 writer 按 Option B 修成 v3.1，附录 D 脚本复算通过。
4. **事实约束**：ICRA 2028 官网日期**尚未公布**（截至 2026-11-02），所有 2027-09-15 均为"预计"，W20 前须核实；ICRA 2027 已于 2026-09 截稿属实。
5. **硬件可行性**：本地 4070 Ti Super 16 GB 可跑 ManiSkill3 RGB N=256（官方 ppo_rgb 默认 256、<15 GB）与 MuJoCo Playground；Isaac Sim 5.1/Isaac Lab 本地不保证（官方测试卡 RTX 6000/5090）→ 大 N 视觉训练租 4090/A100；2×SO-101 真机与 LeRobot 栈已具备。
6. **judge 复核（已完成）**：`06_acceptance_final.md` 给 v3 点估计 **40%（80% CI 29–51%）**，方案质量 S=3.40→p_cond 51%、执行 0.75；明确 writer 42–48% 高估约 5pp、>50% 未达；修复 fixed-K 后 ≈43%。本文件 §0/§10/§11 以 06 为准。

> 注：judge 曾在写完 04 后出现一次后台任务失败（无 closing message），但 04 文件完整；task-5 已重新调度同一 judge 复核并完成 06。**v3.1 修复后，06 §5-1 的"修 fixed-K 预算"建议已落地**；剩余待办见 v3.1 §13 与本文 §14。

---

## 13. 待拍板事项（Lead 建议）

| # | 事项 | Lead 建议 | 理由 |
|---|---|---|---|
| 1 | SGS 官方代码若 W20 前发布，是否强制移植官方版替换重实现？ | **是**（T2-S ¥150 已留） | 审稿人最可能攻击"SGS 重实现不忠实"；官方代码是唯一解 |
| 2 | CoRL 2027 期权是否保留？ | **保留但默认关闭** | W24 超前 ≥3 周才启动；不能与 ICRA 双投；启动条件写进预注册 |
| 3 | 第 3 真机任务 StackCube（+270 trials/2 人日）是否直接砍？ | **默认砍，只作条件项** | 540 trials 的功效已够检验主终点；第 3 任务边际收益 < 排期风险 |
| 4 | 是否在 W1 就租 4090 做 N=1024 的吞吐/显存实测？ | **否，先本地 N=256 + A100 单次显存测试（T2-00b）** | 省 ¥，W2 G0 后再决定租卡窗口 |
| 5 | 真机主实验用 2 台 SO-101 并行还是 1 台串行？ | **2 台并行（若标定稳定）**，W22–W23 pilot 决定 | 540 trials 串行需 ~3 周；并行可压到 ~2 周，但故障耦合风险高 |

---

## 14. W1 立即可执行清单（下周就做）

1. **环境**：装 MuJoCo Playground（JAX）+ ManiSkill3 3.0.1 + Squint；锁版本（`environment.yaml` / `requirements.txt`）。
2. **D12（原 D9）**：在 SO-101 env 上验证 bucket 注入（per-env spawn/target 配置）——这是 R5 的唯一剩余未知；给 4 个任务各写一个最小 patch，跑 1k 步确认不崩。
3. **G0 前置**：`ppo_rgb.py` 在 `SO101LiftCube-v1` N=256 上跑 5M 步，记录 success/吞吐/显存；目标 ≥25%（否则 D1 预案启动）。
4. **算力询价**：AutoDL/恒源云等 4090/A100 实际单价（¥2.5/¥8 是假设）；把报价写进 W2 预算校准。
5. **引用清零**：把 v3 §9/附录 A 中所有"待核实"在 W2 前用 arXiv API 复核（Lead 已核大部分；剩 MT10 实现细节等非文献项）。
6. **预注册骨架**：把 §2 的 H1–H9、主终点、TOST 界、sanity、负结果分支写成 `prereg.md`，W8 冻结哈希。
7. **真机准备**：校准 LeRobot、`tune_camera.py`、黑背景 overlay、3D 打印 bin/cube/can（Squint STL），跑通 `deploy.py` 的 reach 任务。
8. **ICRA 2028 日期**：W10 前持续盯官网（目前未公布）；同时准备 RA-L 的备选时间表。

---

## 15. 交付物清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `00_BRIEF.md` | 任务简报、约束、时间线、近邻工作、14 项清单 | ✅ |
| `01_proposal_v1.md` | 初稿（96 KB） | ✅ 历史 |
| `02a_critic_prep.md` / `02_critique_v1.md` | 批改准备 / C1–C9 + M0–M18 + m1–m12 | ✅ |
| `03_proposal_v2.md` | 修订版（67 KB，843 行） | ✅ 历史 |
| `04a_judge_prep.md` / `04_acceptance_judgment.md` | 评审准备 / 判决 36%（26–46%）+ top-10 + M1–M10 | ✅ |
| `05_proposal_v3.md` | **终稿 v3.1（108.5 KB，1,124 行；fixed-K 计价已修）** | ✅ |
| `06_acceptance_final.md` | 终审：v3 概率 40%（29–51%）、预算审计、剩余 gap、改会建议 | ✅ |
| **`99_FINAL_ICRA_EXPERIMENT_PLAN.md`** | **本文件：Lead 整合终稿** | ✅ |

---

## 附录 A：本文件与 `05_proposal_v3.md` 的关系

- 本文件是**决策层**文档（读这一份就能决定投不投、钱够不够、什么时候做什么）；`05_proposal_v3.md` 是**执行层**文档（公式、协议、run matrix、rebuttal 全文）。
- 两者冲突时：预算/run 数以 v3 §6 为准；概率/改会建议以本文件 §0/§10/§11 为准（judge 06 产出后以 06 为准）。
- 任何后续修改都应在 `05_proposal_v3.md` 上做，再同步本文件；不要在 99 里发明新实验。

## 附录 B：一句话给导师/合作者

> "我们不是再做一个 SGS；我们回答 SGS 没回答的问题——success-band 什么时候够用。方案已按 ICRA 6 页与 ¥5,000/45 周硬约束收敛；judge 终审 40%（80% CI 29–51%），修复 fixed-K 计价后 ≈43%；50% 在现有约束下不可承诺。W8 预注册冻结，W41 投稿；若 W24 真机 pilot <8 pp，自动降级 RA-L。"

---

*本文件由 Lead 基于 00–06 全部材料整合；所有估计值已标注性质；引用已核实；无编造结果。*
