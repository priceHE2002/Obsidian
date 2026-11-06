# DexSkillBench：面向 Agent 操作的灵巧手技能库——经验分配与校准的受控仿真研究

**版本**：v1（2026-11-02，供 critic/judge 评审；后续 v2/v3 按批改意见修订）
**作者**：proposal-writer（研究团队）；目标会议：**CoRL 2027（主投）/ ICLR 2028（兜底）**
**文件**：`/Users/heyuhang/Desktop/frontierguard/11_proposal_dexterous_v1.md`

---

## 0. 时间校正、定位与一句话摘要

### 0.1 投稿策略（2026-11-02 用户更新：不限于 ICRA；以 `10b_VENUE_STRATEGY.md` 为准）

用户明确：**不投 ICRA 也可以，ICML/ICLR/NeurIPS/CoRL 都在考虑范围内**。据此本方案改为 **CoRL 2027 主投 + ICLR 2028 兜底**，ICRA 与 RA-L **不再作为目标**（原因：纯仿真、以测量装置为主的系统类论文在 ICRA 的录取折价明显；judge 对 ICRA 版 sim-only 的先验仅 24%，而 CoRL 对"纯仿真 + 受控基准/分析"没有硬性折价）。

| 顺位 | 会议 | 预计截稿 | 距 W1 | 为什么 / 风险 |
|---|---|---|---|---|
| **主投** | **CoRL 2027** | ≈2027-06 上旬（W31） | ≈30 周 | 机器人学习主会，审稿人懂灵巧手/技能库；8 页可容纳技能表 + scaling + 校准 + agent 全链。**风险**：比 ICRA 早约 12 周，实验必须 **W26 冻结**，scope 压到 6–8 技能 |
| **兜底** | **ICLR 2028** | ≈2027-09 下旬（W46–W47，**待核实**） | ≈46 周 | ML 顶会，重视"分配→校准→组合"的可泛化实证与统计严谨；纯仿真 OK；可用 CoRL 审稿意见改稿。**风险**：超出 45 周窗口约 1–2 周，需要压缩写作或延后 1 个月 |
| 机会项 | NeurIPS 2027 Datasets & Benchmarks | ≈2027-05（W27） | ≈26 周 | 若基准/技能库 MVP 在 W24 前已成熟，可先投 benchmark track；**不主动分兵**，除非 W20 判定 benchmark 已完备 |
| 不投 | ICML 2027 / NeurIPS 2027 main | ≈2027-01 / ≈2027-05 | — | 截稿太早，W8–W24 前无法产出完整证据 |
| 不投 | ICRA / RA-L | ≈2027-09 | — | 纯仿真系统类论文在 ICRA 有录取折价；转投价值低于 ICLR |

> 所有会议日期均为**预计/待核实**（以 2027 年 CFP 为准），论文中不写成官方事实。CoRL 与 ICLR 不能同期双投同一稿件；CoRL 被拒后再按审稿意见改投 ICLR（级联策略见 §12.1）。

**对写作的影响**：论文按 **8 页 CoRL 版**组织（Method 2.5 页 / System & Benchmark 2 页 / Experiments 2.5 页 / Discussion 1 页），统计按 ML 会议口味（预注册、效应量、负结果、消融）而非"系统演示"；机器人细节（技能表、资产来源、接触参数、失败分析、视频）仍要齐全，因为 CoRL 审稿人要看真实灵巧任务。**摘要与贡献的措辞从"我们构建了系统"改为"我们测量并解释了一条因果链"**（见 §0.2）。

### 0.2 定位（最重要的一段，决定全篇写法）

本方案**不写成"我们造了一个 agent 驱动的灵巧手系统"**。原因是审稿现实：纯仿真、纯系统类的论文在机器人类会议上折价明显（judge 对 ICRA 版 sim-only 的先验仅 24%）；能活下来的是**分析/基准类**工作——它们回答一个可检验的科学问题，并给出可复现的测量装置。因此本文的定位是：

> **用一套可复现的纯仿真基准与受控实验，回答一条因果链：训练期的经验分配（allocation）→ 每个技能的可靠性与校准（per-skill calibration）→ LLM agent 在长程精细操作上的组合成功率（composition）。**

技能库、技能 API、LLM agent 在本方案中是**测量装置（instrument）**，不是贡献主体；它们必须足够真实以支撑因果推断，但不追求"系统最全、技能最多"。三条贡献按此排序：

1. **方法（第一贡献）**：把上一轮的不确定性感知学习进度采样（FrontierGuard, FG）从"单任务配置桶"扩展到"**技能库级**的桶空间"，解决跨技能 LP 不可比问题，并让采样过程**同时产出** per-skill 成功概率的校准估计。核心主张：在大规模并行训练下，"把经验花在能力前沿"既提升样本效率，也改善下游 agent 所需的可靠性信号——这是单一 loss 改造做不到的。
2. **基准与校准（第二贡献）**：一个纯仿真、开源、可复现的灵巧手技能库基准（8 技能 + 4 个长程任务 + 统一 API + 评测协议），以及**校准评测套件**（ECE/Brier/风险-覆盖）。审稿人可以直接下载跑通，这是 sim-only 论文最有力的补偿。
3. **实证（第三贡献）**：固定预算下的 scaling 曲线（S×N 网格）、"分配→校准→组合"的因果链证据（含中介分析与"校准替换"干预实验）、与 Play2Perfect-style / OmniReset / SGS / SER 的受控对比。

**我们明确不声称**：不声称超过 Play2Perfect（2606.26428）或 ADEPT（2608.19182）——它们在真机、硬件与算力规模上远强于我们；不声称"LLM 调技能"本身是新的（Being-0 等已建立该范式）；不声称 sim-only 结论可直接迁移到真机。我们声称的是：**在开源手 + 开源仿真栈上，首次把"分配采样对技能校准与组合成功率的贡献"隔离出来并量化**。

### 0.3 一句话摘要

> 我们提出 DexSkillBench：在 MuJoCo Playground（MJX）上用 8 个灵巧手技能、4 个长程任务与一套带校准成功率的技能 API，受控地测量"训练分配 → 技能校准 → agent 组合"的因果链；方法侧把 FG 的桶分配扩展到技能库并联合产出 per-skill 校准，实验侧用 fixed-K 协议、6 个基线与 8 个消融、5–8 seeds 与预注册统计给出可复现证据。

### 0.4 与 Brief 的对齐声明

本文件严格按 `10_BRIEF_DEXTEROUS.md` 的 §5 十四项清单组织；技能数按 Lead/critic 的 scope 收敛为 **8 个 = 6 手内 + 2 臂式**，跨形态（LEAP↔Aero）**降为附录探测**，calibration 提为第二贡献，LLM agent 组合评测为第三贡献/验证。所有引用均给出已核实的 arXiv ID（核实时间 2026-11-02）；无法核实处标"待核实/假设"。文中严格区分 **假设（hypothesis）/ 预期（expected）/ 已观测（actual）**——本方案阶段没有任何"我们已得到"的结果。

---

## 1. 题目、摘要与关键词

### 1.1 英文题目（主选 + 备选）

**主选**：*DexSkillBench: A Controlled Sim-Only Study of Experience Allocation and Calibration for Agent-Operated Dexterous Skill Libraries*

备选 A（更短，强调因果链）：*From Data Diet to Agent Diet: Calibrating Dexterous Skill Libraries for LLM-Operated Fine Manipulation*

备选 B（更"方法"）：*Uncertainty-Aware Learning-Progress Allocation for Dexterous Skill Libraries: Scaling, Calibration, and Composition*

中文题目：**DexSkillBench：面向 Agent 操作的灵巧手技能库——经验分配与校准的受控仿真研究**

### 1.2 英文摘要（≤250 词，v1 草稿 213 词）

Multi-fingered dexterous manipulation is now trainable at scale in simulation, yet two questions remain open: whether the experience allocation used during massively parallel training improves *per-skill reliability*, and whether such reliability actually transfers to long-horizon tasks executed by an LLM agent that calls skills as functions. We present DexSkillBench, a reproducible sim-only benchmark built on MuJoCo Playground (MJX) with two open hands (LEAP, Aero), eight fine-manipulation skills (six in-hand reorientation/translation/gaiting skills and two arm-based insertion/screwing skills) and four long-horizon composition tasks. On top of this instrument we study a single causal chain: allocation → calibration → composition. Methodologically we extend uncertainty-aware learning-progress (LP) sampling from per-task configuration buckets to a skill-library bucket space, with cross-skill LP normalization and a bootstrap-ensemble dispersion signal that doubles as a per-skill success-probability estimator. We compare against uniform sampling, staggered environment resets (SER), success-band sampling (SGS-style), diverse-reset training (OmniReset-style) and a two-stage play-then-specialize baseline (Play2Perfect-style), under fixed-K and fixed-B protocols, five to eight seeds, pre-registered endpoints and equivalence tests. We further expose every skill as a JSON API carrying preconditions, effects and calibrated success probabilities, and evaluate three planners (calibrated LLM, uncalibrated LLM, scripted FSM) plus oracle and random references on the composition tasks. The benchmark, training code, checkpoints and evaluation harness will be released. We explicitly position the work as a controlled measurement of the allocation–calibration–composition chain rather than as a new robotic system, and we discuss the limits of sim-only evidence.

**关键词**：dexterous manipulation; massively parallel reinforcement learning; learning-progress sampling; calibration; skill libraries; LLM agents; sim-only benchmark; reproducible evaluation

### 1.3 中文摘要（约 300 字）

多指灵巧操作已能在大规模并行仿真中训练，但仍有两个问题未回答：训练期的经验分配是否真的改善了**每个技能的可靠性**；以及这种可靠性是否能转化为 LLM agent 调用技能完成长程任务的**组合成功率**。我们提出 DexSkillBench：一个基于 MuJoCo Playground（MJX）、使用 LEAP 与 Aero 两款开源手、包含 8 个精细操作技能与 4 个长程组合任务的纯仿真可复现基准。在此测量装置上，我们只研究一条因果链：分配 → 校准 → 组合。方法上，我们把不确定性感知的学习进度采样从"单任务配置桶"扩展到"技能库桶空间"，给出跨技能 LP 归一化，并让 bootstrap 集成离散度同时充当 per-skill 成功概率估计器。实验上，我们在 fixed-K 与 fixed-B 两种协议、5–8 seeds、预注册终点与等价检验下，对比 uniform、SER、SGS-style、OmniReset-style、Play2Perfect-style 与 FG。我们进一步把每个技能封装为携带前置条件、效果与校准成功率的 JSON API，并评测三种 planner（带校准 LLM、不带校准 LLM、脚本 FSM）以及 oracle 与随机参照。基准、训练代码、checkpoint 与评测脚本将开源。我们明确把工作定位为对"分配—校准—组合"链条的受控测量，而非一个新机器人系统，并讨论纯仿真证据的边界。

### 1.4 三条贡献（审稿人一眼能看懂）

| # | 贡献 | 一句话 | 对应 RQ | 证据 |
|---|---|---|---|---|
| C1 | 方法 | 把 FG 分配从单任务桶扩展到技能库桶，跨技能归一化 LP，并联合产出 per-skill 校准 | RQ1, RQ2 | S×N 网格 + 校准指标 + 消融 |
| C2 | 基准/校准 | 开源 8 技能 + 4 长程任务 + 统一 API + 校准评测套件的纯仿真基准 | RQ2, RQ3 | 开源仓库 + 复现脚本 + 视频 |
| C3 | 实证 | 分配→校准→组合的因果链证据（中介分析 + 校准替换干预） | RQ3 | 3 planner 对照 + 干预实验 |

### 1.5 与 SGS、Play2Perfect 的三句话关系（正面回答"新在哪"）

1. **与 SGS（2610.12465）**：SGS 用"成功历史三段带"决定 reset 采样，本质是**基于结果的成功率加权**；我们用"归一化 LP 导数 × 估计量不确定性"决定**经验分配**，且分配信号在技能库层面归一化。两者是同一层（bucket 采样）的不同信号，因此我们做 head-to-head，并把"SGS 何时足够"写成可证伪假设（H2 的边界条件）。
2. **与 Play2Perfect（2606.26428）**：他们证明"play 预训练 + 稀疏装配 finetune"在**单策略流水线**上极强（33× 样本效率、真机 60%）；我们不与其比真机与规模，而是把**分配采样**从流水线中隔离出来，测它对 per-skill 校准与 agent 组合的影响，并在开源手上重实现其两阶段配方作基线（scaled-down，明确说明非忠实复现）。
3. **与 Being-0（2503.12533）**：Being-0 证明"FM + 模块化技能库 + Connector"可行，但技能是既有控制器、没有"训练期分配"与"校准成功率"；我们的差异是**技能由大规模 RL 训练得到、并携带可校准的成功概率**，agent 侧研究的是"风险感知规划"而非"语言到技能映射"。

---

## 2. 研究问题与可证伪假设

### 2.1 总问题

> 在固定交互预算（fixed-B）与固定更新预算（fixed-K）下，不确定性感知的学习进度分配能否提升灵巧手**技能库级**训练的样本效率与 per-skill 可靠性校准，并进一步转化为 LLM agent 长程组合成功率的提升？

### 2.2 RQ1（scaling）：分配采样是否随 S 与 N 增大而更有效？

**RQ1**：随着技能数 S（4→8）与并行环境数 N（2048→8192）增大，FG 相对 uniform/SER/SGS-band/OmniReset-style/Play2Perfect-style 的样本效率与最终成功率是否更好？uniform 是否在大规模下饱和？

- **H1（主假设）**：fixed-K 主格（S=8, N=8192, B=700M）下，FG 的**库级平均成功率**（8 技能等权）比 uniform 高 **≥5pp**（绝对），且 95% CI 下界 >0。
- **H1b（交互）**：FG−uniform 的差值随 N 增大而增大（N=2048 与 N=8192 的差值差 ≥2pp），即存在"规模 × 方法"交互。
- **H1c（饱和）**：uniform 的库级成功率从 N=2048 到 N=8192 的边际增益 <1pp，而 FG 的边际增益 ≥3pp（"uniform 饱和"的可证伪版本）。

**预期（expected，非结果）**：若 H1 成立，主曲线应呈现"FG 随 N 持续上升、uniform 趋于平台"的形态；若 H1 不成立但 H1c 成立，仍可发表"uniform 饱和"的负结果（见 §12.6 负结果分支）。

### 2.3 RQ2（calibration）：FG 是否给出更校准的 per-skill 成功率？

**RQ2**：FG 训练出的技能是否给出更校准的成功概率估计（ECE、Brier、风险-覆盖曲线），且这是 agent 规划所需的？

- **H2（主假设）**：FG 的 per-skill 成功率估计的 **ECE 相对降低 ≥30%**（相对 uniform），且 Brier 分数降低 ≥15%（相对）。
- **H2b（边界）**：当技能成功率处于"能力前沿"（p∈[0.3,0.7]）时，FG 的校准优势最大；当 p→0 或 p→1 时两者差距 <10%（相对）。这是"SGS 何时足够"的量化版本。
- **H2c（因果）**：校准优势来自**分配**而非单纯更多更新：在 fixed-B 下把 FG 的分配替换为 uniform（其余相同）后，ECE 优势消失 ≥70%。

**校准信号定义**：对每个技能 g，用 held-out 目标分布上 n_g 次独立 rollout 的成功计数 k_g，得到 Beta 后验均值 p̂_g=(k_g+α)/(n_g+α+β)（α=β=1，Jeffreys 先验 α=β=0.5 可作敏感性）；同时报告 bootstrap 集成（E=3 个 EMA 快照策略）在相同 rollout 上的成功率离散度 σ̂_g，作为"估计量不确定性"（**注意：不声称是 model epistemic uncertainty**，见 §3.4）。ECE 用 10 个等宽 bin 计算；风险-覆盖曲线按 p̂ 排序、逐步覆盖率下的实际成功率绘制。

### 2.4 RQ3（composition）：校准是否转化为 agent 组合成功率？

**RQ3**：把技能封装为带前置条件/效果/校准成功率的 API 后，LLM agent 的长程组合成功率是否显著高于无校准 LLM / 脚本编排 / 单体策略 / 随机技能选择？

- **H3（主假设）**：在 4 个长程任务上，**带校准的 LLM planner** 的任务成功率比**不带校准的 LLM planner** 高 **≥15pp**（配对 McNemar，p<0.05）。
- **H3b（机制）**：校准的收益主要通过"减少执行失败后的错误重试"实现：失败归因中"执行失败→错误恢复"的比例下降 ≥30%（相对）。
- **H3c（剂量）**：给 planner 的成功率估计加入 ±10pp 的随机噪声后，组合成功率下降 ≥8pp——证明收益来自**校准质量**而非"API 里多了一个数字"。
- **H6（可分离性）**：agent 失败可稳定分解为"技能执行失败"与"规划错误"两类（人工/规则标注一致率 ≥85%），且校准信息主要改善前者（后者变化 <5pp）。
- **H7（API 粒度）**：API 粒度存在最优：粗粒度（2–3 个大技能）规划负担低但执行失败率高，细粒度（每技能 8–10 个参数变体）规划负担高；中粒度（8 技能 + 每技能 2–3 个目标变体）组合成功率最高，且与相邻粒度差 ≥5pp。

### 2.5 RQ4（morphology，降为附录探测）

**RQ4**：技能库能否跨灵巧手形态迁移（LEAP→Aero）？迁移瓶颈在哪？

- **H5（附录假设）**：LEAP 上预训练的 FG 分配先验迁移到 Aero 手（仅 1 个旋转技能）时，样本效率优于 scratch ≥20%（达到相同成功率所需 env-steps 减少 ≥20%）。
- **边界声明**：跨形态**不作为主贡献**，只做附录探测（6 runs），因为两种手的 DoF/驱动差异大（LEAP 16-DoF 直驱 vs Aero 腱驱动欠驱动），完整迁移研究需要独立课题。若附录结果不支持 H5，正文只报告"未观察到迁移收益"并讨论原因（动作空间不匹配、接触模型差异）。

### 2.6 预注册（W8 冻结）

- 主终点：**H1**（fixed-K 主格，S=8, N=8192, B=700M，FG vs uniform，库级平均成功率差 ≥5pp）。
- 次终点（Holm 分层校正，family 内 α=0.05）：H1b/H1c（RQ1）、H2/H2b（RQ2）、H3/H3b（RQ3）。
- 等价检验：H2c 与"FG≈SGS"用 **TOST**，等价界 Δ=2.5pp（预注册；理由：小于该值的差异无法在 4–5 seeds 下稳定分辨，也不影响 agent 规划决策）。
- 负结果分支（预注册，不做事后选择）：N1 若 H1 不成立但 H1c 成立 → 以"uniform 饱和 + 分配边界"为主叙事；N2 若 H2 成立而 H3 不成立 → 以"校准是训练期属性、但当前 LLM 无法利用"为负结果，附 agent 能力分析；N3 若主格效应 <2pp 且 CI 跨 0 → 降级为 benchmark 论文（贡献 C2 为主），并明确报告。
- 数据与代码：W8 冻结分析脚本与终点定义；所有 run 的 seed、配置、日志在 W30 前上传匿名仓库；W30 后不再改主终点。

### 2.7 假设汇总表

| ID | 陈述（可证伪） | 阈值 | 对应 RQ | 检验 | 预注册层 |
|---|---|---|---|---|---|
| H1 | FG 库级成功率 > uniform | ≥5pp，CI 下界>0 | RQ1 | 混合效应 + bootstrap CI | 主终点 |
| H1b | FG−uniform 随 N 增大 | Δ(N=8192)−Δ(N=2048) ≥2pp | RQ1 | 交互项检验 | 次终点 |
| H1c | uniform 随 N 饱和 | uniform 边际 <1pp 且 FG ≥3pp | RQ1 | 边际差检验 | 次终点 |
| H2 | FG per-skill ECE 更低 | 相对 −30%；Brier −15% | RQ2 | 配对 bootstrap | 次终点 |
| H2b | 校准优势集中在前沿技能 | p∈[0.3,0.7] 优势最大，极端 <10% | RQ2 | 分层回归 | 探索性 |
| H2c | 校准优势来自分配 | 替换分配后优势消失 ≥70% | RQ2 | 干预 + TOST | 次终点 |
| H3 | 校准 LLM > 无校准 LLM | ≥15pp | RQ3 | 配对 McNemar | 次终点 |
| H3b | 收益来自错误恢复减少 | 执行失败→错误恢复比例 −30% | RQ3 | 归因比例检验 | 探索性 |
| H3c | 收益来自校准质量 | 加 ±10pp 噪声后 −8pp | RQ3 | 剂量实验 | 探索性 |
| H4 | S 从 4→8 时 uniform 边际收益下降更快 | 交互 ≥3pp | RQ1 | S×方法交互 | 次终点 |
| H5 | LEAP→Aero 迁移收益 | ≥20% 样本效率 | RQ4 | 附录 | 探索性 |
| H6 | 失败可分离且校准改善执行失败 | 标注一致率 ≥85%；规划失败变化 <5pp | RQ3 | 归因分析 | 探索性 |
| H7 | API 粒度有最优 | 与相邻粒度差 ≥5pp | RQ3 | 三水平对照 | 探索性 |
| H8 | 结论对噪声/分布偏移稳健 | 成功率下降 <5pp 且排序不变 | 全部 | 鲁棒性网格 | 探索性 |

> 说明：H4 在本版中并入 RQ1（S 从 4→8），编号保留以对齐 Brief；H8 的"渲染噪声"在纯状态输入下改为"观测噪声 + 物体质量/摩擦/尺寸随机化 + 目标分布偏移"三档。


---

## 3. 方法：技能库级分配、校准信号与训练集成

> 本节回答三个问题：FG 在"技能库"设定下怎么定义（3.1–3.3）；它如何同时产出可校准的成功概率（3.4）；它如何以最小工程量嵌入 PPO（3.5–3.7）。方法不是"又一个 PPO 变体"，而是**分配器 + 估计器**：分配器决定经验花在哪，估计器把训练过程变成 agent 可用的可靠性信号。

### 3.1 问题形式化：技能库作为一个分配问题

设技能库 $\mathcal{G}=\{g_1,\dots,g_8\}$，每个技能 $g$ 是一个 POMDP $(\mathcal{S},\mathcal{A},\mathcal{O},P_g,r_g,\rho_g,\gamma)$，观测含本体状态与目标（§4.1 表）。所有技能共享同一机器人形态与同一动作空间（LEAP 16-DoF + 7-DoF 臂 = 23 维关节目标），因此可用**一个共享 goal-conditioned 策略** $\pi_\theta(a\mid o, z_g)$ 训练（$z_g$ 为技能嵌入 + 目标编码）；手内技能把臂的 7 维动作固定为 home 姿态（mask），臂技能的 16 维手指作为夹持保持。共享策略是 RQ1 "技能数 S 增大时的 scaling" 的必要条件——否则"技能库"退化为 8 个互不相关的单任务训练，无法测分配效应。

每个技能内部再划分为 $M_g$ 个**配置桶**（bucket）$b=(g,c)$，$c$ 是初始状态/物体位姿/目标区域/难度参数的离散化（§4.2 给每个技能的桶定义）。总桶数 $M=\sum_g M_g$，本方案 $M_g=4$（技能内 2×2：目标区域 × DR 档），故 $M=32$。并行环境池 $N$ 个 env 在每 $T_{\text{alloc}}$ 次 PPO 更新后被重新分配到桶上：分配向量 $n\in\Delta^{M-1}$（$\sum_b n_b=N$，整数化见 3.6）。

**训练目标**：在固定预算下最大化**库级部署成功率**
$$
\max_{\theta,\{n^{(t)}\}} \; \frac{1}{|\mathcal{G}|}\sum_{g}\mathbb{E}_{c\sim\mathcal{D}^{\text{test}}_g}\left[\mathbb{1}(\text{success}_g(\pi_\theta))\right],
$$
其中 $\mathcal{D}^{\text{test}}_g$ 是**部署分布**（不是训练 reset 分布），这是校准评测与 agent 评测共用的分布，也是与"在训练分布上刷成功率"的关键区别。固定预算有两种协议（§5.3）：fixed-B（总 env-steps $B$ 固定）与 fixed-K（PPO 更新次数 $K$ 固定，$B=K\cdot N\cdot L$）——后者避免"N 越大更新越少"的伪影，是**主协议**。

### 3.2 桶空间与跨技能 LP 归一化（核心难点）

上一轮 FG 只在单任务内做配置桶分配；扩展到技能库后，**LP 在技能间不可比**（不同技能的成功率尺度、难度、奖励都不同），这是本方法的第一个技术点。我们采用**两级分配**：

**Level-1（技能间）**：技能 $g$ 的"进度势能"定义为
$$
u_g(t)=\big(\bar p_g^{\text{fast}}(t)-\bar p_g^{\text{slow}}(t)\big)\cdot\big(1-\bar p_g^{\text{slow}}(t)\big) + \lambda\,\hat\sigma_g(t),
$$
其中 $\bar p_g^{\text{fast}},\bar p_g^{\text{slow}}$ 是技能级成功率的快/慢 EMA（半衰期分别为 $H_f$ 与 $H_s$ 个更新，默认 $H_f=20,H_s=200$），$(1-p)$ 项惩罚已接近饱和的技能（"还能学多少"），$\hat\sigma_g$ 是估计量不确定性（3.3），$\lambda$ 控制乐观探索（默认 0.2）。三项都定义在 [0,1] 尺度上，因此**跨技能可比**，无需知道各技能的绝对难度。

**Level-2（技能内桶）**：桶 $b=(g,c)$ 的 LP 用同样形式在桶内计算：
$$
\text{LP}_b(t)=\big(\bar p_b^{\text{fast}}-\bar p_b^{\text{slow}}\big)\cdot\big(1-\bar p_b^{\text{slow}}\big)+\lambda\hat\sigma_b,
\qquad
\widetilde{\text{LP}}_b=\frac{\text{LP}_b-\mu_g}{\sigma_g+\epsilon},
$$
其中 $\mu_g,\sigma_g$ 是该技能所有桶 LP 的滑动均值/标准差（**技能内标准化**，使不同技能内部的桶可比较），$\epsilon=10^{-6}$。最终桶权重
$$
w_b(t)\propto \exp\big(\eta_1 u_g(t)+\eta_2 \widetilde{\text{LP}}_b(t)\big),\quad \eta_1=\eta_2=1.0 \text{（默认，消融扫描）}.
$$

**为什么需要两级而不是直接把 LP z-score 跨技能比**：跨技能 z-score 假设各技能内部桶难度分布相似，这在插入（单峰、窄）与重定向（多峰、宽）之间不成立；两级把"跨技能比较"限制在 $u_g$（同一语义的量），把"技能内精细分配"留给 $\widetilde{\text{LP}}$，避免一个技能的高方差桶吞掉全部预算。消融 A2（去归一化）、A3（改为单级跨技能 z-score）直接检验这一设计。

**与 SGS 的区别（写进方法而非只写在 related work）**：SGS 的权重来自"成功率所处的三段带"（成功过高→减、过低→减、中间→加），是**对当前成功率的静态映射**；我们的权重来自**成功率的导数**（学到多快）**乘**剩余空间与不确定性，并且有技能内标准化与两级结构。当 $H_f=H_s$（无导数）且去掉不确定性与归一化时，FG 退化为 SGS-like 的单调映射——这是消融 A1/A2 的"三档单调近似"对照（§4.6 给了近似成立的精确条件，不声称"SGS 是特例"）。

### 3.3 估计量不确定性：bootstrap EMA 离散度

每个桶维护 $E=3$ 组**不同半衰期**的 EMA 成功率估计（$H=\{10,30,100\}$ 次更新），每组的更新只依赖该桶的 episode 结果。$\hat\sigma_b$ 定义为三组估计的标准差：
$$
\hat\sigma_b=\operatorname{std}\big(\hat p_b^{(H_1)},\hat p_b^{(H_2)},\hat p_b^{(H_3)}\big).
$$
**必须说清它是什么**：这是**估计量不确定性**（有限样本 + EMA 滞后导致的离散度），**不是 model epistemic uncertainty**，也不是贝叶斯后验。它的作用是：(a) bandit 的乐观探索项（不确定的桶获得探索加成）；(b) 置信度 guard（$\hat\sigma$ 过大时限制分配，3.4）；(c) **校准诊断**：$\hat\sigma$ 与"实际误差 $|p̂-p_{\text{true}}|$"的校准图（§5.7 图 F4b），若相关性弱则 guard 退化（G-1 判据）。上一轮审稿人曾正确指出"σ_epi 是采样方差不是模型不确定性"——本版从定义、命名到用途全部改正，并在论文中显式写一段"我们不声称什么"。

### 3.4 校准信号：从训练日志到 agent 可用的成功率

对每个技能 $g$，在**冻结的部署分布** $\mathcal{D}^{\text{test}}_g$ 上做 $n_g$ 次独立 rollout（$n_g=300$，含目标区域×DR 的组合分层），得到成功计数 $k_g$，用 Beta 后验
$$
\hat p_g=\frac{k_g+\alpha}{n_g+\alpha+\beta}\quad(\alpha=\beta=1),\qquad
\text{CI}_{95\%}=\text{Beta 分位数}.
$$
同时报告两种校准形式：(i) **原始后验**（训练日志 + 部署评测的合并，按技能分层的 Beta-Binomial）；(ii) **isotonic 重校准**（在 calibration split 上拟合，test split 报告）——两者都报，避免"校准靠后处理"的质疑。校准指标：ECE（10 bin）、Brier、风险-覆盖曲线（按 $\hat p$ 降序，报告覆盖率 0.5/0.7/0.9 时的实际成功率）、以及 p∈[0.3,0.7] 前沿区的分层 ECE（H2b）。

**H2 的机制叙事**：FG 把经验集中在前沿桶，因此前沿区的样本量更大、成功率估计更接近 0.5，Beta 后验方差更小、ECE 更低；uniform 把大量样本花在 p≈1 的简单桶上，导致前沿区估计稀疏且偏乐观。**注意**：整体 ECE 可能被极端桶主导而差异不大，因此 H2 的主指标是**前沿区分层 ECE**，整体 ECE 作次要指标——这一点必须在预注册中写死（§2.6）。

### 3.5 分配器：Hedge + 三重 guard

桶权重用 **Hedge（指数权重）** 更新（EXP3 作消融 A6）：
$$
w_b^{(t+1)}=w_b^{(t)}\exp\big(\eta\,\tilde r_b^{(t)}\big),\qquad
\tilde r_b^{(t)}=\operatorname{clip}\big(\widetilde{\text{LP}}_b^{(t)},-1,1\big),
$$
得到权重后与均匀分布混合 $\tilde w=(1-\varepsilon)w+\varepsilon/M$（$\varepsilon=0.05$，保证探索与防塌缩），再整数化为每桶 env 数（3.6）。

三重 guard（每步检查，任何触发都写日志并在论文中报告触发率）：

| Guard | 触发条件 | 动作 | 防什么 |
|---|---|---|---|
| G-1 置信度 | $\hat\sigma_b>\sigma_{\max}$ 或 $n_b^{\text{ep}}<n_{\min}$（每桶最少 episode，默认 5） | 该桶权重置为均匀下限，不发额外 env | 高方差桶吞预算（"噪声当进度"） |
| G-2 noisy-TV | 同一桶连续 $W$ 个 episode 成功且 LP≈0 但奖励方差高，或其配置集中在单一 goal 上 | 冻结该桶 50 更新并记录 | 把"反复成功但无信息"当探索（Noisy-TV） |
| G-3 分布塌缩 | 权重熵 $H(w)<H_{\min}$（默认 0.5·log M）或单桶份额 >0.4N | 向均匀混合，临时把 $\varepsilon$ 提到 0.15，持续 20 更新 | 分配塌缩到少数桶，丢技能覆盖 |

**与 SER 的组合**：SER（2511.21011）用 staggered resets 提升并行数据多样性；我们的分配器与 reset 策略正交，可组合为 FG+SER（基线之一）。组合时 guard 的 $n_{\min}$ 按 SER 的有效独立 episode 数调整（SER 下同桶 reset 更频繁，$n_b^{\text{ep}}$ 计数按完成 episode 计，不按 step 计）。

### 3.6 整数化与重分配节奏

权重到整数 env 数用**最大余数法**（largest remainder）：$n_b=\lfloor N\tilde w_b\rfloor$，剩余名额按小数部分降序分配；保证每桶至少 1 个 env（若 $M\le N$）。重分配每 $T_{\text{alloc}}=20$ 次 PPO 更新一次，且**当前 episode 未结束的 env 不迁移**（避免半截 episode 污染桶统计）；新配置在下一次 reset 生效。整数化误差上界 $M-1$ 个 env，对 $N\ge2048$ 可忽略（<0.05%）。

### 3.7 与 PPO / UTD / 训练的集成

- **策略**：共享 actor-critic，MLP 3×256（tanh），输入 = 本体（16 手 + 7 臂 + 速度）+ 物体 pose/vel + 目标 + 技能 one-hot/嵌入 + 桶 id 嵌入（8 维）；输出 23 维高斯均值（对角 log-std 可学习），动作经 tanh 映射到关节目标范围。
- **PPO**：clip 0.2，GAE $\lambda=0.95$，$\gamma=0.99$，lr $3\times10^{-4}$（线性衰减），epochs 4，minibatch = $\max(NL/4, 4096)$；熵系数 0.001（可在消融中扫）。
- **UTD**：主格 UTD=1（每收集 $N\cdot L$ 步做 1 次更新，$L=50$）；fixed-K 即固定更新次数 $K$；高 UTD 仅作附录（§5.3）。
- **fixed-K 主协议**：$K=1221$ 次更新（对应锚点 $N=8192,B=500M,L=50$ 的 $K=B/(N\cdot L)$）；在 $N=2048$ 时 $B=125M$ 保持同 $K$。所有方法（含基线）在同 $N$ 下 $K$ 相同，因此"大 N"不会变成"少更新"。
- **奖励**：统一用 Playground 风格的稠密奖励（距离 + 朝向 + 接触项）**仅在训练期**；评测只看成功判据（阈值化），避免"奖励设计差异"混淆方法对比。所有方法共用同一奖励函数——这是公平性的底线（SGS/OmniReset 类方法同样用该奖励；OmniReset-style 若原生要求单奖励，则用同一 reward 的稀疏版并说明差异）。
- **reset**：默认 reset 到桶配置采样（桶内均匀）；SER 基线用轮转 reset；FG 与 SER 组合时按 SER 的轮转但桶分配由 FG 决定。

### 3.8 伪代码

```
输入: 技能库 G (8), 桶 M=32, env 数 N, 预算 K 更新, 超参 H_f,H_s,λ,η,ε,T_alloc
初始化: 每桶 EMA 估计 (E=3 组), 权重 w←uniform, env→bucket 均分
for t = 1..K:
    分配: w ← (1-ε)·softmax(η1·u_g + η2·zscore(LP_b)) + ε/M     # 3.2
          三重 guard 修正 w                                        # 3.5
          n ← largest_remainder(w·N); 未结束 episode 不迁移       # 3.6
    rollout: for env i: if done: reset(bucket_i); s_i←obs
             收集 L 步 {(o,a,r,done)}                              # 并行 N×L
    更新: PPO 更新 4 epochs on buffer; 记录 success 事件            # 3.7
    统计: 更新每桶 fast/slow EMA 与 σ̂; 更新技能级 u_g               # 3.3
    每 T_alloc 次: 重分配 (上式)                                   # 3.6
输出: 策略 θ, 每技能校准 (p̂_g, CI_g), 分配轨迹 (论文 Fig.2/Fig.4)
```

### 3.9 复杂度与工程量

- **计算复杂度**：每更新分配计算 $O(M)=O(32)$（可忽略）；统计更新 $O(M+|\mathcal{G}|)$；主要开销仍是仿真与 PPO，额外开销 <0.1%（W2 pilot 实测后报告）。
- **内存**：每桶 EMA 常数级，可忽略。
- **代码钩子（100–200 行）**：`dsb/allocation/buckets.py`（桶注册与配置采样，~60 行）、`dsb/allocation/hedge.py`（权重更新 + guard，~50 行）、`dsb/allocation/integerize.py`（整数化 + 迁移保护，~30 行）、`dsb/training/hooks.py`（在 `train_ppo.py` 的 rollout/reset 处插 3 个 hook：`on_episode_end` 记成功、`on_update` 更新权重、`on_alloc` 重分配，~40 行）。
- **工程量**：臂+LEAP 集成与标定 2–3 周；6 个新环境（S3–S8）约 9–10 周；训练/分配代码 2 周；校准与 agent 2 周；benchmark 打包 2 周；合计约 **20 周工程**（含 20% 缓冲，见 §8 人力表）。


---

## 4. 技能库与长程任务（测量装置的具体化）

### 4.1 Embodiment 选择：单形态"LEAP 手 + 7-DoF 臂"（方案 A），并说明为何不是 B/C

MuJoCo Playground 现有的灵巧手任务是**手固定在基座上的手内任务**（`manipulation/leap_hand/reorient.py`、`rotate_z.py`；`manipulation/aero_hand/rotate_z.py`），而装配类任务 `manipulation/aloha/single_peg_insertion.py` 是**双臂 + 平行夹爪**（1 mm 级容差，`ctrl_dt=0.0025`，`impl="warp"`，`naconmax=24*1024`，稠密奖励），`aloha/handover.py` 是双臂交接。**Playground 没有现成的"机械臂 + 多指手"环境**，因此 embodiment 必须自建，三选一：

| 方案 | 描述 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| **A（采用）** | 7-DoF 臂（Franka/UR5e/ViperX MJCF）+ LEAP 手，23 维关节目标 | 一个机器人完成全部 8 技能与 4 长程任务（coherence 成立）；CoRL 审稿人熟悉的形态；臂承担定位、手承担精细操作，技能划分自然 | 需自制 MJCF 装配与标定（+2 周）；臂+手接触场景吞吐低于纯手内（约 12k steps/s @A100，假设） | **主方案** |
| B | 双臂两 LEAP 手对置（无臂） | 不用建臂；handover/双手协作叙事 | "机器人系统"感弱；插入/工具任务的空间范围受限；与主流 arm+hand 形态不符 | 备选（若 A 的 MJCF 在 W6 前无法稳定） |
| C | 混合 Tier：LEAP 手内 5 技能 + Aloha 臂夹爪 3 装配技能 | 最大化复用现成 env，工程量最小 | **跨 embodiment 无法组成同一长程任务**，只能做 benchmark 的 two tracks；审稿人会问"你的系统到底是什么" | 降级方案（W20 若 A 不可行则改 C，标题改为 benchmark） |

**A 的工程细节**：从 MuJoCo Menagerie 取 Franka Emika Panda 或 ViperX 的 MJCF，把 LEAP 手的基座焊接到末端法兰（固定 joint 或 weld equality）；臂 7 维用位置执行器（`position`，kd 阻尼），手 16 维沿用 Playground 的位置执行器参数；标定 home 位姿使手内任务（S1–S4）时臂保持静止且不与环境碰撞。**风险**：接触参数（`solref/solimp`）在臂+手长链条上更难调，插入任务的穿透抖动可能导致 1 mm 容差不可达——W4 先用 Aloha 现成 env 验证 MJX 在我们机器上的 1 mm 可达性（Tier-0 sanity），再决定 S7 的容差是 1 mm 还是 2 mm。若 W6 时 arm+LEAP 仍不稳定，按 Plan B 降级（见 §9 风险 R2）。

### 4.2 技能库：6 个手内 + 2 个臂式（8 技能，按 Tier 组织）

按 Lead/critic 的 scope 收敛：**8 个技能 = 6 个手内技能 + 2 个臂式技能**（judge 原建议 8；critic 给 67 人日/13.5 周的估计，本方案在此基础上加了 20% 缓冲）。统一观测 = 本体（16 手 + 7 臂 pos/vel）+ 物体 pose/vel + 目标编码 + 技能嵌入 + 桶 id；动作 = 23 维关节目标（手内技能时臂 7 维固定 home 并 mask）。

| ID | 技能 | Tier | 环境来源 | 目标 / 成功判据 | 桶定义（M_g=4） | 单技能最低预算 | 人日 |
|---|---|---|---|---|---|---|---|
| S1 | Reorient-Any | T1 手内 | 复用 `leap_hand/reorient.py` | 任意目标四元数；朝向误差 <0.1 rad 并保持 5 步（以 env 判据为准，待核实） | 姿态区域 2 × DR 2 | 80M | 3（输入适配） |
| S2 | Rotate-Z-Continuous | T1 手内 | 复用 `leap_hand/rotate_z.py` | 绕 z 连续旋转 ≥2π 且未掉落 | 角速度 2 × DR 2 | 60M | 2 |
| S3 | In-hand-Translate | T1 手内 | **新建**（基于 reorient 资产/奖励） | 立方体掌内平移到目标位置；误差 ≤5 mm 并保持 5 步 | 目标区域 2 × 尺寸 2 | 90M | 8 |
| S4 | Finger-Gait / Regrasp | T1 手内 | **新建** | 指步态推动 30 mm 或从差姿态重抓；位移 ≥30 mm 且未掉落 | 方向 2 × 摩擦 2 | 100M | 8 |
| S5 | Reorient-Cylinder | T1 手内 | **新建**（圆柱/罐资产） | 把圆柱重定向到目标姿态（物体泛化：cube→cylinder） | 姿态区域 2 × 半径 2 | 90M | 6 |
| S6 | Edge-Align / Fingertip-Push | T1 手内 | **新建** | 绕棱边翻转 90° 或把物体推到指腹区域；目标姿态误差 <0.15 rad | 方向 2 × 摩擦 2 | 80M | 6 |
| S7 | Peg-Insert-2mm（1 mm 参数化） | T2 臂式 | **新建**（臂+LEAP；几何参考 `aloha/single_peg_insertion.py`） | 手持有销插入 2 mm 间隙孔（可调 1 mm）；深度 ≥20 mm 且横向误差 <0.5 mm | 孔位 2 × 销长 2 | 120M | 10 |
| S8 | Screw-Cap-2Turns | T2 臂式 | **新建**（臂+LEAP；螺纹用螺旋副+摩擦近似） | 旋盖 2 圈并轴向推进 ≥2 mm；无滑牙（滑移 <10%） | 螺纹间隙 2 × 初始角 2 | 120M | 10 |
| A1 | Aero-Rotate-Z（附录） | 附录 | 复用 `aero_hand/rotate_z.py` | 同 S2（Aero 腱驱动手） | 角速度 2 × DR 2 | 60M | 1（复用） |

合计：**6 手内 + 2 臂式 + 1 附录**；新环境 **6 个**（S3/S4/S5/S6/S7/S8），复用 3 个（S1/S2/A1）。工程量：技能实现 53 人日 + arm+LEAP 集成/标定 2–3 周（10–15 人日）+ 20% 缓冲 ≈ **68–70 人日 ≈ 13.5–14 周**（与 critic 的 67 人日/13.5 周一致）。

**止血方案（预注册在 §9 风险表）**：
- **W8 止损**：若 6 个手内技能中可稳定训练的 <4 个，则砍到 **4 技能方法论文**（S1/S2/S7/S8 或 S1/S3/S4/S7），主格改为 S=4，RQ2/RQ3 保留但 agent 降为小规模验证；预算按 §7.3 的削减顺序重排。
- **W20 止损**：若臂式技能（S7/S8）在 W20 仍未达 30% 成功率，则**砍臂式技能**，技能库改为 6 手内技能，长程任务改为手内序列（L4 类型），论文标题/贡献改为"手内技能库"版本；仍可投 CoRL（诚实报告"臂式扩展未达预期"作为 negative result）。

**Tier 分层**（写作时按此组织表格）：T1 = 手内 6 技能（可全部在本地 + 小 N 训练）；T2 = 臂式 2 技能（需 arm+LEAP，A100 上跑）；T3 = 附录 Aero。Tier-0 交叉验证用 Playground 现成的 `aloha/single_peg_insertion.py`（1 mm 容差、`ctrl_dt=0.0025`、`impl="warp"`、`naconmax=24*1024`）验证 MJX 在我们机器上的亚毫米插入可达性，再投入自建 S7/S8。

**为什么手内技能占多数**：手内操作是"大量精细操作"的核心（重定向、旋转、平移、指步态、物体泛化、棱边翻转），且 Playground 已有 S1/S2 的成熟资产可复用，工程风险低；臂式技能只保留 2 个高价值接触任务（1–2 mm 插入、旋盖），既保证"技能库覆盖精细装配"，又把 arm+LEAP 的工程风险限制在 2 个环境上。

### 4.3 共享策略 vs 每技能一策略（RQ1 的一部分）

主方案：**共享 trunk + 技能嵌入**（$z_g$ one-hot 8 维 + 目标编码），8 技能联合训练；每技能独立策略作为消融 A10。若 pilot 出现明显负迁移（某技能成功率比独立训练低 >5pp 且 CI 不重叠），退回"共享 trunk + 技能 adapter"，并把负迁移作为机制结果报告（可发表的 negative result）。

### 4.4 4 个长程组合任务（同一 arm+LEAP 机器人完成）

| ID | 任务 | 技能序列（API 调用） | 成功判据 | 为什么需要校准 |
|---|---|---|---|---|
| L1 | 双件装配 | S1(reorient)→S7(insert 2mm)→S8(screw 2 turns) | 销到位 + 盖拧紧（状态判定） | S7 成功率 ~0.6、S8 ~0.7 时，无校准 planner 会反复重试或跳过前置 |
| L2 | 紧配合 + 拧紧（带扰动） | S1→S7(1mm)→S8→(失败检测后 S7 重试) | 插入 + 拧紧完成；扰动 = 插入时施加侧向力 | 扰动后 p̂ 漂移，需风险感知决定重试哪一步 |
| L3 | 三件批量装配 | 3×[S1→S7]（6 次调用） | 三件全部装配（顺序可变） | 长程误差累积；校准用于选择"先做成功率高的件"保底 |
| L4 | 手内精细序列 | S5(reorient cylinder)→S3(translate)→S2(rotate 2π)→S4(regrasp) | 四步全部成功且物体未掉落 | 纯手内链；校准决定何时重试 S3/S4 而非整链重来 |

**Coherence 回答**（审稿人必问）：4 个任务全部由**同一台 arm+LEAP 机器人**、同一策略库、同一 API 完成；L1–L3 是臂式序列，L4 是臂静止的手内序列（同一个机器人），因此"一个机器人完成全部长程任务"成立。若 W20 止损砍掉臂式技能，则只保留 L4 + 一个手内组合任务，并明确论文改为手内技能库版本。

### 4.5 理论直觉与可检验命题（不是定理）

CoRL/ICLR 审稿人会问"为什么 FG 会更好，而不只是工程 trick"。我们给 3 个**可检验命题**（不声称收敛性证明；假设与局限写在每条后面）：

**设定**：库内 $M$ 个桶（技能×配置），总样本预算 $B=N\cdot L\cdot K$；uniform 平均分配，桶 $b$ 得到 $n_b\approx B/M$ 个样本；FG 按权重 $w_b$ 分配，前沿桶获得 $n_b\gg B/M$。桶成功率估计 $\hat p_b$ 的方差 $\approx p_b(1-p_b)/n_b$。

**P1（uniform 的校准误差随桶数/技能数增长）**：在前沿区（$p\in[0.3,0.7]$），uniform 的 ECE 近似满足
$$\text{ECE}_{\text{uni}}\gtrsim c_1\sqrt{\frac{M_{\text{front}}}{B}}\cdot\bar p(1-\bar p),$$
其中 $M_{\text{front}}$ 是前沿桶数、$c_1$ 是常数。**可检验预测**：固定 $B$ 增大 $M$（桶粒度消融 A5）或 $S$（4→8），uniform 的前沿 ECE 上升；FG 的上升更慢（因为 FG 把样本集中到 $M_{\text{eff}}<M$ 个前沿桶）。**假设**：桶间独立、成功率估计近无偏；**局限**：不建模表示学习与接触动力学，对极端 p→0/1 的桶不适用。

**P2（FG 的收益随技能数增长）**：若各技能内部都有"已掌握/前沿/不可达"三段结构，则 uniform 的样本稀释随 $S$ 线性变差，而 FG 通过两级分配把技能间预算集中到 $\Delta p_g>0$ 的技能上；预测 $\Delta(\text{FG}-\text{uniform})$ 随 $S$（4→8）与 $N$（2048→8192）**单调增大**（对应 H1b/H4）。**可检验**：S×N 网格的交互项；若交互项不显著，P2 被证伪（这就是负结果分支 N1 的来源）。

**P3（校准误差到组合成功率的传播）**：设 agent 贪心选择 $\hat p$ 最大的技能执行，真实成功率为 $p$；若 $|\hat p-p|\le\epsilon$，则长程 $H$ 步任务的成功率损失上界约为 $O(\epsilon H)$（最坏情况；随机选择下界更差）。**可检验**：H3c 的"±10pp 噪声注入"实验——把 $\hat p$ 加噪后组合成功率应下降 ≥8pp；若加噪几乎无影响，说明 agent 没有真正使用校准信号，P3 与 H3 的机制叙事被证伪。

> 理论部分的定位是"**可检验的定量直觉**"，不是定理堆砌；正文只保留 P1–P3 各 2–3 句 + 一张 P→H 映射表，完整推导放附录（1 页）。

## 5. 实验设计：平台、矩阵、协议、基线与消融

### 5.1 平台与吞吐假设（全部标注"假设，W2–W8 pilot 校准"）

| 平台 | 角色 | 限制 | 用于 |
|---|---|---|---|
| 本地 RTX 4070 Ti Super 16GB / 32GB RAM / Ryzen 7 7800X3D | 免费；开发、小 N、消融、校准/agent 评测 | 16GB 显存限制 N（LEAP 手内任务估 N≤1024；arm+LEAP 接触任务估 N≤512，待 W8 实测） | Tier-0、Tier-1、agent 评测 |
| 租 A100 80GB（假设 ¥7/h，区间 6–8） | 主 scaling、大 N、P2P-style 基线 | 需批量租用、注意显存/时长 | Tier-2、Tier-3 附录 |
| （可选）A6000 48GB（¥3–3.5/h）/ 4090 24GB（¥1.6–2.5/h） | 若 A100 涨价/排队 | 显存/带宽低于 A100 | 备选，仅当 A100 不可得 |

**吞吐假设（单卡，state-based，MJX）**：LEAP 手内任务 N=8192 约 20k steps/s，N=2048 约 18k；arm+LEAP 接触任务（S7–S8）约 12k；混合库加权约 **15k steps/s @ A100**；Aero 约 25k；本地 4070 Ti Super 在 N=1024 约 6–8k。**这些数字是假设**，W2–W4 用现有 env 实测后更新，所有 GPU 小时按实测比例重算（§7 的预算表给的是"假设吞吐"下的上限估计，实测更快则预算更省）。

**W8 Go/No-Go（硬件可行性）**：在本地 4070 Ti Super 上实测 (a) LEAP reorient 在 N∈{256,512,1024} 的显存与吞吐；(b) arm+LEAP 插入在 N∈{128,256} 的稳定性与 1 mm 可达性；(c) 共享策略 8 技能联合训练 1M 步不崩。三项任一失败 → 触发 §9 的 R2/R3 预案（降级 N 上限或改用方案 C）。

### 5.2 三层实验矩阵（本地 → 租卡大 N → 跨形态附录）

| 层 | 平台 | 技能数 S | N | 方法 | seeds | 预算 | 目的 | 成本 |
|---|---|---|---|---|---|---|---|---|
| Tier-0 | 本地 | 1（Aloha peg，现成 env） | 256/1024 | uniform, SGS-band, FG | 3 | B=30M | 验证 FG 分配在已知 1 mm 装配上不崩（sanity） | 免费 |
| Tier-1 | 本地 | 8 | 256/1024 | uniform, SGS-band, FG | 2 | fixed-K（K=1221） | 小 N 趋势、桶粒度、guard 调试 | 免费（≈30 GPU-h） |
| **Tier-2（主）** | A100 | **8** | **2048 / 8192** | **4 方法（uniform/SER/SGS-band/FG）**；OmniReset/P2P 单独 | **5（N=8192）/ 8（N=2048）** | **fixed-K 主（K=1221）；fixed-B 次（B=112.5M）** | 主 scaling 曲线、主终点 H1 | 租卡（§7） |
| Tier-3（附录） | A100/本地 | 1（Aero） | 2048 | uniform, FG-transfer, scratch | 2 | B=60M | 跨形态迁移探测（H5） | 租卡 ¥39 |
| 校准/agent | 本地 | 8 | eval | 3 planners + oracle + random | 评测 seeds | 300 rollout/技能 | H2/H3/H6/H7 | 免费 + LLM API |

主网格（Tier-2）：**S=8, N∈{2048, 8192}, 方法∈{uniform, SER, SGS-band, FG}, seeds：N=8192 用 5、N=2048 用 8（成本与 Brief 的 8-seed 要求折中）**；OmniReset-style 与 Play2Perfect-style 作为 G4/G6 单独跑（见 §7.2）。技能数扩展（S=4→8）用 G3 单独跑（3 方法×3 seeds×S=4）检验 H4；OmniReset-style 只在 S=8/N=8192 跑（3 seeds），P2P-style 3 seeds（两阶段）。

### 5.3 训练协议：fixed-K 主 + fixed-B 次

- **fixed-K（主协议）**：固定 PPO 更新次数 $K=1221$，总步数 $B=K\cdot N\cdot L$ 随 N 增长（N=2048→112.5M，N=8192→450M，$L=50$）。理由：大规模并行下"N 增大 → 每 env 更新变少"是常见伪影（N=8192 时 fixed-B=112.5M 只有 275 次更新），fixed-K 让所有 N 的更新次数相同，测的是"每次更新用多少并行数据"的收益。主终点用 fixed-K 主格。
- **fixed-B（次协议/部署区制）**：固定 $B=112.5M$（所有 N 相同），N∈{2048,8192}×2 方法（uniform/FG）×3 seeds = 12 runs；回答"如果总交互预算固定，增大 N 是否还有意义"（部署区制的现实问题）。
- **$B$ 的 pilot 校准（不拍脑袋）**：W8 用 uniform 在单技能（S7 臂式插入，最难）上跑 $B\in\{25M,50M,100M\}$，取"成功率首次达到平台 80% 的步数"作为单技能最低预算；若该值 >62.5M（库级均分值），则按比例提高主格 $B$ 并触发 §7.4 的削减（优先砍 G3）。校准结果写进 W8 预注册文档。
- **seeds**：主网格 N=8192 用 5 seeds（seed 0–4）、N=2048 用 8 seeds（seed 0–7，Brief 的 seeds 要求落在这一格）；消融 2 seeds；P2P-style/OmniReset-style 3 seeds；Tier-0/1 3 seeds。每个 seed 用独立的桶采样序列与网络初始化。
- **计算量匹配**：所有方法在同一 (S,N) 下用相同的 $B$/$K$、相同环境步数、相同 PPO epochs/minibatch。P2P-style 两阶段按**等总步数版**匹配：play 300M + finetune 8×37.5M=300M（合计 600M，与 N=8192 主格的 450M 同量级）；完整版（play 600M + finetune 480M）只作上限参考。§7 的 G6 按**完整版**计价（保守），等预算版更便宜。

### 5.4 基线（6 条主 + 2 条参照）

| # | 基线 | 实现要点 | 出处 | 保真度声明 |
|---|---|---|---|---|
| 1 | **uniform** | 桶均匀采样，其余同 FG | 标准做法 | — |
| 2 | **SER** | staggered resets（环境错峰 reset），桶均匀 | 2511.21011 | 按其描述实现，不重训其网络 |
| 3 | **SGS-band** | 按成功率三段带加权（低/中/高），无 LP 导数 | 2610.12465 | **重实现**；先过 G-S1 保真判据（≥1 设置上 SGS>uniform ≥3pp），否则排查/联系作者 |
| 4 | **OmniReset-style** | 程序化多样 reset（多物体/多初始条件），单一奖励、无课程 | 2603.15789 | 风格化重实现（非官方代码）；用同一 reward |
| 5 | **Play2Perfect-style** | Stage-1 play 预训练（随机物体/目标）→ Stage-2 每技能 finetune | 2606.26428 | **scaled-down 重实现**（开源 LEAP 手，非 Sharpa/KUKA）；等预算版为主 |
| 6 | **FG（ours）** | §3 全量 | 本方案 | — |
| R1 | oracle-LP | 用"真实 LP"（评测集上每桶成功率变化）分配，作上界 | 分析工具 | 仅 S=8/N=8192、1 seed |
| R2 | random-skills | agent 随机选技能（下界） | 分析工具 | agent 评测 |

**SGS 保真度判据 G-S1（预注册）**：在 Tier-0 Aloha peg 与 Tier-1 LEAP reorient 两个设置上，若 SGS-band 重实现在 ≥1 个设置上不能复现"SGS>uniform ≥3pp"，则视为实现失败 → W20 前排查（对齐其 reset 权重、success band 边界、warmup），仍失败则改引官方代码（若已发布）；论文中报告"我们尽最大努力复现"并给代码链接。

### 5.5 消融（10 个）

| ID | 消融 | 检验什么 |
|---|---|---|
| A1 | 去导数（fast=slow，退化为成功率单调映射） | LP 的"学习进度"是否必要（与 SGS-like 对照） |
| A2 | 去技能内归一化（桶 LP 直接比较） | 跨技能不可比是否是关键 |
| A3 | 单级分配（跨技能 z-score，无两级） | 两级结构是否必要 |
| A4 | 去不确定性（λ=0，权重只用 LP） | 估计量不确定性是否有用 |
| A5 | 桶粒度 M_g∈{2,8} | 桶粒度敏感性（与 A3/H7 呼应） |
| A6 | EXP3 替代 Hedge | bandit 算法选择 |
| A7 | warm-up：前 100 更新纯 uniform | 冷启动的影响 |
| A8 | 去 guard（G-1/G-2/G-3 全关） | guard 是否防住塌缩/noisy-TV |
| A9 | 后处理校准（isotonic on/off） | 校准是否只靠后处理（H2 稳健性） |
| A10 | 每技能独立策略（不共享） | 共享 trunk + 分配的增益（H4/RQ1） |

所有消融在 S=8, N=2048, 2 seeds, B=112.5M 下跑（与主格固定的 (S,N) 对齐，只改一个部件）；A5/A3 在 N=8192 补 1 seed 验证趋势。

### 5.6 统计规范（预注册）

- **主终点**：H1（fixed-K 主格，FG−uniform ≥5pp）。用**配对**分析：同 seed、同桶配置序列、同评测集下比较；报告 95% CI（bootstrap 10k 次，按 seed 重采样，保留技能内配对结构）。
- **多终点校正**：主终点单独报告（不校正）；次终点（H1b/H1c/H2/H2c/H3/H4）用 Holm–Bonferroni，family 内 α=0.05；探索性假设（H2b/H3b/H3c/H5/H6/H7/H8）只报告效应量与 CI，不做显著性声明。
- **功效分析**：主终点 N=2048 格 8 seeds × 8 技能 = 64 个配对单元（N=8192 格 5 seeds × 8 技能 = 40）；假设 run 间 sd=6pp、评测测量噪声≈1pp（300 rollout），配对差的标准误≈1.2pp → 检测 5pp 的 power>0.95（单侧）；若真实 sd 达 10pp，power≈0.75，因此**同时报告最小可检测效应（MDE）**并在 W30 前计算（若 MDE>5pp 则追加 seeds，预算允许时）。
- **等价检验**：H2c 的"FG vs SGS 在低 S/低 N 下等价"用 TOST，等价界 Δ=2.5pp。
- **校准指标**：ECE 用 bootstrap CI；比较 FG vs uniform 的 ECE 用配对 bootstrap（按技能配对）；Brier 同理。
- **agent 评测**：同一组初始条件（每任务 30 个固定初始状态）下配对比较 3 个 planner，McNemar 精确检验；效应量报告风险差与 OR；失败归因由规则+人工双人标注（一致性 ≥85% 才用于 H6）。

### 5.7 机制指标与鲁棒性（H8）

除成功率外，记录：每技能成功率的训练轨迹、桶分配熵（时间序列）、LP 与 $\hat\sigma$ 的分布、guard 触发率、每桶 episode 数、梯度范数、 critic loss、以及 $|p̂-p_{\text{true}}|$ 校准图（$\hat\sigma$ vs 实际误差）。**鲁棒性网格（H8）**：观测噪声（状态高斯噪声 σ∈{0.01,0.02}）、物理随机化（质量/摩擦 ±20%）、目标分布偏移（评测目标区域外移 20%）；在 2 个技能上测 FG vs uniform/SGS 的成功率排序是否保持；若排名不变且成功率下降 <5pp 则 H8 成立。


## 6. 校准与 Agent：把训练信号变成可调用的技能库

### 6.1 校准协议（RQ2 的测量装置）

对每个技能 $s$ 与桶 $g$，训练结束后（以及每 10% 训练进度）用固定评测集估计成功率：

- **点估计**：$\hat p_{s,g}$ 用 300 次确定性 rollout 的 Wilson 区间；技能级 $\hat p_s$ 用桶按评测分布加权的平均（权重为评测时桶出现频率，不是训练分配频率——这是与"训练成功率"的关键区别）。
- **校准指标**：ECE（15 等宽箱 + 等频箱两版）、Brier、NLL；可靠性图与对角线偏差。报告 per-skill 与 pooled 两种。
- **分布偏移校准**：在 $p_s$ 上做一次"目标分布偏移"重加权（评测桶频率 ×2 某桶）后重算 ECE，检验校准是否只在训练分布内成立（H8 的一部分）。
- **消融对照**：A9（后处理 isotonic）回答"校准是否只是后处理带来的"；若 FG 的 ECE 优势在关闭 isotonic 后消失，则 H2a 降级为"分配改善训练分布覆盖，但校准需后处理"。

### 6.2 技能 API（LLM 可调用函数的具体形态）

每个技能导出为 JSON Schema 工具（python 函数签名 + 自然语言 docstring + 前置/后置条件 + 校准成功率）：

```json
{
  "name": "reorient_any",
  "description": "In-hand reorientation of a cube to a target yaw/pitch; wrist camera only.",
  "parameters": {"type": "object", "properties": {
      "target_yaw_deg": {"type": "number", "minimum": -180, "maximum": 180},
      "target_pitch_deg": {"type": "number", "minimum": -45, "maximum": 45}},
    "required": ["target_yaw_deg", "target_pitch_deg"]},
  "preconditions": ["object_in_hand", "arm_at_home"],
  "postconditions": ["object_grasped", "yaw_error_deg <= 10"],
  "calibrated_success": {"p": 0.86, "ci95": [0.81, 0.90], "n_eval": 300, "ece": 0.045},
  "cost_hint": {"steps": 50, "wall_s": 2.0}
}
```

**关键设计**：`calibrated_success` 由 §6.1 的评测流程写入，agent prompt 中明确说明"这是估计值，不是保证"；同时给出 `recovery_hint`（失败后可用技能，如 `regrasp`）。API 的机器可读性使"技能库"成为可复制交付物，而不是论文里的表格。

### 6.3 三类 planner（RQ3 的 2×2 设计）

| ID | Planner | 是否有校准信号 | 说明 |
|---|---|---|---|
| P1 | ReAct + 反思（Yao et al., 2210.03629） | 有 $\hat p$ | 每步 Thought→Action→Observation；失败后反思重试（≤2 次） |
| P2 | Plan-then-execute（Code as Policies 风格，2209.07753） | 有 $\hat p$ | 一次生成完整计划（JSON 调用序列），执行中仅做前置条件检查与重试 |
| P3 | P1 去掉 $\hat p$（prompt 中给名义"高成功率"） | 无 | 检验"校准信号本身是否被使用"（RQ3 的因果对照） |
| P0 | Random valid plan | 无 | 下界：从技能库随机采样可执行序列 |
| O1 | Oracle planner（用真实 $p_s$ 规划，仅仿真可得） | 真实值 | 上界：估计"好校准"最多能带来多少组合收益 |

主分析：P1(有 p̂) vs P3(无 p̂) 在 4 长程任务上的成功率差（H6）；P1 vs P2 检验 planner 类型是否改变结论（稳健性）；O1−P1 的差 = "校准误差造成的组合损失"（对应 P3 命题）。

### 6.4 组合评测协议（固定、可复现）

- **初始条件**：每长程任务 30 个固定初始状态（物体位姿、目标位置、桌面摩擦参数从训练分布内采样后**冻结**，写入 `eval/init_states.json`），所有方法共用同一组。
- **成功判定**：任务级二元成功（如 L2：peg 进入孔 2 mm 内且完成 2 圈拧紧）；技能级中间检查点单独记录，用于失败归因。
- **预算**：每 trial 最多 20 次技能调用、2 次重试/技能、90 s 墙钟；超时算失败。
- **LLM 固定**：温度 0、同一 system prompt、同一模型版本（型号与版本号在 W20 锁定并写入附录 D）；API 调用缓存到本地（相同 prompt 复用），记录 token 数与费用。
- **指标**：任务成功率、平均调用数、token 成本、失败类型分布（选错技能/参数错误/前置条件违反/执行失败/无法恢复），以及"计划长度 vs 成功率"曲线。
- **公平性**：三种 planner 的 prompt 长度尽量匹配（P3 用占位文本替换 p̂ 字段），避免"更多 token=更好"的混淆。

### 6.5 失败归因（H6 的关键证据）

对每个失败 trial，标注首个致因：①**分配型**（技能成功率被低估/高估导致选错技能或错误顺序）；②**执行型**（技能本身失败）；③**恢复型**（失败后未选择有效恢复技能）。两名标注者独立标注 120 条轨迹，Cohen's κ≥0.8 才用于主分析；分歧由第三人裁决。H6 预期：FG 显著降低①与③，对②影响小（技能本身成功率不变）。

### 6.6 LLM 成本控制（单列，不进租卡预算）

4 任务 × 30 trials × 3 planners = 360 episodes；每 episode 平均 8 次 LLM 调用、每次约 2.5k tokens → 7.2M tokens；加消融/调试 30% 冗余。按 2026 年主流闭源 API 价格区间（假设：$2–$5/M input tokens, $8–$15/M output tokens）估算 **≤¥400**；本文预算取 **¥250**（其余用缓存与更短 prompt 压低）。所有 API 调用记录 `logs/llm/*.jsonl`（模型版本、token、费用、latency）。

---

## 7. 预算：逐行可审计的租卡账本（硬上限 ¥5,000）

### 7.1 单价与吞吐假设（全部标注为假设，W2–W8 pilot 校准）

| 项 | 假设值 | 依据/校准方式 |
|---|---|---|
| A100 80GB 租卡单价 | **¥7.0/GPU-h**（区间 ¥6–8） | 2026 年国内平台常见区间；以实际账单为准，不写成官方定价 |
| 本地 4070 Ti Super | ¥0（电费计入杂项 ¥220） | 自有设备；同时最多 1 个训练进程 + 1 个评测 |
| N=8192 训练吞吐 | **15,000 env-steps/s**（混合技能，含 S7/S8 接触任务更低） | 假设；W2–W8 用 3 个技能实测（目标 ≥12k），低于 10k 则按比例重算 |
| N=2048 训练吞吐 | **6,000 env-steps/s**（显存受限，batch 更小） | 假设；本地 4070 Ti Super 实测外推 |
| N≤1024 本地吞吐 | 4,000–6,000 steps/s | 假设；本地 pilot 实测 |
| 单 run 额外开销 | +0.25 h（启动/评测/保存） | 经验假设，pilot 校准 |
| 训练步数 B（fixed-K） | K=1,221 更新，B=K·N·L，L=50 → N=8192 时 B=500M；N=2048 时 B=125M | 与 Play2Perfect 的 finetune 量级（12,288 envs/1 天）对齐；K 由 pilot 校准 |
| 12% 重跑缓冲 | 覆盖崩溃/OOM/调参重跑 | 经验值 |

### 7.2 实验矩阵 → 每 run GPU 小时 → 总 GPU 小时 → 单价 → 总费用（租卡 A100）

| 组 | 内容（S=技能数，N=并行 env） | runs | h/run | GPU-h | ¥ |
|---|---|---|---|---|---|
| P0 | 工程/pilot/吞吐校准（含 W8 Go/No-Go） | 1 | 24.0 | 24.0 | 168 |
| G1 | 主网格 S=8, N=8192，4 方法（uniform/SER/SGS/FG）×5 seeds | 20 | 8.58 | 171.6 | 1,201 |
| G2 | 主网格 S=8, N=2048，4 方法×**8 seeds**（满足 Brief 的 seeds 要求；N=8192 因成本用 5） | 32 | 2.33 | 74.6 | 522 |
| G3 | 技能数扩展 S=4, N=8192，3 方法（uniform/SGS/FG）×3 seeds | 9 | 8.58 | 77.2 | 541 |
| G4 | OmniReset-style（reset 多样性）S=8,N=8192，3 seeds | 3 | 8.58 | 25.7 | 180 |
| G5 | fixed-B 部署区制（N=2048/8192 × uniform/FG × 3 seeds） | 12 | 2.33 | 28.0 | 196 |
| G6 | Play2Perfect-style 两阶段（play+finetune，缩小版）×3 seeds | 3 | 17.2 | 51.6 | 361 |
| G7 | from-scratch 对照（同 B 不用 play 阶段）×3 seeds | 3 | 9.2 | 27.6 | 193 |
| G8 | 消融 A1–A10 ×2 seeds（N=2048, B=112.5M） | 20 | 2.33 | 46.6 | 326 |
| G9 | Aero 跨形态附录（3 方法×2 seeds） | 6 | 0.92 | 5.5 | 39 |
| G10 | 校准/agent 评测（租卡部分，含 O1 oracle 评测） | 1 | 10.0 | 10.0 | 70 |
| G11 | SGS/SER 重实现调试与对齐 | 1 | 6.0 | 6.0 | 42 |
| **小计** | | **111 runs** | | **548.4** | **3,839** |
| 重跑缓冲 | 12% | | | 65.8 | 460 |
| 杂项 | 电费 ¥220 + 存储 ¥80 + **LLM API ¥250（单列）** | | | | 550 |
| **合计** | | | | **614.2** | **¥4,849** |

**本地免费部分（不计租卡，但计入 GPU-h 账本）**：T0 Aloha peg sanity 19 h；T1 本地 scaling（S=8, N∈{256,1024}, 3 方法×2 seeds，B=N·K·L）约 30 h；工程开发/调试 40 h；校准与 agent 评测（本地小模型/缓存）20 h；鲁棒性测试 10 h → **≈119 GPU-h，¥0**（电费已在杂项）。全部 111 个租卡 run 与本地 run 的配置、日志与 eval 脚本随仓库提交（附录 C 给出预算脚本）。

### 7.3 敏感性分析（±50% 单价 / +50% 时长）

| 情景 | 计算 | 合计 | 结论 |
|---|---|---|---|
| 基准 | 548.4 h × ¥7 = ¥3,839 + 12% ¥460 + ¥550 | **¥4,849** | 余 ¥151 |
| 单价 −50%（¥3.5/h） | 548.4 h × ¥3.5 = ¥1,919 + 缓冲 ¥215 + 550 | ¥2,700 | ✓ |
| 单价 +50%（¥10.5/h） | 548.4 h × ¥10.5 = ¥5,758 + 缓冲 691 + 550 | **¥6,999** | 超 ¥1,999 → 触发削减 |
| 时长 +50%（吞吐低 1/3） | 822.5 h × ¥7 = ¥5,758 + 缓冲 691 + 550 | **¥6,999** | 同上 |
| 双因素（单价+时长各 +50%） | 822.5 h × ¥10.5 = ¥8,636 + 1,036 + 550 | **¥10,222** | 需极限削减 |

**削减顺序（预注册，按论文伤害从小到大）**：

| 级别 | 动作 | 省下（单价+50% 情景） | 累计 |
|---|---|---|---|
| C1 | 砍 G3（S=4 扩展，只保留 S=8 主格与附录讨论） | −¥811 | ¥6,188 |
| C2 | G6 Play2Perfect-style 3→1 seed（保留 baseline，降精度） | −¥361 | ¥5,827 |
| C3 | 砍 G5 fixed-B 部署区制（协议 A 只作附录理论讨论） | −¥294 | ¥5,533 |
| C4 | N=8192 主格 seeds 5→4（H1 power 从 >0.95 降至 ≈0.85，MDE 升到 6pp） | −¥517 | ¥5,016 |
| C5 | G8 消融 10→8 个变体 | −¥98 | **¥4,918 ✓** |

双因素最坏情景：在 C1–C5 后再降 seeds 至 3、消融 8→4、主格只留 uniform/SGS/FG 三方法、G4 仅 1 seed，得 **≈¥4,987**（仍 ≤5,000，但论文降级为"单技能库 + 3 方法 + E1/E2 证据集"）。**FLOOR 包**（只保 H1 最小证据）：G1 3 方法×3 seeds + G2 + 4 消融×2 seeds + P0 6h + 评测 4h = 126.8 h → ¥888 + 缓冲 ¥107 + 杂项 ¥350 = **¥1,345**。

---

## 8. 45 周排期（CoRL 2027 主投口径：W26 冻结、W31 投稿）

| 阶段 | 周 | 里程碑 / Go-No-Go |
|---|---|---|
| 环境与技能 MVP | W1–W6 | 复用 reorient/rotate_z/peg-insert 三个环境；新建 S3/S4；arm+LEAP 集成 W3–W6 |
| 技能库完成 | W5–W10 | S5/S6/S8 + bucket 注入（W3 前预留 1 周工程）；W8 **Go/No-Go G1**：本地 N 上限 ≥1024 且 4 技能可训（uniform 达到 >50% 成功率） |
| Tier-1/T0 本地 scaling | W9–W14 | 本地 N∈{256,1024} 曲线；T0 Aloha peg sanity；W12 校准头 + agent harness 雏形 |
| 主实验第一波 | W14–W19 | G1/G2/G3/G4（S=8、N 扫描、4 方法；N=8192 5 seeds / N=2048 8 seeds）；W16 **主曲线预跑检查** |
| 主实验第二波 | W17–W22 | G5/G6/G7/G8/G9；校准评测；W20 **Go/No-Go G2**：若 agent 2×2 无法在 W26 前完成 → agent 降为验证，主贡献按"分配→校准"收 |
| 长程与 agent | W21–W25 | L1–L4 组合评测（P1/P2/P3/P0/O1）；失败归因；鲁棒性 |
| 确认与补跑 | W25–W26 | 只补 seed/重跑，不加新方法；**W26 实验冻结**（freeze commit + 数据锁） |
| 写作与开源 | W26–W30 | 8 页 CoRL 稿、视频 90 s、代码/技能库/评测 harness 开源、内审 2 轮 |
| 投稿 | W31 | 提交 CoRL 2027（≈2027-06 上旬，**日期待核实**） |
| 兜底 | W31–W45 | CoRL 审稿期并行整理 ICLR 2028 版（扩充 9 页、补真实噪声/更多 seeds）；若 CoRL 被拒则按审稿意见修订后于 W46–W47 投 ICLR 2028 |

**人力核算（1 人）**：工程 ≈20 周（§3.9）、实验执行与监控 ≈12 周（租卡可按 4×A100 并行，墙钟压缩）、分析/写作/视频/开源 ≈10 周、缓冲 3 周 → 合计 **45 周**。若 W8 判定本地 N<1024 或技能训练不稳定，触发止血：砍到 4 技能 + S=4 的"方法论文"，目标改投 ICLR 2028（W46–47），仍保留"分配→校准"主贡献。

## 9. 风险登记表与 Plan B

| ID | 风险 | 概率 | 影响 | 缓解/Plan B | 触发点 |
|---|---|---|---|---|---|
| R1 | arm+LEAP 一体化 MJCF 在 MJX+Warp 下集成失败（接触/驱动不兼容） | 中 | 高 | 退方案 B（双臂两 LEAP 手做 handover/双手装配）或 C（LEAP 手内 5 + Aloha 臂夹爪 3，作 benchmark 两 track）；诚实写"跨形态"，不假装统一 | W6 集成自检 |
| R2 | 1–3 mm 插入在 MJX 中接触不稳定（穿透/抖动） | 中 | 高 | 提高 substep/`naconmax`、容差放宽到 2 mm、S7/S8 用 MuJoCo CPU 后端（慢 5–10×，但预算充足）；必要时把"亚毫米"降为"2 mm 级"并在论文中如实说明 | W10 S7 首次成功率 <10% |
| R3 | 本地 4070 Ti Super 在 N=1024 时不可行（显存/吞吐） | 中 | 中 | 本地上限降 512；把 N=1024 挪到租卡（预算 +¥300，仍在余量内）；W8 Go/No-Go 直接测 | W8 |
| R4 | N=8192 时 PPO 训练不稳定（KL 爆炸/奖励塌缩） | 中高 | 高 | 降 N 到 4096 为主格（结论改为"至 4096"）；调 UTD/学习率/奖励缩放；分批 warm-up | W14 主曲线第一期 |
| R5 | bucket 注入需改 ManiSkill3 env，成本超 1 周或语义不对 | 中 | 中 | 退化为"纯 DR 桶"（每个 episode 随机采 DR 参数并分桶，不需改 env）：仍可测 H1/H2，但 bucket 的"学习进度"语义变弱，需在论文中说明 | W4 集成测试 |
| R6 | LP 信号在技能库级噪声过大（跨技能归一化失效） | 中 | 高 | 三重 guard + 更长 EMA 窗；退化为 SGS-like 单调映射（A1），论文改为"何时 LP 不必要"的负结果 | W16 预跑 |
| R7 | 仿真校准与"真实可用性"脱节（sim-only 局限） | 高 | 中 | 不声称真机效果；把校准定义为"仿真内可靠性"，在 limitation 中明确；用 Play2Perfect/Playground 的 sim2real 证据讨论外推边界 | 写作期 |
| R8 | LLM agent 无法有效利用 $\hat p$（prompt 不敏感） | 中 | 中 | 改为 P2 plan-then-execute 显式规划（把 p̂ 写进计划打分）；agent 降为验证（H6 改为"描述性证据"） | W20 中检 |
| R9 | LLM API 费用/延迟超预期 | 中 | 低 | 缓存、缩短 prompt、换开源权重大模型（记录版本）；预算上限 ¥400 硬顶 | W21 |
| R10 | 租卡实际价格高于 ¥7/h 或吞吐低于假设 | 中 | 高 | 触发 §7.3 削减 C1–C4；FLOOR 包保 H1 | 每月对账 |
| R11 | 45 周排期滑（工程/集成超期） | 中 | 高 | W8/W20 两级 Go/No-Go；W20 后冻结范围；写作期与实验期重叠 1 周 | 每周 |
| R12 | 与 SGS/OmniReset/Play2Perfect 的 novelty 重叠被审稿人质疑 | 高 | 高 | §10 差异表 + 明确"我们测的是什么他们没测"（校准→agent 组合因果链、fixed-K、技能库级分配）；把 OmniReset 作为 baseline 而非对手 | 写作期 |
| R13 | 单人项目的中断风险（疾病/设备） | 低 | 高 | 3 周缓冲 + 所有实验脚本化/可断点续跑；数据在 W26 冻结后不再依赖本地机器 | 全程 |
| R14 | 仿真任务"玩具化"（技能太简单，审稿人认为无挑战） | 中 | 中 | 技能含 1–3 mm 容差、长程多阶段、扰动/偏移评测；报告失败案例视频；与 Aloha peg-insertion 公开任务对齐 | 写作期 |
| R15 | 资产/代码许可证限制开源 | 低 | 中 | 只开源自写代码与配置，第三方 mesh 用脚本下载 + 引用原许可；技能/评测 harness 独立可跑 | W27 |

## 10. 与近邻工作的差异（为什么这不是"又一个采样器/又一个基准"）

| 工作 | 平台/形态 | 规模 | 分配/课程信号 | 校准 | LLM agent 组合 | 与本文的关键差异 |
|---|---|---|---|---|---|---|
| SGS (2610.12465) | Isaac/ANYmal+Franka/UR5e | 2^20 envs | success 历史 buffer 的 success-band 加权 | 无 | 无 | 无技能库、无校准、无 agent；采样函数是 success 阈值带；本文做技能/桶两级 LP×不确定性，并测校准→组合 |
| OmniReset (2603.15789) | 多机器人重置 | 大规模 | reset 多样性（非学习进度） | 无 | 无 | 只改 reset 分布，不分配训练预算；本文把它作为 baseline |
| Play2Perfect (2606.26428) | Isaac Sim，22-DoF Sharpa | 24,576 envs | SAPG（play→finetune 两阶段） | 无 | 无 | 单任务族 + 两阶段训练；本文做多技能库 + 连续分配 + 校准，且对比其缩小版 |
| ADEPT (2608.19182) | 高 DoF 灵巧，visuo-tactile | 大规模 | RL 预训练+后训练 | 无 | 无 | 工程规模大；本文不拼算力，拼"分配→校准→组合"的受控测量 |
| SER (2511.21011) | 并行 env 重置错峰 | 大规模 | 重置调度（非分配） | 无 | 无 | 正交：本文把 SER 作为 baseline 之一（SER+FG 组合在附录） |
| Being-0 (2503.12533) | 人形+灵巧手 | — | 分层 VLA + 模块化技能 | 部分 | 有（LLM 规划） | 真机系统论文；本文纯仿真、可复现、有预注册统计，且不含 VLA 训练 |
| DexSim2Real (2605.05241) | 灵巧手仿真→真机 | — | 域随机化+系统辨识 | 无 | 无 | 关注迁移；本文关注训练分配与校准 |
| Text2Touch (2509.07445) | 触觉+语言 | — | 语言条件 | 无 | 部分 | 触觉模态；本文纯视觉/状态，不依赖触觉硬件 |
| The Price Is Not Right (2602.19260) | sim-only 基准 | — | 无 | 无 | 无 | 说明 sim-only 基准可发表；本文提供技能库级分配+校准基准，非单纯评测 |
| ReAct (2210.03629) / Code as Policies (2209.07753) | LLM agent | — | 无 | 无 | 是 | 本文把"校准成功率"喂给这两类 planner，检验其是否被使用 |

**一句话定位**：SGS/OmniReset/Play2Perfect 分别在"采样信号""重置分布""两阶段训练"上推进大规模 RL；**没有人把"技能库级数据分配 → per-skill 可靠性校准 → LLM agent 组合成功率"作为一条可测因果链**，并用 fixed-K、≥5 seeds、预注册与开源 harness 固定下来。本文的贡献不是更大的系统，而是**这条链的测量方法与首个受控证据**。

## 11. 论文结构、图表清单、视频与开源

### 11.1 8 页 CoRL 结构（页数为估计）

| 节 | 页 | 内容 |
|---|---|---|
| Abstract + Intro | 1.25 | 问题：技能库级训练的资源分配与可靠性校准；三条贡献；Fig.1 因果链 |
| Related Work | 0.5 | 大规模 RL 采样（SGS/SER/OmniReset/P2P）、灵巧技能库、LLM 组合（ReAct/CaP） |
| Method | 2.5 | 两级分配（§3.2）、σ̂ 估计（§3.3）、校准（§3.4）、guard（§3.5）、集成（§3.7）；伪代码 |
| System & Benchmark | 1.5 | 8 技能表、4 长程任务、API schema、评测协议、开源工件 |
| Experiments | 2.0 | RQ1 scaling（Fig.3）、RQ2 校准（Fig.4）、RQ3 agent（Fig.5）、RQ4 附录、消融/鲁棒性（Fig.7） |
| Discussion & Limitations | 0.25 | sim-only、无真机、单形态、LLM 依赖 |

### 11.2 图表清单（每图对应 RQ）

| 编号 | 内容 | 对应 RQ/假设 | 数据来源 |
|---|---|---|---|
| Fig.1 | 因果链示意：分配 → 每技能样本量/覆盖 → 校准误差 → agent 组合成功率 | 全文 | 概念图 |
| Fig.2 | 方法总览：两级分配 + 校准头 + API 层 | 方法 | — |
| Fig.3 | fixed-K 主曲线：成功率 vs N（N∈{2048,8192}）与 vs S；uniform/SER/SGS/FG | RQ1/H1/H2 | G1–G3 |
| Fig.4 | 校准：可靠性图 + ECE vs N + σ̂ 校准 | RQ2/H3 | G1–G3 评测 |
| Fig.5 | agent 组合：P1 vs P3 vs P0 vs O1 的成功率/成本；失败归因堆叠图 | RQ3/H6 | 组合评测 |
| Fig.6 | 长程任务成功率（L1–L4）与调用数 | RQ3/H6 | 组合评测 |
| Fig.7 | 消融（A1–A10）+ 鲁棒性网格 | H4/H5/H8 | G8 + 鲁棒性 |
| Table 1 | 8 技能定义、参数范围、成功判据、校准 p̂ | 方法 | — |
| Table 2 | 主结果：各方法 × N × 指标（成功率/ECE/Brier） | RQ1/RQ2 | G1–G3 |
| Table 3 | agent 结果与失败归因 | RQ3 | 组合评测 |
| Table 4 | 成本/算力表 | 可复现性 | 附录 C |

### 11.3 视频脚本（90 s）

| 秒 | 画面 | 旁白要点 |
|---|---|---|
| 0–10 | 灵巧手在桌面翻转/插入/拧紧的快剪 | 大规模 RL 已能让手做到这些；问题是"技能库怎么训练、怎么被调用" |
| 10–25 | 8 技能分屏 + 每技能 p̂ 与 CI | 每个技能是一个带校准成功率的可调用函数 |
| 25–45 | 训练分配动画：桶的样本流随 LP/σ̂ 变化 | FG 把采样预算给"正在进步且不确定"的桶 |
| 45–60 | fixed-K 曲线：uniform 饱和 vs FG 继续上升 | 同样的更新预算，FG 在更大 N 上继续改善 |
| 60–75 | 可靠性图：FG 更贴对角线 | 校准更好 → agent 的规划更可靠 |
| 75–90 | LLM agent 组合：失败→换技能→成功，字幕给成功率/成本 | 技能库被 agent 真正用起来；代码/基准开源 |

### 11.4 开源计划（W27–W30 打包，投稿时匿名链接，录用后公开）

仓库 `dexskillbench`（MIT）：`envs/`（8 技能 + bucket 注入）、`algo/`（PPO + FG 钩子，约 150 行）、`calib/`（评测与校准）、`agent/`（3 planner + API schema + traces）、`configs/`（全部 run 的 yaml）、`scripts/`（预算/统计/绘图）、`eval/init_states.json`、`docs/`（复现指南、API 文档、失败案例）。发布物：技能库 + 4 长程任务 + 固定初始条件 + 预注册文件 + 全部日志（匿名化）。**不发布**第三方 mesh，只给下载脚本与许可证引用。

## 12. 中稿率提升策略（针对 sim-only 与"系统不够大"的弱点）

### 12.1 投稿路径与自评概率（主观判断，非官方数据）

| 路径 | 时点 | 自评概率（点估计/区间） | 依据与风险 |
|---|---|---|---|
| **CoRL 2027 主投** | W31（≈2027-06，**待核实**） | **40%（30–50%）** | 机器人学习主会；8 页容纳全链；风险：scope 被评"太宽"、sim-only、LLM 依赖、与 SGS/OmniReset 重叠质疑 |
| CoRL 被拒 → **ICLR 2028** | W46–W47（≈2027-09 下旬，**待核实**） | 条件概率 **35–40%** | ML 顶会对"分配→校准→组合"的实证与统计严谨度友好；可用 CoRL 意见改稿；风险：超出 45 周 1–2 周、需要扩 9 页 |
| **级联总概率**（CoRL 或 ICLR 至少中一篇） | 2028 年初 | **≈60%**（区间 45–70%） | 只在"CoRL 被拒后仍按审稿意见补实验"的前提下成立；**>50% 只出现在级联口径，单会口径不成立** |
| NeurIPS 2027 D&B（机会项） | W27（≈2027-05） | 15–25%（若 W20 判定 benchmark 成熟） | 不主动分兵；只投 benchmark track 的 MVP |

> 概率为作者基于"预注册 + 开源 + 视频 + 负结果"的主观判断，**不是录取率统计**；ICRA/RA-L 已排除（见 §0.1）。

### 12.2 针对评审维度的动作（每条都落到本方案的具体位置）

| 评审维度 | 常见质疑 | 本方案的应对 | 位置 |
|---|---|---|---|
| Novelty | "SGS/OmniReset 已经做了采样/reset" | 技能库级两级分配 + 校准信号 + 组合因果链；SGS/OmniReset 均作基线 | §10、§5.4 |
| Technical quality | "PPO 在大 N 不稳定、fixed-B 有更新饥饿" | fixed-K 主协议 + K-sweep pilot + UTD/guard 细节 | §5.3、§3.7 |
| Experimental rigor | "seeds 少、无预注册、CI 缺失" | 预注册 W8 冻结、5–8 seeds、TOST、Holm、MDE、负结果分支 | §2.6、§5.6 |
| Robotics relevance | "纯仿真、技能太玩具" | 1–3 mm 容差、长程多阶段、扰动评测、失败视频、真实任务对齐（Aloha peg） | §4、§11.3 |
| Clarity | "方法+系统+agent 三条线太散" | 全文只讲一条因果链；Fig.1 固定叙事；贡献三条 | §11.1 |
| Reproducibility | "复现不了" | 开源技能库/API/评测 harness/固定初始条件/预算脚本/日志 | §11.4、附录 C |
| sim-only 折价 | "没有真机" | 明确 limitation；引用 sim-only 基准先例（2602.19260）与 Play2Perfect 的 sim2real 证据；不声称真机效果 | §9 R7、§11.1 |
| LLM 依赖 | "prompt 一改就变" | 固定模型/温度、三 planner、P3 无校准对照、oracle 上界、全部 trace 开源 | §6.3–6.4 |

### 12.3 最小充分证据集（MSE）vs 加分实验（预算不足时先保前者）

| 优先级 | 内容 | 成本 | 支撑的结论 |
|---|---|---|---|
| **P0（必保）** | G1（N=8192 5 seeds）+ G2（N=2048 8 seeds）+ G8（8 消融）+ 校准评测 + P1/P3 agent + 本地 Tier-1 曲线 | ≈307.4 GPU-h/¥2,960（含缓冲与杂项） | H1/H2/H3/H6 主证据；没有它论文不成立 |
| P1 | G11 SGS/SER 重实现对齐 + G4 OmniReset baseline | +¥202（含缓冲） | baseline 公平性 |
| P2 | G3（S=4 扩展）、G6（P2P-style）、G7（scratch） | +¥1,226（含缓冲） | H4 与"两阶段"对比 |
| P3 | G5（fixed-B 部署区制）、G9（Aero 附录）、鲁棒性网格 | +¥263（含缓冲） | 加分/附录 |
| P4 | 更多 seeds / 更大 N（若预算与时间允许） | — | 审稿人加分项 |

**结论**：基准 ¥4,849 覆盖 P0–P3；若触发削减（§7.3），按 C1–C5 先砍 P2/P3，保 P0/P1。任何情况下 **P0 的 ¥2,960 必须保住**（这也是 FLOOR 的底线逻辑）。

### 12.4 Rebuttal bank（预写 9 条，投稿前定稿）

1. **"SGS already samples by success band; what is genuinely new?"** → SGS 是单任务 success-band 加权；本文是技能库级（跨技能归一化）+ LP 导数×估计量不确定性 + 校准→组合的因果链，且 SGS 是我们的基线（G11 保真度判据）。
2. **"Sim-only, no real robot."** → 明确 sim-only limitation；基准类工作在 ML/CoRL 有先例（2602.19260）；Play2Perfect 已给出 sim→real 证据，本文贡献是训练分配与校准，不是迁移。
3. **"LLM results are prompt-sensitive."** → 固定模型版本/温度/提示、三种 planner、P3 无校准对照、oracle 上界、全部 trace 开源；主结论只在 P1 vs P3 的配对差上声明。
4. **"N=8192 is not 'massive' compared to 2^20."** → 算力受限（¥5,000）；固定 K 的 scaling 趋势 + 本地 N 曲线；论文只声明"至 8192"。
5. **"Calibration is just training success rate."** → 校准在 held-out 评测分布上测（非训练桶频率），含分布偏移重加权、ECE/Brier、σ̂ 校准图、isotonic 消融。
6. **"Skills are toy tasks."** → 1–3 mm 容差、长程多阶段、扰动/偏移评测、失败视频；技能定义与 Aloha peg-insertion 对齐。
7. **"Only 5 seeds at the largest N."** → N=2048 格 8 seeds；配对设计 + MDE 报告；削减情景明确降级声明。
8. **"Gains may come from exploration, not calibration."** → A1/A4 消融 + 机制指标（分配熵、每桶样本数、LP/σ̂ 分布）+ 失败归因（H6）。
9. **"Budget/claims overreach."** → FLOOR 与负结果分支预注册；论文明确"若 H1 不成立则报告 N1/N2/N3"。

## 13. 决策点、待核实清单与开放问题

### 13.1 需要用户/Lead 拍板的决策（D1–D10）

| ID | 决策 | 选项 | 建议 | 影响 |
|---|---|---|---|---|
| D1 | 技能数最终值 | 8（6 手内+2 臂式）/ 6（W8 止血）/ 10–14（Brief 原建议） | **8**；W8 不达标则 4 技能方法论文 | scope、预算、审稿期望 |
| D2 | 臂的资产 | Franka Panda / ViperX / 自建 | 优先 ViperX（轻、MJX 有 MJCF 先例），W4 验证 | 集成 2–3 周 |
| D3 | S7 容差 | 1 mm / 2 mm | 先 2 mm 跑通，W12 尝试 1 mm；论文写实际值 | 难度与稳定性 |
| D4 | S5/S6 的具体定义 | 圆柱重定向 / 边缘对齐 / 指尖推 | 按 §4.2 表；若工程量超，砍 S6 | 技能库完整性 |
| D5 | Agent 模型 | 闭源 API / 开源权重 | 主实验用闭源 API（固定版本），开源模型作附录对照 | 成本、可复现性 |
| D6 | LLM 预算上限 | ¥250 / ¥400 | ¥250 基准，¥400 硬顶 | 杂项 |
| D7 | W8 Go/No-Go 阈值 | 本地 N≥1024 且 4 技能 >50% | 照此执行 | 是否止血 |
| D8 | W20 agent 降级规则 | agent 2×2 跑不完→降为验证 | 照此执行 | 主贡献范围 |
| D9 | 是否保留 Aero 附录 | 保留 / 砍 | 保留（¥39，低风险） | RQ4 |
| D10 | 开源许可证 | MIT / Apache-2.0 | MIT（代码）+ 第三方资产引用原许可 | 开源合规 |

### 13.2 待核实清单（W1–W4 完成，全部标"待核实"）

- CoRL 2027 截稿/通知/会期；ICLR 2028 截稿（以 CFP 为准）。
- MuJoCo Playground 中 `LeapCubeReorient`/`LeapCubeRotateZAxis`/`AeroCubeRotateZAxis`/`AlohaSinglePegInsertion` 的**精确 env 注册名**与 `max_episode_steps`（W1 用 `gymnasium.registry` 核对，不得凭记忆写）。
- LEAP/Aero 的 MJCF 资产路径与执行器参数；arm+LEAP 组合的可行性（W3–W6）。
- ManiSkill3/Isaac Lab 仅作"平台对比"引用，不承诺实验。
- LLM API 的实际价格与模型版本（W20 锁定）。
- Play2Perfect 的 24,576 envs/7 天/A6000/33×/60%/0.5 mm 等数字已按 arXiv 2606.26428v4 摘要与 Brief 引用，写作时再核对正文。

### 13.3 开放问题（诚实地留给审稿人）

1. 技能库级 LP 的"公平性"：稀有但重要的技能是否会被 FG 长期忽略？本文用 guard G-1/G-3 缓解，但没有理论保证。
2. 校准的"真值"在仿真里是评测分布上的频率；真实部署中分布会漂移，本文只做一次重加权探测。
3. LLM agent 的成功率受规划能力上限约束；本文测的是"校准信号是否被使用"，不是"LLM 能否规划"。
4. 8 技能是否足以代表"大量精细操作"？这是 scope 的取舍，W8 止血方案与附录扩展是诚实回应。
5. sim-only 的结论能否外推到真机？本文不声称，只讨论边界。


---

## 附录 A：参考文献（全部为已核实 arXiv ID；未核实项明确标注）

**核心对比工作**
1. **Play2Perfect** — Lum T. G. W., Kedia K., Liu C. K., Bohg J. *Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?* arXiv:**2606.26428v4**（2026-06-24 首发，2026-10-15 更新）。22-DoF Sharpa 手 + KUKA iiwa；24,576 envs/7 天/单 A6000；finetune 12,288 envs/1 天；33× 样本效率；0.5 mm 插入 60%。
2. **OmniReset** — Yin et al. *Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning.* arXiv:**2603.15789v3**。
3. **SGS** — Zhang O., Guaman Castro M., Yin P., Dagnino I., Gupta A., Scalise R., Boots B. *A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control.* arXiv:**2610.12465v1**（CoRL 2026；代码 "coming soon"，写作时跟踪）。
4. **SER** — Bharthulwar S., Tao S., Su H. *Staggered Environment Resets Improve Massively Parallel On-Policy RL.* arXiv:**2511.21011**。
5. **Being-0** — Yuan et al. *Being-0: A Humanoid Robotic Agent with Vision-Language Models and Modular Skills.* arXiv:**2503.12533v2**。
6. **ADEPT** — Lee et al. *ADEPT: ...* arXiv:**2608.19182v2**（大规模 RL 预训练+后训练、高 DoF、visuo-tactile；judge 称已被 CoRL 2026 接收，**待官方确认**）。
7. **DexSim2Real** — Zeng et al. arXiv:**2605.05241v1**。
8. **2606.22471v2** — Li et al. 语言条件双臂灵巧 RL 数据生成。
9. **Text2Touch** — Field et al. arXiv:**2509.07445**（LLM 设计触觉奖励）。
10. **The Price Is Not Right** — Duggan et al. arXiv:**2602.19260**（sim-only 基准先例）。

**平台与算法**
11. **MuJoCo Playground** — Zakka et al. arXiv:**2502.08844**（RSS 2025；MJX；`pip install playground`）。
12. **ManiSkill3** — Tao et al. arXiv:**2410.00425**（仅作平台对比引用，不作实验平台）。
13. **Isaac Lab** — Mittal et al. arXiv:**2511.04831**（仅作平台对比引用）。
14. **SAPG** — Singla A., Agarwal S., Pathak D. *SAPG: Split and Aggregate Policy Gradients.* arXiv:**2407.20230**。
15. **LeRobot** — Cadene R. et al. arXiv:**2602.22818**（仅作生态引用）。
16. **Squint** — Almuzairee A., Christensen H. I. *Squint: Fast Visual RL for Sim-to-Real Robotics.* arXiv:**2602.21203**（RA-L 2026, doi:10.1109/LRA.2026.3730387；仅作生态引用）。
17. **SO-101 Bench** — Hickok T. arXiv:**2610.36420**（仅作生态引用）。

**LLM agent / 技能库**
18. **ReAct** — Yao et al. arXiv:**2210.03629**。
19. **Code as Policies** — Liang et al. arXiv:**2209.07753**。
20. **SayCan** — Ahn et al. arXiv:**2204.01691**。
21. **RoboCodeX** — arXiv:**2402.16117**（注：**不是** 2411.19309）。
22. **GRAPE** — arXiv:**2411.19309**（与 RoboCodeX 不同；仅背景引用）。

**采样/课程（方法谱系，写作时只作 related work）**
23. **PLR** — Jiang Y., Grefenstette E., Rocktäschel T. *Prioritized Level Replay.* arXiv:**2010.03934**。
24. **RED** — Jiang et al. *Replay-Guided Adversarial Environment Design.* arXiv:**2110.02439**（**不是** RPLR；RPLR 无已核实 arXiv ID，本文不引用）。
25. **PGB** — Xu L., Shen S., Hou Z., Bi C. *Predictive Bandit Curriculum via Gradient-Based Sample Group Discovery.* arXiv:**2610.25704**。
26. **DRATS** — Corrado N. E., Huang W., Hanna J. P. *Distributionally Robust MTRL via Adaptive Task Sampling.* arXiv:**2605.14350**（仅 related work）。
27. **UPER** — Carrasco-Davis R., Lee S., Clopath C., Dabney W. arXiv:**2506.09270**（仅 related work）。
28. **GMC** — Blad S., Längkvist M., Loutfi A. arXiv:**2605.05856**（仅 related work）。
29. **LPM** — Hou Z., An Z., Du W. arXiv:**2509.25438**（仅 related work）。

> 规则：正文引用的每个 arXiv ID 必须在附录出现；附录未列的不得在正文出现。RPLR 因无法核实 arXiv ID，**全文不引用**。

## 附录 B：Brief §5 十四项检查清单对照

| # | 要求 | 本方案位置 | 状态 |
|---|---|---|---|
| 1 | 英文题/摘要/关键词（≤250 词） | §1.1–1.3（摘要 213 词） | ✓ |
| 2 | RQ1–RQ4 + 可证伪 H1–H8（数值阈值） | §2.1–2.7 | ✓ |
| 3 | 方法形式化 + 伪代码（FG + 校准 + API） | §3.1–3.9（伪代码 §3.8） | ✓ |
| 4 | 技能库定义表（obs/action/goal/success/预算） | §4.2（8 技能）+ Table 1 | ✓ |
| 5 | 三层实验矩阵（本地→租卡→跨形态） | §5.2（Tier-0/1/2/3）+ §7.2 | ✓ |
| 6 | 训练协议（fixed-B/fixed-K、seeds、统计、预注册） | §5.3、§5.6、§2.6 | ✓ |
| 7 | ≥6 基线 + 消融 ≥8 | §5.4（6 主+2 参照）、§5.5（10 消融） | ✓ |
| 8 | 校准评测（ECE/Brier/风险-覆盖）+ agent 组合协议 | §6.1、§6.4 | ✓ |
| 9 | LLM agent 设计（planner/API/失败恢复/成本） | §6.2–6.6 | ✓ |
| 10 | 预算表（逐行 GPU-h×单价，±50%，FLOOR） | §7.1–7.3（¥4,849；FLOOR ¥1,345） | ✓ |
| 11 | 45 周时间线 + Go/No-Go + 风险 ≥10 | §8、§9（15 条风险） | ✓ |
| 12 | 与 P2P/OmniReset/SGS/SER/Being-0 逐维差异 | §10 | ✓ |
| 13 | 论文结构（页分配）、图表清单、视频脚本、开源 | §11（8 页 CoRL） | ✓ |
| 14 | 中稿率提升策略（sim-only 弱点） | §12（含 MSE 与 rebuttal bank） | ✓ |

## 附录 C：预算复算脚本（与 §7.2 表一致，可直接运行）

```python
# python3 budget.py  ->  prints per-row and totals used in §7
PRICE = 7.0                      # A100 ¥/GPU-h (assumption; verify at purchase)
# (group, runs, hours_per_run)
rows = [
    ("P0", 1, 24.0), ("G1", 20, 8.58), ("G2", 32, 2.33), ("G3", 9, 8.58),
    ("G4", 3, 8.58), ("G5", 12, 2.33), ("G6", 3, 17.2), ("G7", 3, 9.2),
    ("G8", 20, 2.33), ("G9", 6, 0.92), ("G10", 1, 10.0), ("G11", 1, 6.0),
]
tot_h = sum(r * h for _, r, h in rows)
base = tot_h * PRICE
buffered = base * 1.12
misc = 550                        # electricity 220 + storage 80 + LLM API 250
total = buffered + misc
print(f"GPU-h={tot_h:.1f} base=¥{base:.0f} +12%=¥{buffered:.0f} +misc=¥{total:.0f}")
print(f"+50% price: ¥{tot_h*PRICE*1.5*1.12+misc:.0f}")
```

运行结果：`GPU-h=548.4 base=¥3839 +12%=¥4299 +misc=¥4849`；`+50% price: ¥6999`。削减顺序（C1–C5）与 FLOOR 见 §7.3；本地免费 ≈110 GPU-h。

## 附录 D：LLM 与评测配置（W20 锁定后写死）

| 项 | 设定 |
|---|---|
| LLM 模型 | 主实验：闭源 API（型号与版本号 W20 锁定并披露）；对照：一个开源权重大模型（本地 vLLM） |
| 温度 / top_p | 0 / 1.0（确定性） |
| system prompt | 固定模板（技能文档 + 任务描述 + 输出 JSON schema），全文附于开源仓库 |
| max steps | 20 次技能调用；每技能 ≤2 次重试；90 s 墙钟 |
| 缓存 | 相同 (prompt, state) 复用；`cache/llm/*.jsonl` |
| 成本记录 | 每次调用记录 model/version/tokens/¥/latency；汇总进 Table 4 |
| eval 初始状态 | `eval/init_states.json`：每任务 30 个固定初始状态（物体位姿/目标/摩擦），所有方法共用 |
| 成功判定 | 任务级二元 + 技能级检查点；判定脚本 `calib/success.py` 版本化 |

## 附录 E：版本记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-11-02 | 首版：8 技能（6 手内+2 臂式）、4 长程任务、fixed-K 主协议、6 基线+10 消融、¥4,849 预算、CoRL 2027 主投/ICLR 2028 兜底、45 周排期、15 条风险、14 项清单对照 |

