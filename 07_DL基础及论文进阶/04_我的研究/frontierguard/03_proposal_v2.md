# 03 · FrontierGuard-RL 研究提案 v2（修订全文）

> 修订者：proposal-critic（ICRA 资深审稿人 + 实验设计审计员）　｜　日期：2026-11-02
> 基线：`01_proposal_v1.md`（662 行）　｜　配套批改：`02_critique_v1.md`（C1–C9 / M0–M18 / m1–m12）
> 目标会议：**ICRA 2028**（ICRA 2027 已于 2026-09 截稿，官方截止日期**待核实**）

## 修订说明（200 字内）

v2 相对 v1 的九处结构性改动：①**Tier-2 主实验从"跨任务共享 PPO"改为"单任务内 config 级分配"**（v1 的设计在方法上不成立，见 C9）；②SGS（CoRL 2026）进入差异表首行、基线与理论，并新增"何时 success-band 足够"的判据与负结果分支；③补 fixed-K 协议，分离"更新饥饿"与"分配收益"；④Tier-1 改为含已知真 LP 的构造族、env 名合法化；⑤预算按"本地免费 + 租卡只买大 N"重划（¥3,997→¥1,821），四情景全部闭合；⑥真机 30→45 trials 配对；⑦EXP3-IX 改 full-information Hedge；⑧理论拆成"单轮 SNR + EMA 滞后"两层；⑨排期冻结 W30、提交 W41、留 5 周缓冲。

---

## 0. 定位、投稿目标与本文的一句话

**一句话**：并行 RL 在大 N 下"跑得多、学得少"的根因不是环境数不够，而是**经验被投给了错误的配置**；FrontierGuard-RL（FG）用"**归一化学习进度的导数 × epistemic 不确定性**"在线分配并行环境的 reset 预算，并给出**何时学习进度信号优于成功率信号**的可证伪判据。

**投稿目标（按优先级）**：
1. **ICRA 2028**（主目标；预计截止 2027-09，待核实）——6 页正文 + 视频；
2. **CoRL 2027**（备选；若主要结果在 2027-03 前完成，可提前一版投 CoRL，再扩刊 ICRA）——**不推荐**同时投，二者时间窗重叠不大，但 CoRL 的"learnability"审稿口味更契合本方法；
3. **RA-L**（保底；滚动投稿，若 ICRA 被拒可在 2 个月内转投，代价是放弃 6 页的完整叙事、压到 8 页 journal 格式）。

**为什么不是"又一个采样器论文"**：FG 的主张有三层，且每层都有独立证据链——(i) **机制**：配置级经验分配的收益可以分解为"分配通道"与"估计通道"，v1 把它们混在一起；(ii) **可预测性**：给出 ΔJ 随 N、M、噪声水平的定量预测，并预注册检验；(iii) **边界**：明确说出 success-band 在什么条件下足够、什么条件下失效，并把失效条件做成实验。ICRA 审稿人最认"作者自己定义了方法的失效边界"。

**与 v1 的根本差异**：v1 试图同时做三件事（任务级分配、N 到 16k 的 scaling、真机），但没有一件做扎实；v2 **砍掉任务级的伪命题**（8 个 SO-101 任务没有 task ID，无法共享策略），把预算集中到"**单任务配置级**"这一件干净、可复现、可 head-to-head 比较的事情上，跨任务泛化用 MetaWorld MT10 附录补足。

---

## 1. 题目、摘要、关键词

### 1.1 英文题目（三选一，推荐 A）

- **A（推荐）**：*FrontierGuard-RL: Uncertainty-Gated Learning-Progress Allocation for Massively Parallel Reinforcement Learning*
- B：*When Does Success Rate Suffice? Learning-Progress Signals for Config-Level Allocation in Parallel RL*
- C：*FrontierGuard: Allocating Parallel Environments by the Derivative of Learnability*

> 选 A 的理由：把 novelty 放在"uncertainty-gated"与"learning-progress allocation"两个具体名词上，避免"SGS 2.0"的观感；B 适合负结果分支时启用；C 太窄，会被认为只是一个 trick。

### 1.2 英文摘要（189 词，≤200 词硬上限）

Massively parallel RL promises to turn compute into data, yet returns saturate: a growing share of environments is spent on configurations a policy has already mastered or cannot yet learn. Recent work (SGS, CoRL 2026) addresses this by resampling configurations with a success-rate heuristic. We ask a sharper question: success rate is a *biased, high-variance proxy* for learnability whenever configuration outcomes are stochastic, and no existing method separates the two failure modes it confounds. We introduce **FrontierGuard-RL (FG)**, which allocates parallel environments by the *derivative* of a normalized learning-progress (LP) estimate, gated by an epistemic-uncertainty term and three guards against noise chasing, starvation, and abrupt allocation shifts. FG is head-to-head comparable to success-band sampling under identical budget, seeds, and N. Across four MuJoCo Playground tasks up to N=16,384 and four SO-101 manipulation tasks up to N=2,048, we (i) separate allocation gains from update-starvation artifacts with a fixed-update protocol, (ii) show that success-band sampling degrades predictably as config-level aleatoric noise grows while FG does not, and (iii) validate on a physical SO-101 with 45 paired trials per condition. Code, configs, and raw results will be released.

### 1.3 中文摘要（~300 字）

大规模并行强化学习承诺"以算力换数据"，但收益会饱和：越来越多的并行环境被花在策略已经掌握、或根本学不会的配置上。SGS（CoRL 2026）用成功率启发式重采样缓解了这一点，但成功率是**有偏且高方差**的可学性代理——当配置结果本身带随机性时，它会把噪声误判为进步。本文提出 FrontierGuard-RL（FG）：以**归一化学习进度（LP）的导数**为分配信号，用 **epistemic 不确定性**门控，并加三重保护（噪声追逐、饥饿、分配突变）。方法可与 success-band 采样在**同预算、同 seed、同 N** 下直接对比。我们在四个 MuJoCo Playground 任务（最高 N=16,384）与四个 SO-101 操作任务（最高 N=2,048）上：(i) 用固定更新数协议把"分配收益"与"更新饥饿伪影"分离；(ii) 证明 success-band 的退化随配置级 aleatoric 噪声可预测地加剧、而 FG 不退化；(iii) 在真机 SO-101 上以每条件 45 次配对试验验证。代码、配置与原始结果将开源。

### 1.4 关键词

parallel reinforcement learning; environment allocation; curriculum learning; learnability estimation; uncertainty quantification; manipulation

### 1.5 一句话卖点（for cover letter）

*We show that the success-rate heuristic adopted by the closest prior work (SGS, CoRL 2026) is a special case of our formulation with zero uncertainty weight and zero derivative, and we characterize exactly when that special case suffices.*

### 1.6 与 SGS 的三句话差异（审稿人必读）

1. **同层不同信号**：SGS 与 FG 都在"单任务配置级"上做 reset 分配，但 SGS 的权重来自 success-rate 三档（never-solved / improving / mastered）的**分段常数**函数；FG 用 LP 的**导数**（趋势）乘以 **1/(1+epistemic 不确定度)**。分段常数对噪声不可导，且把"p=0.5 的随机配置"和"p=0.5 的正在进步配置"判成同一档。
2. **SGS 是 FG 的特例**：令不确定性权重→0、导数→常数、guard 全关，FG 的权重退化为 success-band 的单调函数。因此两者的比较不是"两个方法谁好"，而是"**何时需要导数与不确定性**"这一参数化问题——这就是本文 §4.4 的判据。
3. **可复现性差异**：截至 2026-11-02，SGS 代码为 "coming soon"；本文提供 SGS-band 的**忠实重实现**（按原文三档定义）、全部 run 配置与 raw logs，并把它作为**一等公民 baseline** 而非稻草人。

### 1.7 与其余近邻的一句话差异

| 工作 | 差异 |
|---|---|
| SER 2511.21011 | SER 重采样**时序/轨迹**（多久重放），FG 重采样**配置**（重放哪些）；二者正交，可用 2×2 消融组合（扩展基线） |
| DRATS 2605.14350 | DRATS 做**任务级**自适应采样（固定任务集、无 N-scaling、无真机）；FG 主实验在**配置级**，任务级仅 MetaWorld 附录 |
| PGB 2610.25704 | 同为 bandit 分配，但 PGB 在 LLM 领域、分组是 prompt 难度；FG 的 group 是物理配置桶且受控于仿真预算 |
| UPER 2506.09270 | UPER 用"最大化回报/优先未解决"启发式；FG 侧重**不确定性门控**与**scaling 预测** |
| GMC 2605.05856 | GMC 是"按可达性筛选任务"的课程，不做并行 env 的**配额分配** |
| LPM 2509.25438 | LPM 估计"学习进度"但用于课程排序，不做多 env 并行分配、无 N 维度 |
| PLR 2010.03934 / RED 2110.02439 | 基于 regret/value 的 level 重放，面向**生成式 level**；FG 的 group 是固定配置桶，不生成新 level，因此计算开销低一个量级 |

---

## 2. 研究问题与假设（预注册版本）

### 2.1 RQ1：机制归因——收益来自哪里？

在**固定总环境步 B**（协议 A）下，把配置级分配信号从 uniform 换成 FG，收益 ΔJ 需要拆成两条通道：
- **分配通道**（同 N、同 B、仅 w 不同）：FG(B) − uniform(B)；
- **更新次数通道**（同 N、仅 K 不同）：协议 A（固定 B，N 越大 K 越小）与协议 B（固定 K，N 越大 B 越大）在同一 (N, method) 上的差异，归因于 K；
- **交互**：两通道不独立时，报告 2×2 分解（§5.3 协议 A/B 的主格数据即可支持，不需额外 run）。

**H1**：在协议 A（固定 B）下，4 个 Tier-1 族与 4 个 SO-101 任务上 **FG > SGS-band > uniform**（≥3/4 族 + ≥3/4 任务）；在协议 B（固定 K）下排序保持。若 FG ≈ SGS-band（差 <3 pp 且 CI 跨 0），触发 §5.8 负结果分支。

### 2.2 RQ2：scaling——收益如何随 N 与 B 变化？

**H2**：在 fixed-K 协议下，所有方法随 N 单调变好，且 **N×method 交互项不显著**（FG 的优势是"平移"而非"放大"）。——这条是反直觉的**可证伪**预测（若交互显著，说明大 N 让 FG 的估计更准并放大了优势，理论需改）。

**H3**：在 fixed-B 协议下，J(N) 呈倒 U（更新饥饿），**FG 的峰值 N\* 比 uniform 更大**（因为 FG 把省下的样本集中到可学组、等效提高每步有效学习）。

### 2.3 RQ3：边界——success-band 何时失效？

**H4**：注入配置级 aleatoric 噪声 σ_a 后，SGS-band 的分配 regret 随 σ_a 增长（斜率 >0），FG 基本不变（斜率 ≈0）；当 σ_a=0 时两者**无显著差异**（预注册的"无差异"预测，用于验证差异不是自我实现的）。

**H5**：三重 guard 中，只有 **noise guard** 在高 σ_a 下必要（消融 A3 显著变差），greedy-free guard 与 KL guard 主要影响**稳定性**（min-group 成功率、崩溃次数）而非最终均值。

### 2.4 RQ4：任务级与真机——能否迁移？

**H6**：FG 在 4 个 SO-101 任务上独立训练均优于 uniform（≥3/4），在 8 任务 breadth 上方向一致（≥6/8）。

**H7**：MetaWorld MT10（task ID 在 obs）上，FG 的**任务级**版本与 DRATS 可比，且与配置级版本组合（两级）优于任一单级——这是"任务级扩展"的唯一证据，仅作附录，**不作为主贡献**。

**H8**：真机上，FG 与 uniform 的成功率差 ≥20 pp（45 trials/条件，2 任务合并 n=90，McNemar p<0.05）；**不主张**10 pp 级别的差异（功效不足，见 §5.7）。

### 2.5 预注册与负结果分支（必须写进 v2）

- **冻结时点**：W8（G2），冻结 RQ/H/主终点/排除标准/统计模型；冻结前只做 pilot 与吞吐测试，不做主分析。
- **主终点**：fixed-B 协议 A 的 final success rate（最后 10% 步数均值），4 任务 × 6 方法 × 8 seeds 的混合效应模型。
- **负结果分支（预注册，不做事后选择）**：若 H1 中 FG−SGS-band 的均值差 <3 pp 且 95% CI 跨 0，则论文改为 §1.1 的题目 B，主贡献改为"success-band 的适用边界 + 校准研究"，并把 FG 定位为"带不确定性门控的泛化形式"；实验成本已沉没，不追加租卡。
- **降级分支**：若 G0 失败（5M 步 @N=256 成功率 <25%），Tier-2 从 PPO 改为 SAC + FG（复用 Squint 的 SAC 实现，工程量 +1.5 周），并砍 N=2048 行。

---

## 3. 方法

### 3.0 总览与"部件 → 失效模式 → 消融"对应表

FG 是一个**外挂在任意 on-policy 并行 RL 之上的配置分配器**：它不改变 PPO 的目标，只在每个 rollout 轮次决定"下一轮 N 个并行环境里，每个配置桶分几个 env"。

| # | 部件 | 目标失效模式（不加会怎样） | 对应消融 |
|---|---|---|---|
| P1 | 归一化 LP（双 EMA + 组内 z-score） | 不同组的成功率尺度不可比，直接比较会把"简单但没学好"的组误判为前沿 | A1（去归一化） |
| P2 | LP 的**导数**项 Δℓ | 只用 LP 水平 = 追逐已经高的组（贪心），在"高原但仍在进步"的组上停得太早 | A2（去导数，κ=0） |
| P3 | epistemic 不确定性门控 | 小样本组的高方差被误当"在进步"，噪声追逐 | A3（去不确定性） |
| P4 | guard-1 noise（σ_epi 大 → 冻结权重） | 同上，但作为硬保护；与 P3 互补 | A4 |
| P5 | guard-2 starvation（w_min + n_min warm-up） | 某组被永久饿死，最终 min-group 成功率塌陷 | A5（去 w_min）/ A5b（去 warm-up） |
| P6 | guard-3 信任域（ρ 平滑 + 单轮变化上限） | 分配在相邻轮次抖动，PPO 的 advantage 归一化被打乱 | A6 |
| P7 | Hedge（full-information）而非 EXP3-IX | 每组每轮都有 env、都能观测到 reward → 是 full-information 设定，用 EXP3 的 importance weight 是错配且方差更大 | A7 |

**审稿人友好条款**：若 A1–A3 显示"只用 LP 水平"或"只用不确定性"即可解释全部收益，则 v2 的最终方法简化为对应单信号版本，并在论文中如实报告（方法可被自己的消融证伪）。

### 3.1 形式化：单任务配置级 MDP 族

**基本设定**：一个任务 𝒯 定义 MDP ℳ=(𝒮,𝒜,P,r,γ,ρ₀)，其**配置空间** 𝒞 由环境参数 c 张成（初始状态、物体位姿、目标区域、物理参数、光照）。采样 c∼p(c) 得到**配置实例** ℳ_c。策略 π_θ(a|s) 对所有 c **共享**（这是合法的：reward 与成功判定相同，只有初始条件/参数不同）。

> **关键合法性条件（v1 在此出错）**：共享策略要求所有配置实例属于**同一个 MDP 的初始分布扰动**，而不是**不同 reward 函数的任务**。因此 v2 的主实验**固定单个任务**，绝不把多个 SO-101 任务塞进同一个 PPO（C9）。跨任务只出现在 MetaWorld 附录，且那里 **task ID 进入观测**。

**分组**：把 𝒞 划分为 M 个互斥桶 G₁…G_M（构造见 §3.2）。第 t 轮（每轮 L 个 env step）中，组 g 分到 n_g 个 env，权重 w_g=n_g/N，Σ_g w_g=1。

**优化目标**（论文中明确写出，回应"你们到底在优化什么"）：

  J(θ, w) = E_{c∼p_w(c)}[ J_c(θ) ]，  p_w(c) = Σ_g w_g · p(c | c∈G_g)

注意：**w 不改变最优策略**（p_w 与 p 同支撑时 argmax_θ J 相同），w 只改变**有限预算下的学习速度**。这是本文方法的合法性基础，也是审稿人最可能问的第一问——必须显式写出这一句。

### 3.2 分组构造（统一 M 的语义）

> **v1 的错误**：Tier-1 的 M=8 是"任务内初始状态桶"，Tier-2 的 M=4 是"任务"；两种粒度混用同一个 g。v2 全篇只保留一个定义：**group = 同一任务内的一个配置桶**。

#### Tier-1（MuJoCo Playground，本地，N≤16,384）：构造族 A/B/C/D

| 族 | 任务（合法 env 名） | 分组方式 | M | 真 LP 是否已知 | 用途 |
|---|---|---|---|---|---|
| **A** | `PandaPickCube` | 目标区域 4 × 物体质量/摩擦 2 | 8 | **是**（解析可达性/成功概率） | 主族：含真 LP，验证"FG 能否逼近 oracle-LP" |
| **B** | `CheetahRun` | 初始速度 4 档 × 动作噪声 2 档 | 8 | 部分（速度档可解析） | 异质性 + **aleatoric 噪声注入**（H4） |
| **C** | `HumanoidRun` | 初始姿态扰动 4 档 × 地形粗糙度 2 档 | 8 | 否 | 高维、难学性差异大，检验泛化 |
| **D** | `Go1JoystickFlatTerrain` | 目标速度指令 8 档 | 8 | 部分 | 指令条件化、接近真机部署形态 |

- 统一 M=8，便于跨族比较；**H1 要求 ≥3/4 族排序一致**。
- **族 B 的噪声注入**是 H4 的核心：在部分配置上以 p_slip∈{0, 0.3, 0.6} 随机让 episode 提前终止（"滑倒"），制造**与可学性无关的 aleatoric 噪声**。σ_a=0 的族 A/C 是"无差异"预测的对照组。
- **构造性保证异质性**：v1 假设"Cheetah 的初速度桶天然有难度差"，这不成立（δ≈0，各桶几乎同样难）。v2 用**显式构造**：族 A 的目标距离从"机械臂可达半径的 40%→95%"取 4 档，保证成功概率从 ~0.9 单调降到 ~0.1（pilot 校准）；族 D 的速度指令从 0.5→3.0 m/s 取 8 档。**"真 LP 已知"来自解析/半解析计算，不是拟合**。

#### Tier-2（Squint SO-101 视觉操作，租卡 N=1024/2048，本地 N=256）

- **主实验任务（4 个）**：`SO101LiftCube-v1`、`SO101StackCube-v1`、`SO101PlaceCan-v1`、`SO101ReachCan-v1`（env_id 以 Squint 仓库为准，**`-v1` 不是 `-v0`**；W1 核对）。
- **分组（首选）**：**目标区域 4 档 × DR 桶 2 档 = M=8**。SO-101 的 reach/lift/place 任务的**目标位置在观测中**（goal-conditioned；具体字段 W1 核实），因此"目标区域桶"对策略**可辨识**，且 reward 不变 → 满足 §3.1 合法性条件。
- **Fallback（预注册）**：若 W1 核实发现观测**不含目标位置**（不可条件化），则退化为**纯 DR 桶 M=4**（物体初始位姿 2 × 质量/摩擦 2），并在论文中如实写明"配置桶仅在策略可辨识时有效；不可辨识时退化为 DR 采样"。
- **M 的选择**：M=8 主值（4×2）；M=4 与 M=16 作为敏感性（A8）。**注意**：M 增大到 16 以上会因每轮每组的 episode 数过少（n_min=30 约束）而不可行——这一点在 §3.5 的 warm-up 里有量化。

#### MetaWorld 附录（跨任务，本地 CPU，仅附录）

- **MT10**（10 任务），task ID one-hot 拼入 obs（39→49 维；DRATS 亦用该基准，具体实现**待核实**），N≤128（CPU 8 核限制，**不是** 256）。
- 方法：uniform / DRATS（任务级）/ FG（任务级）/ FG 两级（任务级 × 配置级）。
- **只作泛化性证据**，明确写"不作 N-scaling 主证据"。

### 3.3 学习进度（LP）估计：双 EMA + 组内归一化 + 导数

对每个组 g，维护成功率的双 EMA：

  p̄_g^{fast} ← (1−α_f)·p̄_g^{fast} + α_f·p̂_g^{(t)}，  α_f = 1/H_ep^fast，H_ep^fast=20 episodes
  p̄_g^{slow} ← (1−α_s)·p̄_g^{slow} + α_s·p̂_g^{(t)}，  α_s = 1/H_ep^slow，H_ep^slow=100 episodes

- **LP 水平**：ℓ_g = p̄_g^{fast} − p̄_g^{slow}（正值 = 近期进步快于长期平均，即"正在爬升"）。
- **LP 导数**：Δℓ_g = ℓ_g^{(t)} − ℓ_g^{(t−1)}（用 3 轮移动平均去噪）。
- **组内归一化**（M7 修正）：u_g^{raw} = (ℓ_g + κ·Δℓ_g) / (σ̂_g + ε₀)，其中 σ̂_g 是该组最近 20 轮 ℓ 的样本标准差，ε₀=0.05，κ=1.0。归一化保证跨组可比（不同任务/桶的成功率尺度不同）。

> **为什么用导数而不只是水平**：只看 ℓ_g 水平等于"追高 LP 组"，在"LP 高但已进入平台"的组上会持续投入（v1 的贪心问题）。导数项识别"仍在上升"的组。A2 消融（κ=0）直接检验这一项是否必要。

### 3.4 epistemic 不确定性：E=3 集成 + 方差分解

- 用 **E=3** 个独立初始化的"成功率预测头"（轻量 MLP，输入 = 组统计特征 + 全局步数，输出 p̂_g），或等价地：3 个不同 bootstrap 的 EMA 估计器。**不重训策略**（避免 3× 计算）。
- 分解：总方差 = aleatoric（各头内部二项方差均值）+ epistemic（头间均值方差）。
- **只把 epistemic 用于门控**：σ_epi,g = sd(p̄_g^{(e)}, e=1..3)。理由：aleatoric 噪声是配置本身的性质，不应作为"该不该采样"的依据（这正是 SGS 混淆的东西）。

### 3.5 分配器：full-information Hedge + 三重 guard

**为什么不是 EXP3-IX（C6 修正）**：每个组每一轮都分到至少 1 个 env、都能观测到自己的 p̂_g，因此这是 **full-information** 设定，不是 bandit。EXP3 的 importance weighting 在这里只会放大方差。v2 主分配器为 **Hedge**（乘性权重），EXP3-IX 降为 A7 消融（用于证明"full-information 假设是对的"）。

**Hedge 更新**（每轮）：

  w̃_g ← w_g · exp(η · û_g)，  η=0.05，û_g = clip(u_g^{raw}, −1, 1)
  w ← ProjectToSimplex(w̃) 受约束：w_g ≥ w_min = 0.5/M，Σ w_g = 1
  w ← (1−ρ)·w_old + ρ·w，  ρ=0.2（信任域平滑）
  单轮硬约束：|w_g − w_g^{old}| ≤ 0.10

**三重 guard**：

| guard | 规则 | 默认参数 | 作用 |
|---|---|---|---|
| G-1 noise | 若 σ_epi,g > β_noise · \|u_g^{raw}\|，则本轮的 w_g 冻结在上轮值 | β_noise=0.5 | 估计不可信时不改分配 |
| G-2 starvation | w_g ≥ w_min；每组累计 episode < n_min=30 时强制 uniform（warm-up） | w_min=0.5/M, n_min=30 | 防止饿死；n_min 由 §4.2 的 SNR 推出 |
| G-3 信任域 | ρ=0.2 平滑 + 单轮 ≤0.10 | ρ=0.2, Δ_max=0.10 | 防止分配抖动破坏 PPO 的 advantage 归一化 |

> **n_min=30 的推导**（不是拍的）：二项比例在 p=0.5 时 sd=0.5/√n；要求 sd ≤0.09（足以区分 0.4 vs 0.5 的 10 pp 差），n≥30.9 → **n_min=30**。

### 3.6 权重 → env 配额的整数映射

  n_g = max(1, round(w_g · N))，剩余名额按最大余数法分配，保证 Σ n_g = N。

N=256, M=8 时每组的可能配额为 1–64；w_min=1/16 → 至少 16 个 env，满足 n_min 需要每轮每组 ≥30 个 episode——注意 episode 数 ≠ env 数：每轮 L=50 步、episode 长 H=50 步（Squint 默认 horizon 待核实），则每 env 每轮约 1 个 episode，**16 env × 1 = 16 < 30**。因此 warm-up 在 N=256 时需 **2 轮**，在 N=1024 时需 **1 轮**。这一条必须写进论文（否则 n_min 与 N 的关系会被质疑）。

### 3.7 算法伪代码与 PPO 集成

```
输入: N, M, 分组函数 g(c), PPO 超参, FG 超参(η, κ, β_noise, w_min, ρ, n_min)
初始化: w_g = 1/M, 双 EMA 未就绪 → warm-up 阶段用 uniform
for t = 1..K:                       # K = B/(N·L)
    n_g = IntAlloc(w, N)            # §3.6
    并行 rollout: 每个 env 按 n_g 采样配置 c ~ p(·|G_g)，跑 L 步
    收集 {s,a,r,s'}，更新 PPO（标准实现，UTD=1）
    对每组 g: 统计 p̂_g（本轮 episode 成功率）
    更新双 EMA → ℓ_g, Δℓ_g → u_g^{raw}
    更新 3 个预测头 → σ_epi,g
    û_g = clip(u_g^{raw}, −1, 1)
    w ← Hedge(û, η) → 投影(w_min) → 平滑(ρ) → 变化上限(0.10)   # §3.5
    记录: w, p̂_g, σ_epi,g, ℓ_g, 每轮 wall-clock
```

- **PPO 不变**：FG 只在 rollout 采样层作用，不改变 loss、GAE、clipping。这一点让"FG 可直接用于任何并行 RL 代码库"成为卖点（也是审稿人喜欢的"plug-in"定位）。
- **异步版本（可选，附录）**：若 Squint 的 ManiSkill3 实现是同步向量化（默认），无需改；若改为异步（IMPALA 式），分配器需按完成 episode 事件触发而非固定轮次——v2 **不做**，仅在 Limitations 提一句。

### 3.8 默认超参表（v1 缺失，M8）

| 参数 | 默认值 | 搜索范围 | 校准时点 |
|---|---|---|---|
| η（Hedge 学习率） | 0.05 | {0.01, 0.05, 0.2} | W4 pilot |
| κ（导数权重） | 1.0 | {0, 0.5, 1.0, 2.0} | W4 pilot |
| H_ep^fast / H_ep^slow（EMA 窗，episode 数） | 20 / 100 | {10,20,40} / {50,100,200} | W4 pilot |
| β_noise | 0.5 | {0.1, 0.5, 1.0} | W4 pilot |
| w_min | 0.5/M | {0.25/M, 0.5/M, 1/M} | 固定 |
| ρ（平滑） | 0.2 | {0.1, 0.2, 0.5} | 固定 |
| Δ_max | 0.10 | 固定 | 固定 |
| n_min | 30 episodes | 由 §3.5 推出 | 固定 |
| E（集成头数） | 3 | {1,3,5}（A9） | W4 pilot |
| L（每轮 env step） | 50 | 与 PPO rollout 长度一致 | 固定 |

**冻结规则**：以上全部在 W8（G2）冻结；pilot 阶段（W3–W7）只在族 A 的 1 个任务上调参，**不在主格上调**。

### 3.9 计算与内存开销（审稿人必问）

- 分配器本身：O(M) 的标量运算 + 3 个轻量 MLP 前向（每轮一次）→ **<1% 的 step 时间**。
- 额外内存：每组 ~20 轮的统计量 + 3 个 MLP（<1 MB）→ 可忽略。
- **不额外跑 rollout**（与"用 ensemble 估不确定性"的重方法不同）：这是 FG 相对 ensemble-RL 方法的成本优势，必须在论文里明确写。

---

## 4. 理论：一个可检验的"可学性混合"模型

> **诚实声明（写进论文）**：本节不是关于 PPO 收敛性的定理，而是一个**风格化模型（stylized model）**：它把"配置分配是否重要"压缩成可解析的形式，并给出 5 条可证伪预测。论文中必须用这段原话，不能包装成定理。

### 4.1 假设

- **A1（可学性）**：组 g 的成功率按分配到的环境步数 s 演化：p_g(s) = 1 − exp(−a_g·s)，a_g ≥ 0 为**可学性系数**（a_g=0 ⇔ 不可学/纯噪声组）。
- **A2（共享策略）**：所有组共享同一 π_θ，且 a_g 与 θ 无关（在关注的中期尺度上近似成立）。
- **A3（估计）**：p̂_g = p_g + ε_g，ε_g ~ 近似 N(0, p_g(1−p_g)/m_g)，m_g = 每轮该组完成的 episode 数。
- **A4（预算）**：总环境步 B = K·N·L 固定（协议 A），或总更新数 K 固定（协议 B）。

> **A1 的局限必须在 Limitations 写明**：真实 PPO 的 p_g(s) 不是指数曲线（有平台、有遗忘、有课程耦合）。A1 只用于推出**方向性预测**，不用于拟合实验曲线。

### 4.2 单轮噪声与 EMA 滞后（C8 修正）

- **每组每轮的 episode 数**：m_g = w_g·N·L/H（H = episode 长度）。
- **单轮估计噪声**：sd(p̂_g) = √(p_g(1−p_g)/m_g)。
- **LP 的估计噪声**（ℓ_g 为两个 EMA 之差）：sd(ℓ_g) ≈ √(2·p_g(1−p_g)/(m_g·H_ep))，其中 EMA 窗按 **episode 数** H_ep 定义（H_ep^fast=20、H_ep^slow=100）。

由此得到两个**不同**的量（v1 把它们混为一谈，是 C8 的根源）：

1. **单轮精度**：SNR_ℓ = Δp / sd(ℓ_g) ∝ **√(w_g·N·L·H_ep/H)** → **∝ √N**。这是 P1 要检验的量（协议 B 下对 log N 回归，期望斜率 0.5）。
2. **响应滞后**（以"轮"计）：τ_g = H_ep·H/(w_g·N·L) → **∝ 1/(w_g·N)**，即 N 越大，同样的 episode 窗在更少的轮次内填满 → **墙钟响应更快**。

两者不矛盾：一个是"一轮内的精度"，一个是"积累窗口所需的时间"。

### 4.3 分配收益的解析形式

在 A1 下，固定权重 w 的终局目标为

  J(w) = 1 − (1/M) Σ_g exp(−a_g·w_g·B)

- **uniform**：w_g=1/M。
- **oracle**（已知 a_g）：解 a_g·exp(−a_g w_g B)=λ，得 w*_g = (1/(a_g B))·ln(a_g B/λ)（水填充），s.t. Σw*=1。
- **FG**：用估计的 â_g 代替 a_g，误差 ~1/√(m_g·H_ep)。

**三条直接可读出的结论**：
1. **收益随异质性增长**：ΔJ(FG−uniform) 随 {a_g} 的离散度（如 CV(a_g) 或 max/mean）单调增；a_g 全相等时 ΔJ=0。→ **P1**。
2. **收益随 B 增长而衰减**：B→∞ 时所有 learnable 组饱和，ΔJ→(不可学组占比)；因此**收益是"中期限"现象**，论文必须报告"何时收益消失"，而不是只报终点。→ **P2**。
3. **FG 与 oracle 的差距 ∝ 1/√N**：估计精度随 N 提升，但 **FG 与 uniform 的差距（领先项）与 N 无关** → **N×method 交互不显著**。→ **P3**（这就是 H2）。

### 4.4 核心判据：success-band 何时足够？

SGS 的 success-band 把组分成三档（never-solved / improving / mastered），等价于用**分段常数**函数近似 a_g。它足够的条件是**两个不等式同时成立**：

**(i) 可分辨条件**：观测噪声必须小于相邻组的真实可学性差
  σ_obs < Δ_min/2，其中 σ_obs = sd(p̂_g)，Δ_min = min_{g≠g'} |a_g − a_g'|·w·B（在 p 空间换算）

**(ii) 无 aleatoric 混淆条件**：配置级随机性不得把"随机成功"误判为"正在进步"
  p_aleatoric < Δ_min/2，其中 p_aleatoric = 配置本身的不可约成功概率波动（如 p_slip）

**当 (i) 或 (ii) 被破坏时**，success-band 的"improving"判据会被噪声触发，分配退化为"随机地追逐噪声组"；FG 的导数项 + epistemic 门控正是为这个区间设计。**这就是本文相对 SGS 的可证伪差异，也是 §5 中族 B 与 Tier-2 高噪声桶要构造的场景。**

**SGS = FG 的特例**（参数化关系，写进 §1.6 与 §4.6）：令 σ_epi≡0、κ=0、把 u_g 量化为三档、关掉 guards，FG 的权重曲线与 success-band 同构。

### 4.5 可证伪预测 P1–P6 与对应实验

| # | 预测 | 检验方式 | 若不成立说明什么 |
|---|---|---|---|
| P1 | SNR_ℓ ∝ √N（斜率 0.5） | 协议 B，族 A/B，log-log 回归 | 我们的噪声模型错了，A3 需改 |
| P2 | FG−uniform 收益随 B 增大而衰减，且在不可学组占比高时保持 | 协议 A，B∈{50M,150M,400M}（Tier-1 扫描） | "收益是中期现象"错误，需重写 §4.3 |
| P3 | N×method 交互不显著（FG 是平移而非放大） | 协议 A/B 的混合模型交互项 | 大 N 放大了 FG 优势 → §4.3 需加入"估计精度→分配精度"反馈项 |
| P4 | SGS-band 在高 aleatoric 噪声下退化（斜率>0），FG 不退化（斜率≈0） | 族 B 的 p_slip∈{0,0.3,0.6} × 4 方法 | 差异不存在 → 走负结果分支（§5.8） |
| P5 | n_min=30 的 warm-up 必要：去掉后早期误分配导致终点显著变差 | A5b 消融 | warm-up 无必要 → 简化方法 |
| P6 | EMA 滞后 ∝1/N：N 越大，课程切换后的恢复轮数越少 | 人为在 K/2 处改变配置分布，测恢复轮数 | 响应模型错误 |

**每条预测都写"若不成立说明什么"——这是 v2 相对 v1 最重要的写作升级：审稿人看到作者预设了证伪条件，对"过度声称"的怀疑会显著下降。**

---

## 5. 实验协议

### 5.1 平台、环境与工程边界

| 层 | 平台 | 任务 | N 范围 | 算力 |
|---|---|---|---|---|
| **Tier-1** | MuJoCo Playground（JAX，GPU） | `PandaPickCube` / `CheetahRun` / `HumanoidRun` / `Go1JoystickFlatTerrain` | 256 → 16,384 | **本地 4070 Ti Super**，免费 |
| **Tier-2** | ManiSkill3 + Squint（SO-101，RGB） | `SO101LiftCube-v1` / `SO101StackCube-v1` / `SO101PlaceCan-v1` / `SO101ReachCan-v1` | 256（本地）→ 1024 / 2048（租卡） | 本地 + 4090 / A100 80GB |
| **附录** | MetaWorld MT10（CPU，MuJoCo） | 10 个操作任务 + task one-hot | 64–128 | **本地 CPU**，免费 |
| **真机** | 2× SO-101（LeRobot） | Lift-Cube / Place-Can | — | 本地 |

> **明确不做的事（写进 §13 与 Limitations）**：
> 1. **不为 8 个 SO-101 任务自建 task-conditioned 统一环境**（统一 reward/成功判定 + task embedding）：估算 **3–4 人周 + 高失败风险**，v2 不采用；
> 2. **不做跨任务共享策略**（v1 的 Tier-2 设计在方法上不成立，见 C9）；
> 3. **不做 Isaac Lab / MTBench（2507.23172）**：IsaacGym Preview 4 与 40 系卡/新驱动的兼容风险高，且与本课题的科学问题无关；
> 4. **不做 SO-101 Bench（2610.36420）的 VLA 评测协议**：那是语言条件 VLA 基准，不是 RL-from-scratch 的训练环境。

### 5.2 三个协议（核心实验设计）

| 协议 | 固定量 | 变化量 | 回答的问题 | 主要用途 |
|---|---|---|---|---|
| **A：固定步数** | B（Tier-1 150M；Tier-2 15M） | N ∈ {256,1024,4096,8192,16384}（T1）/ {256,1024,2048}（T2） | "实际部署该选多大 N？" | 主结果（H1/H3） |
| **B：固定更新数** | K = B₀/(N₀L) = 150M/(1024×50) ≈ **2,930** 轮 | N 同 A，B = K·N·L | "分配收益是否只是更新次数差异造成的假象？" | **决定性控制（C2）** |
| **C：固定墙钟** | T（Tier-1 4 h；Tier-2 12 h） | N 同 A | "同样等 4 小时，哪个配置学得多？" | 工程相关性（ICRA 关心） |

- **协议 B 的可行性边界**：B = K·N·L 随 N 线性增长，N=16,384 时不可承受。**规则**：协议 B 只跑到 **N≤4,096**，并在论文中写明"N=8,192/16,384 的 fixed-K 点由 N≤4,096 的拟合外推，不外推结论，只标趋势线"。这是诚实做法，比 v1 的"假装所有 N 都能 fixed-K"好。
- **协议 C 的吞吐饱和**：N=16,384 在单卡上墙钟主要由 JAX 编译与显存决定，**必须实测**（G0），并报告"wall-clock 效率 = (steps/s at N) / (steps/s at N=256) × ..."。

### 5.3 基线（6 主 + 4 扩展）

| # | 方法 | 实现来源 | 类别 | 备注 |
|---|---|---|---|---|
| 1 | **uniform** | 原生 | 下界 | 每轮均匀分配 |
| 2 | **SER** 2511.21011 | 重实现（时序重采样） | 近邻 | 只重采样"多久一次"，与 FG 正交 |
| 3 | **SGS-band** 2610.12465 | **重实现**（无官方代码） | 最强近邻 | 三档 success-band 加权 reset；实现细节写进附录 |
| 4 | **PLR / RED** 2010.03934 / 2110.02439 | 官方代码适配 | 经典 | 按原文公式（value/regret 优先级），不做稻草人 |
| 5 | **DRATS** 2605.14350 | 重实现（任务级→配置级） | 近邻 | **明确标注"原方法是任务级，这里做了配置级最小适配，属可争议点"** |
| 6 | **FG（本文）** | 本文 | 方法 | 完整版 |
| 扩展 7 | **UPER** 2506.09270 | 重实现 | 近邻 | 按回报/未解决优先 |
| 扩展 8 | **max-return**（hardest-first） | 原生 | 简单启发式 | 证明"不是越难越好" |
| 扩展 9 | **SGS+FG**（组合） | 本文 | 组合 | band 作先验 + FG 作门控，测"两者是否互补" |
| 扩展 10 | **oracle-LP**（参考线） | 本文 | 上界（非 baseline） | 用真 a_g 的水填充解，画出"还能提升多少" |

> **审稿人预案**：「你们把 DRATS 改成了配置级，是不是在打稻草人？」→ 回答：DRATS 原文是任务级，我们用其"难度感知采样"的核心思想做最小适配，并在附录给出**任务级原版**在 MetaWorld MT10 上的结果（附录实验），两者都报。

### 5.4 消融（A1–A10 + oracle）

| ID | 变体 | 检验的部件 | 平台 |
|---|---|---|---|
| A1 | 去组内归一化（不用 z-score） | P1 | T1（全）+ T2（1 任务） |
| A2 | 去导数（κ=0，只用 LP 水平） | P2 | T1 + T2 |
| A3 | 去 epistemic 不确定性（σ_epi≡0） | P3 | T1 + T2 |
| A4 | 保留不确定性但**不门控**（σ 只记录） | P3/P4 互补性 | T1 |
| A5 | 去 starvation guard（w_min→1/N） | P5 | T1 + T2 |
| A5b | 去 warm-up（n_min=1） | P5 | T1 |
| A6 | 去信任域（ρ=1, Δ_max=∞） | P6 | T1 + T2 |
| A7 | Hedge → EXP3-IX（bandit 版） | 证明 full-information 假设 | T1 + T2 |
| A8 | M ∈ {4, 16}（主值 8） | 分组粒度敏感性 | T1 |
| A9 | E ∈ {1, 5}（主值 3） | 集成规模敏感性 | T1 |
| A10 | **oracle-LP 参考线**（用真 a_g 水填充） | 上界 | T1 + T2（参考线，3 seeds） |

- **Tier-2 消融只做 8 个变体**（A1/A2/A3/A5/A6/A7/A8-M16/A9-E1）× 3 seeds × 1 任务 × N=1024 = **24 runs**（预算已含）。
- **Tier-1 做全消融**（12 变体 × 3 seeds × 2 个 N）= 72 runs，本地免费。
- **A10 是 v1 缺失的最有价值实验（M6）**：它回答"FG 还能提升多少"，把"我们比 baseline 好"升级为"我们离最优有多远"。若 FG 已接近 oracle，说明方法上限已到；若差距大，给出未来工作。

### 5.5 统计规范（预注册）

| 项 | 规定 |
|---|---|
| seeds | 主格 **8**；N=2048 确认格 5；breadth 3；消融 3 |
| 主终点 | 协议 A 的 final success rate（最后 10% 步数均值），4 任务 × 6 方法 × 8 seeds |
| 模型 | 线性混合效应：J ~ method + (1\|task) + (1\|seed)；H2 另加 method×log₂N 交互项 |
| 多重比较 | 5 个预注册对比用 **Holm–Bonferroni**；其余标注 exploratory |
| 效应量 | 报告 Cohen's d、Cliff's δ、95% CI（bootstrap，10,000 次） |
| 功效 | 配对设计（task, seed）n=32，α=0.05 双侧，power=0.8 → 可检 d≈0.50；按 seed 间 σ≈5 pp 估计 ≈ **2.5 pp** |
| N=2048 格 | n=15 对 → 可检 d≈0.77 ≈ 4 pp（故只作"方向一致"证据，不单独下结论） |
| 数据 | 全部 run 进 artifact（含失败/崩溃 run），报告失败率；不做 cherry-picking |
| 排除标准 | 预注册：仅排除硬件故障（日志可证）；算法崩溃计入结果（成功率为 0） |

> **为什么 8 seeds 而不是 v1 的 5 seeds**：本地免费算力让主格多 3 个 seed 只多花 ~200 GPU-h（¥0），但把可检效应从 ~3.5 pp 降到 ~2.5 pp。这是"本地算力换统计功效"的直接收益。

### 5.6 机制指标（回答"为什么有效"，而不只是"更好"）

| 指标 | 定义 | 对应假设 |
|---|---|---|
| **分配 regret** | R_t = Σ_g \|w_g^{(t)} − w*_g\|（w* 为 oracle 水填充） | P3/H1 |
| **浪费率** | 投在 a_g=0 或 p_g>0.95 的组的样本占比 | P2/H4 |
| **饥饿事件** | 任意组连续 >5 轮 w_g=w_min 的次数 | P5 |
| **抖动** | mean_t Σ_g \|w_g^{(t)}−w_g^{(t−1)}\| | P6 |
| **响应滞后** | 人为改变分布后，w 恢复到新最优的 90% 所需轮数 | P6/P6 |
| **min-group 成功率** | 最差组终点成功率（公平性/不放弃） | P5 |
| **墙钟效率** | 协议 C 下 J(T) 与 N 的关系 | ICRA 工程相关性 |

### 5.7 真机协议（2× SO-101）

**硬件与场景（Squint 真机前提，已核实）**：2 台 SO-101 + LeRobot；**3D 打印物体**（bin 白 / can 蓝 / cube 红 / large_cube 蓝，STL 在 `deploy_utils/blender_stls`）；**黑色桌面背景 overlay**（`envs/black_overlay.png`）；相机与关节对齐调参；电机标定。**这些前提条件 v1 未计入工程量**，v2 列为 W20–W24 的显式任务。

**设计**：
- 任务：**Lift-Cube**、**Place-Can**（两个任务的 sim 训练成本最低、真机复现最稳）。
- 方法：**uniform / SGS-band / FG**（各用 1 个训练 seed 的 N=1024 checkpoint，零样本 RGB 迁移）。
- 规模：**45 trials × 3 方法 × 2 任务 = 270 trials/轮**；**2 轮独立训练复现**（不同训练 seed）→ **合计 540 trials**。
- **配对设计**：同一组 45 个场景布局（物体位姿/目标区域随机但记录）在三方法间复用 → **McNemar 检验**（比双比例 z 检验功效更高）。
- 主终点：**合并两任务后的成功率差**（FG−uniform），预注册 **≥20 pp**；**不主张 10 pp 级差异**（功效不足，见下）。
- 评分协议：成功 = 物体被抬起 ≥5 cm 并（Place 任务）落入目标区域，60 s 内完成；**双人盲评**（视频随机编号、方法隐藏），Cohen's κ ≥0.8，分歧由第三人裁定。
- 随机化与漂移控制：方法顺序按 block 随机化；每 10 trials 重新标定/复位；记录环境温度与光照（作为协变量）。
- 硬件故障：预注册"可排除但必须报告"，排除率 >20% 则整轮作废。
- **人工预算**：540 trials ≈ **27 h 执行 + 复位/校准 ≈ 5–6 人日**；3D 打印 2 人日；`deploy.py` 从 SAC 改造到 FG 推理接口 **3–5 人日**（v1 完全未计）。

**真机功效（为什么不主张 10 pp）**：

| 设计 | n/条件 | 20 pp 差（0.6 vs 0.4） | 10 pp 差（0.6 vs 0.5） |
|---|---|---|---|
| v1：30 trials 独立 | 30 | z=1.58, p≈0.11（**无功效**） | z=0.79, p≈0.43 |
| v2：45 trials 配对、合并 n=90 | 90 | McNemar 近似 **p≈0.006** | p≈0.17（仍不可测） |

→ 论文主张限定为"**≥20 pp 或排名一致性**"，Limitations 明写 "underpowered for 10 pp differences"。

### 5.8 负结果与降级路径（预注册，不做事后选择）

| 触发 | 条件 | 动作 |
|---|---|---|
| **N1 平局** | H1 中 FG−SGS-band <3 pp 且 CI 跨 0 | 转题目 B：分析型论文（"success-band 的适用边界"），FG 定位为带不确定性门控的泛化形式；不追加租卡 |
| **N2 噪声无差异** | H4 中 p_slip 的斜率差 <0.02 | 删去"不确定性"卖点，保留"导数"卖点；论文改写为"learning-progress trend vs level" |
| **N3 交互显著** | H2 交互项 p<0.05 且方向为"大 N 放大 FG" | 理论 §4.3 加入"估计精度→分配精度"反馈项；重写 P3，重跑协议 B 的 N≤4096 确认 |
| **D1 PPO-RGB 不收敛** | G0 失败（5M 步 @N=256 <25%） | Tier-2 改 **SAC + FG**（复用 Squint 的 SAC；工程量 +1.5 周）；砍 N=2048；主叙事改为"SAC 并行 RL" |
| **D2 真机不可用** | G5 失败（50 trials 试跑故障率 >40%） | 真机降级为"**定性 demo + 2 任务 × 15 trials**"，论文主证据退回仿真；如实写 Limitations |
| **D3 时间不够** | W26 时主格完成 <60% | 执行 6 页降级版：2 个 Tier-1 族 + 2 个 SO-101 任务 + 真机 2 任务 × 30 trials；删 MetaWorld 附录 |

---

## 6. 预算（本地免费 + 租卡只买大 N）

### 6.1 单价与吞吐假设（全部可复算）

| 项 | 假设 | 来源 |
|---|---|---|
| 4090 租价 | **¥2.5 / GPU-h** | 2026 年国内主流平台区间 ¥2–3；W1 询价确认 |
| A100 80GB 租价 | **¥8 / GPU-h** | 同上；仅用于 N=2048/4096 显存测试与备用 |
| Tier-1 吞吐（N=256） | 25,000 steps/s | MuJoCo Playground JAX，**W1 实测校准** |
| Tier-1 吞吐（N=1024/4096/16384） | 60k / 100k / 120k steps/s | 同上 |
| Tier-2 吞吐（RGB，N=256） | 1,500 steps/s | ManiSkill3 官方示例 + Squint 报告（SAC 1.5M 步≈15 min on 3080+） |
| Tier-2 吞吐（N=1024 / 2048） | 3,000 / 3,800 steps/s | 保守假设，**W1 实测**；v1 的 N=4096:4,000 偏乐观（M1） |
| 每 run 固定开销 | +0.05 h（启动/JAX 编译/日志） | 经验值 |
| 本机电费 | 0.3 kW × ¥0.6/kWh | 4070 Ti Super 满载 |

### 6.2 Tier-2 逐行预算（B2=15M 步，L=50）

| 行 | 设计 | runs | h/run | GPU-h | 平台 | ¥ |
|---|---|---|---|---|---|---|
| T2-00a | pilot（N=256, 3M 步） | 36 | 0.61 | 21.8 | **本地** | 0 |
| T2-00b | 显存/吞吐测试（N=1024/2048） | — | — | 3.0 | A100 | 24 |
| T2-L1 | N=64 sanity（2 任务×2 方法×5 seeds） | 20 | 6.99 | 139.9 | **本地** | 0 |
| T2-L2 | **N=256 主格**（4 任务×6 方法×8 seeds） | 192 | 2.83 | 542.9 | **本地** | 0 |
| T2-R1 | **N=1024 主格**（同上） | 192 | 1.44 | 276.3 | 4090 | 691 |
| T2-R2 | N=2048 确认格（3 任务×4 方法×5 seeds） | 60 | 1.15 | 68.8 | 4090* | 172 |
| T2-R3 | breadth（8 任务×2 方法×3 seeds @1024） | 48 | 1.44 | 69.1 | 4090 | 173 |
| T2-R4 | 消融（8 变体×3 seeds @1024） | 24 | 1.44 | 34.5 | 4090 | 86 |
| T2-R5 | 扩展基线（4 方法×3 任务×3 seeds @1024） | 36 | 1.44 | 51.8 | 4090 | 130 |
| T2-R6 | 2B 确认（3 任务×2 方法×3 seeds @1024, 30M 步） | 18 | 2.83 | 50.9 | 4090 | 127 |
| **小计** | **626 runs** | | | **1,256** | 本地 705 + 租卡 551 | **¥1,403** |
| 12% 缓冲 | 吞吐/调试不确定性 | | | 151 | | **168** |
| 电费/耗材/杂项 | 本机 1,220 h 电费（≈¥185）+ 3D 打印 + 背景板 + 线材 | | | | | **250** |
| **Tier-2 合计** | | | | | | **¥1,821** |

\* 若 N=2048 在 4090 24 GB 上 OOM（**很可能**，W1 判定），该行改 A100 80 GB：¥172 → **¥550**，总价 **¥2,199**，仍在预算内。

### 6.3 Tier-1 预算（本地，¥0，但占墙钟）

| 行 | 设计 | runs | GPU-h | ¥ |
|---|---|---|---|---|
| T1-A | 协议 A 主格（4 族×5 N×6 方法×5 seeds） | 600 | ≈414 | 0 |
| T1-B | 协议 B fixed-K（4 N≤4096×3 方法×3 seeds，K≈2,930） | 36 | ≈36 | 0 |
| T1-C | 全消融（12 变体×3 seeds×2 N） | 72 | ≈65 | 0 |
| **小计** | **708 runs** | | **≈515** | **0** |

**本地总负载**：Tier-1 515 + Tier-2 705 = **1,220 GPU-h**，摊到 45 周 = **27.1 h/周**（本机 16% 占空比）。**这是可行但需要 24/7 待机**；若本机不能长期满载，v2 的降级方案是把 T1-A/N=256 与 T2-L1 各砍 30%（省 ~290 h，结论不变，只少一个低 N 点）。

### 6.4 敏感性分析与 5000 硬上限保底

| 情景 | 触发 | 租卡（含 12% 缓冲） | 其他 | 合计 | ≤5,000 |
|---|---|---|---|---|---|
| **基准** | 单价/吞吐符合假设，N=2048 走 4090 | ¥1,571 | 250 | **¥1,821** | ✓（余 3,179） |
| **单因素 +50%** | 单价 >3.75 或吞吐 <2/3 | ¥2,357 | 250 | **¥2,607** | ✓ |
| **A100 情景** | N=2048 必须 A100 + 单因素 +50% | ¥2,993 | 250 | **¥3,243** | ✓ |
| **双因素 2.25×** | 单价与吞吐同时恶化，N=2048 走 A100 | ¥4,489 | 250 | **¥4,739** | ✓（余 261） |
| **FLOOR 保底** | 双因素 + 最小可发表（4 任务×6 方法×8 seeds×{256,1024}+消融+2B） | ¥2,339 | 250 | **¥2,589** | ✓（安全系数 1.93） |

**结论**：**5000 元不是约束**（最坏情景仍余 ¥261），**人工与墙钟才是**。因此 v2 把省下的钱用于：seeds 5→8（+200 GPU-h 本地，¥0）、N=2048 档（¥172–550）、SGS baseline 与扩展基线（¥130）、以及 ~¥3,000 的意外缓冲（临时补跑/视频渲染/真机耗材）。**不为了花预算而加实验**。

### 6.5 人力与资源声明（写进论文 artifact 附录）

- **人力**：1 名研究生 × 45 周 ≈ **11 人月**（本项目未计人力成本）；
- **算力**：本地 1,220 GPU-h + 租卡 551 GPU-h ≈ **1,771 GPU-h** 总计；
- **真机**：540 trials，2 台 SO-101，耗材 ¥250 内；
- **预算合规**：租卡支出上界 **¥4,739**（双因素情景），低于 5,000 硬上限；基准 ¥1,821。

---

## 7. 时间线（W1–W45，2026-11-02 → 2027-09-15）

| 周 | 任务 | 交付物 | Gate |
|---|---|---|---|
| W1 | 环境落地：Playground / ManiSkill3+Squint / MetaWorld；核实 4 个 SO-101 env_id（`-v1`）、obs 是否含 goal | 环境清单 + 版本锁文件 | |
| W2 | **G0 pilot**：吞吐/显存/单任务收敛（5M 步 @N=256 在 Lift-Cube 上 ≥25% 成功率） | 吞吐表、显存曲线、pilot 曲线 | **G0** |
| W3–W4 | Tier-1 构造族 A/B/C/D 实现；FG 超参 pilot（族 A 单任务） | 分组代码、超参初值 | |
| W5 | 验证族 A 的真 LP 异质性（max/min a_g ≥3） | 异质性报告 | **G1** |
| W5–W8 | Tier-1 协议 A 主格（600 runs，本地） | Fig.3 数据 | |
| **W8** | **预注册冻结**（RQ/H/主终点/统计/排除/负结果分支） | 预注册文档（OSF 或仓库） | **G2** |
| W8–W11 | Tier-1 协议 B（fixed-K，36 runs，本地） | 更新饥饿对照 | |
| W9–W10 | Tier-2 pilot（4 任务 @N=256 本地，含目标区域分组可行性） | Tier-2 分组实现 | |
| W10–W14 | Tier-2 **N=256 主格**（192 runs，本地 543 h） | 主结果表 | **G3（W14，≥50%）** |
| W12–W18 | 租卡：Tier-2 **N=1024 主格**（192 runs，276 h，分 2 批） | 主结果图 | |
| W16–W20 | 租卡：N=2048 确认格（60 runs）+ breadth（48）+ 消融（24）+ 扩展基线（36）+ 2B（18） | 全部主图数据 | **G4（W20）** |
| W18–W24 | MetaWorld MT10 附录（本地 CPU） | 附录图 | |
| W20–W24 | 真机准备：3D 打印（2 人日）、Squint `deploy.py`→FG 推理接口（3–5 人日）、相机对齐/标定、50 trials 试跑 | 可用真机管线 | **G5（W24）** |
| W24–W29 | 真机主实验（540 trials，约 6 人日） | Fig.5 数据 + 视频素材 | |
| **W30** | **实验冻结**（不再改代码/超参/分组） | 冻结 tag + 数据快照 | **G6** |
| W31–W38 | 写作（含图表、附录、artifact） | 全文初稿 | **G7（W38）** |
| W37 | 内部评审（导师 + 同组 2 人） | 评审意见 | |
| W39–W40 | 图表精修、附录、artifact 打包、**投稿日历核对** | camera-ready 级 PDF | |
| W40 | 视频拍摄与剪辑（90 s + 3 min 两版） | 视频文件 | |
| **W41** | **投稿**（≈2027-08-09） | 提交回执 | |
| W42–W45 | 缓冲：补实验/修格式/开源仓库 + **投稿日历**（abstract 注册、全文、video 三截止，各留 ≥3 天；日期**待核实**） | 开源仓库、备投 CoRL/IROS 决策 | |

**缓冲设计**：实验冻结 W30（v1 是 W34），比正文截止 W41 早 **11 周**；比预计 ICRA 2028 截止（~2027-09-15，待核实）早 **5 周**。任何 1 周滑期都不会致命。

**投稿日历（占位，W20 前核实官方日期）**：
| 事件 | 预计时间 | 本方案内部截止 |
|---|---|---|
| 摘要/标题注册 | 2027-09-01 前后 | W40（08-30） |
| 全文截止 | 2027-09-15 前后 | **W41（08-09）** |
| 视频截止 | 2027-09-20 前后 | W40（08-30，已完成） |

---

## 8. 风险表（R1–R13）与 Plan B/C

| # | 风险 | 概率 | 影响 | 触发信号 | 缓解 / Plan B |
|---|---|---|---|---|---|
| R1 | **SGS 压制 novelty** | 高 | 致命 | 审稿意见出现"incremental over SGS" | §1.6 三句话差异 + 同层 head-to-head + 边界判据；预注册负结果分支 |
| R2 | PPO+RGB 在 ManiSkill3 不收敛 | 中 | 高 | G0 失败 | 改 SAC+FG（+1.5 周）；砍 N=2048 |
| R3 | N=2048 显存不足 | 高 | 中 | W1 显存测试 | 4090→A100（¥+378）；或砍该行（结论仍成立） |
| R4 | 视觉吞吐远低于假设 | 中 | 中 | W1 实测 <2/3 假设 | 触发单因素情景（仍闭合）；砍 breadth seeds |
| R5 | Tier-2 分组不可行（obs 无 goal） | 中 | 高 | W1 核实失败 | **Fallback：纯 DR 桶 M=4**（预注册） |
| R6 | 真机零样本迁移差 | 中高 | 高 | G5 试跑成功率 <10% | 降级为"定性 demo + 15 trials"；或加 1 天 sim-to-real 微调（在真机上 200 步 BC 校准，明确标注） |
| R7 | 真机硬件故障（SO-101 舵机/相机） | 中 | 中 | 试跑故障率 >20% | 备用舵机 + 第二台 SO-101；预注册排除标准 |
| R8 | 本地机器不可 24/7 | 中 | 中 | 周均 <20 h 可用 | 砍 T1-A/N=256 与 T2-L1 各 30%；或临时租 4090 跑 N=256（+¥300） |
| R9 | 排期滑期 | 高 | 高 | W26 主格 <60% | 6 页降级版（D3） |
| R10 | 负结果（FG≈SGS-band） | 中 | 中 | H1 差 <3 pp | 转分析型论文（§5.8 N1）；不追加拿卡 |
| R11 | 统计功效不足 | 低 | 中 | 8 seeds 的 CI 过宽 | 主终点用混合模型 + 配对；报告效应量而非仅 p 值 |
| R12 | 真机评分主观性 | 中 | 中 | κ<0.8 | 双盲 + 第三人裁定；预注册 rubric |
| R13 | MetaWorld/CPU 并行度不足 | 中 | 低 | N=128 跑不动 | 降到 N=64；附录只做 MT10 不做 MT50 |

**Plan C（最坏情况）**：若 W26 时 Tier-2 与真机都未成型，论文降为"**Tier-1 的 4 族 × N≤16k scaling + 理论判据 + 2 任务仿真真机视频**"，投 ICRA 的 **short paper / workshop**，并把完整版留给 CoRL 2027（2027-06 截止，待核实）。

---

## 9. 与近邻工作的差异表（SGS 首行）

| # | 工作 | 分配单元 | 信号 | 不确定性 | guard | 理论 | N 规模 | 真机 | 代码 | 与我方关系 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **SGS** 2610.12465 (CoRL 2026) | 单任务 task-config | success-band 三档 | ✗ | ✗ | ✗ | 2^20 | ✓（RGB 蒸馏） | ✗（coming soon） | **最近邻：同层，本文是其"加导数+不确定性+判据"的推广** |
| 1 | SER 2511.21011 | 轨迹/时序 | 重采样频率 | ✗ | ✗ | 部分 | 中 | ✗ | 待核实 | 正交，可组合（扩展基线 9） |
| 2 | DRATS 2605.14350 | 任务 | 难度/自适应 | ✗ | ✗ | ✗ | 小 | ✗ | 待核实 | 任务级；本文配置级 + 附录任务级 |
| 3 | PGB 2610.25704 | prompt/组 | bandit | 部分 | ✗ | ✓（bandit regret） | — | ✗ | 待核实 | LLM 域；本文是物理配置 + 并行 env |
| 4 | UPER 2506.09270 | 任务 | 回报/未解决 | ✗ | ✗ | ✗ | 中 | ✓ | 待核实 | 启发式；无 scaling 预测 |
| 5 | GMC 2605.05856 | 任务 | 可达性筛选 | ✗ | ✗ | ✗ | 小 | ✗ | 待核实 | 课程排序 ≠ 配额分配 |
| 6 | LPM 2509.25438 | 任务 | 学习进度 | ✗ | ✗ | ✗ | 小 | ✗ | 待核实 | 用 LP 做排序，不做并行配额 |
| 7 | PLR 2010.03934 | level | regret/value | ✗ | ✗ | ✓ | 中 | ✗ | ✓ | 生成式 level；本文固定桶，开销低 |
| 8 | RED 2110.02439 | level | value 优先级 | ✗ | ✗ | ✓ | 中 | ✓ | ✓ | 同上 |
| 9 | 2506.03404 | 待核实 | 待核实 | — | — | — | — | — | — | 需在 Related Work 引用并标注 |
| 10 | 2507.23172 MTBench | 任务 | benchmark | — | — | — | — | ✓ | ✓ | 基准，非方法；本文**不用**（IsaacGym Preview 4 风险） |
| 11 | 2610.33420 | 待核实 | 待核实 | — | — | — | — | — | — | 需核对后引用 |
| 12 | 2603.01741 | 待核实 | 待核实 | — | — | — | — | — | — | 需核对后引用 |
| 13 | Squint 2602.21203 (RA-L 2026) | — | — | — | — | — | — | ✓ | ✓ | **平台**（SO-101 训练/部署），非竞争方法 |
| 14 | SO-101 Bench 2610.36420 | — | — | — | — | — | — | — | 待核实 | VLA 评测基准，**不作训练环境** |
| 15 | LeRobot 2602.22818 | — | — | — | — | — | — | ✓ | ✓ | 硬件/软件栈 |

> **写作要求**：表中所有"待核实"项必须在 W2 前用 arXiv API 核实标题/作者/方法要点；**凡无法核实的，宁可不引用，也不许编造**（v1 的 RPLR 错标就是教训）。

---

## 10. 论文结构、图表、视频、开源

### 10.1 6 页正文分配（ICRA 双栏）

| 部分 | 页数 | 内容要点 |
|---|---|---|
| 1 Introduction | 1.0 | uniform 的浪费（图 1）；SGS 已解决一半；本文 sharper question + 3 条贡献 |
| 2 Related | 0.5 | SGS/DRATS/SER/PLR 四条线，一段一个；**差异表放 Table 1** |
| 3 Method | 1.5 | 形式化 + group 构造 + LP 估计 + Hedge + guards + 伪代码（Alg.1） |
| 4 Theory | 0.5 | 可学性模型 + 判据（§4.4）+ P1–P6 表 |
| 5 Experiments | 2.0 | 主结果（Fig.3）、fixed-K 控制（Fig.4a）、噪声退化（Fig.4b）、消融（Table 3）、真机（Fig.5） |
| 6 Conclusion | 0.25 | 边界 + 负结果分支 + 未来工作（任务级扩展） |

### 10.2 图表计划

| 图/表 | 内容 | 数据来源 |
|---|---|---|
| **Fig.1** | 概念图：uniform 把 env 花在"已掌握/不可学"桶上 vs FG 的重新分配 | 示意 + pilot 真实数据叠加 |
| **Fig.2** | 方法框图（PPO 与 FG 的接口，强调 plug-in） | — |
| **Fig.3** | Scaling：J vs N（4 个 Tier-1 族，4 方法，8 seeds，均值±95%CI） | T1-A |
| **Fig.4a** | fixed-K 控制：同图但 B∝N | T1-B |
| **Fig.4b** | 噪声退化：J vs p_slip（SGS-band 下坡、FG 平） | T1 族 B |
| **Fig.5** | 真机：2 任务成功率 + 配对 CI | 真机 540 trials |
| **Table 1** | 近邻差异表（§9 的精简版） | 文献 |
| **Table 2** | Tier-2 主结果（4 任务 × 6 方法 × 3 N） | T2-L2/R1/R2 |
| **Table 3** | 消融 A1–A10 | T1-C/T2-R4 |
| 附录 | MetaWorld MT10、超参表、SGS 重实现细节、预算审计 | — |

### 10.3 视频（ICRA 要求，待核实具体时长上限）

- **90 s 版**（主）：0–15 s 问题（uniform 浪费的可视化）；15–45 s 方法（FG 分配动画 + 伪代码）；45–75 s 结果（scaling 曲线 + 噪声实验）；75–90 s 真机。
- **3 min 版**（备用）：加 fixed-K 与消融。
- **拍摄素材**：真机 540 trials 中预选 20 段成功/失败对比；仿真渲染 4 段。

### 10.4 开源与 artifact

- 仓库结构：`fg/`（分配器）、`configs/`（全部 run YAML）、`scripts/`（复现脚本）、`results/`（raw logs + 聚合 JSON）、`sgs_reimpl/`（SGS-band 重实现）、`robot/`（SO-101 部署）。
- **release 时点**：投稿同日匿名化版本（Anonymous.4open.science）；中稿后完整版。
- 复现命令：`python -m fg.train --task PandaPickCube --N 1024 --B 150M --seed 0 --method fg` 等（见 §12）。

---

## 11. 中稿率策略与审稿人预案（rebuttal bank）

### 11.1 中稿率估计（**估计值，非事实**）

| 版本 | 我估计的接收概率 | 依据 |
|---|---|---|
| v1 原案 | 25–35% | novelty 被 SGS 压制 + Tier-2 方法硬伤（C9）+ 真机功效不足 |
| **v2（本方案）** | **35–45%** | 同层 head-to-head + 可证伪判据 + 8 seeds + 真机 45 trials + 诚实边界；仍受"方法增量"与"单一 embodiment"拖累 |
| v2 + 真机 10 pp 显著 | 45–55% | 需 ≥90 trials/条件，时间不允许 |

> ICRA 近年录取率约 35%–42%（**估计，以官方统计为准**；v1 写的"40%"没有出处，v2 改为区间 + 标注）。

### 11.2 Rebuttal bank（8 条预写）

1. **「Incremental over SGS」** → 我们给出 SGS 失效的**解析条件**（§4.4 两个不等式）+ 构造性实验（族 B 的 p_slip 扫描）+ 同预算 head-to-head；且 SGS 的 success-band 是我们的特例（§1.6）。
2. **「Only one real robot / two tasks」** → 承认；用 540 trials 配对设计把可检效应做到 20 pp，并在 Limitations 明写；仿真侧 4+8 任务弥补广度。
3. **「Why not任务级?」** → C9 的合法性论证：无 task ID 的共享策略不成立；MetaWorld MT10 附录提供任务级证据；自建 task-conditioned env 需 3–4 周，成本/风险不成比例。
4. **「Guards look ad-hoc」** → 每个 guard 有对应失效模式 + 消融（Table 3）；n_min=30 由二项方差推出（§3.5）。
5. **「Hyperparameters tuned on test tasks」** → 冻结在 W8，只在族 A 上 pilot 调参；主格从未用于调参；预注册文档公开。
6. **「Fixed-B saturates because of update starvation」** → 这正是我们加协议 B 的原因；fixed-K 结果单独报告（Fig.4a）。
7. **「Reward-agnostic claim?」** → 我们不声称 reward-agnostic；方法只用成功率与 LP，属于"outcome-based"；对稀疏奖励任务需先用 demo/curriculum 启动（Limitations）。
8. **「Reproducibility of SGS comparison」** → SGS 无代码，我们提供重实现并公开全部超参与日志；若原作者发布代码，我们承诺在 camera-ready 前对齐。

---

## 12. 实验运行清单（run matrix）

### 12.1 Tier-1（本地，MuJoCo Playground）

| Run ID | 任务 | N | B | 方法 | seeds | 数量 | 预计 GPU-h |
|---|---|---|---|---|---|---|---|
| T1-A-{task}-N{n}-{m}-s{k} | 4 族 | {256,1024,4096,8192,16384} | 150M | 6 | 5 | 600 | 414 |
| T1-B-{task}-N{n}-{m}-s{k} | 族 A/B | {256,1024,2048,4096} | K≈2,930·N·L | 3 | 3 | 36 | 36 |
| T1-C-{variant}-s{k} | 族 A | 1024 | 150M | 12 变体 | 3 | 72 | 65 |

### 12.2 Tier-2（本地 N=256 / 租卡 N≥1024）

| Run ID | 任务 | N | B | 方法 | seeds | 数量 | 平台 |
|---|---|---|---|---|---|---|---|
| T2-L2-{task}-N256-{m}-s{k} | 4 | 256 | 15M | 6 | 8 | 192 | 本地 |
| T2-R1-{task}-N1024-{m}-s{k} | 4 | 1024 | 15M | 6 | 8 | 192 | 4090 |
| T2-R2-{task}-N2048-{m}-s{k} | 3 | 2048 | 15M | 4 | 5 | 60 | 4090/A100 |
| T2-R3-{task}-N1024-{m}-s{k} | 8 | 1024 | 15M | 2 | 3 | 48 | 4090 |
| T2-R4-{variant}-s{k} | 1 | 1024 | 15M | 8 变体 | 3 | 24 | 4090 |
| T2-R5-{method}-{task}-s{k} | 3 | 1024 | 15M | 4 扩展 | 3 | 36 | 4090 |
| T2-R6-{task}-N1024-{m}-s{k} | 3 | 1024 | 30M | 2 | 3 | 18 | 4090 |

### 12.3 命令样例（env 名已合法化，`-v1` 已修正）

```bash
# Tier-1（本地）
python -m fg.train --suite playground --task PandaPickCube --N 1024 --B 150e6 \
  --method fg --seed 0 --groupId goal_x_mass --M 8 --logdir runs/T1-A/PandaPickCube/N1024/fg/s0

# Tier-2（本地 N=256；Squint env 名以仓库为准，-v1）
python -m fg.train --suite squint --task SO101LiftCube-v1 --N 256 --B 15e6 \
  --method fg --seed 0 --groupId goal_region_x_dr --M 8 --logdir runs/T2-L2/lift/N256/fg/s0

# Tier-2（租卡 N=1024）
python -m fg.train --suite squint --task SO101LiftCube-v1 --N 1024 --B 15e6 \
  --method sgs_band --seed 0 --logdir runs/T2-R1/lift/N1024/sgs/s0

# Fixed-K 协议（Tier-1）
python -m fg.train --suite playground --task CheetahRun --N 1024 --K 2000 --L 50 \
  --method fg --seed 0 --logdir runs/T1-B/cheetah/N1024/fg/s0
```

> **注**：`--method` 取值 `{uniform, ser, sgs_band, plr_red, drats_cfg, fg,...}`；`--B` 与 `--K` 二选一（互斥校验）。

### 12.4 运行总数核对

- Tier-1：600 + 36 + 72 = **708 runs**（本地）
- Tier-2：36 + 20 + 192 + 192 + 60 + 48 + 24 + 36 + 18 = **626 runs**（v1 是 661，v2 更精简且设计正确）
- 合计 **1,334 runs**；总 GPU-h ≈ 1,220（本地）+ 551（租卡）≈ **1,771**

---

## 13. 决策点（D1–D9）与 Go/No-Go

| ID | 时点 | 决策 | 依据 |
|---|---|---|---|
| D1 | W2（G0） | 是否继续 PPO-RGB，还是改 SAC | 5M 步 @N=256 ≥25% |
| D2 | W5（G1） | 族 A 的 LP 异质性是否足够（≥3×） | 若不足，换更强的构造（目标距离跨度加大） |
| D3 | W8（G2） | 预注册冻结 | 冻结后不改主终点 |
| D4 | W14（G3） | 是否租卡跑 N=1024（还是延后） | N=256 主格完成度 |
| D5 | W20（G4） | 主结果方向是否支持 H1；是否触发负结果分支 | 效应量与 CI |
| D6 | W24（G5） | 真机是否可用；是否降级 | 50 trials 试跑 |
| D7 | W26 | 是否执行 6 页降级版 | 主格完成度 <60% |
| D8 | W30（G6） | 实验冻结 | — |
| D9 | W1 | **obs 是否含 goal → 决定 Tier-2 分组首选/fallback**（C9 的关键判定） | 代码核查 |

---

## 附录 A：参考文献（真实 + 标注）

> 规则：只列**已核实**的条目；未核实的标 "待核实"，且不得出现在正文引用编号中（宁缺毋滥）。

1. **SGS**: Zhang, O., Guaman Castro, M., Yin, P., Dagnino, I., Gupta, A., Scalise, R., Boots, B. *A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control.* arXiv:2610.12465, 2026. (CoRL 2026; 项目页 https://sgs-rl.github.io/)
2. **Squint**: Almuzairee, A., Christensen, H. I. *Squint: ...* IEEE RA-L, 2026. doi:10.1109/LRA.2026.3730387（arXiv:2602.21203）
3. **SO-101 Bench**: Hickok, T. *SO-101 Bench: Accessible Spatial Manipulation.* arXiv:2610.36420, 2026.
4. **SER**: arXiv:2511.21011（待核实完整作者/标题）
5. **DRATS**: arXiv:2605.14350（待核实）
6. **PGB**: arXiv:2610.25704（待核实）
7. **UPER**: arXiv:2506.09270（待核实）
8. **GMC**: arXiv:2605.05856（待核实）
9. **LPM**: arXiv:2509.25438（待核实）
10. **PLR**: Jiang, M., Grefenstette, E., Rocktäschel, T. *Prioritized Level Replay.* ICML 2021. arXiv:2010.03934.
11. **RED**: Jiang, M., et al. *Replay-Guided Adversarial Environment Design.* NeurIPS 2021. arXiv:2110.02439. **（v1 误标为 RPLR，v2 已修正）**
12. **RPLR**: 未能核实到 arXiv ID——**v2 不引用**（如需引用须先核实，不得编造）。
13. **MTBench**: arXiv:2507.23172（待核实；v2 不使用，仅 Related Work 提一句）
14. **LeRobot**: arXiv:2602.22818（待核实）
15. 其他 Brief 4.1 中 2506.03404 / 2610.33420 / 2603.01741：**待核实**，核实前不引用。

## 附录 B：对 Brief 第 5 节 14 项清单的逐项对照

| # | 清单项 | v2 状态 | 位置 |
|---|---|---|---|
| 1 | 英文题目/摘要/关键词 + 中文摘要 | ✅ 189 词（≤200） | §1 |
| 2 | RQ1–RQ4 + 可证伪假设 H1–H8 + 预注册 | ✅ 含负结果分支 | §2 |
| 3 | 方法形式化 | ✅ 单一 group 定义 + 超参默认表 | §3 |
| 4 | 理论/分析 | ✅ 可学性模型 + 判据 + P1–P6 | §4 |
| 5 | 实验矩阵（三层/固定预算/基线/消融/统计） | ✅ 三协议 + 6 主 4 扩展基线 + A1–A10 + 8 seeds | §5 |
| 6 | 真机实验 | ✅ 45 trials×3 方法×2 任务×2 轮，McNemar，盲评 | §5.7 |
| 7 | 预算表（≤5000 + 敏感性 + 砍单） | ✅ 本地/租卡重划，四情景闭合，¥1,821 基准 | §6 |
| 8 | 时间线 + Go/No-Go | ✅ W1–W45 + G0–G7，冻结 W30 | §7 |
| 9 | 风险表 + Plan B | ✅ R1–R13 + Plan C | §8 |
| 10 | 近邻差异表 | ✅ SGS 首行 + 15 行 | §9 |
| 11 | 论文结构/图表/视频/开源 | ✅ 6 页分配 + 图表 + 90s/3min 视频 + artifact | §10 |
| 12 | 中稿率提升策略 | ✅ 35–45%（估计）+ rebuttal bank 8 条 | §11 |
| 13 | 实验运行清单 | ✅ 1,334 runs，env 名合法，`-v1` | §12 |
| 14 | 开放问题/决策点 | ✅ D1–D9 | §13 |
| 15 | **新增**：SGS 专项对照 + 预算审计脚本 | ✅ 附录 C/D | 附录 |

## 附录 C：SGS 专项逐维对照

| 维度 | SGS（2610.12465） | FG（本文） | 差异是否可测 |
|---|---|---|---|
| 分配单元 | 单任务内的 task configuration | 单任务内的配置桶（目标区域×DR） | 同层 → **可 head-to-head** |
| 信号 | success-rate 三档（never/improving/mastered） | 归一化 LP 的导数 × epistemic 门控 | 可测（A1/A2/A3 消融） |
| 不确定性 | 无 | aleatoric/epistemic 分解 | 可测（族 B 的 p_slip 扫描） |
| 噪声鲁棒性 | 无保护 | guard-1 noise + guard-3 信任域 | 可测（H4） |
| 理论 | 无 | 可学性模型 + 判据 + P1–P6 | 可证伪 |
| 规模 | 2^20 env | N≤16,384（单卡） | **我方弱项，必须如实写** |
| 机器人 | 四足 + 装配，RGB 蒸馏真机 | 4 SO-101 任务 + 真机 540 trials | 方向不同，各有证据 |
| 代码 | 未开源 | 全开源 + SGS 重实现 | 我方优势 |

## 附录 D：预算审计脚本（可直接运行）

```python
THRU = {64: 600., 256: 1500., 1024: 3000., 2048: 3800.}
PRICE = {"local": 0.0, "4090": 2.5, "A100": 8.0}
OV = 0.05  # 每 run 固定开销 (h)

def hours(B, N):
    return B / THRU[N] / 3600 + OV

rows = [
    ("T2-L2", 4*6*8, 15e6, 256, "local"),
    ("T2-R1", 4*6*8, 15e6, 1024, "4090"),
    ("T2-R2", 3*4*5, 15e6, 2048, "4090"),
    ("T2-R3", 8*2*3, 15e6, 1024, "4090"),
    ("T2-R4", 8*3,   15e6, 1024, "4090"),
    ("T2-R5", 4*3*3, 15e6, 1024, "4090"),
    ("T2-R6", 3*2*3, 30e6, 1024, "4090"),
]
tot = 0.0
for name, runs, B, N, plat in rows:
    h = runs * hours(B, N); cost = h * PRICE[plat]
    print(f"{name}: {runs} runs, {h:8.1f} GPU-h, ¥{cost:7.0f}")
    tot += cost
print(f"租卡小计 ¥{tot:.0f}；含 12% 缓冲 ¥{tot*1.12:.0f}；+杂项 ¥250 → ¥{tot*1.12+250:.0f}")
# 期望输出：租卡小计 ¥1403；含缓冲 ¥1571；+杂项 → ¥1821
```

---

## 版本与变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-11-02 | 初稿（`01_proposal_v1.md`） |
| **v2** | 2026-11-02 | 本文件。修 C1–C9、M0–M18、m1–m12；Tier-2 重构为 config 级；预算重划为 ¥1,821；8 seeds；真机 540 trials；排期冻结 W30 |
| v2 → v3 待办 | — | ①W1 核实 4 个 SO-101 env_id 与 obs goal 字段（D9）；②W2 实测吞吐/显存替换 §6.1 假设；③W2 前核实附录 A 全部"待核实"条目；④W20 前核实 ICRA 2028 官方截止日期并更新投稿日历 |
