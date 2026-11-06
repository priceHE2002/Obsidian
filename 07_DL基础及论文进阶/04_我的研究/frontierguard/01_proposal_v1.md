# FrontierGuard-RL：面向可扩展并行强化学习的不确定性感知学习进度采样
## ICRA 2028 实验方案（v1，2026-11-02）

> 撰写者：proposal-writer。本文件是**实验方案**，不是论文；文中所有效果均为 **hypothesis（假设）** 或 **expected result（预期）**，不存在任何 actual result（已获得结果）。
> 事实分级标注约定：**【事实】**=有 arXiv/官方仓库/官网依据；**【假设】**=工程判断，可由 pilot 校准；**【待核实】**=需用户确认或购买前复核。

---

## 0. 时间校正：ICRA 2027 已不可能，本方案按 ICRA 2028 排期

**【事实】** ICRA 2027 官网（https://2027.ieee-icra.org/）Call for Papers 页面标注 "Submission Closed"；论文提交约在 2026 年 9 月中旬，视频提交期为 2026-09-18～22（官网 2026-09-17 公告），会议在首尔。**今天是 2026-11-02，ICRA 2027 的投稿窗口已经关闭，任何"投 ICRA 2027"的计划都不成立。**

因此本方案的目标会议改为 **ICRA 2028**：按 ICRA 2026/2027 连续两届的规律，预计 **2027 年 9 月中旬截稿**（**【待核实】** 以 2028 官网 CFP 为准，通常在前一年 12 月至当年 1 月上线）。主排期按 **2026-11-02 → 2027-09-15，共 45 周**设计，见第 7 节。

**对冲选项（optional，不得干扰主排期）：**

| 选项 | 预计截稿 | 优点 | 代价/风险 | 建议 |
|---|---|---|---|---|
| **IROS 2027** | 约 2027-03（**【待核实】**） | 比 ICRA 2028 早 6 个月拿到"已发表"；IROS 接受系统/应用型工作 | 需在第 19 周就有完整故事；会分掉第 14–19 周的写作与实验时间 | **建议作为"半程对冲"**：若第 14 周 Go/No-Go 显示 Tier-1+Tier-2 已能闭环回答 RQ1–RQ3，则投 IROS 2027 的 6 页版本；否则放弃，不追 |
| **CoRL 2027** | 约 2027-06（**【待核实】**） | CoRL 更看重 RL 方法本身，对"机器人相关性"要求低于 ICRA；与本文 framing 天然契合 | CoRL 对 novelty 与 baseline 强度的要求不低于 ICRA；6 月截稿会挤压 Tier-3 与真机 | **建议作为"备选主投"**：若 ICRA 评审画像显示"方法偏 RL、机器人贡献弱"，改投 CoRL 更合身；二者只选其一，不能一稿两投 |
| RA-L | 滚动 | 周期短（~3 个月首轮） | 页数少（6 页），放不下 3 层实验 + 真机 | 不建议作为主投，可作为 ICRA 被拒后的转投 |

**给用户的一句话结论**：ICRA 2027 已经赶不上；本方案按 ICRA 2028（2027-09 中旬）自洽设计，并在第 14 周设一个"是否顺手投 IROS 2027"的决策门。若用户的实际目标是"尽快发表"，请在第 13 节决策点 D1 明确选择。

---

## 1. 题目、摘要、关键词、一句话卖点

### 1.1 英文题目
**FrontierGuard: Uncertainty-Aware Learning-Progress Sampling for Scalable Parallel Reinforcement Learning**

### 1.2 中文题目
**FrontierGuard：面向可扩展并行强化学习的不确定性感知学习进度采样**

### 1.3 一句话卖点（one-sentence pitch）
> 在固定交互预算下，把并行环境这份"经验预算"从"平均分给所有环境"改成"跟着学习前沿走、并用不确定性守卫防止被噪声骗走"，从而让增加并行环境数 N 从"只是更快"变成"确实更好"。

### 1.4 英文摘要（draft，投稿前再压缩到 150–200 词）
Massively parallel simulators have made data collection cheap, yet at a fixed interaction budget more environments do not automatically produce better policies: uniform allocation keeps spending samples on environments that are already solved or fundamentally unlearnable, and with short rollouts the composition of each update becomes increasingly stale as the environment count grows. This paper asks whether *allocation*, rather than throughput, is the missing lever for scalable on-policy RL. We propose **FrontierGuard**, a drop-in sampler for PPO that (i) partitions N parallel environments into M instance/config groups (difficulty buckets of initial states, object poses, and goal regions inside a task, all sharing one policy), (ii) estimates group-level learning progress with a two-timescale, saturation-corrected success-rate estimator, (iii) separates epistemic uncertainty from aleatoric noise using a lightweight ensemble of outcome heads, and (iv) allocates environments with a constrained EXP3/UCB bandit under three guards — noisy-TV, collapse, and confidence. FrontierGuard is a ~150-line hook and composes with staggered environment resets. We give a toy analysis that predicts *when* adaptive allocation helps and derive four falsifiable scaling predictions, then test them in pre-registered, compute-matched experiments on MuJoCo Playground (N up to 16,384), ManiSkill3 with the SO-101 Task Set (N up to 4,096), and two real SO-101 arms. We expect gains to concentrate in the high-N / short-rollout / heterogeneous-difficulty regime, and we will report negative results for the regimes where they do not appear.

### 1.5 中文摘要
大规模并行仿真让采样变得廉价，但在固定交互预算下，环境数 N 的增加并不会自动带来更好的策略：均匀分配会持续把样本花在"已经学会"或"根本学不会"的环境上；而当 rollout 很短时，N 越大，每次更新所看到的数据构成越"陈旧"、越偏。本文追问：**分配**（而非吞吐）是不是可扩展 on-policy RL 缺失的那根杠杆？我们提出 **FrontierGuard**——一个可直接插在 PPO 上的采样器：把 N 个并行环境划分为 M 个实例/配置 group（同一任务内的初始状态、物体位姿、目标区域等难度桶）；用双时间尺度、去饱和的成功率估计器估计 group 级学习进度（LP）；用轻量结果预测器 ensemble 把 epistemic 不确定性与 aleatoric 噪声分开；在 noisy-TV、塌缩、置信度三重 guard 约束下，用 EXP3/UCB bandit 分配环境；与错峰 reset（SER）可组合。方法约 150 行钩子。我们给出 toy 分析，预测"何时自适应分配有用"，导出四条可证伪的 scaling 预测，并界定"单任务 success-band 采样（同期 balanced data diet 工作的机制）何时足够、何时必须做配置级不确定性感知分配"，在预注册、计算量匹配的三层实验中检验：MuJoCo Playground（N 至 16,384）、ManiSkill3 + SO-101 Task Set（N 至 4,096）、两台真机 SO-101。我们**预期**增益集中在"高 N / 短 rollout / 难度异质"区域，并会**如实报告**不出现增益的区域。

### 1.6 关键词
scalable reinforcement learning; curriculum learning; exploration; uncertainty estimation; sample efficiency; massively parallel simulation; sim-to-real

---

## 2. 研究问题（RQ1–RQ5）与可证伪假设

### 2.1 核心命题
把"可扩展性"定义为：**在固定交互预算 B（env-steps）与固定计算预算（GPU-hours）下，最终策略性能随 N 单调改善的程度**；而不是吞吐量。由此把用户原始问题拆成五个可实验回答的子问题。

| RQ | 问题 | 主要实验 | 主要指标 | 判定"是"的门槛（预注册） |
|---|---|---|---|---|
| **RQ1** | 固定 B 下，均匀/随机经验分配的样本效率是否随 N 增大而饱和或退化？在什么 N 出现拐点？ | Tier-1 全 N 扫描 + Tier-2 4 独立任务（诊断） | 最终成功率/归一化回报；AUC；机制指标（组覆盖、已解组浪费率、LP-SNR） | 在 ≥2 个任务上 PPO-uniform 的 final-perf(N) 曲线在 N≥1024 后出现统计显著平台或下降（配对检验 p<0.05） |
| **RQ2** | FrontierGuard 能否让 scaling 曲线在更大 N 上继续改善？增益随 N 如何变化？ | Tier-1/2/3 主矩阵 | Δ = FG − PPO-uniform 在固定 B 的差值；增益–N 曲线斜率；N* 位移 | FG 在 2 个以上任务、N≥1024 上显著优于 PPO-uniform（Holm 校正后 p<0.05，且效应量 ≥3 个百分点成功率或 ≥5% 归一化回报） |
| **RQ3** | 增益是否来自"分配"本身，而不是"更多探索/更大 batch/更多算力"？ | 消融 + 计算量匹配 + 机制诊断 + 合成噪声组实验 | 固定 B 与固定 GPU-h 两条协议下的差值；消融落差；LP 信号与噪声的分离度 | 去掉 bandit 分配（退回均匀）后增益消失 ≥70%；在 GPU-hour 匹配下仍显著；不确定性/guard 的消融各自造成可测退化 |
| **RQ4** | 在 SO-101 真机上，sim 训练得到的采样优势是否迁移（零样本 + 少量真机数据）？ | 真机评测（2 任务 × 3 方法 × 30 trials） | 真机成功率 + Wilson 95% CI；sim 排名与真机排名的一致性（Spearman） | FG 训练策略在真机零样本成功率高于 PPO-uniform，且 95% CI 不重叠或配对检验 p<0.1（真机样本量有限，故放宽） |
| **RQ5（v1 追加）** | 在什么条件下"单任务内 task-config 级 success-band 采样"（SGS 式）已经足够，什么条件下必须用"配置级 LP × epistemic 不确定性"？ | Tier-1 的 M=8 难度桶 + Tier-2 单任务的 M=4–8 配置桶（SGS / SGS+FG 对比；4 任务独立重复）+ §4 的 P1/P2 预测 | Δ = FG − SGS-band 随 N、M、L 的变化；离线指标 σ_p（每轮每组 success 估计噪声）与 LP-SNR | 见 H9/H10 |

### 2.2 可证伪假设（每条都写明"什么结果会杀死它"）

| 编号 | 假设（H） | 若为真，预期观测 | 被证伪的条件 |
|---|---|---|---|
| **H1（RQ1）** | 固定 B 下，uniform 的 final-perf 随 N 呈"先升后平/降"的凹形，拐点 N* 由 rollout 长度 L 与 UTD 决定 | Tier-1 中至少 3/4 任务出现显著平台；N* 随 L 变短而左移 | 4 个任务上 perf(N) 全部单调不降（在 CI 内），则 RQ1 的"饱和"命题被证伪，改写为"大规模并行不饱和"的正面结果 |
| **H2（RQ2）** | FG 把 N* 向右推，且在高 N 区（N≥1024）增益最大 | Δ(N) 随 N 单调或先增后平；Δ(4096) > Δ(256) | Δ(N) 在高 N 反而更小（即 FG 只在低 N 有用）→ 说明"分配"不是 scaling 杠杆，须改写结论并转向"低算力场景的样本效率"论文 |
| **H3（RQ3-分配）** | 增益主要来自分配，而非额外算力 | 把 FG 的辅助计算折算成额外 PPO 更新后（compute-matched 控制），增益仍保留 ≥50% | 计算量匹配后增益消失（<20%）→ 增益来自隐藏算力，方法贡献不成立 |
| **H4（RQ3-guard）** | 去掉 guard 后，在含"不可学噪声组"的环境中 FG 会显著退化（noisy-TV 效应） | 噪声组注入实验中，unguarded 版本的噪声组权重与真噪声水平正相关（ρ>0.4），guarded 版本 ρ<0.15 | guarded 与 unguarded 无差异 → guard 是冗余设计，应从方法中删除（负结果，但可写成"LP 本身足够鲁棒"的发现） |
| **H5（RQ3-LP）** | LP 信号优于"难度/最大回报"启发式 | LP 变体 > 难度采样 ≥3 个百分点；LP 与 return-slope 的差异在异质配置桶上更大 | LP ≈ 难度采样（CI 重叠）→ 说明 LP 的额外不确定性建模无价值，退化为简单 curriculum |
| **H6（RQ4）** | sim 中的方法排名在真机上保持（Spearman ≥0.5），零样本可迁移 | 真机成功率排序与 sim 一致；FG 绝对成功率 ≥ uniform + 10pp | 真机排名反转或全部接近 0 → sim-to-real gap 主导，RQ4 降级为"定性视频 + 失败分析" |
| **H7（理论 P1）** | group 级 LP 估计的信噪比随每轮每组样本数按 N^{1/2} 增长，但受"轮数 K=B/(NL)"限制 | 测得 SNR 与 √(N·L·w) 的回归斜率在 [0.3,0.7]（理论 0.5），R²>0.7 | 斜率显著偏离或与 N 无关 → toy 模型的关键假设（A2）不成立，理论节须重写 |
| **H8（预算/工程）** | B2=15M 步足以让 PPO-uniform 在 N=1024 达到"平台期 50% 性能"，即 fixed-B 协议有意义 | pilot 测得 PPO 收敛曲线在 15M 内进入平台 | pilot 显示 15M 远未收敛（<30% 平台值）→ 触发第 6.4 节预算重规划（B2→20M，砍 Tier-3 与诊断实验） |
| **H9（RQ5-scal.)** | 当 N 增大、L 不变时，每轮每组的样本数 n_g = N_g·L 下降，二值 success 的估计噪声放大，FG 的平滑/guard 相对 SGS-band 的增益随 N 增大 | Δ(FG − SGS) 在 N∈{1024,4096} 为正且 Δ(4096) > Δ(256)；Δ 与 1/√(n_g) 正相关（Spearman ρ>0.3） | Δ(FG − SGS) 随 N 不增或为负 → success-band 已足够，FG 的配置级分配没有额外价值（任务级只在 MT10 附录讨论），novelty 须收缩到"理论 + 真机"两条 |
| **H10（RQ5-等价）** | 在 M 小（M=2）、L 长、N≤256 的"低并行"设置下，FG 与 SGS-band 的差距小于 1 个 CI（二者等价） | 配对检验不显著（p>0.05），效应量 <1pp | 出现显著大 gap → 先做实现审计（SGS 未开源，不排除我们的 reimplementation 不公），再解释为 low-N 下 success-band 也失效 |

**预注册**：H1–H10、主指标、排除规则（崩溃 run 判定：CUDA error / NaN loss / 成功率恒为 0 且 entropy 爆炸）、统计脚本，在第 12 周冻结为仓库内 `preregistration.md` 并打 git tag。**主分析只做一次**，事后分析一律标注 exploratory。

---

## 3. 方法：FrontierGuard

### 3.1 问题设定与形式化
设并行环境数 N，交互预算 B（env-steps），rollout 长度 L，则更新轮数 K = ⌊B/(N·L)⌋。把 N 个环境划分为 M 个 group g∈{1..M}，每个 group 对应一个"可学单元"（任务、任务×难度桶、或初始状态桶）。分配策略在每个 episode 边界给出权重 w∈Δ^{M-1}（单纯形），环境中第 i 个 env 在 reset 时按 w 抽一个 group g_i 并加载该 group 的初始条件/任务参数。

**与已有工作的一句话区分**：SER（2511.21011）改的是 reset 的**时序**；PLR（2010.03934）与 RED（2110.02439）改的是**层级回放**；DRATS（2605.14350）在小规模任务集上按"离解决最远"分配**任务**；UPER（2506.09270）用不确定性给**回放池 transition** 排序；同月出现的 **SGS**（2610.12465，CoRL 2026）在**单任务内部**按 success-rate 分带对 task configuration 做加权 reset（二值 success、无不确定性建模、代码未开源）。因此本方案与 SGS **处于同一层**（单任务内的配置级分配），精确定位是：**不确定性感知的 LP 信号（epistemic/aleatoric 分解 + 三重 guard）× 大 N 下可检验的 scaling 预测 × 三层实验 + 真机零样本证据**；跨任务分配仅作 P1 附录（MT10），且不声称"此前无人做过并行环境层分配"（详见 §14 的定位修订与 §14.6 的设计修正）。

### 3.2 group 划分与 N→M 映射
- **Tier-1（单任务，每个任务一个策略）**：M = 8 个初始状态/难度桶（**CheetahRun** 的初速度分位桶、**HumanoidRun** 的初始姿态扰动桶、**Go1JoystickFlatTerrain** 的地形粗糙度桶、**PandaPickCube** 的立方体初始位姿/目标区域桶）；另注入 2 个**不可学噪声组**（奖励为纯噪声、策略无法改进）用于 H4。
- **Tier-2（每个 SO-101 任务内）**：**同一任务**内的 M = 4–8 个配置桶（物体初始位姿分位、目标区域分位、域随机化档位：摩擦/光照/相机抖动），同一 reward、同一成功判定、共享同一策略；8 个 SO-101 任务**各自作为独立实验**分别做 N 扫描（**不把不同任务混进同一策略**，原因见 §14.6）。粒度消融 M ∈ {4,8,16}（A4）。
- **Tier-3 / 真机**：每个任务内 M = 2–4 个配置桶（初始位姿/目标区域）。
- **跨任务分配（仅 P1 附录，可选）**：MetaWorld MT10/MT50（task ID 在 obs 中，DRATS 原论文同基准），N≤256、CPU 向量化本地跑；不作主 scaling 证据。MTBench（2507.23172，IsaacGym Preview 4 + 50 任务）因 IsaacGym 停更、40 系驱动风险，只引用为 related platform。

映射规则：每个 group 至少分到 ⌈w_min·N⌉ 个 env（w_min=0.5/M，保证可估）；余下的 env 按最大余数法分配，保证 ΣN_g = N 且 N_g 为非负整数；env 的 group 只在 episode 结束时改变（与 staggered reset 天然兼容，见 3.7）。

### 3.3 learning progress（LP）估计
对 group g，用窗口 W 内的 episode 结果（成功 0/1 或归一化回报）计算两个 EMA：
- 快尺度：$\hat{s}^{f}_g(t)=\mathrm{EMA}_{\alpha_f}(s)$，α_f 对应约 20 个 episode；
- 慢尺度：$\hat{s}^{s}_g(t)=\mathrm{EMA}_{\alpha_s}(s)$，α_s 对应约 200 个 episode。

**去饱和 LP**：$\widehat{LP}_g = \max\left(0,\ \frac{\hat{s}^f_g-\hat{s}^s_g}{\Delta t}\right)\cdot\big(1-\hat{s}^s_g\big)^{\gamma}$，γ=0.5。
设计理由：直接用成功率差分会在接近 1 时饱和（已解组的 LP→0，这是对的），但也会让"高成功率但仍在缓慢进步"的组被过早放弃；乘以 (1−ŝ) 相当于"剩余可提升空间"加权，让已解组的分数自然衰减而不会突然归零（避免分配振荡）。
**多信号**：同时记录 (a) 成功率斜率、(b) 归一化回报斜率、(c) episode 长度（若任务有提前终止）。主实验用 (a)（跨任务可比），(b)(c) 作为 LP 代理消融（对照 GMC 2605.05856 的参数空间代理）。

### 3.4 epistemic 与 aleatoric 不确定性的轻量分离
对每个 group 训练一个**结果预测头**：输入是环境的低维可观测量（本体状态 + 任务参数 + 一个小的图像特征，若用视觉则复用 PPO 的 CNN 特征），输出"本 episode 成功"的 logit。用 **E=3 个头的 ensemble**（共享 PPO 主干特征，仅头部独立初始化 + 不同数据子采样），成本约为主干的 0.5%。
- **epistemic**：$U^{epi}_g = \mathrm{Var}_e[\hat{p}_e]$（ensemble 方差）；
- **aleatoric**：$U^{ale}_g = \frac{1}{E}\sum_e \hat{p}_e(1-\hat{p}_e)$（Bernoulli 方差）；
- **互信息分解**（用于诊断与 guard 判定）：$I_g = H(\bar{p}) - \frac{1}{E}\sum_e H(\hat{p}_e)$，其中 H 为二元熵。$I_g$ 高 = "模型们不同意，但各自很自信" = 认知不确定性高、值得继续采样；$U^{ale}_g$ 高而 $I_g$ 低 = "本质上就是随机/不可学" = 应降权。

这套分解是 noisy-TV guard 的理论依据：**LP 的噪声来自 aleatoric，而 LP 的信号来自 epistemic**。这也解释了为什么 UPER（2506.09270）用 epistemic 排序能抑制噪声转移，而我们在 group 层做同样的事。

### 3.5 优先级分数与三重 guard
**基础分数**（未加 guard）：
$$\tilde{q}_g = \widehat{LP}_g \cdot \left(1+\kappa\, \tilde{U}^{epi}_g\right),\qquad \tilde{U}^{epi}_g = \frac{U^{epi}_g}{\frac{1}{M}\sum_j U^{epi}_j + \epsilon}$$

**Guard 1（noisy-TV guard）**：若 $U^{ale}_g > \tau_{ale}$ 且 $\widehat{LP}_g$ 的自相关（lag-1）< 0.1（说明"进步"不可持续、是噪声），则 $q_g \leftarrow q_g \cdot \beta_{noise}$，β_noise=0.1。判据的直觉：真学习进度是**可持续**的（这周的进步预测下周的进步），噪声不是。
**Guard 2（collapse guard）**：分配权重投影到可行域 $\mathcal{W}=\{w: w_g \ge w_{min},\ \mathrm{KL}(w\|\mathrm{unif}) \le \epsilon_{KL}\}$，w_min=0.5/M，ε_KL=0.5。作用：防止 bandit 把某个组饿死导致其统计永远无法恢复（探索下限），也防止极端分配破坏 PPO 的 batch 多样性。
**Guard 3（confidence guard）**：若组 g 在本窗口的样本数 n_g < n_min（如 30 个 episode）或 LP 的 bootstrap CI 宽度 > δ（如 0.05），则该组使用**乐观初始化**：$q_g \leftarrow \max(q_g, \bar{q} + c\sqrt{\ln t / n_g})$（UCB 式的乐观项）。作用：新组/低样本组不被过早淘汰。

最终分数 $q_g$ 归一化为分配 logits，交给 bandit。

### 3.6 分配：EXP3 与 UCB，约束投影
- **默认 EXP3-IX**（对抗性 bandit，对非平稳更稳）：权重 $w_g \propto \exp(\eta \hat{Q}_g)$，$\hat{Q}_g$ 为 q_g 的无偏重要性加权估计，η = sqrt(ln M/(M K))，并加隐式探索项（IX）。
- **对照 UCB1**：$w_g \propto \hat{q}_g + c\sqrt{\ln K / n_g}$，c=0.5。
- **对照 Boltzmann**（DRATS/PGB 风格）：$w_g \propto \exp(q_g/T)$，T 退火。
- **约束投影**：每轮对 w 做一次"截断 + 归一化 + KL 检查"的 3 步投影（M≤16 时 O(M log M)，可忽略）。
- **分配粒度**：w 只在 episode 边界生效，且使用**错峰重置**：新 episode 的开始时刻在 env 之间错开（SER 的做法），保证任一时刻 batch 内 env 的 phase 分布均匀——这一点直接消除了"分配导致 batch 内时间步同质化"的副作用。

### 3.7 与 PPO / UTD / staggered reset 的集成
- **数据管线**：每个 env 自带 group id；PPO 的 rollout buffer 记录 group id 与 episode 边界；GAE 不跨 episode；batch 内不重排 group（保持 on-policy 性质，无需重要性修正——每个 env 的数据仍是其自身策略采的）。
- **UTD 与 batch**：固定 UTD=1（每批数据一次更新），但 minibatch 数与 epoch 数随 N 调整以保持"每 env-step 的梯度步数"恒定——这是计算量匹配的关键（评测 2506.03404 关于并行度/rollout 长度/UTD 的权衡结论）。
- **rollout 长度 L**：Tier-1 L∈{16,32,64}；Tier-2 L=24（ManiSkill3 默认量级）；L 作为显式实验因子（因为理论预测 FG 的增益与 L 交互，见 P3）。
- **与 SER 组合**：SER 作为独立 baseline，同时提供 "SER+FG" 组合——这是本文相对 SER 的**正交性证据**，也是审稿人最可能问的问题（"你是不是只是 SER 的变体？"）的答案。
- **reset 协调**：group 在 reset 时抽样；若某 group 被"降权"，其正在运行的 env 不强制中断（避免破坏 on-policy 与 episode 完整性），只影响下一次 reset 的选择。

### 3.8 伪代码、复杂度、实现位置与工程量
```text
# 每个 rollout 结束后调用一次（M<=16，开销可忽略）
def frontierguard_step(rollout, fg_state):
    for g in groups:                      # O(M)
        s_g = recent_success(rollout, g)  # 窗口 W 内 episode 结果
        fg_state.update_ema(g, s_g)       # O(1)
        lp[g]  = max(0, (ema_f[g]-ema_s[g])/dt) * (1-ema_s[g])**0.5
        u_epi[g], u_ale[g] = ensemble_heads(g)   # O(E*d)，E=3，只在有足够样本时更新
        q[g] = lp[g] * (1 + kappa * normalize(u_epi)[g])
        if u_ale[g] > tau_ale and autocorr(lp_hist[g], lag=1) < 0.1:
            q[g] *= beta_noise                    # Guard 1
        if n_episodes[g] < n_min or ci_width[g] > delta:
            q[g] = max(q[g], mean(q) + c*sqrt(log(t)/n_episodes[g]))  # Guard 3
    w = exp3_or_ucb(q, fg_state.bandit)           # O(M)
    w = project(w, w_min=0.5/M, eps_kl=0.5)       # Guard 2，O(M log M)
    assign_envs_to_groups(w, N, fg_state)         # O(N)，episode 边界生效
```

**复杂度**：每轮 O(M + N)，内存 O(M·W)；E=3 个结果预测头参数量为主干 <1%，实测 wall-clock 开销目标 <5%（pilot 必测，超标则减到 E=2 或降频更新）。
**实现位置（约 100–200 行钩子）**：
- `frontierguard/group_sampler.py`（~120 行）：group 定义、w 采样、env 分配、约束投影；
- `frontierguard/estimators.py`（~60 行）：双 EMA、ensemble 头、MI 分解、guard 判定；
- `frontierguard/bandit.py`（~40 行）：EXP3-IX / UCB1 / Boltzmann；
- 集成点：Tier-1 在 brax `PPO.train` 的 `env.step` 之后、`reset` 之前插入（brax 的 `Env.reset` 支持 `rng` 与自定义 reset 函数）；Tier-2 在 ManiSkill3 的 `gym.vector` 包装层用 `reset(options={"env_ids":..., "task":...})` 指定 group；Tier-3 在 Isaac Lab `manager_based_rl_env` 的 `_reset_idx(env_ids)` 里按 group 设置任务参数。
**工程工作量（1 名研究生，人周，**【假设】**）**：

| 工作 | 人周 | 验证方法 |
|---|---|---|
| ManiSkill3 上自建 PPO（向量化 RGB 编码 + GAE + 梯度裁剪，CleanRL 风格） | 2.5 | 在 1 个 SO-101 任务上**先复现标准 PPO 收敛**（sanity check）：训练曲线与 Squint 报告的 SAC 15 min/1.5M steps 同量级算力下，PPO 需 3–10× 样本但能在 30 分钟内达到 >50% 成功率 |
| Squint 8 任务环境接入 + 奖励/成功判定复现 | 1.5 | 每个任务跑 1M 步，成功率与 Squint 仓库 README/论文图对齐（允许 2× 样本差） |
| FrontierGuard 模块 + 单元测试（合成 bandit 环境验证 EXP3 收敛） | 2.0 | 合成 4 臂 bandit：regret 曲线符合 O(√(MT log M)) 量级 |
| Baselines（PLR、DRATS、SER、UPER-env、难度采样） | 3.5 | 每个 baseline 在 1 任务×1 N×2 seeds 复现出"合理"曲线（不追求复现原文数字） |
| Tier-1 brax 集成 + 多任务包装 | 1.5 | PPO-uniform 在 CheetahRun 上 100M 步达到公开参考量级回报 |
| 真机部署（LeRobot + Squint deploy.py）+ 评测协议 | 2.5 | 1 个零样本策略在真机上跑通 10 trials |
| 数据/日志/分析流水线（wandb + 统一 eval + bootstrap 脚本） | 1.5 | 端到端跑通一个小矩阵并自动出图 |
| Isaac Lab 可选 Tier-3（环境搭建、任务筛选） | 2.0 | 2 个任务在 A100 上跑通 PPO 1M 步 |
| **合计** | **17 人周** | 其中 4 周与实验并行（脚本/分析），关键路径约 13 周 |

---

## 4. 理论：toy 模型与可检验的 scaling 预测

> 本节目的不是"证明定理"，而是**把直觉写成可算的式子、导出可被实验杀死的预测**。所有结论都标注了假设；假设不成立时结论作废，这与本方案的诚实性要求一致。

### 4.1 模型设定与假设
设 M 个 group，group g 的"可学进度"为 $p_g(n)=A_g(1-e^{-n/\tau_g})$，n 为该组累计采到的 episode 数，A_g∈(0,1) 为该组天花板，τ_g 为学习时间常数。固定总预算 B 个 env-step，每个 episode 长 H 步，则总 episode 数 E=B/H。均匀分配给 n_g=E/M；自适应分配给 n_g=w_g E。

假设：
- **A1（可加性）**：总性能 $P=\sum_g \pi_g p_g(n_g)$，π_g 为任务/评测权重；忽略组间迁移（sim 中大致成立，真机上不成立，见 4.5）。
- **A2（信号可辨）**：LP 的信号来自 $A_g/\tau_g$ 的差异，噪声来自伯努利采样与不可学噪声组；后者 aleatoric 方差不可通过学习降低。
- **A3（轮次结构）**：固定 B 与 rollout 长度 L 时，更新轮数 $K=B/(NL)$，每轮每组样本 $n_g^{round}=w_g N L$。这是"并行度"进入理论的唯一通道。
- **A4（非平稳）**：在一轮内，策略以速率 ρ 变化，故用窗口 W 估的 $\hat p$ 相对"当前策略"存在滞后偏差 $\propto \rho W$。

### 4.2 LP 估计噪声随 N 的缩放
窗口内成功率估计 $\hat s\sim$ 二项分布，$\mathrm{Var}(\hat s_g)=p_g(1-p_g)/n_g^{win}$。LP 估计
$$\widehat{LP}_g=\frac{\hat s^f_g-\hat s^s_g}{\Delta t},\qquad \mathrm{Var}(\widehat{LP}_g)=\frac{p^f(1-p^f)/n^f_g+p^s(1-p^s)/n^s_g}{\Delta t^2}.$$
在均匀分配、每轮每组样本 $w_g N L$ 的设定下，$n^{win}_g\propto N L W/H$，于是
$$\sigma_{LP}\ \propto\ \frac{1}{\Delta t}\sqrt{\frac{M}{N L W}}\ .$$
**可检验预测 P1**：LP-SNR $= \widehat{LP}_g/\sigma_{LP}$ 对 $\sqrt{N L w_g}$ 的回归斜率为 0.5（允许 [0.3,0.7]）。若实测斜率≈0，说明我们的窗口估计被策略非平稳主导（A4），而非采样噪声——这直接决定 guard 该防哪一类误差。
**重要含义**：σ_LP 随 N **下降**，所以"大 N 下 LP 估不准"不是好理由；真正随 N 恶化的是 A4 的偏差与轮数 K 的减少（见 4.3）。

### 4.3 uniform 与 adaptive 的 regret：为什么增益可能与 N 同向
把问题简化成"两臂 bandit"：前沿组（可学，边际进步 g）与饱和组（已学，边际进步 0）。均匀分配浪费一半预算，单位步 regret 为 g/2；自适应需要先"识别"哪个是前沿，识别成本 ≈ $c\,\sigma^2/\delta^2$ 个 episode，δ 为两组 LP 之差。
- **uniform regret**：$R_u \approx \frac{g}{2}B$（线性浪费）。
- **adaptive regret**：$R_a \approx g\cdot c\sigma^2/\delta^2 + g\cdot(\text{识别后残余})$。在 δ 足够大时，$R_a \ll R_u$，即"识别成本 ≪ 浪费量"。
- 代入 σ² ∝ 1/(N L W)（4.2）与每轮识别所需样本，可得：**识别成本占总预算的比例 ≈ p(1-p)/(B δ²)，与 N 无关**。也就是说，纯粹的"估计噪声"通道解释不了"为什么需要大 N 才值得做分配"。

真正随 N 增长的是 **A4 的滞后偏差与 batch 组成失真**：固定 B 时 K=B/(NL) 随 N 减少，每次更新的数据构成（哪些组占多少）对最终结果的影响被放大；而短 L 下 batch 内的时间多样性下降（这正是 SER 2511.21011 报告的非平稳机制）。形式上，把每次更新的有效梯度写成组贡献加权和 $\sum_g w_g \nabla_g$，则组成偏差的方差 $\propto \sum_g w_g^2 \sigma_g^2$，在 $\sigma_g$ 差异大时（难组方差大、已解组方差≈0）均匀权重并非最优；N 越大、更新越少，这个"一次定生死"的效应越强。
**可检验预测 P2**：FG 相对 uniform 的增益随 N 增大，**但只在短 L（16–32）下成立**；当 L≥64 时，增益对 N 不敏感（因为每轮数据已足够多样、偏差被平均掉）。
**可检验预测 P3（阈值）**：FG 有增益的条件是异质度 δ 超过 LP 噪声的若干倍（δ ≳ 2σ_LP）；当所有组同样难学时（δ→0），FG 应不劣于但也不优于 uniform。实验上用 Tier-1 的"难度阶梯"环境族扫描 δ 验证。

### 4.4 四条可检验的 scaling 预测（汇总）
| 编号 | 预测 | 实验 | 若不符 |
|---|---|---|---|
| **P1** | LP-SNR ∝ (N·L·w)^{1/2} | Tier-1 记录每轮每组的 LP 估计与真值（仿真里可算真 LP） | A4 主导，理论退化为"偏差受限"版本 |
| **P2** | 增益–N 斜率在短 L 下为正、长 L 下≈0 | Tier-1 N×L 网格（L∈{16,32,64}） | "分配"与"非平稳"的耦合解释错误，须撤回 RQ2 的机制叙述 |
| **P3** | 增益是 δ 的 S 形函数，阈值 δ*≈2σ_LP | Tier-1 难度阶梯（4 档异质度） | LP 信号选择机制不同，改写成经验观察 |
| **P4** | 注入噪声组后，unguarded 的权重–噪声相关 ρ>0.4，guarded ρ<0.15 | Tier-1 噪声组实验 | guard 冗余，删除并作为负结果报告 |

### 4.5 局限（必须写进论文的 Limitations）
1. toy 模型假设组间独立、可加，真机上组间存在迁移与干扰，A1 明显不成立；
2. A4 的偏差项我们只给了定性缩放，没有闭式解；论文里只作为"机制解释"而非定理；
3. M 的上限（≤16）使理论在大 M 区的行为未被覆盖；
4. 我们的 regret 是"性能差"的代理，不是标准 bandit regret，不可直接与 bandit 文献的界比较（论文中不使用"我们证明了 O(√T)"这类表述）。

---

## 5. 实验协议

### 5.1 平台与层级
**【事实】** 平台均已核实存在（2026-11-02）：

| 层级 | 平台/仓库 | 角色 | 本地/租卡 | N 范围 | 备注 |
|---|---|---|---|---|---|
| **Tier-0** | 同上 | pilot 校准 B、测吞吐、测 FG 开销 | 本地免费（Tier-1）；租 4090/A100 少量（Tier-2/3） | — | 预算内已列 |
| **Tier-1** | MuJoCo Playground + Brax（https://github.com/google-deepmind/mujoco_playground） | 高吞吐 scaling、理论验证、N×L 网格、噪声组 | **本地 4070 Ti Super（免费）** | 64 / 256 / 1024 / 4096 / 16384 | 状态输入、吞吐最高；主任务 **CheetahRun / HumanoidRun / Go1JoystickFlatTerrain / PandaPickCube**（后者为机器人操作，提升 ICRA 相关性）；group = 各任务内初始状态/难度桶 |
| **Tier-2（主）** | Squint SO-101 Task Set（https://github.com/aalmuzairee/squint，ManiSkill3 3.0.1，8 任务：ReachCube/ReachCan/LiftCube/LiftCan/PlaceCube/PlaceCan/StackCube/StackCan；1.5M timesteps≈15 min；RTX 3080+/≥10GB；含域随机化、腕部相机、deploy.py） | 视觉 scaling 主实验（**每任务一个策略**，M = 任务内配置桶）+ 真机策略来源 | **租 4090 24GB** | 64 / 256 / 1024 / 4096（4096 需 pilot 验显存，默认上限 1024–2048） | Squint 默认 SAC；我们用 **ManiSkill3 环境 + 自建 PPO**（CleanRL 风格），并用"标准 PPO 复现单任务收敛"做 sanity check（见 3.8） |
| **Tier-3（可选/附录）** | SO-101 Bench（https://github.com/5hadytru/so101_bench，Isaac Sim 5.1 + Isaac Lab 0.54，4 任务 bin/next-to/between/directional move）或 MuammerBay/isaac_so_arm101（Isaac Lab 2.3） | 数字孪生一致性、附录级证据 | **租 A100 80GB，严格限时** | 256 / 1024 / 4096 | 该 benchmark 面向 VLA/policy-server 评测，**不适合 RL 从头训练做主 scaling 实验**；只做 2 任务 + 1 任务的附录，不占核心预算 |
| **真机** | 2× LeRobot SO-101 + 腕部/第三人称相机 | sim-to-real 验证（RQ4） | 自有（不占租卡） | — | 用 Squint deploy.py；HIL-SERL 微调为 stretch |

**本地 vs 租卡的分界（明确写死）**：凡是"状态输入、N 可以大到 4096+"的实验一律本地；凡是"视觉渲染、需要 ManiSkill3/Isaac Sim"的一律租卡；真机训练（HIL-SERL 微调）用本地 4070 Ti Super（策略小、可离线）。**Isaac Sim 5.1 在本机 32GB 内存/16GB 显存不保证可跑**（官方测试卡为 RTX 6000/5090），因此 Tier-3 不做本地尝试，直接租 A100。

**Tier-2 N=4096 的显存可行性（v1 追加的硬约束）**：ManiSkill3 官方 RGB PPO 默认 `num_envs=256`（官方报告 <15GB 显存）；本方案 4096 个 RGB env 需要 4096 份渲染上下文/相机缓冲，**很可能超过 4090 24GB**（**【假设】**，须 pilot 实测：在 N=512/1024/2048 上二分 `num_envs`，记录峰值显存与吞吐）。**默认视觉档上限设为 1024–2048**；若 pilot 证明 4096 不可行，则 Tier-2 最大 N 降为 1024，省下的 GPU 小时（T2-03 约 131 h ≈ ¥328）改投 (a) N=512 新扫描点 与 (b) 主比较 seeds 5→8（约 +¥200），论文结论明确写成"**视觉档 N≤1024、状态档 N≤16384**"，视觉档的 scaling 外推一律标注为假设。

**跨任务分配降为 P1 附录（v1 追加的边界）**：共享 PPO 策略没有 task ID 条件输入时，不同 Squint 任务（不同 reward/物体/成功判定）**不能**作为同一策略下的 group——否则 LP/return 跨任务不可比、分配信号被 reward 尺度差驱动（详见 §14.6）。因此 Tier-1/Tier-2 的 group 一律是**同一任务内的配置桶**；跨任务分配只用 MetaWorld MT10/MT50（task ID 在 obs 中；DRATS 原论文同基准；CPU 向量化、本地 7800X3D、N≤256、无租卡成本）作为附录，人工约 1.5 人周、本地 GPU ≈60 h（或纯 CPU 更多）。**不建议**为 8 个 SO-101 任务自建统一 task-conditioned env（3–4 周工程量 + 高失败风险，会挤掉主实验）。

### 5.2 固定预算协议（两种，都要报）
**协议 A：固定 env-step 预算 B。** 每个 run 训练到恰好 B 步（不看墙钟）。这是主协议，保证"同样的交互预算"下比较。B 必须由 pilot 校准，见 5.3。
**协议 B：固定 GPU-hour 预算。** 用协议 A 的学习曲线 + 实测吞吐换算成 GPU-hours 曲线（同一 N 下各方法吞吐差异 <5%，用 pilot 实测修正），报告 (i) 固定 1.0 / 3.0 GPU-h 时的性能；(ii) "time-to-threshold"：达到 PPO-uniform 最终性能 80% 所需 GPU-hours。**为什么这很重要**：审稿人会问"你的方法多了 ensemble/bandit 计算，公平吗"——协议 B 直接回答。协议 B 不额外消耗租卡（曲线复用），只多花分析时间。

### 5.3 B 的 pilot 校准（不拍脑袋）
- **Tier-1**：2 任务 × 3 seeds × 10M 步测吞吐（本地，~2h）；再跑 PPO-uniform @N=1024、300M 步找渐近回报；取 B1 = 使 PPO-uniform@N=1024 达到渐近值 70% 的步数（**【假设】** 100–200M，落定后固定）。
- **Tier-2**：4 任务 × 3M 步 × 2 N × 3 seeds（预算表 T0b，21 GPU-h）；同时跑 1 个任务 40M 步找渐近成功率。取 B2 = 使 PPO-uniform@N=1024 达到"渐近成功率 50%"或"30% 绝对成功率"的最小步数（**【假设】** 10–20M，默认 15M）。
- **Tier-3**：1 任务 10M 步测吞吐 + 2 任务 5M 步验数据；B3 默认 20M（**【假设】**）。
- **决策规则**：若 pilot 显示 B2 > 18M，则 **B2 封顶 18M**（此时总价约 ¥4,420，仍在 5,000 内但预备金降到 ¥580），并同步执行 C3（N=64 诊断缩水，省 ¥350）以保住 ≥¥900 的预备金；若 B2 < 10M，把省下的钱加到 seeds（5→10，N=1024 主比较，+¥432）。B1/B2/B3 一旦 pilot 落定即写入预注册，后续不得调整。

### 5.4 Baselines（≥6，全部有出处；实现难度与公平性逐条说明）
| # | Baseline | 出处 | 适配方式 | 公平性说明 |
|---|---|---|---|---|
| 1 | **PPO-uniform** | 标准 on-policy 基线（Schulman et al., 2017, arXiv:1707.06347） | 同一 PPO 实现，N 个 env 均匀分到 M 组 | 主对照 |
| 2 | **PLR / RED** | 2010.03934 / 2110.02439 | 把"level replay 优先级"映射到 group 分配：按 TD-error/return 的优先级采样 group（RPLR 为该路线的鲁棒变体，但无可核实 arXiv 记录，不单独引用） | 经典 curriculum 采样，代表"非 LP"路线 |
| 3 | **DRATS** | 2605.14350 | **env 级改造**：把"离解决最远"（成功率最低/进度最慢）信号映射到同一任务内的配置桶；另在 MT10 附录跑其原版任务级 DRATS（任务采样）。**两种都标注 adaptation** | 概念最近的竞争工作；主实验用配置桶版，附录用原版 |
| 4 | **SER** | 2511.21011 | 错峰 reset，均匀分配 | 最直接威胁；同时用于验证"正交性" |
| 5 | **SER + FG** | 本文 | SER 的 reset 时序 + FG 的分配 | 组合 baseline，证明正交增益 |
| 6 | **UPER-env** | 2506.09270 | 把 epistemic 不确定性优先级从"replay transition"改造成"env/group 级"：优先采样预测不确定性高的组 | 机制最接近的不确定性工作 |
| 7 | **难度/最大回报采样** | 常规 practice（如 PLR 的变体；不单独引用） | 按成功率最低或 return 最高固定贪心分配 | 代表"简单启发式"，用来证明 LP 不是"难度采样换皮" |
| 8 | **RND/ensemble 探索**（可选） | 2603.01741（ensemble policy gradient at scale）讨论的多样性视角 | 只在 Tier-1 做 1 个任务，检验"探索 vs 分配"的混淆 | 若预算紧张可砍（见 6.4） |
| 9 | **SGS-band（reimplementation）** | 2610.12465（CoRL 2026；代码 "coming soon"，未开源） | 按论文/项目页实现：固定 task configuration 池 + success 历史 buffer + 三段 success band 加权 reset；与 FG **同层**：直接用在 Tier-1 的 M=8 难度桶、Tier-2 单任务的 M=4–8 配置桶上（组即 config 池）。**必须标注 "our reimplementation; no official code released at submission time"** | S0 级近邻；检验"config 级 success-band 是否已足够"（RQ5/H9/H10） |
| 10 | **SGS + FG（两级）** | 本文 | 同一组配置桶上，把 SGS 的 success-band 权重与 FG 的 LP×uncertainty 权重做门控/凸组合（同层两级信号） | 证明二者正交可组合（与 SER+FG 一起构成"组合性"证据链） |

**公平性铁律**：所有方法共用同一 PPO 实现、同一超参搜索预算（每方法 8 组配置、随机搜索、按验证集选）、同一 B、同一 seed 集、同一评测协议（每 0.5M 步固定 eval 20 episodes）。FG 的额外计算通过协议 B 与"compute-matched 控制"（把 FG 的辅助 FLOPs 折算成额外 PPO 更新给基线）双重控制。

### 5.5 Ablations（≥5，全部在 Tier-1 + Tier-2 各做一遍的关键项）
| 编号 | 消融 | 回答什么 |
|---|---|---|
| A1 | 去掉 epistemic 不确定性（κ=0） | 不确定性是否必要（H4/H5） |
| A2 | 去掉 LP（用均匀权重 + 仅不确定性） | LP 是否必要 |
| A3 | 去掉全部 guard（raw argmax/greedy） | guard 是否必要（noisy-TV） |
| A4 | group 粒度 M ∈ {4, 8, 16} | 分配粒度–收益关系 |
| A5 | 分配器：EXP3 vs UCB1 vs Boltzmann vs greedy | bandit 选择是否敏感 |
| A6 | EMA 窗口 W ∈ {10, 50, 200}；Δt | LP 估计的时间尺度 |
| A7 | κ ∈ {0, 0.5, 1, 2}；ε_KL ∈ {0.1, 0.5, 2}；w_min | 超参敏感性 |
| A8 | LP 代理：成功率 vs 回报斜率 vs GMC 风格（2605.05856） | LP 信号的选择 |
| A9 | 不确定性估计器：E ∈ {2,3,4} 头 vs 最后层 Laplace | 轻量实现是否够用 |
| A10 | N/M 比（固定 M，变 N） | 分配粒度相对并行度 |

**主消融集（进正文）**：A1、A2、A3、A5、A6；其余进附录。

### 5.6 统计与分析规范（预注册）
- **seeds**：Tier-1/2 核心设置 **5 seeds**（seed 即初始化 + 环境种子 + 采样种子）；Tier-3 3 seeds；真机 30 trials/条件。
- **主指标**：Tier-1 = 最后 10% 步的归一化回报（按 PPO-uniform 的最终值归一，跨任务平均用几何均值）；Tier-2 = 最后 20% 步的成功率（连续 20 次 eval 的平均）。
- **主分析（预注册，只做一次）**：混合效应模型 `perf ~ method * log2(N) + (1|task) + (1|seed)`，主检验为 `method × log2(N)` 交互项；若显著，报告 FG 在各 N 的边际效应。备选（更稳健）：在每个 N 上做配对 permutation test（10,000 次重排），Holm–Bonferroni 校正。
- **CI**：连续指标用 bootstrap（10,000 次，percentile）；成功率用 Wilson 区间。
- **效应量**：成功率差（百分点）+ 优势比；回报差用 Cliff's delta。
- **计算量匹配**：报告 env-steps、梯度步数、wall-clock GPU-h、估算 FLOPs（按参数量×步数粗算）；协议 B 曲线共存。
- **排除规则**：run 崩溃（CUDA OOM/NaN/成功率恒 0 且 entropy 爆炸）记为失败并重跑，最多 2 次；连续失败则该 cell 报"不可用"而不是删除。所有排除记录进 `exclusions.csv`。
- **不做事后挑 seed**；所有图用全部 seed 的均值±95% CI。

### 5.7 机制指标（用于 RQ1/RQ3 的"为什么"）
1. **组覆盖**：每 10 轮内至少被采样 1 次的 group 比例（uniform 在 N 大时若不降则说明"覆盖"不是问题）；
2. **已解组浪费率**：p̂_g ≥ 0.95 的 group 消耗的 env-step 比例；
3. **LP-SNR**：每组的 LP 估计与其"真 LP"（仿真可算）的相关系数与噪声；
4. **梯度组成**：按组统计 advantage 方差与梯度范数贡献占比（回答"batch 里谁在起作用"）；
5. **策略熵 / 成功率方差**：排除"FG 只是探索更多"；
6. **分配–真值一致性**：w_g 与真 LP 的 Spearman 相关；regret 代理 $\sum_g w_g(1-\widehat{LP}_g/\max LP)$；
7. **reset 相位分布**：验证 SER 组合确实改变了相位分布（否则组合无效）。

### 5.8 真机实验（RQ4）
- **任务**：从 Squint 8 任务中 pilot 后选 **2 个**（建议 LiftCube 与 StackCube 或 PlaceCube；选择标准：sim 成功率 >60%、真机可重复、单次 trial <15 s）。**【待核实】** 最终由第 4 周真机 pilot 决定。
- **策略来源**：Tier-2 中用 3 种方法（PPO-uniform / SER / FG）训练的 sim 策略，每种取**预注册规则选出的 1 个 seed**（"sim 验证成功率中位数"的 seed，不看真机结果），共 3 方法 × 2 任务 = 6 个策略。
- **零样本评测**：每条件 **30 trials**（Wilson 95% CI 半宽约 ±0.17，可接受）；初始位置在固定范围内随机（记录并分层）；成功判定用脚本化 rubric（物体提升高度/目标区域），双人独立打分不一致时第三人裁决。
- **两臂并行**：两台 SO-101 同时跑不同条件（臂效应通过条件–臂轮换平衡），把评测时间减半。
- **人工时间核算**：单 trial ≈ 35 s（reset 20 s + 执行 10 s + 判定 5 s），30 trials ≈ 18 min；6 条件 × 30 = 180 trials ≈ 1.75 h 纯评测，含 setup/校准/失败重试按 **3× 膨胀 ≈ 6–8 人时/轮**；计划 2 轮（不同日期）→ **16 人时 + 2 个半天 setup**，约 **4 人日**，现实可承受。
- **HIL-SERL 微调（stretch，标注降级方案）**：用 LeRobot `src/lerobot/rl` + HIL-SERL 指南（https://github.com/michel-aractingi/lerobot-hilserl-guide），每任务 1–2 h 人工干预数据 + 30 min 训练，目标是在 FG 策略上再 +10pp。**降级方案**：若 HIL-SERL 在 2 周内不稳定，则只报零样本结果，并在论文中明确"HIL 微调留待 future work"。
- **真机安全**：力矩限幅、工作空间围栏、急停；每次上电前跑 3 次空载轨迹。

---

## 6. 计算预算与 compute accounting（硬上限 5000 元）

### 6.1 单价假设（保守上限，购买前必须复核）
| 型号 | 假设单价 | 依据 | 备注 |
|---|---|---|---|
| RTX 4090 24GB | **2.5 元/h** | 市场常见 2.0–2.5 元/h，取上限 | **【假设/待核实】** 平台实时价无法抓取（AutoDL 页面不可访问），下单前截图存证 |
| A100 80GB | **8.0 元/h** | 市场常见 6–8 元/h，取上限 | 同上；若实际 6 元/h，本方案总价下降约 20% |
| H100 | 不使用 | 12–18 元/h，性价比不划算 | 明确不租 |
| 本地 4070 Ti Super | **0 元（免费）** | 自有 | 只计电费，不计入预算 |

**存储与网络**：本地 2TB SSD 足够存 checkpoint 与日志（只存 best/last，约 200–400 GB）；租卡平台存储按 0.5–1 元/GB/月估算，全程预计 **<100 元**（**【假设/待核实】**，已计入预备金）。

**耗材（与租卡预算分开列，≤500 元）**：USB 相机 2 个（~300 元）、3D 打印夹具/线缆/扎带（~150 元）、标注用标签纸（~50 元），合计 **≤500 元**，不计入 5,000 元租卡硬上限；购买前需用户确认（决策点 D6）。

### 6.2 实验矩阵 → 每 run GPU 小时 → 总 GPU 小时 → 单价 → 总费用
**吞吐假设（【假设】，pilot 必测，见 6.3）**：Tier-2（4090，ManiSkill3 视觉 PPO）稳态吞吐 N=64: 0.6K、256: 1.5K、1024: 3.0K、4096: 4.0K steps/s（对照：Squint 报告 SAC 1.5M steps≈15 min≈1.7K steps/s 端到端，故上述 PPO 数字属合理量级）；每 run 另加 0.05 h 固定开销（JIT/评测）。Tier-3（A100，Isaac Lab）N=256: 5K、1024: 10K、4096: 15K steps/s，另加 0.15 h 启动开销。

| 分组 | GPU | runs | h/run | GPU-h | ¥ |
|---|---|---|---|---|---|
| T0b Squint PPO pilot（3M 步，4 任务×2N×3 seeds） | 4090 | 24 | 0.88 | 21 | 53 |
| T2-a N=64 诊断（4 任务×3 方法×3 seeds） | 4090 | 36 | 6.99 | 252 | 630 |
| T2-b N=256 主比较（4 任务×6 方法×5 seeds） | 4090 | 120 | 2.83 | 339 | 848 |
| T2-c N=1024 主比较（同上） | 4090 | 120 | 1.44 | 173 | 432 |
| T2-d N=4096 主比较（同上） | 4090 | 120 | 1.09 | 131 | 328 |
| T2-e SER+FG 组合（4 任务×2N×5 seeds） | 4090 | 40 | 1.27 | 51 | 127 |
| T2-f 任务广度（4 个 held-out 独立任务×3 方法×3 seeds@N=1024；共 8 个独立任务） | 4090 | 36 | 1.44 | 52 | 130 |
| T2-g 主消融（6 变体×3 seeds@N=1024，1 任务） | 4090 | 18 | 1.44 | 26 | 65 |
| T2-h 2B 预算确认（4 任务×3 方法×2 seeds×2N） | 4090 | 48 | 1.27 | 61 | 152 |
| T0c Isaac Lab 环境搭建（10 h）+8 个 1M 步 sanity run | A100 | 18 | 0.82 | 15 | 118 |
| T3-a Tier-3 主（2 任务×3N×3 方法×3 seeds） | A100 | 54 | 0.83 | 45 | 358 |
| T3-b Tier-3 第 3 任务（1 任务×3N×3 方法×3 seeds） | A100 | 27 | 0.83 | 22 | 179 |
| **小计（4090）** | | **562**（含 T0b 24；核心主表 538） | | **1105** | **2,765** |
| **小计（A100）** | | **99** | | **82** | **655** |
| **重跑/鲁棒缓冲 12%** | | | | **130 + 8** | **390** |
| **合计** | | | | **1,235（4090）+ 90（A100）** | **3,810** |
| **剩余预备金** | | | | | **1,190** |

**可选加分项（不改变基准 ¥3,810）**：T2-09（Tier-2 的 SGS / SGS+FG 对比，¥144）与 T2-f 任务广度（4 个 held-out 任务，+¥130）从预备金出；两项都做 → 总价 **¥4,084**，仍 ≤5,000（预备金 ¥916）。两项都不做也不影响 RQ1–RQ4。

**Tier-1（本地，0 元）**：4 任务 × 5 N × 8 方法 × 5 seeds = **800 runs**（含 v1 追加的 SGS、SGS+FG）；B1=150M 步；按吞吐 N=64:150K、256:400K、1024:900K、4096:1.3M、16384:1.5M steps/s 估算，主网格 ≈ **78 GPU-h**（每 run 0.05–0.30 h）；加 eval（每 0.5M 步 20 episodes）、pilot、L 网格/噪声组/难度阶梯等附加实验按 1.5× 计 ≈ **117 h**，再给 2× 保守余量 ≈ **235 h**，折合 45 周内每周 ≈5.2 h，**本地完全可承受**（边缘任务 HumanoidRun/PandaPickCube 更重，PandaPickCube 吞吐按 CheetahRun 的 1/2【假设】计，已含在 1.5× 内）。若本地吞吐低于假设 50%，Tier-1 改按优先级 P0 子集执行（见第 12 节），不租卡补。

### 6.3 compute accounting（每个分组 → GPU 小时 → 元）
| 分组 | runs 的来源（任务×N×方法×seeds） | 每 run 步数 × GPU 型号/数量 | 计算式 → GPU-h | ¥ |
|---|---|---|---|---|
| T2-a | 4×1(N=64)×3×3 | 15M 步 × 4090×1 | 36 × (15M/0.6K/3600 + 0.05) = 252 | 630 |
| T2-b | 4×1(256)×6×5 | 15M × 4090×1 | 120 × (15M/1.5K/3600 + 0.05) = 339 | 848 |
| T2-c | 4×1(1024)×6×5 | 15M × 4090×1 | 120 × (15M/3.0K/3600 + 0.05) = 173 | 432 |
| T2-d | 4×1(4096)×6×5 | 15M × 4090×1 | 120 × (15M/4.0K/3600 + 0.05) = 131 | 328 |
| T2-e | 4×2(1024,4096)×1(SER+FG)×5 | 15M × 4090×1 | 40 × 1.27 = 51 | 127 |
| T2-f | 4×1(1024)×3×3 | 15M × 4090×1 | 36 × 1.44 = 52 | 130 |
| T2-g | 1×1(1024)×6×3 | 15M × 4090×1 | 18 × 1.44 = 26 | 65 |
| T2-h | 4×2(1024,4096)×3×2 | 30M × 4090×1 | 48 × 1.27 = 61 | 152 |
| T3-a | 2×3×3×3 | 20M × A100 80GB×1 | 54 × (20M/6.7K/3600 + 0.15) = 45 | 358 |
| T3-b | 1×3×3×3 | 20M × A100 80GB×1 | 27 × 0.83 = 22 | 179 |
| T0b | 4×2×1(PPO)×3 | 3M × 4090×1 | 24 × 0.88 = 21 | 53 |
| T0c | 2×2×1×2 + 10 h 搭建 | 1–10M × A100×1 | 8 × 0.60 + 10 = 15 | 118 |

**本地免费 vs 租卡分界（一句话）**：Tier-1（状态输入、N≤16384、600 runs）全部本地，0 元；Tier-2（视觉、562 runs，含 pilot 24；不含 pilot 538）全部租 4090；Tier-3（Isaac Sim、99 runs）租 A100；真机训练与评测用本地 4070 Ti Super，0 元。**租卡只花在"本地跑不了的渲染/大显存"上。**

### 6.4 敏感性分析与砍实验顺序
**±50% 敏感性**（单价与小时数各 ±50%）：

| 情景 | 总费用 | 是否 ≤5000 | 动作 |
|---|---|---|---|
| 基准（本方案） | **3,810** | 是，余 1,190 | 正常执行 |
| 单价 −50% | 1,905 | 是 | 把省下的钱加 5→10 seeds（N=1024 主比较，+432 元）与 Tier-3 加 1 个任务（+179 元） |
| 单价 +50% | **5,715** | **否，超 715** | 执行砍单 C1 |
| 单价 +50% 且小时 +50%（双因素叠加，最坏情景） | 8,573 | 否，超 3,573 | 执行 C1+C3+C4+C5+C6 → 估算 **4,469**（见下方"硬上限保底规则"） |
| B2 上调至 18M（pilot 上限） | 4,420 | 是，余 580 | 冻结其余可选项（C4/C5 不执行但也不加新实验） |
| B2 上调至 20M | 4,820 | 是，余 180 | 必须同时执行 C3（N=64 诊断缩水），否则预备金不足以重跑 |
| B2 下调至 12M | 3,200 | 是，余 1,800 | 余钱加 seeds 与 8 任务广度 |

**单因素 vs 双因素必须分开陈述**：±50% **单因素**最坏是"单价 +50%"→ 5,715 元（超 715），只需执行 C1（−721）即回到 **4,994 元** ✓；"小时 +50%"单独出现同理。**双因素叠加**（1.5 倍单价 × 1.5 倍小时 = 2.25 倍）才是真正需要动骨的情景（8,573 元），不能拿单因素结论覆盖它。

**硬上限保底规则（强制，写入项目 checklist）**：在每个 checkpoint（W10/W14/W19/W24/W28/W32）重算"已花 + 剩余预计"；一旦预计总价 >5,000 元，按下列顺序执行，直到预计 ≤4,900（留 100 元执行缓冲）：
1. **C1 砍 Tier-3**：基准价 −721 元（双因素下 ≈ −1,622）→ 8,573 → 6,951
2. **C3 砍 N=64 诊断**：−350（≈−788）→ 6,163
3. **C4 砍 8 任务广度**：−130（≈−293）→ 5,870
4. **C5 砍 2B 确认**：−152（≈−342）→ 5,528
5. **C6 seeds 5→3**：−471（≈−1,060）→ **4,468** ✓（≤5,000，余 532）
   等价说法：C1+C3+C4+C5+C6 按基准价共砍 1,824 元，剩余 1,986 元 × 2.25 = **4,469 元**。
6. **保底包 F（若实测恶化 >50%，如 4090 单价 3.5 元/h 或吞吐只有假设 1/3）**：B2=10M、N ∈ {256,1024}、5 方法、3 seeds、无 Tier-3 / 无 8 任务广度 / 无 2B 确认 / 真机 2 条件 → 基准价约 **¥557（≈223 GPU-h）**，即使 3 倍恶化也 ≈¥1,670；论文改写为"受限算力下的 scaling 研究"，RQ2 显著性门槛相应下调并在预注册中说明。

**代价披露**：C1–C6 全执行后统计功效下降（3 seeds、无 Tier-3），这是硬上限约束下的显式取舍，会写进论文 limitations；**任何情况下不得通过"不加缓冲"或"少报重跑"来假装 ≤5,000**。

**砍实验顺序（严格按此顺序，不跳步）**：
1. **C1 砍掉全部 Tier-3（Isaac Lab）**：−721 元（−90 A100 h）。理由：Lead 已确认 SO-101 Bench 面向 VLA 评测、不适合 RL 从头训的 scaling 主实验，Tier-3 本就是附录级。
2. **C2 真机条件数 3 方法→2 方法**（去掉 SER，保留 uniform 与 FG）：省租卡 0 元、省人工约 2 人日；真机结果从"三方法排序"降级为"两方法对比"，RQ4 仍成立但 Spearman 一致性检验改为定性。
3. **C3 Tier-2 N=64 诊断缩水**（3 方法→2 方法、3 seeds→2 seeds）：−350 元（−140 h）。
4. **C4 砍任务广度（T2-f，held-out 4 任务）**：−130 元。
5. **C5 砍 2B 预算确认（T2-h）**：−152 元。
6. **C6 Tier-2 seeds 5→3（N=256 与 N=4096）**：−471 元；同时把统计口径从"5 seeds 主分析"降级为"3 seeds + 预注册说明"，**这是最后手段**，因为会削弱 ICRA 最看重的统计严谨性。
7. **C7 B2 15M→10M**：−900 元，但会让 fixed-B 协议失去意义，**仅在前述全部执行后仍超支时使用**，并在论文中报告"budget-limited regime"。

**超支预警线**：实际单价或吞吐偏差导致预计总价 >4,500 元时，立即执行 C1；>5,000 元时执行 C1+C2+C3。

---

## 7. 时间线（45 周）与 Go/No-Go 门

按 **W1 = 2026-11-02** 起算，W45 = 2027-09-06～12，距 ICRA 2028 预计截稿（2027-09 中旬）留 3 天余量。

| 周 | 日期 | 工作内容 | 交付物 | 预算累计 | 门 |
|---|---|---|---|---|---|
| W1–W2 | 11-02～11-15 | 环境：Squint/ManiSkill3、MuJoCo Playground、LeRobot 装好；PPO 骨架（向量化 + GAE） | `env_setup.md`、可跑通的 PPO 脚本 | ¥0 | — |
| W3–W4 | 11-16～11-29 | **PPO 复现 sanity check**（1 个 SO-101 任务）；Tier-1 单任务跑通；真机连通性测试（不训练） | 复现曲线；真机 10 trials 手动跑通 | ¥0–50 | **G0** |
| W5–W7 | 11-30～12-20 | FrontierGuard 模块 + 合成 bandit 单测；Tier-1 接入（group 抽样、SER 组合） | `frontierguard/` 三个文件 + 单测 | ¥0 | — |
| W8–W10 | 12-21～01-10 | Tier-1 pilot（测吞吐、定 B1）；Tier-1 前两个 N（64/256）的 5 方法对比 | B1 校准报告；初版 scaling 图 | ¥50 | **G1** |
| W11–W13 | 01-11～01-31 | Tier-2 PPO 在 Squint 上跑通 + pilot（T0b，定 B2）；真机任务 pilot（选 2 任务） | B2 校准报告；任务选择报告 | ¥300 | — |
| W14 | 02-01～02-07 | **IROS 2027 决策门**：评估是否值得用当前结果投 3 月截稿 | 决策记录 D2 | ¥400 | **G2/D2** |
| W15–W19 | 02-08～03-14 | Tier-1 全网格（N×L、噪声组、难度阶梯）；Tier-2 N=256/1024 主矩阵 | RQ1/RQ2 主结果（Tier-1 完整、Tier-2 部分） | ¥1,500 | — |
| W20–W24 | 03-15～04-18 | Tier-2 N=4096 + SER+FG + 消融；真机第一次评测（3 方法×2 任务×30 trials） | RQ3 主结果；真机第一轮数据 | ¥2,600 | **G3** |
| W25–W28 | 04-19～05-16 | （若保留）Tier-3；任务广度（held-out 4 任务）；2B 确认 | 附录结果 | ¥3,400 | — |
| W29–W32 | 05-17～06-13 | 真机第二轮评测（重复性）；补充实验与失败分析；**CoRL 2027 决策门** | 真机终版数据 | ¥3,700 | **D3** |
| W33–W34 | 06-14～06-27 | **实验冻结**；数据整理、统计脚本定稿、图初稿 | 冻结数据集 + 所有图 v1 | ¥3,810 | **G4** |
| W35–W39 | 06-28～08-01 | 论文写作（ICRA 6 页正文 + 引用）；图表精修；附录 | 论文 full draft v1 | ¥3,810 | — |
| W40–W42 | 08-02～08-22 | 内部评审（找 2 位同方向同学读）；修改；**视频拍摄与剪辑** | draft v2 + 90 s 视频 | ¥3,810 | — |
| W43–W44 | 08-23～09-05 | 提交材料：author kit 格式、附录、代码/数据开源整理、reproducibility checklist | 提交包 | ¥3,810 | — |
| W45 | 09-06～09-12 | 提交（留 3 天余量） | 提交回执 | — | **提交** |

**Go/No-Go 判据与失败动作：**

| 门 | 时间 | 通过条件 | 不通过的动作 |
|---|---|---|---|
| **G0** | W4 | 标准 PPO 在 1 个 SO-101 任务上 30 min 内达到 >50% 成功率；Tier-1 CheetahRun 100M 步达到公开量级 | 若 PPO 训练不稳：降级用状态输入（proprio-only）先把方法学验证跑通，视觉作为后续增量；若 Si 环境不通：换 ManiSkill3 官方任务 |
| **G1** | W10 | Tier-1 至少 3 个 N 的曲线可用；LP-SNR 斜率可估 | 若 LP 信号测不出（SNR<1）：把窗口 W 放大、改用回报斜率，并在论文中把 LP 降级为"经验信号" |
| **G2/D2** | W14 | Tier-1 完整 + Tier-2 初步；若决定投 IROS，需 6 页故事闭环 | 不投 IROS，专注 ICRA 2028；IROS 决策只做一次，不反复 |
| **G3** | W24 | Tier-2 主矩阵完成 ≥70%；FG 在 ≥2 任务上显示正向趋势 | 若 FG 无正向趋势：转向"诊断 + 边界条件"论文（RQ1 的负结果本身有价值），并加做 guard 的鲁棒性实验作为主贡献 |
| **G4** | W34 | 全部主实验 + 真机数据冻结；统计脚本通过 | 砍 C4/C5 补时间；绝不延期到写作期做实验 |

**预算–时间对齐**：¥1,500 在 W19 前花完（Tier-1 pilot + Tier-2 前两档 N），与"主结果雏形"里程碑一致；¥3,810 在 W34 前花完，留 ¥1,190 预备金给 W34 后的重跑（若有）。**任何一周的实际花费超过计划 20%，当周上报并触发砍单预演。**

---

## 8. 风险登记表与 Plan B

| # | 风险 | 概率 | 影响 | 触发信号 | 缓解 | Plan B |
|---|---|---|---|---|---|---|
| R1 | **FG 相对 uniform 无显著增益**（最致命） | 中（30–40%） | 高（论文无主贡献） | W20 时 3 个任务上 Δ<1pp | 早做 Tier-1 的"高异质度"环境族，把方法放在最有利的 regime；强调 guard/鲁棒性作为第二贡献 | 改写为"何时分配有效/无效"的诊断论文 + scaling 边界条件；投 CoRL/workshop |
| R2 | **视觉 PPO 在 Squint 上训不起来**（样本效率低） | 中 | 高（Tier-2 崩塌） | W5 复现失败：1 任务 10M 步 <20% 成功率 | 先 proprio-only 跑通再逐步加 RGB；用更大 batch/更长 rollout；参考 Squint 的 PPO 相关设置 | Tier-2 降级为状态输入（ManiSkill3 proprio），视觉只做 1 任务附录；主叙事仍成立（方法层面是 env 分配，不依赖视觉） |
| R3 | **Isaac Lab / Isaac Sim 跑不起来**（本地 32GB 内存、16GB 显存不够；A100 上版本不兼容） | 高（60%） | 低（Tier-3 可砍） | W25 首次尝试失败 | 直接租 A100；用 Docker 一键版 so101_bench_quickstart；锁 Isaac Sim 5.1/Isaac Lab 0.54 版本 | 砍 Tier-3（C1），论文中说明"数字孪生一致性留待 future work"，不影响 RQ1–RQ4（RQ4 用真机） |
| R4 | **真机 sim-to-real gap 大，零样本全 0** | 中（35%） | 中（RQ4 弱） | 第一轮 30 trials 成功率 <10% | 域随机化增强（Squint 自带）、腕部相机对齐、初始位置收窄 | 只报"sim 中训练策略在真机上可运行"的定性视频 + 失败模式分析；或用 HIL-SERL 微调后报"少量真机数据后的性能" |
| R5 | **预算超支**（单价上浮 / 吞吐低于假设 / 重跑多） | 中（30%） | 中 | 月度核算 >4,500 | 第 6.4 节砍单顺序；pilot 先测吞吐再全量开跑 | 执行 C1→C2→C3，保底方案为 2,500–3,000 元的"Tier-1 + Tier-2 三任务"精简版 |
| R6 | **审稿人认为 novelty 与 SER/DRATS/UPER 重叠** | 高（50%） | 高 | 预投内部评审意见 | 差异表（第 9 节）+ SER+FG 正交性实验 + 明确"分配层 vs reset 时序层 vs replay 层"的定位 | 把贡献重述为"可扩展性诊断 + 不确定性守卫的分配框架"，并加"组合性"作为一等贡献 |
| R7 | **人力/时间冲突**（1 人全职但实验排队） | 中 | 中 | 周计划连续 2 周延期 | 严格优先级 P0/P1/P2（第 12 节）；自动化脚本；写作与实验并行（W29 起） | 砍 P2 全部；把 8 任务广度与 Tier-3 移到"journal extension" |
| R8 | **统计被质疑**（5 seeds 不够、多重比较） | 中 | 中 | 评审意见 | 预注册 + 5 seeds + 配对检验 + Holm 校正 + 效应量 + 全部 seed 公开 | 补 5 seeds（¥432，来自预备金）；或把主结论限制在 Tier-1 大样本结论上 |
| R9 | **基线实现不公**（PLR/DRATS 被削弱） | 中 | 高（拒稿理由） | 基线曲线明显差于文献 | 每个基线做"超参搜索 8 组"+ 与原文设置对齐的说明；公开实现 | 若某基线复现不出合理性能，改为"引用原文数字 + 说明不可比"，并把它移出主表 |
| R10 | **HIL-SERL 真机微调不稳**（stretch 项） | 高 | 低 | 2 周内无法稳定收敛 | 限时 2 周；只做 1 任务 | 放弃 HIL 微调，明确写"future work"；RQ4 只报零样本 |

---

## 9. 与近邻工作的差异表（全部 arXiv ID 已用 arXiv API 核实，2026-11-02）

| 工作（arXiv） | 采样/分配单元 | 核心信号 | 研究 N 扩展？ | 真机？ | 与 FrontierGuard 的关键差异 |
|---|---|---|---|---|---|
| **SGS**（2610.12465，CoRL 2026） | 单任务内 task configuration（s0/goal/env） | 二值 success 三段带（never/improving/mastered）加权 reset | **是（2²⁰ envs）** | 是（蒸馏 RGB 零样本） | **S0 级近邻且同层**（都在单任务内做 config 级分配）：差异在 LP 斜率信号、epistemic/aleatoric 分解、三重 guard、可检验 N-scaling 理论；代码未开源。SGS+FG 为同层组合 baseline（§14） |
| **SER**（2511.21011） | 并行环境（reset 时序） | 无（错峰 reset 降低非平稳） | **是** | 否 | 它调"何时 reset"，我们调"reset 到哪个 group"；**正交、可组合**（SER+FG 是本文 baseline） |
| **DRATS**（2605.14350） | 任务 | "离解决最远"（成功率最低）的鲁棒分配 | 否（MetaWorld MT10/MT50，小规模） | 否 | 无 LP 估计、无 epistemic 不确定性、无 guard；我们的 δ/噪声分析解释了它何时失效 |
| **PGB**（2610.25704） | 样本组（梯度聚类） | 梯度聚类 + 时序预测 + Boltzmann | 否（LLM RL） | 否 | 领域不同；我们用成功率 LP + 不确定性 + 约束 bandit，且聚焦并行 env 层与 scaling |
| **UPER**（2506.09270） | replay transition | epistemic 不确定性排序 | 否（off-policy 小规模） | 否 | 它用于**回放池**、我们是**on-policy 并行 env 分配**；共享"用 epistemic 抑制噪声"的思想，我们把它提升到 group 层并加 aleatoric 分解 |
| **GMC**（2605.05856） | 任务/样本 | 参数空间代理（梯度×动量） | 否 | 否 | 它是 LP 的**代理信号**；我们把它作为 LP 代理消融 A8，而不是竞争方法 |
| **LPM**（2509.25438） | 探索模块 | learning progress monitoring 抗 noisy-TV | 否 | 否 | 直接支撑我们的 Guard 1；我们在分配层实现并做消融，二者是"监控器 vs 分配器"的关系 |
| **PLR / RED**（2010.03934 / 2110.02439） | level（ProcGen） | TD/return 优先级 replay | 否（单机小规模） | 否 | 经典 curriculum；我们做并行 env 分配 + 不确定性 + 约束，且规模到 16,384 |
| **并行数据收集分析**（2506.03404） | —（分析） | 并行 actor 数 × rollout 长度 × UTD 的 bias-variance | 是（分析） | 否 | 提供我们的机制解释与超参依据（L、UTD、N 的交互，对应 P2） |
| **大规模并行 MTRL benchmark**（2507.23172） | —（benchmark） | 并行多任务 RL 评测协议 | 是 | 否 | 我们借鉴其协议设计，但不做 benchmark 贡献 |
| **RL scaling laws**（2610.33420） | —（经验规律） | Starpilot/Craftax 上的 scaling law | 是 | 否 | 我们用其方法论拟合 success(N) 曲线，并报告拟合参数与 CI |
| **Ensemble PG at scale**（2603.01741） | 策略多样性 | ensemble/SAPG/PBT | 是 | 否 | 它是"探索/多样性"视角，我们是"数据分配"视角；作为可选 baseline（RND/ensemble）对照 |
| **Squint**（2602.21203） | —（平台） | 视觉 SAC + 并行仿真 + 分布 critic + squinting | 部分 | **是** | **平台**而非竞争方法；我们用其 SO-101 Task Set 与真机部署，但算法是 PPO+FG（on-policy），且明确说明与 Squint 的 off-policy 差异不可直接比样本效率 |
| **SO-101 Bench**（2610.36420） | —（平台） | Isaac Lab 数字孪生 + 语言条件任务 | 否 | 是 | **平台**；其任务偏 VLA，RL 从头训难度高，只做附录，不占核心预算 |
| **LeRobot**（2602.22818） | —（软件栈） | 端到端机器人学习库 + RL 模块 | 否 | 是 | 真机软件栈基础；HIL-SERL 微调作为 stretch |

**一句话定位（v1 修订 2）**：SER 改 reset 时序、DRATS 改小规模任务集的任务权重、UPER 改回放优先级、GMC/LPM 改 LP 信号本身、**SGS 在单任务内用二值 success 带改 config 权重**；本方案与 SGS **同层**（单任务内的配置桶分配），novelty 边界是"**不确定性感知的 LP 信号（epistemic/aleatoric 分解 + 三重 guard）× 大 N（64–16384）下可检验的 scaling 预测 × 三层实验 + 真机零样本证据**"的组合；跨任务分配仅作 P1 附录（MT10），且**不声称**任何"此前无人做并行环境层分配"。

---

## 10. 论文结构、图表清单、视频脚本、开源计划

### 10.1 论文结构（ICRA 6 页正文 + 引用，**【待核实】** 以 2028 author kit 为准）
1. **Introduction**（0.75 页）：大规模并行 ≠ 更好；图 1 的 motivating 观察（Tier-1 的 uniform 饱和曲线）；贡献 3 条。
2. **Related Work**（0.6 页）：curriculum/level sampling、MTRL 任务采样、不确定性 RL、并行 RL 系统；用第 9 节差异表压缩成 2 段。
3. **Method**（1.5 页）：3.1–3.7；伪代码 1 个；复杂度 1 段。
4. **When does allocation help?**（0.6 页）：第 4 节的 P1–P4 与 toy 分析（不放定理，放"预测 vs 实测"对照图）。
5. **Experiments**（2.0 页）：协议（1 段）+ 图 2–5 + 表 1–3。
6. **Real-robot**（0.6 页）：图 6 + 表 4。
7. **Limitations & Conclusion**（0.3 页）。
8. **References**（约 40 条）。

### 10.2 图表清单（每张图对应哪个 RQ）
| 编号 | 内容 | 数据来源 | 回答 | 备注 |
|---|---|---|---|---|
| **Fig.1** | 方法总览 + motivating 曲线（uniform 在 N=1024 后饱和） | Tier-1 | RQ1（引出） | 单栏/双栏视排版 |
| **Fig.2** | Scaling 曲线：perf vs N（4 方法 × 4 任务，均值±95% CI），叠加理论预测 | Tier-1 + Tier-2 | **RQ1, RQ2** | 主图，必须最漂亮 |
| **Fig.3** | 增益 Δ(N) 与机制指标（组覆盖、已解组浪费率、LP-SNR） | Tier-1/2 日志 | **RQ1, RQ2, RQ3** | 双 y 轴或 2×2 小图 |
| **Fig.4** | 消融：去掉 uncertainty / LP / guard 的落差（条形图 + 每任务的点） | Tier-1/2 | **RQ3** | 含 compute-matched 对照 |
| **Fig.5** | 噪声组实验：注入不可学组后权重–噪声相关性（unguarded vs guarded） | Tier-1 | **RQ3（guard）** | 直接对应 H4 |
| **Fig.6** | 真机：3 方法 × 2 任务的 Wilson CI + sim/real 排名一致性 | 真机 | **RQ4** | 配 3 张真机帧 |
| **Fig.7（附录）** | N×L 网格热图（增益） | Tier-1 | P2 | 支撑理论 |
| **Table 1** | 与近邻工作差异（第 9 节精简版） | 文献 | novelty | 5 行以内 |
| **Table 2** | 主结果：各 N 下 6 方法 × 4 任务的 final perf（均值±CI，最优加粗） | Tier-1/2 | RQ2 | 必须含 p 值与效应量 |
| **Table 3** | 消融汇总 | Tier-1/2 | RQ3 | 6 行 |
| **Table 4** | 真机成功率 + CI + trials 数 | 真机 | RQ4 | 含评测协议脚注 |
| **Table 5（附录）** | 计算量与预算明细（compute accounting） | 本方案第 6 节 | 可复现性 | 审稿人加分 |

### 10.3 视频脚本（90 s，ICRA 视频对录取影响明显）
| 时间 | 画面 | 旁白/字幕 |
|---|---|---|
| 0–10 s | 左右分屏：左边 16384 个 env 的网格闪烁（均匀分配），右边同一画面但高亮"前沿组" | "More environments should mean more learning. Often it doesn't." |
| 10–25 s | 方法动画：N 个 env → M 组 → LP/不确定性 → bandit → 分配回 env | "We allocate environments by learning progress, guarded by uncertainty." |
| 25–50 s | Tier-1 scaling 曲线生长动画（uniform 平、FG 继续升）；叠 P1–P4 预测 | "Gains concentrate where uniform allocation stalls." |
| 50–70 s | Tier-2 视觉任务 rollouts（2–3 个 SO-101 任务，成功率对比数字） | "Same interaction budget, better policies." |
| 70–85 s | 真机双 SO-101 同时执行（LiftCube/StackCube），成功/失败各一次 | "And the ranking transfers to two real SO-101 arms." |
| 85–90 s | 结尾：代码/数据链接 + 团队 | "Code, seeds, and protocols are open." |

### 10.4 开源计划
- **仓库**：`frontierguard/`（方法，~300 行含测试）+ `experiments/`（配置、脚本、预注册）+ `results/`（全部 seed 的 JSON/CSV，不含大 checkpoint）+ `figures/`（可复现脚本）。
- **发布内容**：代码 MIT；配置与随机种子全公开；每个 run 的 wandb 链接；真机评测视频与 rubric；预注册文件；统计脚本。
- **不发布**：预训练大 checkpoint（>5GB）改为提供训练脚本 + 小模型；Squint/ManiSkill3 的模型权重按其 license 处理。
- **时间**：W43 整理，投稿时提供 anonymized 链接（ICRA 双盲要求以当年 author kit 为准），录用后转公开。

---

## 11. 中稿率提升策略（针对 ICRA 评审维度）

### 11.1 最小充分证据集（Minimum Sufficient Evidence, MSE）——必须有，否则不投
1. **Tier-1 完整 scaling 曲线**（4 任务 × 5 N × 4 核心方法 × 5 seeds）：证明 RQ1 饱和 + RQ2 增益；
2. **Tier-2 视觉主结果**（4 任务 × 3 N × 4 方法 × 5 seeds）：证明不是玩具；
3. **≥4 个消融**（去 uncertainty / 去 LP / 去 guard / 换分配器）：证明 RQ3；
4. **真机零样本 30 trials × 2 任务 × 3 方法**：证明 RQ4；
5. **统计规范**（CI、配对检验、预注册、计算量匹配）：应对"统计不严谨"的常见拒稿理由；
6. **差异表 + 组合性实验（SER+FG）**：应对"novelty 重叠"。

### 11.2 加分项（有时间/钱才做，按 ROI 排序）
| 加分项 | 成本 | 对中稿率的估计影响 | 说明 |
|---|---|---|---|
| 计算量匹配（固定 GPU-h 协议） | 0 元（复用曲线） | **高** | 直接回应"不公平比较" |
| 真机 HIL-SERL 微调（+10pp） | 本地 0 元，2 周 | 高 | 机器人审稿人喜欢"真机可用" |
| 任务广度（held-out 4 任务） | ¥130 | 中 | 证明不是挑任务 |
| 理论预测 vs 实测对照图（P1–P4） | 0 元 | 中高 | 提升 technical quality 维度 |
| Tier-3 数字孪生 | ¥721 | 低 | 附录级，ROI 最低 → 首先砍 |
| RND/ensemble 对照 | ¥50 | 低 | 可选 |

### 11.3 针对评审维度的具体动作
| 维度 | 常见失分 | 我们的动作 |
|---|---|---|
| Novelty & Contributions | "只是 curriculum 采样的变体" | 差异表 + 组合性实验 + 明确的"分配层"定位；把"guard/不确定性估计的统计可靠性"作为独立贡献 |
| Technical Quality | 方法细节缺失、复杂度不明 | 伪代码 + 复杂度 + 开销实测 + 消融；理论节只放可检验预测 |
| Experimental Evaluation | baseline 弱、统计弱、只仿真 | 6+ 强基线（含 SER/DRATS/UPER）+ 5 seeds + 配对检验 + 计算量匹配 + 真机 |
| Clarity & Presentation | 图乱、写作差 | 图 2 为主图（scaling curves 最直观）；表格精简；写作 W35 起专门打磨 |
| Robotics Relevance | "纯 RL 论文" | 真机 + SO-101 平台 + 明确的机器人任务；引言从"并行仿真在机器人中的普及"切入 |
| Reproducibility | 代码不开源、超参缺失 | 全开源 + 预注册 + compute accounting 表 |

### 11.4 中稿率的主观估计（**明确标注为判断，不是事实**）
在"ICRA 2028 录取率估计 **35%–42%**（无官方统计来源，属经验估计；ICRA 官方不公布精确接收率）"的前提下，本方案若按 MSE 全部完成且结果符合预期，我对中稿率的**主观先验估计为 45%–55%**：方法新意中等偏上（有明确差异），实验扎实（三层 + 真机 + 统计），主要风险是 R6（novelty 重叠）与 R1（结果不显著）。若只完成 Tier-1+Tier-2 而无真机，估计降到 30%–40%；若 FG 无增益，降到 <15%（此时应转投 CoRL 或 workshop）。**这个数字是方案作者的判断，不是统计推断，也不构成任何承诺。**

---

## 12. 实验运行清单（run-level，含成本与优先级）

优先级：**P0 = 必做（进正文）**；**P1 = 应做（进附录/加分）**；**P2 = 可选（预算允许）**。负责人均为 1 名研究生（本方案假设），"自动"表示脚本化、无需人工。

**关于表内"任务"列（v1 追加）**：T1/T2 的每个 run 都是**单任务、单策略**；"4 独立任务"表示 4 个彼此独立的实验（各自 N 扫描，不做跨任务分配）。FG 的 M 指**该任务内的配置桶数**（T1 默认 8；T2 默认 4–8），不是跨任务 group（原因见 §14.6）。

| ID | 平台 | 任务 | N | 方法 | seeds | steps | GPU | h | ¥ | 优先级 |
|---|---|---|---|---|---|---|---|---|---|---|
| T1-01 | Playground | CheetahRun | 64/256/1024/4096/16384 | PPO-uniform, PLR, SER, DRATS, FG, SER+FG, SGS, SGS+FG（8 方法） | 5 | 150M | 本地 | 0.05–0.30 | 0 | P0 |
| T1-02 | Playground | HumanoidRun | 同上 | 同上 | 5 | 150M | 本地 | 同上 | 0 | P0 |
| T1-03 | Playground | Go1JoystickFlatTerrain | 同上 | 同上 | 5 | 150M | 本地 | 同上 | 0 | P0 |
| T1-04 | Playground | PandaPickCube | 同上 | 同上 | 5 | 150M | 本地 | 同上 | 0 | P0（机器人操作；若 pilot 显示吞吐过低则降 P1） |
| T1-05 | Playground | CheetahRun × L 网格 | 256/1024/4096 | FG, uniform | 3 | 150M | 本地 | — | 0 | P0（P2 验证） |
| T1-06 | Playground | CheetahRun 噪声组注入（2 组） | 1024 | FG, FG-no-guard, uniform | 5 | 150M | 本地 | — | 0 | P0（H4） |
| T1-07 | Playground | CheetahRun 难度阶梯（4 档 δ） | 1024 | FG, DRATS, uniform | 3 | 150M | 本地 | — | 0 | P1（P3） |
| T2-01 | Squint/ManiSkill3 | 4 独立任务（pilot 选） | 256 | 6 方法 | 5 | 15M | 4090 | 2.83 | 848 | P0 |
| T2-02 | 同上 | 4 独立任务 | 1024 | 6 方法 | 5 | 15M | 4090 | 1.44 | 432 | P0 |
| T2-03 | 同上 | 4 独立任务 | 4096 | 6 方法 | 5 | 15M | 4090 | 1.09 | 328 | P0 |
| T2-04 | 同上 | 4 独立任务 | 64 | 3 方法 | 3 | 15M | 4090 | 6.99 | 630 | P1（诊断点） |
| T2-05 | 同上 | 4 独立任务 | 1024/4096 | SER+FG | 5 | 15M | 4090 | 1.27 | 127 | P0（组合性） |
| T2-06 | 同上 | 4 个 held-out 任务 | 1024 | 3 方法 | 3 | 15M | 4090 | 1.44 | 130 | P1（广度） |
| T2-07 | 同上 | 1 任务 × 6 变体 | 1024 | 消融 | 3 | 15M | 4090 | 1.44 | 65 | P0（RQ3） |
| T2-08 | 同上 | 4 独立任务 × 3 方法 | 1024/4096 | 2B 确认 | 2 | 30M | 4090 | 1.27 | 152 | P1 |
| T2-09 | 同上 | 4 独立任务 | 1024 | SGS, SGS+FG | 5 | 15M | 4090 | 1.44 | 144 | P1（从预备金出，可砍） |
| T0b | Squint | 4 独立任务 × 2 N | 64/1024 | PPO-uniform | 3 | 3M | 4090 | 0.88 | 53 | P0（校准） |
| T3-01 | SO-101 Bench/Isaac Lab | 2 任务 | 256/1024/4096 | 3 方法 | 3 | 20M | A100 | 0.83 | 358 | P2 |
| T3-02 | 同上 | 1 任务 | 256/1024/4096 | 3 方法 | 3 | 20M | A100 | 0.83 | 179 | P2 |
| T3-00 | 同上 | 环境搭建 + sanity | — | — | — | 1M | A100 | 0.82 | 118 | P2 |
| R-01 | 真机 SO-101 ×2 | 2 任务 | — | 3 方法 × 30 trials | 1（规则选 seed） | — | 本地 | 4 人日 | 0 | P0（RQ4） |
| R-02 | 真机 | 1 任务 | — | FG + HIL-SERL | 1 | 1–2 h 干预 | 本地 | 3 人日 | 0 | P2（stretch） |

**命令级样例（Tier-2 单 run；单任务、单策略）**：
```bash
# P0 主 run：4 个独立任务之一（lift_cube），N=1024，FG，seed=0，B=15M
# env-id 以 Squint 仓库实际注册名为准【待核实】；M=8 为该任务内的配置桶数
python experiments/train_ppo.py \
  --env-id SO101LiftCube-v0 --num-envs 1024 --rollout-len 24 \
  --total-timesteps 15000000 --seed 0 \
  --sampler frontierguard --groups 8 --bucket-type pose_goal \
  --lp-window 50 --exp3-eta 0.05 \
  --guard-noise --guard-collapse --guard-confidence \
  --eval-interval 500000 --eval-episodes 20 \
  --wandb-project frontierguard-tier2 --run-name t2_lift_N1024_fg_s0
```
**执行与记账纪律（v1 追加）**：每个 run 结束必须落盘 JSON（含 commit_hash、实际 wall-clock、实际 GPU-h）并在 `compute_ledger.csv` 追加一行；每周五用实际 GPU-h × 实际单价对账预算表，偏差 >10% 即触发 §6.4 的砍单预案；禁止手工改配置重跑（配置全部来自 `configs/`、seed 由脚本生成）；任何失败 run 必须写进 `exclusions.csv`（原因、是否纳入分析），预注册后不得因结果调参；租卡实例一律用"按量 + 自动关机 + 启动即拉最新 commit"的镜像，避免空转计费；真机 trial 的证据（视频、成功/失败、耗时）当日归档，禁止事后补标。

**目录与日志规范（**【假设】**，可按团队习惯调整）**：
```
frontierguard/
  frontierguard/{group_sampler,estimators,bandit,guards}.py
  experiments/{train_ppo.py,eval.py,launch_*.sh,configs/}
  results/{tier1,tier2,tier3,real}/*.json
  analysis/{bootstrap.py,regression.py,figures.py}
  preregistration.md  exclusions.csv  README.md
```
每个 run 的 JSON 至少含：method/task/N/seed/steps/status/wall_clock/gpu_hours/final_metrics/eval_history/commit_hash/price_assumption；`exclusions.csv` 记录所有失败 run 与原因。

---

## 13. 开放问题与需要用户确认的决策点

| 编号 | 待决问题 | 选项 | 我的建议 | 影响 |
|---|---|---|---|---|
| **D1** | 目标会议确认：ICRA 2027 已截稿，是否接受 ICRA 2028（2027-09）主目标？ | (a) ICRA 2028；(b) 改 IROS 2027（2027-03，更快）；(c) 改 CoRL 2027（2027-06） | **(a)**，并在 W14 设 IROS 对冲门 | 决定整条时间线 |
| **D2** | 是否投 IROS 2027 做半程对冲？ | 是/否 | 视 W14 结果，**默认不投**（保护主排期） | 分掉 4–6 周写作时间 |
| **D3** | 真机任务偏好 | LiftCube / PlaceCube / StackCube（从 Squint 8 任务选 2） | **LiftCube + StackCube**（难度分层，且 Squint 已验证 sim-to-real） | 影响 RQ4 的说服力 |
| **D4** | 是否保留 Tier-3（Isaac Lab 数字孪生） | 保留（+721 元、+3 周）/ 砍掉 | **默认砍掉**，作为 P2；Lead 已确认其不适合做主 scaling 实验 | 预算与时间 |
| **D5** | 本地 4070 Ti Super 是否可长期独占（每周 ≥10 h）？ | 是/否 | 需要确认；若不能，Tier-1 需租 4090（约 +1,000 元） | Tier-1 主结果 |
| **D6** | 是否允许购买少量耗材（≤500 元）？ | 相机 / 3D 打印件 / 线缆 | 建议：2 个 USB 相机（~300 元）+ 打印件（~150 元） | 真机评测稳定性 |
| **D7** | 论文语言与投稿策略：英文投稿，是否需要中文版本/报告？ | 是/否 | 论文英文；中文方案与结题报告保留 | 写作工作量 |
| **D8** | 是否愿意公开代码与全部 seed 结果？ | 是/否 | **强烈建议公开**（ICRA 可复现性加分） | 中稿率 |
| **D9（v1 追加）** | 是否确认 Tier-2 主实验 = **单任务内 M=8 配置桶 + 4 个独立任务重复**（跨任务分配降为 P1 附录 MT10）？ | (a) 确认（方法上成立，且与 SGS 同层可比）；(b) 仍做跨任务，则需自建 task-conditioned 统一环境（+3–4 周、高失败风险）；(c) 只做 1 个任务，不做任务广度 | **(a)**：共享 PPO 无 task ID 时跨任务 group 不可比；MT10 可满足审稿人"多任务"期待 | 锁定 v2 实验矩阵；critic 不得再按"4 任务当 4 组"审 |

**需要 Lead/用户拍板的 3 件事（按紧急度）**：① D1 目标会议；② D5 本机可用时长；③ D3 真机任务。其余可延后到 W8 前。

---

## 14. SGS（2610.12465）对照与定位修订（v1 追加，2026-11-02）

### 14.1 SGS 是什么，为什么它迫使本方案改写定位
arXiv:2610.12465，*A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control*，CoRL 2026，UW + NVIDIA（Octi Zhang, Mateo Guaman Castro, Patrick Yin, Ignacio Dagnino, Abhishek Gupta, Rosario Scalise, Byron Boots），项目页 https://sgs-rl.github.io/ ；截至 2026-11-02 代码标注 "coming soon"，**未开源**。

按项目页与摘要可核实的内容：固定 N 个 task configuration（初始状态 s0、目标 goal、环境参数 env），每个 config 记录 success 历史；episode 结束后按 success rate 分带加权 reset（never solved 给低权重、improving 给高权重、mastered 给低权重）；外层就是标准 PPO；用到 2^20 个并行环境；任务为 ANYmal 运动控制与 Franka/UR5e 装配；最后把策略蒸馏到 RGB 输入做零样本真机。它的核心论点与本文 RQ1 **正面冲突**：uniform 采样浪费经验，正是 N 增大却看不到 scaling 的原因。

**对 v1 的影响（必须承认）**：v1 原 §3.1 写"没有人把并行环境这一层的分配与 LP 统计可靠性结合"——**这句在 SGS 面前是错的**（SGS 在单任务内的 config 层做了加权分配，且明确针对 mega-scale 的 exploration bottleneck）。本版已改写 §3.1 定位、在 §9 增加 SGS 行，并在本节给出机制级对照、实验补丁与 v2 的定位方案。

### 14.2 机制级对照表（SGS 放首行）
| 维度 | SGS（2610.12465） | DRATS（2605.14350） | SER（2511.21011） | FrontierGuard（本文） |
|---|---|---|---|---|
| 分配单元 | 单任务内的 task configuration（s0/goal/env） | 小规模任务集（MT10/MT50）的任务 | 并行环境的 reset 时序 | **同一任务内的 M 个配置桶**（初始状态/物体位姿/目标区域；大 N 下的 env 槽位）；跨任务仅 P1 附录 |
| 信号 | 二值 success 的三段带（never/improving/mastered） | "离解决最远"（成功率/进度最低） | 无（时间错峰） | LP 斜率（双时间尺度平滑）× epistemic 不确定性 × aleatoric 归一化 |
| 不确定性 | 无显式建模 | 无 | 无 | ensemble 三头分解（epistemic / aleatoric / MI）+ 置信度 guard |
| 稳定性机制 | success 历史 buffer 平滑（未见显式 guard） | 分布鲁棒优化 | 错峰 reset 降低非平稳 | 三重 guard：noisy-TV 过滤、置信度门控、分布塌缩检测 |
| 规模 | 2²⁰ envs（单任务 config） | 小规模（~50 任务） | 大规模（N 扫描） | N = 64…16384（状态）/ 64…4096→1024（视觉） |
| 理论 | 经验 scaling；无 regret 分析 | 鲁棒性分析（非 N 缩放） | 非平稳性论证 | toy MAB：LP 噪声 ∝ 1/√(N L w) + 两臂 bandit regret（P1–P4） |
| 真机 | 蒸馏 RGB 零样本 | 无 | 无 | 零样本 + stretch HIL-SERL |
| 代码 | 未开源（"coming soon"） | 未核实 | 未核实 | 计划开源 |
| 与本文关系 | **同层直接对比**（都在单任务内做 config 级分配）：差异在 LP×不确定性、三重 guard、N-scaling 理论；SGS+FG = baseline #10 | 任务级但无 LP/不确定性/大 N | reset 时序正交（SER+FG = baseline #5） | — |

### 14.3 何时 success-band 足够、何时必须 LP×uncertainty（可证伪）
把 SGS 的权重看成 LP 的**单变量二值代理**：它用 success 的**符号变化**（是否在进步）代替 LP 的**幅度**，且忽略估计噪声。设每轮每组样本 $n_g = w_g N L$，success 估计的标准误 $\sigma_p \approx \sqrt{p(1-p)/n_g}$（二项近似）。由 §4 的 P1/P2 直接推出三条可检验预测：

1. **高并行、短 rollout、大 M** → $n_g$ 小、$\sigma_p$ 大，band 边界剧烈抖动，SGS 权重退化为在几个组间反复横跳、甚至近似 uniform（方差主导）；此时 FG 的双时间尺度平滑 + 置信度 guard 应有增益（**H9**）。
2. **低并行（N≤256）、长 L、M 小** → $\sigma_p$ 小，二值 success 已足够，FG 与 SGS 差距应 <1 CI（**H10**）。
3. **稀疏奖励/长时程任务**（部分 SO-101 装配任务）→ 二值 success 本身不可得，需要 success classifier 或阶段进度；FG 用 return/进度差分，不依赖二值 success，适用面更宽（定性论点，不做强 claim）。

无论结果如何都能写成"success-band 的适用边界"这一有信息的结论——这是面对 SGS 时最稳的落点，也避免"我们又做了一个采样器"的平庸叙事。

### 14.4 实验补丁（并入 §5.4、§12；预算影响很小）
- **baseline #9 SGS-band（reimplementation）**：按论文/项目页实现（固定 config 池 + success 历史 buffer + 三段 band 权重），在 Tier-1 的 M=8 难度桶、Tier-2 单任务的 M=4–8 配置桶上可直接实现（与 FG 同层）。Tier-1 从 6 方法扩到 8 方法：**600 → 800 runs**（本地免费：主网格 ≈78 GPU-h，含 eval/附加实验 1.5× ≈117 h，2× 保守余量 ≈235 h）。**必须标注** "our reimplementation; no official code released at submission time"，附录报告实现假设与超参。
- **baseline #10 SGS+FG（两级分配）**：同一组配置桶上 SGS 的 success-band 权重与 FG 的 LP×uncertainty 权重的门控/凸组合；与 SER+FG 一起构成"组合性"证据链（回答"正交性"质疑）。
- **Tier-2 可选**：N=1024 补 SGS / SGS+FG 两方法 × 4 任务 × 5 seeds = 40 runs ≈ 58 GPU-h ≈ **¥144**，从预备金出（T2-09）；预算收紧时优先砍，不影响 RQ1–RQ4。
- **不新增图**：RQ5 结果叠加在 Fig.3（增益–N 曲线）与 Fig.6（机制指标）上，用不同 marker 区分 SGS；§10 图清单不变。
- **M=4 vs M=8**：这里的 M 是**同一任务内的配置桶数**：Tier-1 默认 8、Tier-2 默认 8（A4 做 M∈{4,8,16} 粒度消融）；**不存在**"把 8 个任务当 8 个 group"的方案（见 §14.6）。8 个 SO-101 任务是 8 个**独立实验**；T2-f 扩到 8 任务 = 72 runs（+36 runs ≈ ¥130）仅作任务广度，超支先砍（C4）。

### 14.5 定位修订与 v2 计划
**修订后的一句话定位**：SGS 管"单任务内的 config 吃多少"、DRATS 管"小规模任务集吃哪个"、我们管"**大 N 下同一任务内的配置桶怎么分**"，并回答"**这种分配带来的增益是否随 N 增长**"——这是 SGS/DRATS 都没回答、且需要新理论（P1–P4）与新实验（N 扫描 + 固定预算 + 真机）的问题。

**主贡献表述（v2 采纳）**：(C1) 同一任务内的大 N 配置桶经验分配（M 组、N→M 映射、与 PPO/staggered reset 的集成；跨任务分配降为 P1 附录 MT10）；(C2) 随 N 的可检验 scaling 预测（H1–H10）；(C3) 三层 + 真机的低成本验证与零样本证据。**不再声称**"首次做并行环境层分配"。

**v2 待办（交给 critic/judge 输入）**：①把 Tier-2 主实验写成"单任务 M=8 配置桶 + 4 个独立任务重复"并按新预算重排（可能砍 Tier-3 换取）；明确 DRATS-env 改造版与 MT10 附录原版的边界；②写清 SGS reimplementation 细节与公平性（超参预算、评测步数一致）；③§4 新增 4.6"二值 success band 是 LP 的特例"的形式化对应，强化理论叙事。


### 14.6 设计修正（v1 末轮，**优先级高于 §14.4/14.5 中的旧措辞**）：Tier-2 的 group 是"同任务内配置桶"，不是"不同任务"

**问题**：v1 早期稿（及 §14.4/14.5 的部分旧句）曾打算把 Tier-2 的 M=4–8 设为"4–8 个 SO-101 任务"，让 FG 在任务间分配并行 env。Lead 审阅指出，这在方法上不成立，我认同：(i) PPO 的 advantage 是策略内的相对量，N 个并行 env 共享同一策略网络，实验里**没有 task ID 条件输入**；(ii) Squint 8 个任务是独立的 ManiSkill 单任务环境，reward 形状、物体、成功判定都不同；(iii) 把不同任务放进同一策略时，LP 的 return/success 信号跨任务不可比，分配信号会被 reward 尺度差而非"可学习性"驱动（DRATS 原版需要 task ID 且在其 MTRL 设置下才有条件网络）；(iv) 更关键的是：**与 SGS 的对比只有在同一层才是 apples-to-apples**。

**修正（已同步写入 §1 摘要、§2 RQ5/H9、§3.2、§5.1、§5.4、§9、§12、§13 D9）**：
1. **Tier-2 主 = 单任务内的配置桶**：每个 SO-101 任务内定义 M=4–8 个桶（物体初始位姿分位、目标区域分位、域随机化档：摩擦/光照/相机抖动），同一 reward、同一成功判定、共享策略；N 扫描在**每个任务内**独立进行。
2. **4（可扩 8）个任务 = 独立实验**：各自做 N 扫描/截断，报告"3 个任务上 FG>uniform、1 个持平"这类**跨任务一致性**，而不是把任务混进一个策略。
3. **Tier-1 同理**：M=8 难度桶（CheetahRun 初速度分位、HumanoidRun 姿态扰动、Go1JoystickFlatTerrain 地形粗糙度、PandaPickCube 立方体初始位姿/目标区域），单任务、单策略。
4. **跨任务分配 → P1 附录（可选）**：MetaWorld MT10/MT50（task ID 在 obs 中，DRATS 原论文同基准；N≤256、CPU 向量化、本地 7800X3D），人工 ≈1.5 人周、本地 GPU ≈60 h（或纯 CPU 更多），**不占租卡预算**；MTBench（2507.23172）仅作 related platform 引用。
5. **不做**：为 8 个 SO-101 任务自建统一 task-conditioned 环境（3–4 周工程量 + 与"scaling"叙事冲突 + 高失败风险）。

**对 SGS 对比是净收益**：FG 与 SGS 现在都在"单任务内 config 级分配"这一层，RQ5/H9/H10 的 Δ(FG−SGS) 是干净的同等条件比较；我们与该近邻的差异因此必须（也只能）落在：(a) LP 斜率 × epistemic/aleatoric 分解（SGS 是二值 success 带）；(b) 三重 guard（noisy-TV / 置信度 / 塌缩，SGS 只用 success 历史 buffer 平滑）；(c) 随 N 的**可检验 scaling 预测**与 toy MAB 理论（SGS 无 regret 分析）；(d) 三层实验 + 真机零样本证据。**v2 的 contribution 表述据此收紧**：不再说"任务级分配"，改说"配置级分配 + 早期 guard + scaling 预测"。

**预算/人力影响**：0 元新增（只是每组内部的组结构变化，runs 数与 GPU 小时不变）；若做 MT10 附录，+1.5 人周（本地免费）。T2-f 的 8 任务扩广（+¥130）仍成立（8 个独立实验）。

**风险与 Plan B**：审稿人若追问"为什么不做任务间分配"，回答：task ID 条件化的多任务采样是 DRATS/MT10 的问题，我们放在附录用原版 DRATS 对比；**Plan B** 若主结果强依赖"多任务"叙事，用 MT10 附录 + 明确 limitations 段兜底（不重排主实验）。

## 附录 A：引用清单（全部经 arXiv API 核实，2026-11-02）

**方法与近邻工作**
1. Staggered Environment Resets Improve Massively Parallel On-Policy Reinforcement Learning. arXiv:2511.21011.
2. Distributionally Robust Multi-Task Reinforcement Learning via Adaptive Task Sampling (DRATS). arXiv:2605.14350.
3. Predictive Bandit Curriculum Reinforcement Learning via Gradient-Based Sample Group Discovery (PGB). arXiv:2610.25704.
4. Uncertainty Prioritized Experience Replay (UPER). arXiv:2506.09270.
5. Gradient-Momentum Coupling: A Parameter-Space Proxy for Learning Progress (GMC). arXiv:2605.05856.
6. Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring (LPM). arXiv:2509.25438.
7. Prioritized Level Replay (PLR). arXiv:2010.03934.
8. Replay-Guided Adversarial Environment Design (RED). arXiv:2110.02439.（作者：Jiang, Dennis, Parker-Holder, Foerster, Grefenstette, Rocktäschel；**注：该 ID 不是 RPLR**）
9. The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks. arXiv:2506.03404.
10. Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks. arXiv:2507.23172.
11. An Empirical Study of Reinforcement Learning Scaling Laws. arXiv:2610.33420.
12. Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning. arXiv:2603.01741.

**平台**
13. Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics. arXiv:2602.21203. 代码：https://github.com/aalmuzairee/squint
14. SO-101 Bench: Accessible Spatial Manipulation. arXiv:2610.36420. 代码：https://github.com/5hadytru/so101_bench（Docker 版：https://github.com/kabilankb/so101_bench_quickstart）
15. LeRobot: An Open-Source Library for End-to-End Robot Learning. arXiv:2602.22818. 代码：https://github.com/huggingface/lerobot；HIL-SERL 指南：https://github.com/michel-aractingi/lerobot-hilserl-guide
16. MuJoCo Playground / Brax：https://github.com/google-deepmind/mujoco_playground
17. Isaac Lab SO-101 外部工程：https://github.com/MuammerBay/isaac_so_arm101；NVIDIA workshop：https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop

**会议/组织**
18. ICRA 2027 官网（Submission Closed）：https://2027.ieee-icra.org/
19. PPO（Schulman et al., 2017）：arXiv:1707.06347（标准算法引用，非近邻工作）

**v1 追加（2026-11-02 二次核实）**

20. A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control (SGS). arXiv:2610.12465. CoRL 2026. 项目页：https://sgs-rl.github.io/ （代码 "coming soon"，2026-11-02 查证未开源）
21. 注：RPLR（Robust Prioritized Level Replay，PLR 的鲁棒变体，常被引作 ICLR 2022）经 arXiv 全库标题检索无记录，**本方案不单独引用**；该路线由 PLR（2010.03934）与 RED（2110.02439）代表。

> 说明：本方案不引用任何未核实的论文、作者、实验结果、价格或录取率；所有"约/假设/待核实"均已显式标注。

## 附录 B：对 Brief 第 5 节 14 项检查清单的覆盖对照

| Brief 项 | 本方案位置 |
|---|---|
| 1 英文题目/摘要/关键词 + 中文 | §1.1–1.6 |
| 2 RQ1–RQ5 与可证伪假设 | §2.1–2.2（H1–H10；RQ5/H9/H10 为 v1 追加） |
| 3 方法形式化 | §3.1–3.8（含伪代码、复杂度、工程量） |
| 4 理论/toy 分析 | §4.1–4.5（P1–P4） |
| 5 实验矩阵（三层 + N 扫描 + 固定预算 + 基线 + 消融 + 统计） | §5.1–5.7 |
| 6 真机实验 | §5.8 |
| 7 计算预算表（≤5000 + 敏感性 + 砍单顺序） | §6.1–6.4 |
| 8 时间线（45 周）+ Go/No-Go | §7 |
| 9 风险登记表 + Plan B | §8 |
| 10 近邻工作差异表 | §9 |
| 11 论文结构/图表/视频/开源 | §10 |
| 12 中稿率提升策略 | §11 |
| 13 实验运行清单 | §12 |
| 14 开放问题与决策点 | §13 |
| 15 SGS 对照与定位修订（v1 追加） | §14；RQ5/H9/H10 见 §2；baseline #9/#10 见 §5.4 |
