# 02 · 对 `01_proposal_v1.md` 的批改意见

> 批改者：proposal-critic（ICRA 资深审稿人 + 实验设计审计员）
> 日期：2026-11-02　｜　被批对象：`01_proposal_v1.md`（662 行，§0–§13 + 附录 A/B）
> 判决：**大修（Major Revision）**。v1 的工程完成度在 ICRA 投稿平均线以上，但**不能按现状执行**：①**Tier-2 把 4 个独立任务塞进同一个未条件化 PPO，方法上不成立（C9，最致命）**；②novelty 定位已被 CoRL 2026 的 SGS 击穿；③归因实验缺一个决定性协议；④真机统计功效不足；⑤预算与排期各有一处不闭合。

## 0. 六维评分（1–5，5 为 ICRA 强接收水平）

| 维度 | v1 | 主要扣分点 |
|---|---|---|
| Novelty & Contribution | **2.0** | §9 定位句"没有人把并行环境分配与统计可靠性结合"在 SGS（2610.12465, CoRL 2026）面前**不成立**；RQ1 的"uniform 饱和"诊断已被 SGS 当作 motivation 做完（且做到 2^20 环境） |
| Technical Quality | **2.5** | EXP3 用在 full-information 设定（错）；理论 §4.2 的 σ_LP 与 §3.3 的 EMA 定义自相矛盾；超参缺默认值表 |
| Experimental Evaluation | **3.0** | 缺 fixed-K（固定更新数）协议 → 固定 B 下"N 增大变差"可能只是更新次数变少；视觉 N=4096 显存不可行；PLR 基线是稻草人 |
| Robotics Relevance | **3.0** | 真机只有 2 任务 × 3 方法 × 30 trials，且 30 trials 无统计功效；Squint 真机的 3D 打印物体/黑背景/颜色匹配前提未计入 |
| Clarity & Presentation | **3.5** | 结构清晰，但摘要 240 词超 150–200；6 页里 Method 占 1.5 页偏多 |
| Reproducibility | **4.0** | 预注册、compute accounting、开源计划是好设计；但缺超参默认值表与 run 级清单的成本一致性 |
| **综合** | **≈3.0** | 对应中稿概率我判断 **25%–35%**（低于用户 >50% 目标），主因是 novelty 与真机证据 |

---

## 1. 我做了什么独立复核（所有结论可复算）

1. **文献**：用 arXiv API 逐个拉取 Brief 4.1 的 15 个 ID + 新发现的 SGS，核对标题/作者/摘要。**结论：15 个 ID 全部真实，但与 Brief 描述有 1 处不符**（见 M0）。
2. **平台**：用 GitHub API 拉 `aalmuzairee/squint` 的 README 与目录，核实：8 个任务 env_id 为 `SO101{Reach,Lift,Place,Stack}{Cube,Can}-v1`、Squint 已是 **IEEE RA-L 2026**（doi:10.1109/LRA.2026.3730387）、`deploy.py` 存在、真机前提含 **3D 打印物体（指定 PLA 颜色）+ 黑色桌面背景 overlay + 相机对齐调参 + LeRobot 电机标定**。
3. **预算**：把 §6.2/§6.3 每一行按 v1 自己的吞吐假设重算，找出 1 处漏算（M3）、1 处口径不一致（m2）、1 处缓冲算错（m4）。
4. **统计功效**：对 §5.8 的 30 trials 做双比例功效计算（见 §6）。
5. **N=4096 视觉显存**：按 ManiSkill3 RGB PPO 的 batch 结构做量级估算（见 C4/M1）。

---

## 2. CRITICAL（必须修，不修则 v2 不可执行/必被拒）

### C1【最严重】SGS（arXiv 2610.12465, CoRL 2026）完全缺席，novelty 定位失效

**事实**（我用 arXiv API 核实）：*A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control*，Octi Zhang, Mateo Guaman Castro, Patrick Yin, Ignacio Dagnino, Abhishek Gupta, Rosario Scalise, Byron Boots（UW + NVIDIA），2026-10-08，CoRL 2026，项目页 https://sgs-rl.github.io/ 。摘要原话：**"uniformly sampling over this distribution wastes a growing fraction of learning experience on task configurations the policy has already mastered or cannot yet attempt. This makes it challenging to see the expected benefits of scaling parallel environments for RL"**；方法 Success Guided Sampling：**按 success-rate 对 task configuration 加权 reset（含 never-solved / improving / mastered 三档）**；规模 **2^20 (>1M) 并行环境**；任务为**多地形四足运动 + 接触丰富装配**；**RGB 蒸馏零样本迁移到真机**。

**问题**：v1 §9 的定位句（"没有人把并行环境这一层的分配与 LP 估计的统计可靠性结合"）与 §2.1 的 RQ1 前提（"uniform 分配在大 N 下浪费经验、看不到 scaling 收益"）**被 SGS 逐字占据**。v1 摘要里"uniform allocation keeps spending samples on environments that are already solved or fundamentally unlearnable"几乎就是 SGS 摘要的复述。

**审稿人会怎么问**：
- 「This is SGS with an ensemble uncertainty term. What is the incremental contribution?」
- 「SGS already shows success-weighted resampling at 2^20 envs on real robot tasks. Your Table 1 doesn't mention it.」
- 「Your RQ1 'discovery' that uniform allocation saturates is the motivation paragraph of a CoRL 2026 paper already published.」

**为什么是 CRITICAL 而不是 MAJOR**：CoRL 2026 早于 ICRA 2028，审稿人**必然**知道；不引用 = 学术不端风险，引用但无机制级差异 = desk-reject 级 novelty 质疑。且 v1 目前连"相关但不同"的辩护都没有。

**具体修改**（v2 已全部落实）：
1. 差异表**首行**改为 SGS，逐维对照（分配单元 / 信号 / 是否含不确定性 / guard / 理论 / N 规模 / 真机 / 代码）。
2. 机制级差异必须写成三段：①SGS 是**单任务内的 task-config 级** success-band 三档；我们在**同一层**（单任务 config 级）用更强的信号（**LP 导数 × epistemic 不确定性** + 三重 guard），因此是**同层 head-to-head**，而不是"我们多做了一个维度"；②SGS 的 success-band 隐含假设"success rate 是无偏进度信号"，在 **aleatoric 噪声配置**（p≈0.5 的高随机/不可学配置）上会把噪声误判为"在进步"，我方的不确定性分解 + guard 正是为此设计——这是**可证伪的差异化主张**；③SGS 无 aleatoric/epistemic 分离、无 guard、无 N-scaling 预测、无理论、无官方代码（截至 2026-11-02 "coming soon"）；跨任务维度的差异放到 **MetaWorld MT10 附录**（C9）。
3. baseline 加 **SGS-band 重实现**（注明 reimplementation，按原文描述的 success 分档 + 加权 reset，无官方代码），并加 **SGS+FG 组合**。
4. RQ 与理论新增一条："**何时 success-band 足够、何时必须 LP×不确定性**"（见 v2 §2.3 与 §4.6），把 SGS 从竞争者变成被解释的对象——这是把威胁转成贡献的唯一办法。
5. 贡献陈述从"新采样器"升级为"**单任务配置级的可靠性感知分配 + 随 N 的可预测 scaling + 低成本真机验证**（跨任务泛化以 MetaWorld 附录补足）"。**注意**：Lead 早期建议的"异构任务集任务级分配"在 SO-101/Squint 上不成立（C9），因此 v2 采用"config 级为主、MetaWorld 任务级为辅"的分层叙事。
6. §11.4 的中稿率估计必须把 SGS 计入：**若不与 SGS 正面对比，中稿概率再降 8–10 个百分点**。

---

### C2【最严重】缺 fixed-K 协议：固定 B 下 N 增大必然减少更新次数，RQ1 的"饱和"极可能是假象

**问题**：固定 B、固定 rollout L 时，更新轮数 K=⌊B/(N·L)⌋。v1 自己的设定（B1=150M、L=16）：
- N=64 → K≈146,484 次更新；
- N=4096 → K≈2,288；
- **N=16384 → K≈572**。

也就是说，"N 越大越差"至少有三种解释：(a) 分配浪费（v1 的主张）；(b) **更新次数少了 256 倍**；(c) 每轮 batch 更大但 PPO 的 epoch 设计没跟上。v1 只做了 fixed-B 与 fixed-GPU-hour 两种协议（§5.2），**没有任何实验能把 (a) 与 (b) 分开**。而 §6.2 里 N=16384 的 run 只跑 B1=150M → 572 次更新，PPO 在这个更新数下本来就学不动，实验设计上是"注定失败"的单元。

**审稿人会怎么问**：
- 「At N=16384 you perform 572 gradient updates. Your "saturation" is an optimization-budget artifact, not a sampling phenomenon.」
- 「Did you show the same N-trend at fixed number of updates? If not, your RQ1 is unfalsifiable.」
- 「Why should anyone care that 16k envs with 150M samples and 572 updates underperforms 64 envs with the same 150M samples and 146k updates? That comparison is not informative.」

**修改**（v2 已落实）：
1. 增加**协议 C：固定更新数 K**（B=K·N·L，K=2,000，L=16）→ 这是唯一能干净归因"N 变化"的协议，放在 Tier-1（免费）。
2. 固定 B 主协议保留（回答用户原问题的"固定交互预算"），但**RQ1 的判定门槛改为必须在协议 B 与协议 C 下同向**；只在 fixed-B 下饱和 = 改写为"更新预算不足的伪 scaling 现象"（这本身也是有价值的诚实发现）。
3. Tier-1 的 N 上限与 B1 绑定：要求 **K(N_max) ≥ 2,000**。若 N_max=4096、L=16 → B1≥131M（v1 的 150M 刚好够）；**N=16384 必须单独用 B1'=512M**（K=2,000），并在同一 B1' 格内自比较，不与小 N 混排。成本可忽略（N=16384 吞吐 1.5M/s，512M 步 = 0.095 h/run，120 runs ≈ 17 GPU-h，本地免费）。
4. 三种协议的结论若不一致，必须在论文里给出**决策树式表述**（"fixed-B 下 X，fixed-K 下 Y，fixed-clock 下 Z"），而不是挑一个报。

---

### C3【最严重】Tier-1 的"异质性"可能是空的：唯一能到 N=16384 的证据层可能根本没信号

**问题**：§3.2 把 Tier-1 的 group 定义为"Cheetah 初速度分位桶 / Humanoid 初始姿态扰动桶 / Quadruped 地形粗糙度桶"，另注入 2 个噪声组。但 P3/§4.3 自己说"当所有组同样难学时（δ→0），FG 应不劣于但也不优于 uniform"。**Cheetah 的初速度分位桶之间可学性差异极小**（都能学，只是初始状态不同），A_g≈1、τ_g≈相同 → δ≈0 → 按 v1 自己的理论，FG 在 Tier-1 上应该**没有增益**。而 Tier-1 恰恰是唯一能到 N=16,384、也是 RQ1/RQ2 主曲线的来源。**主证据层与理论预测自相矛盾**。

**审稿人会怎么问**：
- 「Your own theory (P3) predicts no gain at δ→0. Your main scaling experiment uses initial-state buckets where δ is small. Why do you expect a positive result?」
- 「Cheetah with different initial velocities is not a heterogeneous task distribution — it's the same task.」

**修改**（v2 已落实）：
1. Tier-1 改用**构造性异质族**，让 A_g、τ_g **由构造决定且已知**：
   - **族 A（τ-异质）**：PointMass/Reacher 的 M=8 个目标距离档（0.2–0.9 工作空间）→ A_g≈1、τ_g 随距离单调增，δ 大且可算真 LP；
   - **族 B（A-异质 + 噪声）**：Cheetah 的 8 个目标速度档 {0.5,1,2,3,4,5,6,8} m/s（高速档可能 A_g<1）+ 2 个奖励恒 0 的不可学组 → 同时测 Guard 1；
   - **族 C（同构负对照，δ=0）**：所有 group 完全相同 → FG 必须**不劣于** uniform（预注册"无害性"检验）；
   - **族 D（标准 sanity）**：Cheetah-Run、Humanoid 原版（与公开数字对齐，证明管线没写错）。
2. 因为族 A/B 的 τ_g 已知，"**真 LP**"可以精确算出来 → H7/P1 的"LP 估计 vs 真 LP"不再是空话，且可做 **oracle-LP 消融**（见 M6）。
3. 论文叙事改为"**我们构造可算真值的异质族来验证机制，再用标准 benchmark 证明普适性**"——这比"在标准任务上碰运气"强得多，且成本为 0。

---

### C4【严重】Tier-2 的 B2=15M 是全案"单点故障"；视觉 PPO + N=4096 在 4090 上不可行

**问题 A（B2）**：整个 ¥3,810 的预算建立在 "PPO-uniform@N=1024 用 15M 步达到渐近成功率 50%" 上。但 Squint（RA-L 2026）用 **SAC** 在 8 个任务上分别只要 2–9 分钟 / 1.5M 步收敛；PPO 在视觉操作上通常比 SAC 差 1–2 个数量级的样本效率。若真实 B2=40M，v1 的 Tier-2 主格成本 ×2.67 → 总价约 ¥8,500，**直接破 5000 硬上限**。这不是"风险"，这是**没有被定价的方案主干**。

**问题 B（N=4096 视觉）**：v1 的 Tier-2 设 N=4096，每 run 15M 步。按 64×64×3 观测、帧堆叠 2、rollout L=24 估算，单帧缓冲 4096×64×64×3×4B ≈ 201 MB，×24×2 ≈ 9.7 GB，再叠加 PPO 的激活值与 CNN 中间张量 → 24 GB 的 4090 上**大概率 OOM**（ManiSkill3 官方 PPO baseline 默认 `num_envs=256`，本身就是显存受限的信号）。v1 §5.1 却把 4096 列为 P0 主档（¥328）。

**审稿人会怎么问**：
- 「You report N=4096 with RGB on a 24GB card. What was the peak memory?」
- 「Your baseline uses PPO while the platform's validated results are SAC — are you comparing algorithms or samplers?」

**修改**（v2 已落实）：
1. 视觉档 **N 上限砍到 2048**，且 2048 本身标注"pilot 显存测试 gate"；4096 只在显存测试通过时才做（写进 run 清单 P2）。Tier-1（状态输入）保留到 16384，承担"N 很大"的叙事。
2. **B2 改为三分支预算**（S: ≤15M / M: 25M / L: 40M），每分支给独立可审计的成本表与网格（见 §4 与 v2 §6.3）。三分支都在 ≤¥5,000 内闭合。
3. 增加 **pilot 显存/吞吐测试行**（T0d，12 runs）：测 N∈{128,256,512,1024,2048} 的峰值显存与 steps/s，要求 ≥4 GB 余量，否则降档。
4. G0 判据从"30 min 内 >50% 成功率"（几乎不可能）改为可测的：**N=256、5M 步、LiftCube 成功率 ≥25%，且 15M 步外推 ≥45%**；未达标 → 走 Plan B（proprio-PPO 或 SAC+FG）。
5. 增加 Plan B 的**定价**：SAC+FG 走 Squint 原生路径（`train_squint.py` + `deploy.py` 直接可用），Tier-2 成本可降到分支 S 的 1/3，且真机路径最长的一段被复用。

---

### C5【严重】真机 30 trials/条件没有任何统计功效

**问题**：§5.8 每条件 30 trials（Wilson 半宽 ±0.17，v1 自己算对了），但 H6 却要求 "FG 绝对成功率 ≥ uniform + 10pp" 且 "p<0.1"。双比例检验（p1=0.6, p2=0.5, n=30/组）：SE=0.126, z=0.79, p≈0.43；即使差距 20pp（0.6 vs 0.4），z≈1.58, **p≈0.11**，也不达 p<0.05。**功效 ≈ 0.35**。也就是说，真机章节按 v1 设计有 **2/3 概率测不出真实存在的 20pp 差距**，而 reviewer 只会看到"CI 重叠"。

**审稿人会怎么问**：
- 「30 trials gives you ±17 points. Your claimed effect is 10 points. This experiment cannot support the claim.」
- 「Did you do a power analysis?」

**修改**（v2 已落实）：
1. **配对设计**：每任务预生成 45 组固定初始物体位姿，3 个方法在**同一组初始条件**上各跑一次 → McNemar 配对检验，方差大幅降低；同时天然消除"初始条件不公"的质疑。
2. **合并分析为主终点**：2 任务 × 45 = **90 trials/方法**（双比例 n=90、20pp 差距 → z≈2.74, p≈0.006 ✓）；单任务结果作为次要终点。
3. 2 轮重复（不同日期）→ 报告 batch 间一致性（Cohen's κ）。
4. 主终点改为"**合并成功率差（配对）+ sim/real 排名一致性（描述）**"；显著性目标从 p<0.1 提到 p<0.05（合并后可达）。
5. 人工小时重算：2 任务 × 3 方法 × 45 = 270 trials/轮；双臂并行 + 平均 50 s/trial → 实测 4.5 h/轮 + setup 2 h + 评分 1 h ≈ 1 人日/轮；2 轮 + 物体打印/背景搭建/相机对齐（Squint README 明示的前提）≈ **5–6 人日**（v1 的 4 人日偏乐观但不荒谬）。
6. 新增耗材明细：Squint 任务物体需 3D 打印且**颜色必须匹配**（bin 白 / can 蓝 / cube 红 / large_cube 蓝），加黑背景布 → 预算 ¥300–400，在 Brief 的 ≤500 元耗材额度内，但必须显式列入（v1 §6.1 只写了"USB 相机 2 个"）。

---

### C6【严重】EXP3 用在 full-information 设定上：方法描述有技术错误

**问题**：§3.6 默认分配器是 **EXP3-IX**（对抗性 bandit，部分信息、重要性加权）。但 FG 每轮其实观测到**所有** group 的反馈：Guard 2 保证 w_g ≥ w_min = 0.5/M > 0 → 每组每轮都有 env 在跑，`recent_success(rollout, g)` 对每个 g 都有样本。这是标准的 **full-information**（专家/对冲）设定，此时：(a) 不需要 importance weighting，加上反而引入无谓方差；(b) EXP3 的 √(T log M) 保证依赖的部分信息结构不成立，理论引用会错。

**审稿人会怎么问**（bandit 背景审稿人）：
- 「You use EXP3 with importance weights, but you observe losses for all arms each round. EXP3's estimator is unbiased only in the bandit setting; here it just adds variance. Why not Hedge?」
- 「Your ablation says EXP3 vs UCB vs Boltzmann — the differences you'd see are estimator noise, not algorithmic insight.」

**修改**（v2 已落实）：
1. 主分配器改为 **full-information Hedge/EXP3 的去重要性加权版本 + 非平稳折扣**（或直接 **discounted UCB1 / Thompson**），并在文中写明 "we observe group-level feedback for all groups each round, hence full-information"。
2. EXP3-IX 保留为消融 A5 的一个选项，并**明确写出其适用条件与为何不选**——这反而会加分（说明作者懂 bandit 文献）。
3. 理论节与 bandit 文献的对接改为 "expert advice with switching costs" 或 "restless bandit 的启发式"，**不要**写 O(√(MT log M)) 这类会被抓的表述（v1 §3.8 的验收标准里就有这条，必须删）。

---

### C7【严重】排期零缓冲：W45 与 2027-09-15 只差 3 天，且未计入摘要/视频截止

**问题**：§7 的 W45=09-06～09-12，距预计截稿 09-15 仅 3 天。ICRA 的实际流程通常是：**摘要/标题注册截止（约早 1 周）→ 全文截止 → 视频截止（全文后 3–5 天）**。v1 在 W43–W44 才做"author kit 格式、附录、代码开源整理"，W45 提交；任何一周滑期（这在 45 周的 solo 项目里概率 >70%）就直接错过，且**没有考虑 video deadline 晚于 paper deadline**——视频如果按 paper deadline 也交，那就必须在提交前完成（v1 W40–W42 拍视频，看似合理，但一旦实验滑期就被挤掉）。

**审稿人不会看到**，但这是**项目失败风险**（比拒稿更严重）。

**修改**（v2 已落实）：
1. **实验冻结提前到 W30**（v1 是 W34）；写作 W31–W38；内部评审 + 图表精修 W39–W40；视频拍摄 W40；提交 W41（≈2027-08-09）；留 **5 周**缓冲。
2. 单列"**投稿日历**"：abstract registration（预计 2027-09-01 前后，待核实）/ paper deadline（预计 2027-09-15，待核实）/ video deadline（预计 2027-09-20，待核实），三者各自留 ≥3 天。
3. 设 **W26 硬性 checkpoint**：若 Tier-2 主格完成 <50%，立即执行"降级到 2 benchmark + 真机"的 6 页版本（在社会上比"什么都做一半"强）。
4. 对冲选项（IROS 2027）的决策门从 W14 移到 W12，避免与 Tier-2 pilot 抢时间。

---

### C8【严重】理论自相矛盾：§4.2 的 σ_LP 与 §3.3 的 EMA 定义不能同时成立

**问题**：§3.3 定义 EMA 平滑窗口为"**约 20 个 episode**"（per group），也就是 n^win 恒等于 20（或 200），**与 N 无关**。但 §4.2 写 Var(ŝ) = p(1−p)/n^win 且 n^win ∝ N·L·W/H，进而 σ_LP ∝ √(M/(N·L·W))，于是 P1 声称 LP-SNR ∝ (N·L·w)^{1/2}。**这两个定义互相打架**：如果窗口按 episode 数定义，σ_LP 与 N 无关；如果按 env-step 窗口定义，就不是 §3.3 说的"20 个 episode"。

**审稿人会怎么问**：
- 「Your Eq. (4) claims σ ∝ 1/√(NLW), but your estimator's EMA horizon is 20 episodes per group, independent of N. Which is it?」
- 「P1 is the only quantitative prediction you test; it's derived from an estimator you don't implement.」

**修改**（v2 已落实，见 v2 §4.2）：把两层噪声分清楚——
1. **单轮噪声**（一次 rollout 内，组 g 得到 m_g = w_g·N·L/H 个 episode）：Var(ŝ_g^{round}) = p(1−p)/m_g ∝ 1/(w_g N L) → 单轮 SNR ∝ √(w_g N L/H)。**这是 P1 应该检验的量**（对 √N 回归，斜率 0.5）。
2. **EMA 滞后**：EMA horizon 固定为 20 episodes ⇒ 滞后轮数 ℓ_g = 20·H/(w_g N L) ∝ 1/(w_g N)。→ P1b：**估计滞后随 N 线性下降**（可测，也解释了为什么大 N 反而"反应更快"）。
3. 相应地把 §4.3 的推导改成"**累积策略误差 ∝ √(N L/B · Σ_g w_g²σ_g²)**"（K 次更新的随机游走），给出 P2 的**定量**预测：Δ(N) ∝ √N（小误差区），并在论文中明确这是启发式模型而非定理。

---

### C9【最严重·方法硬伤】Tier-2 把"不同任务"当 group 塞进同一个未条件化 PPO：实验根本不成立；且 M 的语义前后不一致

**问题（两层）**：

**(a) 跨任务共享策略是错的方法定义，不是工程风险。** §3.2 写 "Tier-2（SO-101 任务集）M=4"，§5.4 写 "N 个 env 均匀分到 M 组"，§12 的 run 清单与命令样例都是 `--M 4` 把 4 个 SO-101 任务（Reach/Lift/Place/Stack，Cube/Can 变体）当作 4 个 group 同时训练，**所有 env 共用同一个 π(a|s)**。但 Squint 的 8 个 env 是 8 个**独立的 ManiSkill 单任务环境**，各自有不同的 **reward 函数、物体几何、初始分布、成功判定**；v1 的 policy 输入是本体感知 + RGB，**没有任何 task ID / goal 条件输入**（§3.7 的 actor-critic 就是标准 PPO MLP，没有 embedding）。后果：

- 同一图像/状态在 Reach-Cube 与 Place-Cube 下语义相同，但最优动作不同；未被条件化的 π 只能学"混合动作分布"，等于对 4 个 MDP 做行为克隆平均 → **4 个任务全部失败**；
- value function 拟合的是 4 个不同 reward 的混合，advantage 符号在不同 group 间互相抵消，LP 估计（§3.3）也失去意义；
- 更根本：**"1 个策略 + 4 个 MDP" 不是 PPO 能优化的目标**——PPO 的 objective 是 E_{s~d^π}[A^π(s,a)]，而 A^π 依赖 reward 函数；没有条件变量时这是不可定义的。

这不是"跑得慢"的问题，是**照 v1 写代码会直接得到 8 个全 0 成功率的实验结果**，等于白烧 1,200 GPU-h 和 20 周。审稿人（尤其做多任务 RL 的）会第一页就看出来。

**(b) M 的语义混乱。** §3.2 里 Tier-1 的 M=8 是"**任务内初始状态/难度桶**"，Tier-2 的 M=4 是"**任务**"；§3.3–§3.6 的 LP、双 EMA、bandit 公式却统一写成对 g=1..M 的同一套记号。审稿人会问"你的 group 到底是什么粒度"，而且两种粒度对应的**可辨识性、真实 LP 的可定义性、归一化方式完全不同**（M7 已指出 LP 跨任务不可比——这正是因为 v1 把两种粒度混在同一个 M 里）。

**审稿人会怎么问**（三连，任何一句答不上就 Reject）：
- 「PPO optimizes a single policy for a single MDP. Your Tier-2 trains one policy on 4 environments with different reward functions and no task identifier. What objective is being optimized?」
- 「How does the policy know which task it is in? Nothing in your observation space encodes the task.」
- 「Your LP estimator is indexed by group g. In Tier-1, g is a difficulty bucket of one task; in Tier-2, g is a task. These are not the same object. Please define g precisely.」

**修改（v2 已落实，这是 v2 最大的结构性改动）**：

1. **Tier-2 主实验改为"单任务内 config 级分配"**：固定一个 SO-101 任务（同一 reward、同一成功判定、同一 policy），把该任务的**配置空间**切成 M=4–8 个 group。group 的合法定义（按优先级，v2 同时给出）：
   - **首选：goal-region × DR 桶的笛卡尔积**。SO-101 Reach/Lift/Place 的**目标位置在 obs 中**（goal-conditioned，待核实具体 obs 字段），因此"目标区域桶"是策略**可辨识**的子任务；再乘 2–4 个域随机化桶（物体初始位姿范围/质量/光照）→ M=8–16。reward 仍是同一函数 → **合法 MDP 族**；
   - **Fallback（若 obs 不含 goal 或目标不可条件化）**：纯 DR 配置桶（M=4–8，物体位姿/质量/摩擦/光照），单层。这一条写进**预注册的 fallback 分支**，不做事后选择。
2. **4–8 个 SO-101 任务改为"独立实验"**：每个任务各自跑 N 扫描（每个任务内部仍是 config 级分配），**不跨任务共享策略**。这构成 v2 的 breadth 证据（"FG 在 8 个不同操作任务上一致有效"），而不是"一次训练 8 任务"。
3. **task-level 叙事移到附录**：用 **MetaWorld MT10/MT50**（obs 里拼 task one-hot，MT10 的 39 维 obs → 49 维；DRATS 也用该基准）做"跨任务分配"的附录实验，N≤128–256，**本地 CPU 并行**（MetaWorld 是 CPU/MuJoCo，不吃 GPU），只作泛化性证据，**不作为 N=16k scaling 的主证据**。MTBench（2507.23172）基于 IsaacGym Preview 4，40 系卡 + 新驱动风险高，v2 明确**不做核心**。
4. **明确写出不采用的方案**：为 8 个 SO-101 任务自建 task-conditioned 统一环境（统一 reward/成功判定 + task embedding）估算 **3–4 周 + 高失败风险**，v2 **不采用**；若审稿人问"为何不做真正的多任务"，答案就是这条成本/风险论证 + MetaWorld 附录。
5. **统一 M 语义**：全篇只保留一个定义——"**group = 一个 (任务, 配置桶) 单元；主实验固定单任务，因此 group 退化为配置桶**"；Tier-1 与 Tier-2 的 g 定义一致。§3.2/§5.1/§5.4/§12 全部同步。

**为什么这样反而更强**：改完后 FG 与 **SGS 在同一层**（单任务 config 级的 success-band 采样），**head-to-head 直接可比**，差异表第一行就能写清"同层、同预算、同 seed，只是信号不同"——这比 v1"我在任务级做 SGS 没做的事"要干净得多；而且 v1 的"双层"卖点可以保留为 **"配置空间的两级分解（目标区域 × DR）"**，仍在单任务内，不涉及跨 MDP 的策略共享。

---

## 3. MAJOR（应修；不修会被审稿人抓，但不至于直接拒）

### M0 事实错误：2110.02439 不是 RPLR
- 实测标题：*Replay-Guided Adversarial Environment Design*（RED, NeurIPS 2021）。v1 §3.1、§5.4（baseline #2）、§9、附录 A #8 四处都写成 "PLR / RPLR（2010.03934 / 2110.02439）"。**RPLR（Robust Prioritized Level Replay）我没有找到可核实的 arXiv ID，不敢替 v1 编一个**。
- 攻击：「The authors cite RPLR but give the RED paper's ID. Sloppy.」（这句在 ICRA 足以让一个 reviewer 打出 "technical quality: 2"）
- 修改：统一改为"**PLR 2010.03934 + RED 2110.02439**"，RPLR 若要引用必须标注"待核实"或直接删。

### M1 视觉 N=4096 与 v1 自己的吞吐表不一致
§6.2 假设 N=4096 时 4.0K steps/s（比 N=1024 的 3.0K 还高 33%）。ManiSkill3 在 RGB 下扩展性通常在 N≥512 后趋平甚至下降（渲染线程/PCIe/显存带宽受限），4.0K 是乐观值。修改：改为 N=1024:3.0K、N=2048:3.6K、N=4096:3.8K，并在 T0d 实测校准；同时对吞吐 −30% 情景做预算复核。

### M2 §6.2 "524 runs" 与逐行相加 538 不符
逐行加：T2-a 36 + T2-b 120 + T2-c 120 + T2-d 120 + T2-e 40 + T2-f 36 + T2-g 18 + T2-h 48 = **538**。加上 T0b(24) 为 562，A100 侧 T0c(18)+T3-a(54)+T3-b(27)=99，总计 **661 runs**。修改：所有口径统一为 661 total / 538 Tier-2。

### M3 T2-h 按 15M 步定价，实际写 30M → 漏算 ¥146
§12 的 T2-08 写 `steps=30M`，§6.2/§6.3 却用 h/run=1.27（=15M 的平均值）。按 30M 重算：N=1024 → 2.83 h，N=4096 → 2.13 h，均值 2.48 h；48 runs × 2.48 = **119 GPU-h = ¥298**，v1 写 61 h / ¥152，**少算 58 h / ¥146**。修正后 v1 总价 = **¥3,994**（含 12% 缓冲），预备金从 ¥1,190 降到 ¥1,006。

### M4 12% 缓冲算错
1105×12% = 132.6（v1 写 130）、82×12% = 9.8（v1 写 8）。修正后缓冲 = ¥350 + ¥79 = ¥429（v1 写 ¥390）。

### M5 砍单序列 C2 放错位置
C2 是"真机 3 方法→2 方法"，省 **¥0**（真机不花租卡），却被排进"省租卡"的预算砍单链。修改：拆成两条独立清单——**省钱清单**（C1/C3/C4/C5/C6/C7）与**省人工清单**（C2/D2/HIL-SERL），避免在预算压力下误砍零成本项。

### M6 缺 "oracle-LP" 消融（最有价值的一个实验）
若用**真 LP**（Tier-1 构造族里已知的 τ_g/A_g）替代估计值，FG 性能 = 估计版 → 说明瓶颈在"分配"；若 oracle 明显更好 → 说明瓶颈在"估计精度"，方法叙事要改。这个实验在 Tier-1 上**零成本**（只需改一个 flag），却能直接回答 RQ3 的核心质疑。v1 完全没有。

### M7 LP 跨任务不可比（M=8 任务池时致命）
§3.3 的 LP 直接建立在成功率差分上，但 8 个 SO-101 任务的成功率量级差异巨大（Reach 类 2 分钟收敛 vs Stack 类 9 分钟）→ LP 尺度差异可能达到 10×，bandit 会被高难度任务主导，而不是"按学习前沿分配"。修改：LP 必须**组内归一化**（per-group z-score 或 relative LP = (ŝ^f−ŝ^s)/(1−ŝ^s) 的无量纲形式），并做 A8 消融对比"原始 LP vs 归一化 LP"。

### M8 缺失的超参数默认值表（可复现性）
v1 给了 γ=0.5、ε_KL=0.5、w_min=0.5/M、n_min=30、δ=0.05、c=0.5、β_noise=0.1、η 公式，但 **τ_ale 没给值**、**W 主值没给**（只列了 {10,50,200} 消融）、**κ 主值没给**（只列 {0,0.5,1,2}）、**E=3** 给了但更新频率没给。修改：v2 §3.9 加"默认超参表"并声明"**跨 N、跨任务固定不调**"（直接回应"每个任务都要重调不叫 scalable"的攻击）。

### M9 Guard 1 的判据不可靠
"lag-1 自相关 < 0.1" 需要至少 10–20 个窗口点才能稳定估计，而 v1 的窗口是 20/200 episodes，在 N 大时每轮就有很多 episode → 自相关几乎一定显著。修改：改用**两样本趋势检验**（窗口前半 vs 后半的 LP 差 + bootstrap CI）或 LPM 式的"进步是否可持续"检验，并给 τ_ale 的确定方式（按 aleatoric 分位数取 top-k，k=1 或 2）。

### M10 "去掉全部 guard"（A3）会连带删掉 w_min → 估计饥饿混淆
§5.5 A3 写 "raw argmax/greedy"，这会让部分组 w=0 → 没有样本 → LP 永远无法恢复。这样的消融结果只能说明"删掉探索下限会坏"，不能说明 "noisy-TV guard 有用"。修改：拆成 A3a（去 Guard1+3，**保留** w_min）与 A3b（另去 w_min，单独报"饥饿"现象）。

### M11 PLR/RED 基线是稻草人
§5.4 把 PLR 映射成"按 TD-error/return 优先级采样 group"，丢掉了 PLR 的核心：**level buffer + staleness 混合 + score = max(return/TD)**。修改：按 PLR 公式实现 group 级版本（buffer 存 group 历史分数、混合 staleness × max-score），超参照原文；并在文中写清"我们做的是 group 级适配，非原文的 level 级"。否则 reviewer 一句"strawman baseline"就能否掉整个对比。

### M12 compute-matched 控制的定义缺失
§5.4 说"把 FG 的辅助 FLOPs 折算成额外 PPO 更新给基线"——折算比例怎么算？修改：给出公式 **N_extra = (τ_FG/τ_step)**，其中 τ_FG 为每轮 FG 开销（实测，目标 <5%）、τ_step 为单次 PPO 更新耗时；同时在协议 B 里报"同等 wall-clock"曲线。两处一起才构成"算力匹配"。

### M13 论文体量与图表不一致
§10.1 给 Method **1.5 页**（6 页论文的 25%），而 Method 里的 guard/bandit/不确定性三项在正文只值 1.0 页；§10.2 的 Fig.2 写"4 方法 × 4 任务"但主格是 6 方法 × 8 任务（v2）。修改：Method 1.0 页、When-does-allocation-help 0.5、Experiments 2.3、Real robot 0.7；所有图表标题与最终矩阵一致。

### M14 摘要 240 词，超出 150–200 词要求
§1.4 英文摘要实测约 245 词。修改：压到 190 词以内，并把 SGS 的差异化写进第一句之后（"Unlike SGS's success-band, ..."）。

### M15 Tier-3（Isaac Lab）应默认删除，而不是 P2
理由：①SO-101 Bench 是语言条件 VLA 基准（arXiv 2610.36420 摘要核实：4 任务 5 族 / 138 物体 / 916 评测 episode / 3629 演示 / Spearman ρ=1.00），**从零 RL 训练不是它的设计用途**；②Isaac Sim 5.1 在本地 32 GB 内存不保证可跑，租 A100 也有版本坑（+2 人周）；③它对 RQ1–RQ4 没有不可替代的贡献。砍掉后省 ¥721 + 2 人周，正好补 Tier-2 的 M=8。

### M16 8 任务广度（T2-f）与 M=8 主格重复——**此条在 C9 修正后部分撤回，只保留成本问题**
C9 修正后，"8 任务"不再是"跨任务共享策略"，而是**8 个独立任务各自做 config 级分配**，因此 breadth 有独立价值（证明不是单个任务的偶然现象），**不再是重复实验**。保留的意见只有成本：v1 的 T2-f（8 任务 × 2 方法 × 3 seeds @N=1024）需要 8 个任务的 env/资产全部走通 + 评分脚本，人工 ≈2 人周；v2 把它降为 **N=1024、2 方法（uniform/FG）× 3 seeds × 8 任务**（正是这个规模），并明确"**若 W26 前 4 个 deep 任务未跑通，breadth 降为 4 任务**"。省 ¥130 的说法作废。

### M17 事实错误：Tier-1 的 "Ant"、"Quadruped" 不是 MuJoCo Playground 的合法 env 名
v1 §5.1、§6.2、§12 的 Tier-1 任务写 "Cheetah / Ant / Quadruped / Humanoid"。核对 MuJoCo Playground（`mujoco_playground`）源码，合法注册名是 **`CheetahRun`、`HumanoidRun`、`HumanoidStand`、`Go1JoystickFlatTerrain`、`Go1JoystickRoughTerrain`、`PandaPickCube`、`PandaPickCubeOrientation`、`BarkourJoystick`** 等；**没有叫 "Ant" 或 "Quadruped" 的 env**（Ant 是 Gym/MuJoCo 老基准的名字；Playground 的四足是 Go1 / Barkour）。

- 攻击：「The authors list `Ant` and `Quadruped` as MuJoCo Playground environments. I could not find either. This calls into question whether the experiments were run at all.」——"环境名不存在"在 ICRA 是致命的可复现性质疑，它暗示作者没真正跑过。
- 修改（v2 已落实）：Tier-1 主任务定为 **`CheetahRun` / `HumanoidRun` / `Go1JoystickFlatTerrain` / `PandaPickCube`**。前三个给运动控制 + 域随机化多样性；**`PandaPickCube` 是状态输入的操作任务，直接把方法连到 ICRA 的机器人操作主线**（部分回应"这只是 locomotion 论文"的攻击）。同步改 §5.1、§6.2、§12 的 run 清单与命令样例。

### M18 "模块堆叠"观感：FG 的 5 个部件需要一张"部件→失效模式→消融"对应表
v1 §3 依次给了双 EMA LP、3 头 ensemble MI 分解、三重 guard（噪声/饥饿/KL）、三种 bandit、PPO 集成。审稿人第一反应是 "over-engineered, each component lacks independent evidence"。v2 必须在 §3.0 加一张 **部件 → 目标失效模式 → 对应消融（A1–A10）** 的三列表，并写明"**若 A1/A2 显示 LP 或不确定性单独即可解释全部收益，则 FG 简化为单信号版本**"——让方法本身可被自己的消融证伪，reviewer 认这种诚实。

---

## 4. MINOR（可修可不修，但修了更专业）

| # | 问题 | 修改 |
|---|---|---|
| m1 | §12 命令用 `SO101LiftCube-v0`，Squint 实际注册的是 **`-v1`**（README 表格核实） | 全部改 `-v1` |
| m2 | Squint 应引 **RA-L 2026**（doi:10.1109/LRA.2026.3730387）+ arXiv 2602.21203，v1 只写了 arXiv | 补 doi 与"期刊版"说明 |
| m3 | §5.1 说"用 Squint deploy.py"，但 deploy.py 适配 Squint 的 SAC 观测/动作接口，PPO 需改（obs 键、帧堆叠、动作裁剪、squint 分辨率调度） | 工程量 +1 人周，并在风险表登记 |
| m4 | 存储/网络 <¥100 未单列 | 列为独立行 |
| m5 | §5.6 混合模型 `(1|seed)`、`(1|task)`，5 seeds/4 tasks 的方差分量不稳 | 主分析改配对置换 + Holm；混合模型降为"探索性" |
| m6 | §5.6 主指标写"最后 10%/20% 步"，且 RQ2 门槛混用"3pp 成功率或 5% 回报"两个终点 | 每个 RQ 只留 1 个主终点 + 二手终点清单 |
| m7 | H7 的"真 LP"没定义 | 用 C3 的构造族，真 LP = τ_g/A_g 的解析式 |
| m8 | §2.2 预注册在 W12 冻结，但 Tier-1 主格 W8–W10 已在跑 | 写明"冻结前只做 pilot 可视化，不做主分析" |
| m9 | §11.4 "ICRA 2028 录取率 40%" 不存在的数字 | 改为"近届 35%–42%（估计，以官方统计为准）" |
| m10 | 视频脚本 90 s；ICRA 通常 2–3 min | 同时准备 90 s 与 3 min 两版 |
| m11 | §6.3 口径把 T0c 归入"Tier-3 99 runs"，T0b 归入"Tier-2" | 统一用"平台层"而非"Tier"口径 |
| m12 | Fig.5 噪声组实验只做 Tier-1 | 至少在 Tier-2 加 1 个"含不可学组"的补充实验（成本已含） |

---

## 5. 预算重算（基于 v1 自己的吞吐与单价假设）

### 5.1 逐行复核

| 行 | v1 runs | v1 GPU-h | v1 ¥ | 复核 GPU-h | 复核 ¥ | 说明 |
|---|---|---|---|---|---|---|
| T0b | 24 | 21 | 53 | 21.2 | 53 | ✓ |
| T2-a | 36 | 252 | 630 | 251.8 | 630 | ✓（N=64 最贵，6.99 h/run） |
| T2-b | 120 | 339 | 848 | 339.4 | 848 | ✓ |
| T2-c | 120 | 173 | 432 | 172.8 | 432 | ✓ |
| T2-d | 120 | 131 | 328 | 130.8 | 327 | ✓（但吞吐假设偏乐观，见 M1） |
| T2-e | 40 | 51 | 127 | 50.6 | 127 | ✓ |
| T2-f | 36 | 52 | 130 | 51.8 | 130 | ✓（v2 删除） |
| T2-g | 18 | 26 | 65 | 25.9 | 65 | ✓ |
| **T2-h** | 48 | **61** | **152** | **119.1** | **298** | **✗ 按 15M 计但实际 30M，+58 h / +¥146** |
| T0c | 18 | 15 | 118 | 14.8 | 118 | ✓ |
| T3-a | 54 | 45 | 358 | 44.8 | 358 | ✓ |
| T3-b | 27 | 22 | 179 | 22.4 | 179 | ✓ |
| **小计** | **661** | **1,188** | **3,421** | **1,246** | **3,567** | |
| 12% 缓冲 | — | 138 | 389 | 150 | 430 | v1 写 130+8，少算 |
| **合计** | 661 | — | **3,810** | — | **3,997** | **v1 少算 ¥187** |

### 5.2 v1 敏感性分析的三处逻辑不自洽

1. **"单价+50% 且小时+50% = 8,573，C1–C6 只砍 1,824 → 6,749 > 5000"** —— 这里把**基准价下算出的砍单收益**直接减到**压力情景**里。正确做法是在压力情景下重算砍单收益（单价 ×1.5、小时 ×1.5 ⇒ 省下的钱也 ×2.25）。按 2.25× 重算，C1–C6 的收益是 4,104，8,573 − 4,104 = 4,469 ≤ 5,000 → **其实闭合**。v1 的结论"不闭合"是算错导致的，但结论方向（需要砍很多）是对的。
2. **C1 说砍 Tier-3 省 ¥721**，但 Tier-3 的 A100 行是 655 元 + 缓冲 ≈ 734；压力情景下应是 ¥1,650。数字需按情景参数化。
3. **C2（真机 3→2 方法）省 ¥0** 却排在预算砍单链的第 2 位（见 M5）。

### 5.3 修正设计（C9 之后）下的 v2 预算：**本地免费 + 租卡只买大 N**，总价降到 ¥1,800 量级

v1 的预算压力主要来自一个错误假设：**所有 run 都要租卡**。实际上 ManiSkill3 RGB PPO 的官方示例默认 **N=256、显存 <15 GB**，本地 4070 Ti Super 16 GB 完全跑得动；只有 N=1024/2048 才需要租。重划后（B2=15M 步、L=50）：

| 行 | 设计 | runs | h/run | GPU-h | 平台 | ¥ |
|---|---|---|---|---|---|---|
| T2-00a pilot | N=256, 3M 步 | 36 | 0.61 | 21.8 | **本地** | 0 |
| T2-00b 显存/吞吐测试 | N=1024,2048 | — | — | 3 | A100 | 24 |
| T2-L1 N=64 sanity | 2 任务×2 方法×5 seeds | 20 | 6.99 | 139.9 | **本地** | 0 |
| T2-L2 N=256 主格 | 4 任务×6 方法×8 seeds | 192 | 2.83 | 542.9 | **本地** | 0 |
| T2-R1 N=1024 主格 | 同上 | 192 | 1.44 | 276.3 | 4090 | 691 |
| T2-R2 N=2048 确认格 | 3 任务×4 方法×5 seeds | 60 | 1.15 | 68.8 | 4090* | 172 |
| T2-R3 breadth | 8 任务×2 方法×3 seeds @1024 | 48 | 1.44 | 69.1 | 4090 | 173 |
| T2-R4 消融 | 1 任务×8 变体×3 seeds @1024 | 24 | 1.44 | 34.5 | 4090 | 86 |
| T2-R5 扩展基线 | 4 方法×3 任务×3 seeds @1024 | 36 | 1.44 | 51.8 | 4090 | 130 |
| T2-R6 2B 确认 | 3 任务×2 方法×3 seeds @1024 | 18 | 2.83 | 50.9 | 4090 | 127 |
| **小计** | **626 runs** | | | **1,256** | 本地 705 + 租卡 551 | **¥1,403** |
| 12% 缓冲 | | | | 151 | | 168 |
| 电力/耗材/杂项 | 本机 1,220 h 电费（≈¥185）+ 3D 打印/背景/线材 | | | | | 250 |
| **合计** | | | | | | **¥1,821** |

\* N=2048 的 68.8 h 若 4090 24 GB 装不下（很可能），改 A100 80 GB，该行从 ¥172 变 **¥550**，总价 ¥2,199。**这一行是唯一需要 pilot 决定的租卡项，且无论哪种都闭合。**

**本地负载核对**：本地 GPU-h = 21.8 + 139.9 + 542.9 = 705（Tier-2）+ 515（Tier-1：414+36+65）= **1,220 h**，摊到 45 周 = **27.1 h/周**，即本机 16% 占空比——完全可行（甚至可以在夜间批量跑）。**这正是 v1 漏掉的免费算力**：v1 的 T2-a（N=64，252 h）与 T2-b 的一部分（N=256，339 h）共 591 h 本可以本地跑，白花了 ¥1,478。

### 5.4 三情景 + 5000 硬上限保底（v2 §6.5）

| 情景 | 触发 | 动作 | 租卡花费 | 合计 | ≤5000 |
|---|---|---|---|---|---|
| 基准 | 单价/吞吐符合假设 | 全网格（N=2048 走 4090） | ¥1,403 | **¥1,821** | ✓（余 3,179） |
| 单因素 +50% | 单价 >3.75 或吞吐 <2/3 | 全网格，不改设计 | ¥2,357 | ¥2,607 | ✓ |
| 最坏现实 | N=2048 必须 A100 + 单价/小时 +50% | 全网格 | ¥2,993 | ¥3,243 | ✓ |
| 双因素 2.25× | 单价与吞吐同时恶化；N=2048 走 A100 | 全网格（**不砍实验**） | ¥4,489 | **¥4,739** | ✓（余 261） |
| 极端保底 FLOOR | 双因素 + 需砍到最小可发表 | 砍 N=2048/breadth/扩展基线，保留 4 任务×6 方法×8 seeds×{256,1024}+消融+2B | ¥2,339（双因素价） | **¥2,589** | ✓（安全系数 1.93） |

**关键结论（与 v1 相反）**：**5000 元不是约束，甚至不需要砍单链**。本地免费算力吃掉了 Tier-2 的 705 GPU-h（Tier-1 再 515 h，合计 1,220 h），租卡只剩 551 h；双因素压力下也只是 ¥4,739（最坏情景仍余 ¥261）。所以 v2 应当**把省下的钱用于统计功效与覆盖度，而不是"为了花完预算"加实验**：
1. seeds **5 → 8**（主格）；若愿意，8 任务 breadth 的 seeds 3 → 5（+¥115）；
2. N=2048 视觉档**默认保留**（v1 直接砍掉，是过度保守）；
3. **SGS-band baseline 与 SGS+FG 组合各追加一组 run**（已含在 T2-R5）；
4. 余下 ~¥2,000 作为**真机耗材 + 意外租卡**的缓冲（A100 临时补跑、视频渲染等），不预分配。

### 5.5 一条新增的诚实性要求：预算表必须写"人力成本"这一行
¥5,000 只覆盖租卡，但项目的真实成本是 **1 人 × 45 周**。v2 预算表必须加一行"**人力：45 周 × 1 FTE，折算 ≈ 33 人月**"，并在 Limitations 里写"本工作未计人力成本"。v1 完全没提这一点，而 ICRA 的 reproducibility/资源讨论越来越看重"compute & human budget"的透明。

---

## 5bis. 砍单链重写（v1 的 C1–C7 在新预算下基本不需要）

在新预算（§5.3）下，v1 的砍单链可以整体作废，只保留两条**真正的**决策：

| v1 砍单项 | 省 | v2 处置 |
|---|---|---|
| C1 砍 Tier-3 | ¥721 | **直接永久删除**（M15：SO-101 Bench 是 VLA 评测基准，非 RL-from-scratch；Isaac Sim 5.1 在 32 GB 内存无保证） |
| C3 砍 2B 确认 | — | **保留**（成本已降到 ¥127，且是 RQ4 的唯一证据） |
| C4 砍 N=2048 | — | **保留**（本地/租卡重划后不再贵） |
| C5 砍 breadth | ¥130 | **保留但设阈值**：W26 前 4 任务未跑通才降为 4 任务 |
| C2 真机能耗砍单 | ¥0 | **移出预算**：真机不花租卡，只花人工；改为"**人工预算**"条目（见下） |
| C6 砍 HIL-SERL | — | v1 已有 Plan B（SAC），保留为 D4 决策点 |
| C7 砍 seeds 5→3 | — | **反向操作**：seeds 5→8（见 §5.4） |

**人工预算（v1 缺失，v2 必须补）**：真机 45 trials × 3 方法 × 2 任务 × 2 轮 = **540 trials**；按每 trial 准备+执行+记录 ≈ 2.5 min + 每 10 trials 一次复位/校准 5 min，合计 ≈ **27 h 纯执行**，加上夹爪/物体复位与失败重跑 ≈ **5–6 人日**；3D 打印 4 类物体 + 背景板 ≈ 2 人日；Squint 的 `deploy.py` 从 SAC 改成 FG 的部署接口 ≈ **3–5 人日**（v1 §12 完全没算这一项）。这 3 项共 ≈ **12 人日 ≈ 2.4 周**，必须进 W24–W30 的排期。

---

## 6. 真机统计功效计算（C5 的依据）

| 设计 | n/条件 | 20pp 差距（0.6 vs 0.4）双比例 z / p | 10pp 差距（0.6 vs 0.5） | 结论 |
|---|---|---|---|---|
| v1：30 trials，独立 | 30 | z=1.58, p=0.11 | z=0.79, p=0.43 | **无功效** |
| v2：45 trials × 2 任务合并，配对 | 90 | z≈2.74, p≈0.006（未配对近似） | z≈1.37, p=0.17 | 20pp 可测，10pp 不可测 |
| 若要测 10pp @80% 功效 | ~93/条件 | — | — | 不可行（时间不够） |

**因此**：论文的真机主张必须限定为"**≥20pp 的差距或排名一致性**"，不能写成"我们的方法在真机上显著更好（10pp）"。这一点必须写进预注册与 Limitations——否则 reviewer 会用"underpowered"一刀切。

---

## 7. 近邻威胁重估（SGS 加入后）

| 工作 | 威胁度 | 变化 | 理由 |
|---|---|---|---|
| **SGS 2610.12465（CoRL 2026）** | **极高（新）** | 新增 | 同动机（uniform 浪费 → scaling 无收益）、同层（并行 env reset 分配）、更强规模（2^20）、有真机、有强 lab；且无代码（可复现性是我方唯一硬优势） |
| SER 2511.21011 | 高 | 不变 | 同 scaling 问题；但只调时序，可组合 |
| DRATS 2605.14350 | 高 | 不变 | 任务级自适应采样；但固定任务集、无 N-scaling、无真机 |
| PGB 2610.25704 | 中高 | 不变 | 形态最近（bandit 分配），但 LLM 域 |
| PEBBLE/其他未见（未核实） | 未知 | — | 我无法保证没有其他未检索到的工作；v2 的 Related Work 必须写"as of 2026-11" |

**定位结论**：在 SGS 存在的前提下，v1 的"第一贡献 = 发现 uniform 在大 N 下饱和"**不成立**。v2 必须把贡献顺序改为：

1. **方法（同层可比，但信号更强）**：**单任务内的 config 级分配**（目标区域 × 域随机化桶，M=8–16），信号 = **归一化 LP 的导数 × epistemic 不确定性** + 三重 guard。**与 SGS 同层、同预算、同 seed 直接 head-to-head**——SGS 只有 success-band（三档阈值、无不确定性、无导数），我方在"结果噪声大 / 组间成功率不可分"的配置上应当胜出，这是**可证伪的差异化主张**，而不是"我们做了更多"。
2. **理论/可预测性**：给出"**何时 success-band 足够、何时必须 LP×不确定性**"的判据（SGS 的隐含假设是"success rate 是无偏的进度信号"，该假设在 aleatoric 噪声组上失效）+ Δ(N) ∝ √N 的可证伪预测（SGS/DRATS/SER 都没有）。
3. **证据**：Tier-1（N≤16,384，含已知真 LP 的构造族，本地）+ Tier-2（**4 任务 deep grid @ N∈{256,1024,2048} + 8 任务 breadth，每个任务独立训练**）+ **MetaWorld MT10 跨任务附录**（task ID 在 obs，DRATS 同基准，本地 CPU）+ **低成本真机（2×SO-101，配对 90 trials）**。

**一个必须提前写进 v2 的风险对冲**：如果 Tier-2 结果显示 **success-band 已经足够好（SGS 简单版打平 FG）**，本文的卖点就退化为"理论 + 一个更贵的等价方法"。因此 v2 必须**预注册"负结果分支"**：若 H2 的组间差 <3 pp，则论文改写成 **"When does success-band suffice? A calibration study of progress signals for parallel RL"**（分析型论文，仍可投 ICRA/CoRL workshop，且实验成本已沉没）。这个分支要在 §2.4 写出来——审稿人看到作者预设负结果，会显著提高可信度。

这个 framing 我评估后认为**成立**且比 v1 强：它把一个"被 SGS 压制"的增量方法，转成"**在 SGS 的边界之外给出可预测的分配理论 + 一个 SGS 失效的可测场景**"。

---

## 8. 对 Brief 第 5 节 14 项清单的逐项勾选

| # | 清单项 | v1 | 问题 |
|---|---|---|---|
| 1 | 英文题目/摘要/关键词 + 中文 | ✅ | 摘要超词（M14） |
| 2 | RQ1–RQ4 + 可证伪假设 | ✅ | RQ1 归因缺协议 C（C2）；H7 定义缺真 LP（C8/m7） |
| 3 | 方法形式化 | ✅ | EXP3 误用（C6）、超参缺省（M8）、LP 未归一化（M7） |
| 4 | 理论/分析 | ⚠️ | σ_LP 公式与 EMA 定义矛盾（C8）；无 success-band 对比 |
| 5 | 实验矩阵（三层/固定预算/基线/消融/统计） | ⚠️ | **跨任务共享策略不成立（C9）**；视觉 N=4096 不可行（C4B）；PLR 稻草人（M11）；缺 oracle-LP（M6） |
| 6 | 真机实验 | ⚠️ | 30 trials 无功效（C5）；耗材/背景未计；`deploy.py` 改造工程量未计（§5bis） |
| 7 | 预算表（≤5000 + 敏感性 + 砍单） | ⚠️ | 加总少 ¥187（M3/M4）；压力情景算法错（§5.2）；**漏掉本地免费算力（§5.3）**；无人力成本行（§5.5） |
| 8 | 时间线 + Go/No-Go | ⚠️ | 零缓冲（C7）；真机 12 人日未进排期 |
| 9 | 风险表 + Plan B | ✅ | 需加 R11（SGS 压制）、R12（PPO-RGB 不收敛→改算法）、R13（Tier-2 主结果打平→转分析型论文） |
| 10 | 近邻差异表 | ❌ | **SGS 缺席（C1）**；RPLR 错标（M0） |
| 11 | 论文结构/图表/视频/开源 | ✅ | 体量分配（M13）、90s/3min（m10） |
| 12 | 中稿率提升策略 | ⚠️ | 需按 SGS 重估（C1.6）；v1 的 45–55% 过于乐观，我评估 **25–35%** |
| 13 | 实验运行清单 | ⚠️ | T2-h 步数/成本不一致（M3）；env 名非法（M17）；`-v0` 应为 `-v1`（m2） |
| 14 | 开放问题/决策点 | ✅ | D5（本机可用时长）应在 W1 前确认；需新增 D9（obs 是否含 goal → 决定 C9 首选/fallback 分支） |

**结论**：14 项中 6 项完全通过、7 项需修、1 项（近邻差异表）**必须重写**；此外 **C9 使得"实验矩阵"一项从"需修"升级为"必须重构"**。

---

## 9. v2 修改对照（供 Lead 审计）

| 类别 | v2 改动 | 对应问题 |
|---|---|---|
| 定位 | 贡献重排为"同层可比但信号更强 + 可预测 scaling + 低成本真机"；新增 §1.7 "与 SGS 的三句话差异"；预注册负结果分支 | C1, C8, C9 |
| 方法 | **Tier-2 改单任务 config 级分配（goal-region × DR，M=8–16）+ 8 任务独立 breadth**；主分配器改 full-information Hedge；LP 组内归一化；默认超参表；oracle-LP flag；部件→失效模式表 | **C9**, C6, M7, M8, M6, M18 |
| 理论 | §4.2 重写为双层噪声模型（单轮 SNR + EMA 滞后）；新增 §4.6 "success-band vs LP×不确定性"判据；P2 改为 Δ(N)∝√N 的定量预测 | C8, C1 |
| 实验 | 新增协议 B（fixed-K）与协议 C（fixed wall-clock）；Tier-1 改构造性异质族 A/B/C/D 且 env 名合法化（CheetahRun/HumanoidRun/Go1JoystickFlatTerrain/PandaPickCube）；Tier-2 改 4 任务 deep grid @N∈{256,1024,2048}；MetaWorld MT10 附录 | C2, C3, C4, M1, M17, **C9** |
| 基线 | 加 SGS-band（重实现）、SGS+FG；PLR 按原文公式重写；DRATS 给超参并注明"任务级→配置级的最小改动" | C1, M11 |
| 真机 | 45 trials 配对 × 2 任务 × 2 轮；合并主终点；耗材/背景明细；人工 12 人日（含 `deploy.py` 改造 3–5 人日） | C5 |
| 预算 | **本地免费/租卡重划（本地总计 1,220 GPU-h，租卡 551 h）→ ¥1,821 基准**；四情景 + FLOOR 全部闭合；新增人力成本行 | M2–M5, C4, §5.3–5.5 |
| 排期 | 冻结 W30，提交 W41，5 周缓冲；投稿日历（摘要/全文/视频）；真机 2.4 周嵌入 | C7 |
| 引用 | SGS 首行；2110.02439 改 RED；Squint 补 RA-L（doi:10.1109/LRA.2026.3730387）；env_id 改 `-v1` | C1, M0, m1, m2 |
| 清单 | 新增 §14 "SGS 专项 + 预算审计 + 投稿日历"，附录 B 更新到 15 项；新增 D9（goal 可条件化判定） | — |

**给 Lead 的一句话总结**：v1 最大的问题不是写作或预算，而是 **C9——按 v1 的实验设计去跑，Tier-2 在方法上就不成立**；其次是 **C1——SGS 让 v1 的第一贡献失效**。这两条修完后，v2 的预算反而降到 ¥1,800 量级、seeds 升到 8，整体可信度从"25–35%"抬到 **"35–45%（估计）"**。
