# 15 · DexSkillBench v3.2（终稿：CoRL 2027 主投 / ICLR 2028 兜底）

> **v2 → v3.1 改动清单（≈400 字）**：本版只执行 judge 的 **P1 精简锐化包**，不堆实验。①**理论**：Prop 1 加回重试因子 `(R+1)`（`2H(R+1)ε`），并明确它是**方向/斜率预测而非幅度预测**；Neyman 命题进正文（0.5 页，式 6/8/9），用 26–40% 上界给 H2 的 −30% 阈值提供依据（v3.2 修正：该上界是全-M 异质上界，H2 阈值改为 20% 相对，见 ⑧）；新增"EMA 滞后机制 + 局部灵敏度引理"，把启发式→可控近似做实。②**测量**：校准改**配对 CRN**（16 cell × 1,000 个 held-out 初始状态（4 技能；6 技能 gate 通过后扩 24 cell），所有方法/种子共用），主表报告 **decision-weighted ECE**，uniform-weight ECE 放附录；预注册降级条款。③**机制**：预测器改 **learned calibration head** + EMA age–误差回归（0 GPU），否则 H2 降级。④**H3 改剂量-反应**（噪声 0/2/5/10pp）为主图，二元 12pp 移附录，power 写 0.82。⑤**排期**：砍 stretch 技能与 G3/G4/G5，W1–16 只做 **4 技能**（S1/S2/S7/S8），6 技能为 W12 gate 的加分目标。⑥预算 **¥4,528**（余 472；v3.2 修正为 **¥4,809/余 191**），全文改用 judge 的 CoRL/ICLR/级联三值；不投 ICRA/RA-L。⑦**v3.1 小修**：CRN cell 口径统一为 **16（4 技能）/24（6 技能）**；D10 决策（v3.1 四级 → v3.2 六档）；S7 接受 **5 mm + Aloha 代理**降级；A7 附录条件项。⑧**v3.2 补丁（judge 终审 G1–G3）**：H2 阈值从 −30% 改为 **20% 相对**（26–40% 明确标为全-M 异质上界，不是阈值）；CRN 补方法因子 **512k rollouts/32 h/¥224**，LLM **¥641** → 基准 **¥4,809（余 191）**，削减链加 **C4 G1b 8→6 seeds**，FLOOR 触发时 H2 降 4-cell CRN；G0 顺延 W5、阈值统一；审计项（CI 行、时间线 GPU-h 列和、§3.0 编号、B=500.1M、D10 六档）全部闭合。

---

## 0. 定位、投稿目标与一句话

**一句话**：在固定预算下，灵巧手**技能库级的数据分配**决定了每个技能成功率的**可信度（校准）**，而可信度决定了 LLM agent 长程组合的成功率——我们用 4 个核心技能（6 个为加分目标）、受控干预、配对 CRN 评测与剂量-反应实验，把"分配 → 校准 → 组合"这条链拆开并量化。

**贡献（三条，按 CoRL/ML 混合口径）**：
1. **方法（可直接落地，不声称最优）**：**LGA**（Library-level Gated Allocation）——把单任务的"学习进度采样"提升到**技能库级**：Level-1 用成功率导数 × 估计量不确定性给技能加权，Level-2 在技能内按配置桶分配，并用 compute-normalized 权重保证"每 GPU 秒"而非"每环境步"的公平；输出每个技能的**校准成功率区间**而非单点。我们给出它与 Neyman 最优分配的同向关系（引理 4a/4b），并明确它是未知 `p_b`、非平稳下的**在线近似，不是最优解**。
2. **测量与基准**：一个纯仿真、开源、可复现的灵巧手技能库基准（CoRL 版 4 核心技能、2 个长程任务、类型化技能 API、held-out 目标划分），以及**"分配 → 校准 → 组合"的受控协议**：fixed-K / fixed-B / fixed-wall-clock 三协议、**配对 CRN 校准评测**、**剂量-反应噪声注入**、2×2 干预。
3. **理论与可证伪预测**：**Prop 1**（per-skill 校准误差 ε 经 H 步、每技能 R 次重试放大为计划选择损失 `≤2H(R+1)ε`，并给 margin 条件）与 **Prop 2**（Neyman 分配的 L2/L1 最优形式、uniform 的相对劣化 O(M)、ECE 收益上界 26–40%）。两者的价值在三个可证伪预测：斜率（TP1）、margin 翻转（TP2）、分配替代（TP5）。

**明确不声称**：不声称超过 Play2Perfect/ADEPT 的真机与算力规模；不声称 LLM 调技能本身是新的；不声称 sim-only 结论可直接迁移到真机；不声称 LGA 达到 Neyman 最优或给出收敛保证；不声称 4–6 个技能覆盖灵巧操作全空间。**声称的是**：在开源手 + 开源仿真栈上，把"分配 → 校准 → 组合"作为受控变量隔离，给出可复现的测量协议、一个可检验的传播界与一个可检验的分配刻画。

**投稿策略（口径固定）**：主投 **CoRL 2027**（预计 ≈2027-06，W31；**日期待核实**）；兜底 **ICLR 2028**（预计 ≈2027-09 下旬，W46–47；**待核实**）。不投 ICRA/RA-L（纯仿真系统论文折价明显）；NeurIPS 2027 D&B 只在 W20 显示 benchmark 已成熟时作机会项，不分兵。**本版执行 judge 的 P1 包**：砍 stretch/G3/G4/G5（−64 GPU-h、−¥450、腾 2 周），加 CRN（+¥28）、剂量-反应（+120 episodes/¥80）与 M1–M8 全部修正；judge 终审 v3.1 = **35%**（质量 47%，80% CI 26–44；ICLR 31%、级联 42%），补 G1–G3 后 ≈**38%**（仍单会 <50%，见 §11）。

---

## 1. 题目、摘要、关键词

### 1.1 题目（推荐 A）
- **A（推荐）**：*Calibration, Not Scale: How Library-Level Allocation Shapes Dexterous Skill Reliability and LLM Agent Composition*
- B：*DexSkillBench: A Controlled Study of Allocation, Calibration, and Composition in Dexterous Skill Libraries*
- C：*When Does Success-Band Suffice? A Calibrated Library Test of Learning-Progress Allocation*

### 1.2 英文摘要（≈210 词）
> Large-scale reinforcement learning can now train individual dexterous skills, but a skill library is useful to a planner only if each skill's success probability is *known and calibrated*. We study how allocating a fixed interaction budget across a skill library shapes per-skill calibration and, through it, the success of an LLM agent that composes skills into long-horizon tasks. We introduce **LGA**, a library-level allocation rule that weights skills by the derivative of learning progress times an estimator-uncertainty term, allocates within each skill over configuration buckets, and normalizes by measured per-step compute; it outputs calibrated success intervals rather than point estimates. We build **DexSkillBench**, an open, purely simulated benchmark with four core dexterous skills (two in-hand with a LEAP hand, two arm-mounted contact skills), two long-horizon composition tasks, a typed skill API, held-out goal splits, and a **paired common-random-number calibration protocol** (1,000 fixed held-out states per cell shared by all methods and seeds). Under a fixed-update protocol (K=1221, N=8192 and 2048, 8 seeds), we test LGA against uniform, staggered-reset, success-band, and level-replay allocation. We further prove that per-skill calibration error ε propagates through H steps with R retries to plan-selection loss bounded by 2H(R+1)ε, yielding a margin condition, and show that Neyman-style allocation implies an ECE improvement ceiling of 26–40% for a fully heterogeneous library — reported as a full-library upper bound, not as a threshold; our pre-registered H2 asks for a 20% relative reduction on frontier cells under paired CRN, with power recomputed for the same protocol. A dose–response noise-injection experiment tests whether composition success tracks calibration.

### 1.3 中文摘要
大规模并行 RL 已能训练单个灵巧技能，但技能库对规划器有用的前提是：每个技能的成功率**已知且校准**。本文研究固定交互/算力预算下，技能库级分配如何塑造 per-skill 校准，并通过校准影响 LLM agent 的长程组合成功率。我们提出 LGA：按"学习进度导数 × 估计量不确定性"给技能加权、技能内按配置桶分配、按实测每步算力归一化，输出校准后的成功率区间。我们构建 DexSkillBench：纯仿真、开源，含 4 个核心技能（2 个 LEAP 手内技能 + 2 个臂式接触技能）、2 个长程组合任务、类型化技能 API、held-out 目标划分，以及**配对 CRN 校准协议**（每个 cell 固定 1,000 个 held-out 初始状态，跨方法/种子共用；4 技能 = 16 cell，6 技能 gate 通过后扩为 24）。在 fixed-K（K=1221，N=8192/2048，8 seeds）下对比 uniform、SER、success-band 与 PLR 式分配。我们证明 per-skill 校准误差 ε 经 H 步、每技能 R 次重试后的计划选择损失 `≤2H(R+1)ε`，给出 margin 条件；并给出 Neyman 型分配在**全 M 异质**假设下的 ECE 改善上界 26–40%（这是上界，不是阈值）；H2 的预注册阈值是 frontier 上 **20% 相对降幅**（配对 CRN，power 按 8 seeds 重算）。最后用剂量-反应噪声注入检验组合成功率是否随校准误差单调变化。

### 1.4 关键词
dexterous manipulation; skill libraries; reinforcement learning; calibration; LLM agents; benchmark

### 1.5 与最近邻的三句话（“不声称”写在里面）
- **vs SGS（2610.12465）**：SGS 按 task-config 成功率加权 reset、单任务、2^20 环境；我们在**技能库级**分配，信号是成功率导数 × 估计量不确定性（对 aleatoric 噪声稳健），输出 per-skill 校准区间，并给出"何时 success-band 足够"的 margin 判据；**不声称**我们的成功率信号比 SGS 的 success-band 更"正确"，只声称在库级 + 校准输出这一层是新的可测问题。
- **vs Play2Perfect（2606.26428）**：P2P 是 play 预训练 → 单任务装配 finetune，其 limitation 明确把 task sequencing/composition 留给未来；我们不训练更强的单技能，而把技能库的**分配与校准**作为自变量。
- **vs OmniReset（2603.15789）**：OmniReset 用多样 reset + 单一 reward 做 scaling；我们把 reset/分配显式参数化，区分"多样性"与"按学习进度分配"，并用 fixed-K/fixed-B/固定墙钟三协议分离二者。
- **vs ADEPT（2608.19182）/ Being-0（2503.12533）**：前者是"更强单技能 + 真机"路线，后者是全栈 agent 系统；我们承认规模与真机劣势，只卖"固定预算下的可靠性测量与因果干预"。

---

## 2. 研究问题与假设（预注册）

### 2.1 RQ
- **RQ1（分配 → 库级成功率）**：fixed-K 下，LGA 是否比 uniform/SER/SGS-band/PLR-style 获得更高库级成功率？收益是否随 N 增大？
- **RQ2（分配 → 校准）**：LGA 是否给出更校准的 per-skill 成功率（held-out、decision-weighted frontier ECE、Brier、风险–覆盖）？该差异是否能在配对 CRN 下被可靠测出？
- **RQ3（校准 → 组合）**：把校准区间接入 LLM agent，是否提高长程组合成功率？收益是否随注入噪声单调下降（剂量-反应），且来自校准而非分配本身（2×2 + 分配替换干预）？
- **RQ4（边界）**：什么时候 success-band（SGS 式）足够、什么时候必须用 LP 导数 + 校准？（由 Prop 1 的 margin 条件与 Prop 2 的异质性条件给出可检验预测。）

### 2.2 H1（主假设，co-primary）
fixed-K 主格（`K=1221`、`S=4 核心`、`N=8192`、`B=500M`、`L=50`、8 seeds、4 方法）：
- **H1a（主终点 1）**：LGA 的**库级平均成功率**（4 技能等权、held-out 目标）比 uniform 高 **Δ ≥ max(6pp, W6 pilot 实测 MDE)**，95% CI 下界 >0；mixed model `success ~ method + (1|seed) + (1|skill)`。**M5**：阈值在 W6 用 pilot sd 写死（sd≤5→6pp；5<sd≤7→7pp；sd>7→先砍 G3/G4/G5 换 2 seeds，若仍 >7 则改 7pp 并在论文标注"未达原计划 power"）。
- **H1b（非劣）**：**min-skill 成功率**非劣——LGA 任一核心技能不得比 uniform 同技能低 >5pp（δ=5pp，95% CI 下界 >−5pp）。
- **H1c（utility-weighted）**：按 2 个长程任务等权加权的成功率（每任务 = 所需技能成功率之积）Δ≥5pp。
- **H1d（scaling，描述性）**：`Δ(N=8192) − Δ(N=2048)` 的 95% CI；下界 >0 才支持"收益随 N 增大"。
- **H1e（等价性，次要）**：uniform 在 N=8192 相对 N=2048 的边际增益 <2pp（TOST，margin=2pp）。

### 2.3 H2（校准；配对 CRN + 双口径）
**H2（主，v3.2 修订）**：LGA 的 **held-out decision-weighted frontier ECE**（p∈[0.3,0.7] 分层，权重=评测时该桶被查询的频率）相对 uniform **降低 ≥20%**（≈2pp 绝对 @基线 10pp），**配对 CRN** 下 ΔECE 的 95% CI 排除 0。**为什么不是 30%**：Prop 2 的 26–40% 是 M=8 全异质例的上界；而 H2 实测限定在 frontier（v∈0.21–0.25）时，Neyman 比值≈**1.000**（样本集中本身无收益），即便计入 Level-2 桶内集中（≤2×），可实现相对降幅上界约 **29.3%——30% 不可达**。故预注册阈值改为 20%（若 W6 用引理 4b + 实测份额预测的 ΔECE 更低，则取"预测值 −5pp"中的较大者）。**power**：配对 CRN 下 ΔECE 的 seed 间 SD 假设 1.5–2.0pp，8 seeds 的 MDE≈1.5–2.0pp，power 0.80–0.95；若 W16 实测 sd>2.5pp，按预注册降为 **15%** 或补 2 seeds（二选一，W16 前写死）。**H2 若只达 20%**：仍支撑"校准更好的方向性"与 H3 的传播预测，但**不足以支撑 30% 叙事**——论文措辞相应降级（§5.6/§11）。
- **H2b**：优势在 frontier 区（p∈[0.3,0.7]）最大（分层图）。
- **H2c（因果）**：fixed-B 下把 LGA 的桶分配替换为 uniform（策略训练日志复用，只换分配），ECE 优势消失 ≥70%；用**置换检验**（10,000 次重排分配标签）正式化，p<0.05。
- **H2d（机制）**：`|p̂−p|` 对 EMA age（自上次刷新以来的更新数）与 allocation share 的回归斜率显著为正（staleness 机制），且 learned calibration head 相对 EMA+isotonic 的 ECE 再降 ≥5%（否则 H2 降级，走 N2）。
- **口径纪律**：主表 **decision-weighted ECE**；附录报 **uniform-weight ECE**（`17_theory` §3.3b：分配会主动饿死低权重桶，uniform 口径可能变差——这是必须报告的"最坏口径"，不是反例）。

### 2.4 H3（agent 组合；剂量-反应为主）
**H3（主）**：向 planner 看到的 `p̂` 注入噪声 η∈{0,2,5,10}pp（4 档 × 2 任务 × 30 trials = 240 episodes；H3 主实验另 200 配对），组合成功率随 η **单调下降**（线性混合模型斜率 95% CI 排除 0，方向与 Prop 1 一致：损失 ∝ η·H·(R+1)）。剂量-反应是主图；二元 12pp（P1 vs P3，50 trials/task×2 任务=200 配对）放附录。
- **H3b**：错误重试比例降低 ≥30%；**H3c**：P1 vs P3 的二元效应 ≥12pp（附录，McNemar；power 0.82 @ρ=0.3；0.67 @ρ=0；报告敏感性）。
- **理论口径**：Prop 1 给的是**方向与斜率**（损失 ∝ ε·H·(R+1)），**不是幅度预测**；H=4,R=2,ε=1pp 时最坏界 24pp，超过 H3 想检验的 12pp，故不能写"理论预测 12pp"。

### 2.5 H4–H9
- **H4（技能数）**：S=4→6（W12 gate 通过时）时，uniform 的边际增益下降快于 LGA；交互效应报告 CI（描述性）。
- **H5（归因）**：失败归因一致率 ≥85%（双人标注 κ≥0.8），allocation-swap 后"分配型失败"比例变化 ≥10pp。
- **H6（API 粒度，探索性）**：粗/中/细三种 API 粒度下 agent 成功率最优与次优差 ≥5pp。
- **H7（跨手迁移，探索性）**：LEAP 上训练的分配规则零样本用于 Aero（附录 6 runs，ICLR 版）；本节不作主张。
- **H8（鲁棒性，ICLR 版）**：观测噪声、质量/摩擦 ±20%、held-out 目标外移 20% 下排序不变。
- **H9（prediction head）**：learned calibration head 的 ECE 低于 EMA+isotonic（≥5% 相对），且与 H2d 的 age 斜率一致；若 head 无改善 → H2 降级为"测量协议"。

### 2.6 预注册与分析计划
- **W6 冻结**：协议表（K/B/N/seeds）、H1–H9、MDE 表、主/次终点、排除规则、mixed model 公式、Holm family = {H1a,H1b,H1c,H2,H3-dose}。
- **W16 冻结**：实现细节（env config hash、模型快照、prompt 模板、CRN 初始状态集 SHA256、head 训练脚本）。
- **重复单元**：seed；skill 为固定水平，用 `(1|skill)` 随机截距；校准用 per-seed cell 级 cluster bootstrap（10k）。
- **MDE 表**（sd=6pp，双侧 5%，power 80%）：5 seeds 7.5pp、6 seeds 6.9pp、8 seeds 5.9pp、10 seeds 5.3pp。**M5 规则**：W6 实测 sd 决定阈值与是否补 seeds。
- **校准功效**（M3）：250/cell 的二项 SE=3.16pp、`E|误差|` 偏差 2.52pp，测不出 30% 相对改善；1,000/cell + 配对 CRN 把 ΔECE 方差降一个量级——这是 H2 可检验的前提。
- **负结果分支**：N1 H1 不成立但 H2 成立 → "分配影响校准而非库级成功率"；N2 H2 不成立 → "success-band 何时足够"边界研究（margin 条件 + TP2 翻转）；N3 S7 不可行 → 手内版 + 臂式失败分析。
- **降级**：D1 N=8192 OOM → 主格 N=4096、K=1221（B=250M），阈值不变、MDE 附录更新；D2 LLM 超 ¥700 → 砍剂量档位与锚点，H3 主检验不动；D3 W16 主曲线 <3pp → 按"分配 → 校准"收，agent 降为验证。

### 2.7 假设–实验–图表闭环

| 假设 | 实验 | 图/表 |
|---|---|---|
| H1a–H1e | G1a/G1b 主网格（4 方法×8 seeds×2 N） | Fig.2 主曲线、Table 2 |
| H2/H2b/H2c/H2d | G1 + G6 CRN 评测 + A6 分配替换 + 置换检验 | Fig.3 可靠性图、ECE–N；Table 2/3 |
| H3/H3b | G7 agent 2×2 + 剂量-反应 | Fig.4 剂量曲线、Table 3 |
| H4 | A7（S=4 vs 6） | 附录 Fig.A2 |
| H5 | G7 + 归因 + allocation-swap | Fig.4b |
| H6 | G7 三种 API 粒度 | 附录 Table A3 |
| H9 | head vs EMA+isotonic（0 GPU） | Fig.3b 校准散点/age 回归 |

---

## 3. 方法

### 2.8 为什么是这四个 RQ（而不是更多）

四个 RQ 各自对应一条"若失败就换叙事"的退路：RQ1（分配是否改变性能）失败→论文降为"校准测量研究"（H2 仍可独立成立）；RQ2（校准是否改善）失败→论文退为"分配的系统性研究"（H1 仍可成立），且必须按预注册降级条款处理；RQ3（校准是否被 agent 使用）失败→agent 层降为相关工作，主贡献收为"分配→校准"；RQ4（技能数与 scaling）失败→4 技能版是保底，6 技能只是加分。也就是说，四个 RQ 是**可分级交付**的最小链，而不是"多写几个问题显得完整"——在 1 人 45 周、¥5,000 的约束下，可分级交付比覆盖面更重要。

### 3.0 部件 → 失效模式 → 对应消融

| 部件 | 若缺失/失效会怎样 | 消融 |
|---|---|---|
| 共享 trunk + per-skill adapter | 4 技能互相负迁移，手内技能被臂式接触拖垮 | A11（附录结构消融，不占主表预算） |
| Level-1 技能级权重（LP 导数 × σ̂） | 退化为"按成功率加权"（≈SGS），噪声桶被误判为进步 | A1 |
| Level-2 技能内桶分配 | 技能内部难桶饥饿、前沿区估计稀疏 | A3 |
| compute 归一化（c_g） | 分配偏向每步便宜的手内技能，库级成功率虚高 | A2 |
| 三 guard（G-1/G-2/G-3） | 高方差权重抖动、单桶垄断、熵坍缩 | A5 |
| 校准区间/head（vs 名义成功率） | agent 只能按名义成功率规划，长程错误累积 | A9 |
| held-out 目标 split | 校准在训练分布内"自证"，OOD 失效 | A8 |
| 配对 CRN 评测集 | 两臂独立采样，ΔECE 方差过大、20% 相对阈值测不出 | （评测设计，见 §5.7） |

### 3.1 统一 embodiment 与 MDP 族

- **Embodiment A（主）**：7-DoF 机械臂（Franka Panda，MuJoCo Menagerie MJCF）+ LEAP 手（16-DoF）刚接法兰；obs = 手部关节角/速度（16）+ 臂关节角/速度（14）+ 物体位姿（7）+ 上一动作（23）+ 目标（16 或 7，按技能类型）；action = 23 维位置目标（16 手 + 7 臂），手内技能时臂动作置零（mask），臂式技能时臂头激活。
- **统一决策率 20Hz**：所有技能 `ctrl_dt=0.05`；物理 `sim_dt`：手内 0.01（2 substeps）、臂式接触 0.0025（20 substeps，400Hz 接触求解）。**1 步 = 50ms 物理时间，对所有技能一致**。
- **MDP 族**：每个技能 g 是同一 robot 上的一个目标条件 MDP `M_g = (S, A, P_g, R_g, γ, ρ_g, φ_g)`；`φ_g` 是成功谓词（几何阈值），`ρ_g` 是目标/DR 分布；共享策略 `π_θ(a|o,z_g)`，z_g 为技能目标编码。**合法性**：同一 robot、同一动作空间、同一控制率；技能间差异只在物体/目标/成功谓词/奖励塑形，因此"共享策略 + 技能级分配"在方法上成立。
- **为什么不是"跨任务共享"**：跨任务（不同 reward/成功判定）需要 task ID 条件化或分离策略；本库是**同一任务族内的技能**，不跨任务共享。

### 3.2 技能库（CoRL 版：4 核心 + 2 gate 可选 + ICLR stretch）

| ID | 技能 | 来源/实现 | 成功谓词 | 训练步数 | 人日 | CoRL 版 |
|---|---|---|---|---|---|---|
| S1 | Reorient-Any | `LeapCubeReorient` 复用改目标 | 立方体姿态误差 <0.1 rad（保持 0.2s） | 80M | 3 | ✅ 核心 |
| S2 | Rotate-Z | `LeapCubeRotateZAxis` 复用改目标 | 绕 z 目标角误差 <0.15 rad | 60M | 2 | ✅ 核心 |
| S7 | Peg-Insert-2.5mm | 新建（臂+LEAP；参考 Aloha peg 流水线） | 销（20mm）入孔（26mm，单边 3mm），深度 ≥15mm，横向 ≤2mm，无卡死 | 120M | 10 | ✅ 核心 |
| S8 | Cap-Twist | 新建（臂+LEAP；弹性盖，无螺纹） | 旋盖 2 圈 + 轴向 2mm，滑脱 <10% | 100M | 8 | ✅ 核心 |
| S3 | In-hand-Translate | 新建（reorient 改奖励/目标） | 物体质心到目标区 ≤3mm | 90M | 8 | ⚠️ W12 gate |
| S4 | Finger-Gait | 新建 | 指步态推进 ≥15mm 且不跌落 | 100M | 8 | ⚠️ W12 gate |
| S5/S6 | Reorient-Cylinder / Edge-Align | 新建 | 圆柱轴对齐 / 棱边对齐 | 90M/80M | 6/6 | ❌ ICLR |
| A1 | Aero-Rotate-Z | `AeroCubeRotateZAxis` 复用 | 绕 z 目标角误差 <0.15 rad | 60M | 1 | ❌ ICLR 附录 |

- **CoRL 核心 4 技能 = S1/S2/S7/S8（23 人日）**；若 W12 检查点显示进度超前 ≥2 周，则加做 S3/S4（+16 人日）形成 6 技能版；否则 CoRL 版就是 4 技能，S3–S6 与 G3/G4/G5 移 ICLR 版（M7）。
- **精度承诺**：S7 主容差写**实测值**（W4 三容差探针 1/3/5mm；目标 3mm、回退 5mm，后置条件横向 ≤2mm）。依据：Playground `AlohaSinglePegInsertion` 是 8mm 单边间隙，我们比它紧 1.6–2.7×。
- **S8 不建模螺纹**：弹性盖 + 摩擦旋合（MJX 接触可稳定表达）。

### 3.3 技能内配置桶、held-out 目标与 CRN 评测集

- 每个技能 4 个桶：`(目标区域 2) × (DR 档 2)`；4 技能共 **16 桶**（6 技能 24 桶）。DR：摩擦 0.5–1.5、质量 ±20%、初始位姿噪声。
- **目标划分**：每技能 8 个 goal cell → **6 训练 / 2 held-out**；held-out cell 只在校准评测与 agent 评测中采样，训练中从不出现；W6 冻结并记录 SHA256（`eval/goal_split.json`、`eval/init_states.json`）。
- **配对 CRN 评测集（M3，关键）**：对每个 (skill×bucket) cell，固定 **1,000 个 held-out 初始状态**（由固定 seed 生成，W6 冻结；**16 cell = 4 技能×4 桶；若 W12 gate 扩到 6 技能则为 24 cell**），**所有方法、所有 seed 共用同一组**。评测时只跑策略，不再采样初始状态——两臂的评测噪声由此共享，ΔECE 的方差降一个量级。frontier 子集（p∈[0.3,0.7]）约 8–12 cell，主表用 decision-weighted 权重。
- **桶难度标签**：训练后按 held-out 成功率把桶分为 easy/medium/frontier/hard（阈值 0.8/0.5/0.2），仅用于 H2b 分层与 margin 分析，不参与训练。

### 3.4 LGA：库级分配

**Level-1（技能级，每 T_alloc=20 次 PPO 更新重算一次）**：
- 学习进度：`u_g = (p_fast,g − p_slow,g)(1 − p_slow,g) + λ σ̂_g`，`p_fast` 为最近 20 episode、`p_slow` 为最近 100 episode 的成功率（按 episode 计），`λ=0.2`。
- **compute 归一化（主）**：`ũ_g = u_g / c_g`，`c_g` = W4 实测每 env-step GPU 时间（手内归一 1.0，S7/S8≈6.5）；同时报告 step-normalized 作对照（A4）。
- 权重：Hedge `w_g ← w_g·exp(η ũ_g)/Z`，`η=1.0`，混合 `ε=0.05`。
- **三 guard（M3 保留）**：G-1 `n_g<30` 或 `σ̂_g>q90` → 冻结上一次权重；G-2 相邻两次权重 TV>0.2 → 冻结 50 更新并线性插值；G-3 权重熵 <0.5logM 或单技能 >40%N → 温度×1.2。
- 配额：`n_g = max(1, round(w_g·N))`，largest remainder 取整；**未结束 episode 不迁移**。

**Level-2（技能内桶级）**：`LP_b = (p_b,fast − p_b,slow)(1 − p_b,slow) + λ σ̂_b`；技能内 z-score 后 `w_b ∝ exp(η2 LP̃_b)`，`η2=1.0`；每技能至少 10% N、至多 50% N。

**与 Neyman 的关系（§4.4 引理 4a/4b）**：在"LP 导数 ∝ v_b、σ̂_b≈√(v_b/n_b)"的建模假设下，分数 `s_b=LP/σ̂=κ√(v_b n_b)` 与 L1 最优 `α*∝(w_b√v_b)^{2/3}` 对 v_b 的单调方向一致；我们**只声称同向**，并用 oracle 分配（真实 p_b 代入式 (8)）测量 LGA 离最优多远（P2）。

### 3.4b 为什么是"LP 导数 × 估计量不确定性"这两个信号（机制解释）

单用成功率导数（LP）在并行 env 数量大时会退化：`p̂` 的噪声随 `1/√n_b` 下降，但也随"更新太快"而上升（EMA 跟不上策略；见 Prop 1b 的 `(τṗ)²` 项）。也就是说，**LP 大的桶有两种可能**：真的在进步，或者只是在噪声里跳。前者值得多给数据，后者是典型的 noisy-TV 陷阱——把数据投给一个随机回报的桶，会同时浪费预算、并污染共享策略的梯度。我们的做法是给 LP 乘一个"可信度项"：用 3 条不同滞后（5/20/100 次更新）的 EMA 的离散度 `σ̂` 作为**估计量不确定性**（不是 model epistemic uncertainty，§3.5 有显式声明），并把分位数 guard 挂在 `σ̂` 上。

这个设计与经典 bandit 的"乐观面对不确定性"方向相反：我们不是给不确定的臂更高优先级，而是**给"不确定到无法判断"的臂降温**。原因是这里的"臂"是数据源而不是决策对象——探索一个无信号的数据源不会带来信息收益，只会带来方差。这一取舍是 LGA 与 PLR/SGS 家族的关键分歧之一：PLR 的优先级里也含 `σ̂`，但用于在"level"上主动制造难度；我们不制造难度，只在**技能库**这个层级上做经验预算的再分配。也正因如此，LGA 的消融 A1（关掉 `σ̂`）预期在简单技能上几乎无差、在臂式接触技能上差异明显——这是我们在 §5.5 里预注册的**异质性预测**，而不是事后挑选。

### 3.5 预测器、不确定性与机制诊断

- **预测器（只允许训练期统计，避免循环）**：基线为 EMA 成功率（窗口 {20,100} episode，按训练桶频率加权）；**M4 新增 learned calibration head**：在训练日志上拟合（0 GPU，10 分钟级）`held-out 成功率 ~ f(EMA p, EMA age, allocation share, value head V, bucket difficulty, skill one-hot)`，输出 `p̂` 与区间；用 calibration split 拟合、test split 评测（与 isotonic/Beta 温度缩放对照，A9）。
- **σ̂_g**：3 个 EMA 滞后 {5,20,100} episode 的成功率标准差（**估计量离散度**；显式声明"不是 model epistemic uncertainty，也不是贝叶斯后验"）；同时给 `σ̂` vs `|p̂−p|` 的校准散点（Fig.3b），相关性弱则 guard 退化并如实报告。
- **EMA age（机制诊断，M4）**：记录每桶自上次"被显著更新"以来的更新数 `age_b`；回归 `|p̂_b−p_b| ~ age_b + allocation_share_b`。若 age 斜率显著为正 → staleness 机制成立；若为 0 → 机制不成立，H2 走 N2 降级。
- **校准评测口径**：主表 decision-weighted ECE（权重=评测时该桶被查询频率）；附录 uniform-weight ECE；同时报 Brier、NLL、风险–覆盖（0.5/0.7/0.9）。

### 3.6 校准与技能 API

**API（类型化 JSON）**：
```json
{
  "skill_id": "S7",
  "args_schema": {"object_pose": "SE3", "target_region": "AABB"},
  "preconditions": {"hand_holding": "peg", "clearance_ok": true},
  "postconditions": {"inserted_depth_m": ">=0.015", "lateral_m": "<=0.002"},
  "calibrated_success": {"p": 0.62, "ci95": [0.55, 0.69], "n_eval": 1000,
                          "ece_dec": 0.061, "source": "head", "age_updates": 140,
                          "split": "heldout_cells_crn"},
  "compute": {"steps_per_call": 120, "gpu_ms_per_step": 0.08},
  "recovery_hint": "retry_if(p<0.5); reorient_if(clearance_fail)"
}
```
Agent 只通过该 schema 调用，不能读训练日志或 env 内部状态。`source` 字段与 `age_updates` 让审稿人能检查"校准来自 head 还是 EMA 滞后"。

### 3.7 Planner 与 agent 实验设计

- **P0 random**：均匀随机选可行技能序列。
- **P1 ReAct + p̂**：ReAct 循环，动作空间 = API 调用，prompt 中含 `calibrated_success`（p 与 CI）。
- **P2 plan-then-execute + 风险感知**：先枚举计划（≤3 技能、≤2 分支），按 `∏ q_i` 选最大（`q=1−(1−p)^{R+1}`，R=2），执行中失败按 `recovery_hint` 重试；CI 下界作风险约束。
- **P3 no-p̂（对照）**：prompt 只给"技能描述 + 名义成功率"，不给校准区间；其余同 P1。
- **P4 calibrated + uniform 分配**：用 uniform 训练的技能库 + P1 planner（2×2 第四格）。
- **O1 oracle**：用 held-out 真 p 规划（上界）；**S scripted**：无 LLM 的贪心规划器（分离"技能质量"与"LLM 规划"）。
- **剂量-反应（M6）**：对 P1 看到的 `p̂` 注入 `η∈{0,2,5,10}pp` 的独立噪声（固定 seed、每个 episode 重新抽），主分析 = 成功率对 η 的斜率与单调性；二元 12pp（η=0 时 P1 vs P3）放附录。
- 触发条件：`preconditions` 不满足则跳过并记录（归因用）；最多 20 次调用、2 次重试/技能、90s/episode。

**2×2 干预**：{LGA, uniform} × {calibrated, no-p̂}；主分析 H3 = calibrated vs no-p̂（LGA 策略上），P4 用于 H2c。

### 3.8 PPO 与共享策略结构

- PPO：clip 0.2、GAE 0.95、γ 0.99、lr 3e-4、epochs 4、minibatch `max(N·L/4, 4096)`、ent 0.001、UTD=1、L=50；**所有方法同 K、同 L、同 minibatch 规则**。
- 结构：共享 trunk（2×256 MLP）+ per-skill adapter（64 维）+ 值头（per-skill 输出 + 共享 baseline，A1 消融）。手内技能臂动作 mask，梯度不回传臂头。
- 训练稳定性：梯度裁剪 1.0；obs 归一化 running stats；每 20 更新记录一次 `p_fast/p_slow/σ̂/age`。

### 3.9 伪代码

```text
for update t in 1..K:
    if t % T_alloc == 0:
        u_g = (p_fast - p_slow)(1 - p_slow) + λ σ̂_g
        ũ_g = u_g / c_g
        w ← hedge_update(w, ũ, η); w ← mix(w, ε)
        apply guards G-1/G-2/G-3
        n_g ← largest_remainder(w·N)
    sample rollouts: n_g envs per skill, bucket weights w_b
    PPO update on collected data (L=50, UTD=1)
    update p_fast/p_slow/σ̂/age per skill and bucket
every 20 updates: log weights, n_g, entropy, TV, age
# evaluation (offline, 0 training GPU)
fit calibration head on calibration split (logs + CRN states)
evaluate on CRN test set: p̂, ECE_dec, ECE_unif, Brier
regress |p̂ - p| ~ age + allocation_share   # H2d mechanism
```

### 3.10 工程量（CoRL 版，4 技能）

| 项 | 人日 |
|---|---|
| 核心 4 技能（S1/S2/S7/S8） | 23 |
| arm+LEAP 集成（MJCF/IK/标定） | 10–15 |
| 槽/盖/夹具资产 | 3–5 |
| 20% 缓冲 | 9–11 |
| harness/API/校准 CRN/planner/agent 代码 | 15–18 |
| **合计** | **60–72 人日 ≈ 12–15 周** |

排期按 **W1–16**（含 1–2 周缓冲）；S3/S4（+16 人日）只在 W12 检查点通过后加入，否则 CoRL 版保持 4 技能。S5/S6、Aero、G3/G4/G5 全部移 ICLR 版（W31–45）。

---

## 4. 理论（正文 0.75–1 页；完整推导进附录 E）

> 口径声明：本节给的是**可验证的弱命题与方向性预测**，不是收敛性/最优性证明。所有常数取保守值；当界在最坏情形退化（如 `2H(R+1)ε>1`）时，我们如实说明它只提供方向与斜率。

### 4.1 假设
- **A1（桶内 i.i.d. 与平稳）**：同一桶内 episode 独立同分布，训练窗口内 p_b 不变。真实训练不严格成立，故 §4.2.2 给非平稳修正。
- **A2（桶异质性）**：技能内桶成功率随难度单调；前沿桶（p≈0.5）的 LP 导数最大。
- **A3（估计误差）**：`p̂_b` 由 n_b 次 rollout 得到，`|p̂_b−p_b|≤ε` 以概率 1−δ 成立（Hoeffding/Beta 区间）。
- **A4（加性预算）**：库级总步数 `B=Σ_g b_g`；compute 归一化时以 `Σ c_g b_g = C` 为约束。
- **A5（重试独立）**：每技能最多 R 次重试，重试成功率与首次同分布（不独立时用 §4.6 的衰减修正）。

### 4.2 命题 1：uniform 分配的校准噪声

**4.2.1 采样噪声界（M1/Lead 修订）**。uniform `n_b=B/M`，Hoeffding + union bound：`Pr[max_b|p̂_b−p_b|≥ε] ≤ 2M exp(−2Bε²/M)`，即式 (1) `max_b|p̂_b−p_b| ≤ √(M log(2M/δ)/(2B))`。由 `E|X−EX|≤√Var` 与 `p(1−p)≤1/4` 得**严格**界

$$\mathbb{E}[\mathrm{ECE}] \le \frac{1}{2}\sqrt{\frac{M}{B}}, \tag{2a}$$

而 `≈0.399√(M/B)` 只在**正态近似/大 n 渐近**下成立，写作式 (2b)，正文与表格并列两列。技能级加权估计的方差 `Var(p̂_g)≤(M/(4B))Σ_b w_b²`，等权时 `SD≤1/(2√B)`（与 M 无关）。

**数值（两列；B=112.5M 与 B=500M 对应 fixed-B 次格与 fixed-K 主格 N=8192）**：

| M | 式(1) 高概率界 B=112.5M | 严格 ECE B=112.5M | 渐近 ECE B=112.5M | 式(1) B=500M | 严格 ECE B=500M | 渐近 ECE B=500M |
|---|---|---|---|---|---|---|
| 4 | 3.0e-4 | 9.4e-5 | 7.5e-5 | 1.4e-4 | 4.5e-5 | 3.6e-5 |
| 8 | 4.5e-4 | 1.3e-4 | 1.1e-4 | 2.1e-4 | 6.3e-5 | 5.0e-5 |
| 16 | 6.8e-4 | 1.9e-4 | 1.5e-4 | 3.2e-4 | 8.9e-5 | 7.1e-5 |

结论：主格 B=500M 下纯采样噪声比我们在意的 pp 级校准差**小两个数量级**；预算越大（500M > 112.5M）噪声越小，"观测 ECE 差不是采样噪声"的结论越强。因此观测差只能来自：分布偏移（评测权重错）、非平稳（EMA 滞后）、训练期统计冒充 held-out（偏差）、分配偏斜（饿死桶）。

**4.2.2 非平稳修正（命题 1b）**：设桶成功率以速率 ṗ_b 漂移，用最近 τ_b 个样本估计，则 `MSE_b(τ_b) ≤ (τ_b ṗ_b)² + p_b(1−p_b)/τ_b`（式 4）；最优 `τ_b*=(p_b(1−p_b)/(2ṗ_b²))^{1/3}`，`MSE_b* ≤1.89(p_b(1−p_b)ṗ_b²)^{1/3}`（式 5）。**含义**：LP 导数估计趋势（偏差项），σ̂ 估计噪声（方差项）——两个信号都不可省（A2 去 σ̂、A9 换 head 分别检验两项）。

### 4.3 命题 2：Neyman 型最优分配（M2，进正文）

**L2 准则**（最小化加权方差）：`min Σ_b w_b v_b/n_b, s.t. Σ n_b=B`，Lagrange 给

$$n_b^\*=\frac{B\sqrt{w_bv_b}}{\sum_{b'}\sqrt{w_{b'}v_{b'}}},\qquad v_b=p_b(1-p_b). \tag{6}$$

uniform 的相对劣化 `(M Σ_b w_bv_b)/(Σ_b√(w_bv_b))² ≥1`，等号仅当 `w_bv_b` 为常数；异质时**可差至 O(M) 倍**（式 7）。

**L1 准则**（规划关心绝对误差）：`min Σ_b w_b√(2v_b/(πn_b))` 给出

$$n_b^\* \propto (w_b\sqrt{v_b})^{2/3}=w_b^{2/3}\,v_b^{1/3}. \tag{8}$$

最优 ECE 为 `√(2/π)B^{-1/2}Z^{3/2}`（`Z=Σ_b(w_b√v_b)^{2/3}`），uniform 为 `√(2/π)B^{-1/2}√M U`（`U=Σ_b w_b√v_b`），故

$$\frac{\mathrm{ECE}^\*}{\mathrm{ECE}_{\text{uniform}}}=\frac{Z^{3/2}}{\sqrt{M}\,U}\le1. \tag{9}$$

**数值（附录 B 例 2）**：`M=8`、`w=(0.4,0.2,0.2,0.1,0.05,0.03,0.01,0.01)`、`v=(0.25,0.25,0.16,0.09,0.04,0.01,0.0025,0.0025)` → 比值 **0.74**（降 26%）；权重更集中（最大桶 0.9）可达 0.35–0.45（降 55–65%）。**结论（v3.2 修订）**：**26–40% 是 M=8 全异质情形的上界，不是 H2 的阈值**。把 H2 限定在 frontier（v∈0.21–0.25）后，式 (9) 的比值≈**1.000**（仅靠样本集中在 frontier 内几乎无 ECE 收益）；即便计入 Level-2 桶内集中（最多 2×），可实现相对降幅上界约 **29.3%**，仍 <30%。因此 H2 的预注册阈值改为 **20% 相对**（§2.3），26–40% 只用于说明"全库异质时理论上界在哪里"；桶同质/权重均匀时收益趋 0（与 H2c / P5 一致）。

**定位（必须写进正文）**：式 (6)(8) 需要真实 p_b 与 w_b，二者未知且非平稳；**LGA 是它们的在线近似，不声称达到最优**。我们只声称：①方向与最优解同向（引理 4a）；②用 oracle 分配量化差距（P2）。

**命题 2b（标准结论，不作贡献）**：Hedge/EXP3 的静态 regret 为 `O(√(T log M))`，非平稳用 EXP3.S/折扣 Hedge 得 `O(√(TM log M))`（Auer et al. 2002；Cesa-Bianchi & Lugosi 2006）。它把"分配器切换代价"与"统计误差"解耦。

### 4.4 命题 3：校准误差 → 长程组合损失（M1：加回重试因子）

规划规则：计划 π=(s_1,…,s_H)，每技能最多 R 次重试，`q_i=1−(1−p_i)^{R+1}`，`q̂_i=1−(1−p̂_i)^{R+1}`；planner 选 `π̂=argmax_π ∏_i q̂_i`（式 10/11），真值 `S(π)=∏_i q_i`。

**引理 3.1**：`|q_i−q̂_i| ≤ (R+1)ε_i`（q'(p)=(R+1)(1−p)^R≤R+1）。
**引理 3.2**：`|∏a_i−∏b_i| ≤ Σ_i|a_i−b_i|`。
**命题 3a（最坏情况，线性）**：`|S(π)−Ŝ(π)| ≤ H(R+1)ε`（式 12），故相对 oracle 的损失

$$\boxed{\,S(\pi^*)-S(\hat\pi)\le 2H(R+1)\varepsilon\,} \tag{13}$$

**数值**：H=4,R=2 → 24ε；ε=1pp → 最坏 24pp，ε=0.5pp → 12pp。**这是最坏界，常常偏松（>1 时退化）**；它的作用是给**方向与斜率**（随 H 线性、随 ε 线性、随重试数线性），**不是幅度预测**。论文中不得写"理论预测 12pp"。

**命题 3b（独立误差，√H 更紧）**：若 p̂_i 相互独立、`Var(p̂_i)≤σ_p²`，则 `Var(Ŝ) ≤ S²H(R+1)²σ_p²/q_min²`（式 14，δ-method 一阶 `(q')²Var`；严格 Lipschitz 口径带因子 2，整体 ×√2，O(·) 不变），对 N_π 个计划 union bound 得 `max_π|Ŝ−S| ≤ σ_S√(2log(2N_π/δ))`，故 `S(π*)−S(π̂)=O(ε√(H log N_π))`（式 15）。

**推论（rollout 下界，修正常数）**：以概率 1−δ 选对计划需每技能 `n ≳ 2H²(R+1)²log(2S/δ)/m²`（Hoeffding+union；H=4,R=2,S=6,δ=.05,m=.12 时约 **1.1e5**/技能）。在 B=500M/6 技能下（n≈8e7）**不绑定**，故 v3 不依赖此推论做预算，只把它作为"组合越长、间隔越小时校准越贵"的定性结论。

### 4.5 引理 4a/4b：LGA 与 Neyman 的定量连接（M2）

- **引理 4a（单调比较静态）**：在 A1 与非平稳近似下，若 `LP导数 ≈ κv_b`、`σ̂_b≈√(v_b/n_b)`，则 FG 分数 `s_b=LP/σ̂=κ√(v_b n_b)`；L1 最优 `α*_b∝(w_b√v_b)^{2/3}`。两者对 v_b 的偏导同号（均 >0）。**只声称方向一致，不声称数值相等。**
- **引理 4b（局部灵敏度）**：若实际分配 `α_b=α*_b(1+δ_b)`，把 ECE(α)=Σ_b w_b√(2v_b/(πα_b B)) 在最优解处二阶展开，得

$$\frac{\mathrm{ECE}(\alpha)}{\mathrm{ECE}^\*}\le 1+\frac{3}{8}\max_b\delta_b^2. \tag{16}$$

因我们记录每桶实测 n_b，可**直接算 δ_b 并预测 ECE**，与 CRN 实测对照（P2 的可证伪版本：预测与实测相对偏差 ≤10% 才算支持）。ρ>0.6 只作弱代理，不由 ρ 直接推界。
- **附注（Lead 修订口径）**：式 (14) 的方差传递若只用 Lipschitz，常数带因子 2（整体 ×√2，O(√H) 不变）；正文用 δ-method 一阶、附录给严格口径与来源。

### 4.6 可检验预测（理论 → 实验 → 失败含义）

| # | 预测 | 实验 | 若失败说明 |
|---|---|---|---|
| TP1 | 固定 ε 下，组合损失随 H 线性增长（斜率 ≈ 2(R+1)ε） | H3 剂量-反应 + 短链 H=2 评测（仅评测，不训练） | 斜率≈0 → planner 未用 p̂（RQ3 否定）；远大于 → 未建模传播 |
| TP2 | margin `m > 2H(R+1)ε` 时校准收益→0（success-band 足够） | 构造 m∈{0.02,0.05,0.10,0.20} 的计划对 | 无翻转 → RQ4 答案是否定 |
| TP3 | 对 p̂ 注噪后成功率**下降**（η 单调） | H3 主实验（M6） | 不降 → planner 忽略 p̂ |
| TP4 | 分配替换后 ECE 优势消失 ≥70% | H2c + 置换检验 | 不消失 → 优势来自后处理/评测伪影 |
| TP5 | LGA 的桶份额与 oracle Neyman 份额 Spearman ρ>0.6、且 δ 预测 ECE 与实测一致 | P2/oracle-LP | 相关弱 → LGA 分数不是好的分配代理 |
| TP6 | 重试成功率几何衰减 ρ 可估；R_eff<R+1 时用 R_eff 修正 | 失败归因日志 | 强衰减 → 式 (10) 换模型，界更保守 |

### 4.7 假设边界与"不成立时数据里会看到什么"

| 假设 | 不成立时 | 数据信号 | 应对 |
|---|---|---|---|
| 失败可检测 | 观测成功率偏离 | 记录 vs 评测系统偏差 | 独立评测器、混淆矩阵 |
| 重试独立 | 二次成功率下降 | 第 2/3 次成功率显著低 | 衰减模型、报 ρ 与 R_eff |
| 计划集有限、前置条件准 | 非法计划被选 | violation rate>0 | 失败归因、max 调用限制 |
| 技能间误差独立 | 共享 trunk 相关误差 | 跨技能 ε 相关 | 用 3a 线性界、报相关系数 |
| p_i 评测期平稳 | 策略自适应 | 前后半段漂移 | 分段估计 |
| planner 使用 p̂ | P1≈P3 | 注噪后成功率不变 | 即 RQ3 负结果判据 |

### 4.8 写作安排
正文（0.75–1 页）：式 (13) + 式 (6)(8)(9) 与 26–40% 上界 + 精简版 TP 表 + 一段"为什么两个信号都要"。附录 E（约 2 页）：式 (1)(2)(4)(5)(7)(14)(15)(16) 的推导、数值表、假设边界表、corollary 常数修正。**篇幅核算**：正文 0.75 页 + 附录 2 页；若 8 页紧张，正文压到 0.5 页（只留式 13 + 表）。

---

## 5. 实验设计

### 5.1 平台、吞吐与本地/租卡划分

| 资源 | 配置 | 用途 | 吞吐假设（effective steps/s，W4 实测后写入预注册） |
|---|---|---|---|
| 本地工作站 | RTX 4070 Ti Super 16GB，32GB RAM，Ryzen 7 7800X3D | 手内技能（S1/S2）、Tier-0/1、pilot、校准 rollout | N=256：2,000；N=1024：3,500 |
| 租用 | 1×A100 80GB（**¥7/h，假设区间 ¥6.5–8，W4 比价后锁定**） | 臂式技能（S7/S8）与 N≥2048 的库级训练、CRN 评测 | N=2048：12,000；N=8192：15,000；arm 加权 6,000 |
| 存储 | 2TB 本地 + 对象存储 200GB | checkpoint（每 50 更新）、轨迹 10% | 存储 ¥80 |

- **吞吐口径**：`effective steps/s` 已按技能算力权重 c_g 归一（手内 1.0、臂式 6.5）；W4 用 3 技能×3 N 实测拟合 `thru(N)`，写入预注册。**若实测低于假设 >20%，触发 §6.4 的削减链（C1–C3）**。
- **显存**：N=8192 手内状态输入在 A100 80GB 上 W4 实测（目标 <70GB）；若 OOM → 按 §8 R6 降 N=4096（K=1221，B=250M），H1 阈值不变、MDE 附录更新。
- **本地 24/7 假设**：132 GPU-h / 30 周 ≈ 4.4 h/周（占空比 2.6%），不依赖通宵；所有本地 run 可断点续训。

### 5.2 协议（主协议 A，fixed-K；K=1221 与 B=500M 固定不变）

| 协议 | 固定量 | 公式 | 用途 |
|---|---|---|---|
| **A（主）fixed-K** | 更新次数 K=1221 | `B(N)=K·N·L`，L=50 | 分离"分配"与"总交互"；所有方法更新数相同 |
| B（次）fixed-B | 总步数 B=125M | `K(N)=B/(N·L)` | 回答"固定交互预算下分配是否仍有效" |
| C（描述性）fixed-wall-clock | 墙钟 6h | 由 G1 日志直接计算实际 B、K | 工程视角，**不单独跑**（省 GPU-h），只在论文附录报告 |

**A 的 B(N)**：N=2048→125.0M；N=8192→500.1M；N=16384→1,000.2M（仅 scaling 附录，需 A100×2，v3 不做）。
**B 的 K(N)**（B=125M）：N=2048→1221；N=8192→305。
**更新饥饿检查（M1 相关）**：N=8192/K=1221 给 1221 次 PPO 更新（每 skill 每次分配约 20–40 更新），W9 pilot 确认不出现"更新饥饿主导"（对比 K=305 的 fixed-B 结果，若 fixed-B 显著更差则承认 fixed-K 是限制）。

### 5.2b 为什么主协议是 fixed-K 而不是 fixed-B

fixed-B（"所有方法都跑 125M 环境步"）看起来最公平，但它隐含一个错误前提：在 N 很大时，500M 步的预算对应的**策略更新次数** K=B/(N·L) 会极小（N=8192 时 K≈1221；若再大两个数量级，K 会掉到个位数）。此时任何一个方法都停在"策略还没更新几次"的阶段，比较的其实是采样器的 cold-start 行为，而不是"经验分配是否让有限更新更有效"。这正是大规模并行 RL 文献里被反复指出的 update-hunger 问题（2506.03404；SGS 用 2^20 envs 时也以"更新次数"为单位讨论）。

因此我们采用 **fixed-K 为主**：每个 run 有相同的策略更新次数 K=1221，N 越大则 B=K·N·L 越大；fixed-B 125M 只作为"部署区制"的次要协议（N≤2048 时可与 fixed-K 对照），并明确标注"在 fixed-B 下 N=8192 等价于把 K 压到 305"。两个协议都报，主结论只来自 fixed-K；这避免"用吞吐赢论文"，也让"分配是否值得"这个问题有可比的决策节奏。需要坦白的代价：fixed-K 下 N=8192 的单 run 要 9.51 GPU-h，是 N=2048 的 3 倍，预算因此被 N=8192 主导（304.3/493.3=61.7%）——这也是我们砍掉 stretch 与 G3/G4/G5 的直接原因。

### 5.3 基线与公平性规则

| 类别 | 方法 | 实现与保真度 | 预算 |
|---|---|---|---|
| 主基线 | **uniform** | 每技能/桶等概率采样 | 协议 A |
| 主基线 | **SER**（2511.21011） | staggered resets：技能内固定相位错开 reset | A |
| 主基线 | **SGS-band**（2610.12465 最小重实现） | task-config 成功率加权 reset，三档（never/improving/mastered），保留 success-band 判据；**G-S1**：若任一设置不能 >uniform 3pp，判重实现失败并降附录 | A |
| 主基线 | **PLR-style**（PLR/RED，2010.03934/2110.02439） | 按 regret/LP 给配置打分，库级扩展为技能×桶 | A |
| 主基线 | **LGA（本文）** | §3.4 + §3.5 head | A |
| 扩展 | **SGS+LGA** | SGS 权重作 LGA 先验 | A |
| 扩展 | **oracle-LP** | 真实 p_b 代入式 (8) 的 Neyman 分配（**离线日志重加权，0 GPU**；测量 LGA 离最优多远） | 离线 |
| 参照（不进 H1） | **P2P-style**（2606.26428 最小适配） | play→finetune 套到每技能，等总步数 | 协议 D；**移 ICLR 版** |

**CRN 口径（全文统一）**：主格 **16 cell**（4 技能×4 桶）；24 cell 只在 W12 gate 扩到 6 技能时启用（成本 12h/¥84，从缓冲出）。

**公平性规则（写入预注册）**：①同一 PPO 超参、网络、K、评测协议；②LGA 分配开销 <1%；③SGS/PLR 最小适配由同一作者实现并开源，说明放附录；④P2P-style 给足原文超参搜索（3 组）并报最好；⑤外部方法不能复现原文趋势时按 G-S1 降附录、如实报告，不静默删除。

### 5.4 实验矩阵（CoRL 版；砍 stretch 与 G3/G4/G5）

| 组 | 内容 | runs | 平台 | 用途 |
|---|---|---|---|---|
| G1a | 4 方法 × 8 seeds × N=8192，协议 A | 32 | A100 | H1 主 / H2 |
| G1b | 4 方法 × 8 seeds × N=2048，协议 A | 32 | A100 | H1 副 / scaling |
| G2 | 消融 A1–A8 + A10 × 2 seeds @N=2048（9 变体×2） | 18 | A100 | 机制归因 |
| G6 | **CRN 校准评测批**（4 方法×16 cell×1,000×8 seeds = 512k rollouts；6 技能时 24 cell） | 1 批（32 GPU-h） | A100 | H2/H2b |
| G7 | agent 2×2 + 剂量-反应（LLM 调用，非训练） | — | 本地/API | H3/H5 |
| L1 | Tier-0 Aloha peg sanity 3 runs（B=30M） | 3 | 本地 | W4 可行性 |
| L2 | Tier-1 协议 B 子集：3 方法×4 技能×1 seed + LGA 第 2 seed @N=1024 | 13 | 本地 | 低成本 scaling |
| L3 | pilot：吞吐/显存/校准 10 runs | 10 | 本地 | W4/W6 校准 |
| L4 | 校准 smoke + CRN 随机化验证 | 5 | 本地 | 测量可靠性 |

合计：**租卡 83 runs + 1 评测批（493.3 GPU-h）**；**本地 31 runs（132 GPU-h，免费）**；**agent 600 + 240 episodes**；oracle-LP 与 A6 的日志重加权是**离线分析（0 GPU）**。

### 5.5 消融（A1–A10）

| # | 变体 | 去掉/替换 | 预期（若机制正确） | 关联 |
|---|---|---|---|---|
| A1 | LGA−σ̂（只用 LP 导数） | σ̂ 项 | 前沿 ECE 变差，高噪声技能被过度分配 | H2 |
| A2 | LGA−compute 归一 | c_g=1 | 臂式技能饥饿，库级成功率下降 | H1 |
| A3 | LGA−Level-2 | 桶级分配 | 桶内异质时收益下降 ≥1/3 | H1 |
| A4 | LGA−Hedge（改 UCB1） | 指数权重 | 抖动/饥饿指标变差 | §3.4 |
| A5 | LGA−guards | G-1/G-2/G-3 | 训练不稳、方差增大 | §3.4 |
| A6 | 分配替换 | LGA 分配→uniform（重跑） | H2c：ECE 优势消失 ≥70% | H2c |
| A7 | S=4 vs S=6 | 技能数 | H4 交互（W12 gate 通过才跑） | H4 |
| A8 | 无 held-out（训练分布内校准） | 评测划分 | ECE 虚低、agent 收益夸大（负面证据） | 测量 |
| A9 | EMA+isotonic vs learned head | 预测器 | head ECE 再降 ≥5%（H9）；否则 H2 降级 | H2d/H9 |
| A10 | 冻结分配（用 W12 快照固定） | 自适应更新 | 动态 vs 静态归因 | H1 机制 |

**A11 为附录结构消融**（共享 trunk vs per-skill head，0 预算、不占主表）；**A7 为附录条件项**（仅 W12 gate 扩 6 技能时才跑）；**A9 为分析型消融（0 GPU）**：在已训练 checkpoint + CRN 日志上拟合/对照两种预测器；**A10 进主图**（动态 vs 静态归因）。三者均**不固定进主表**（Lead 已拍板，2026-11-02）。

### 5.6 统计与功效

- **重复单元 = seed**；mixed model `success ~ method + (1|seed) + (1|skill)`；per-skill 效应与 95% CI（bootstrap 10k，按 seed 聚类）。
- **MDE**（sd=6pp，双侧 5%，power 80%）：5/6/8/10 seeds → 7.5/6.9/5.9/5.3pp。**H1a 阈值 = max(6pp, W6 实测 MDE)**（M5 规则写死：sd≤5→6pp；5–7→7pp；>7→先砍 G3/G4/G5 换 2 seeds，仍 >7 则 7pp 并标注）。
- **H2 配对分析**：CRN 下 ΔECE 用 cell 级配对 bootstrap（10k，按 seed 聚类），报 95% CI 与相对改善；**降级条款**：若相对 ΔECE<10% 且不显著 → 不主张"更校准"，改报 N1/N2。
- **H3 剂量-反应**：混合模型 `success ~ η + (1|seed) + (1|task)`，主检验 = 斜率 <0（单侧，Holm family 内）；同时报单调性（4 档顺序的 Page 检验）。
- **二元 H3c（附录）**：200 配对 McNemar；power 0.82 @ρ=0.3（0.67 @ρ=0；0.92 @ρ=0.5），报告敏感性。
- **多重比较**：Holm，family = {H1a, H1b, H1c, H2, H3-dose}；H4–H9 探索性，报告效应量与 CI，不做显著性主张。
- **H1b 非劣**：δ=5pp（单侧 α=0.025）；**H1c**：按 2 个长程任务的技能乘积计算，缺失技能用校准 p̂ 补齐并做敏感性。

### 5.6b 多重比较、效应量与"什么才算赢"

主检验族固定为 `{H1a, H1b, H1c, H2, H3}` 五个端点（H4–H9 为探索性，不进 Holm 族，报告中标注 exploratory）。H1a–H1c 用预注册的 min(6pp, W6 MDE) 阈值：8 seeds、sd≈6 时 MDE≈5.9pp，power 0.81；若 W6 实测 sd=7，规则自动升到 7pp——这一"阈值随 pilot 调整、但调整规则提前写死"的做法，是为了避免"先看结果再定阈值"（v2 的 30% ECE 阈值就是被 judge 指出的问题）。H2 的主指标是 decision-weighted frontier ECE 的**配对**相对下降，**预注册阈值 20% 相对**（v3.2：30% 不可达，见 §2.3/§4.3）：因为同一批 1,000 个 held-out 状态被所有方法共用（CRN），方法间差异的方差主要由模型行为差异贡献，而不是状态抽样差异；配对把 250/cell 的二项 SE 3.16pp 压到可测 2pp 量级（见 §5.7）。power：配对 SD 1.5–2.0pp、8 seeds → MDE 1.5–2.0pp，power 0.80–0.95；W16 实测 sd>2.5pp 时按预注册降 15% 或补 2 seeds。**解读纪律**：若只达 20%，论文只主张"方向性校准改善"，不写"30%"。H3 用剂量-反应的斜率检验（混合模型里噪声水平 × 方法的交互项），而不是只用单个 12pp 的二元比较；后者留在附录作为"审稿人熟悉的读法"。

我们同时报效应量与 CI，且**不把 p<0.05 当作结论**：若 H1a 显著但 Δ<3pp，论文的措辞降为"方向一致、幅度不足以支撑部署主张"；若 H2 显著但相对下降 <10%，按预注册降级条款不主张"更好校准"。这些规则写在 W6 预注册里，W16 冻结实现，W24 前不得修改。

### 5.7 校准评测协议（M3：配对 CRN + 双口径）

1. **CRN 评测集**：每个 (skill×bucket) cell 固定 **1,000 个 held-out 初始状态**（W6 冻结）；4 技能 = 16 cell，6 技能 = 24 cell。
2. **采样**：每 cell 跑 1,000 次确定性 rollout（策略确定性动作），所有方法/种子共用同一状态序列——两臂的评测噪声共享。
3. **指标**：主 = **decision-weighted frontier ECE**（权重=评测分布中该桶被查询频率；frontier 限 p∈[0.3,0.7]）；辅 = 全 cell decision-weighted ECE、Brier、NLL、风险–覆盖（0.5/0.7/0.9）；**附录** = uniform-weight ECE（最坏口径）。
4. **预测器对照**：EMA+isotonic/Beta vs learned calibration head（A9），在 calibration split 拟合、test split 报 ECE；head 特征见 §3.5。
5. **机制**：`|p̂−p| ~ age + allocation_share` 回归（H2d）；LGA 份额 vs oracle Neyman 份额的 Spearman ρ（P2）；δ 预测 ECE vs 实测（引理 4b）。
6. **报告**：正文一张可靠性图（per-skill）+ 一张 ECE–N 主图；cell 级细节进附录。**成本（v3.2 修正）**：4 方法×16 cell×8 seeds×1,000 = **512k rollouts ≈32 GPU-h/¥224**（原写 128k 漏了方法因子）；6 技能 768k/48h，从缓冲出或按 C3 削减。**FLOOR 触发时** H2 降为 **4-cell 小 CRN（8h/¥56）**，并在论文中降级 H2 为"方向性证据"。

### 5.8 LLM agent 协议（2×2 + 剂量-反应）

- **条件**：{LGA, uniform} × {calibrated, no-p̂}，四格 P1/P3/P4/P3u；另加 P0 random、O1 oracle、S scripted 锚点。主检验：P1 vs P3（50 trials/task×2 任务=200 配对）+ 剂量-反应（4 档×2 任务×30 trials=240 episodes）。
- **任务（CoRL 版 2 个）**：L1 双件装配 **S1→S7→S8**（H=3）；L2 紧配合 + 扰动重试 **S2→S7→S8**（H=3）。短链 **S1→S7**（H=2）只用于 TP1 斜率核验（评测，不训练）。L3/L4 移 ICLR。
- **每个任务 50 个固定初始状态**（held-out goal cell 采样，W6 冻结，与 CRN 独立）。
- **LLM 配置**：W10 选定并锁定快照（provider、model id、version string、temperature=0、max_tokens=512、system prompt hash）；W16 冻结；调用缓存（相同 prompt 命中不重复计费）；失败重试 ≤2；单 episode ≤20 调用、90s。
- **剂量注入**：对 P1 的 `p̂` 注入 `η∈{0,2,5,10}pp` 独立噪声（每 episode 重抽、固定 seed），主图 = 成功率 vs η。
- **成本（v3.2 重算）**：600 base episodes + 240 剂量 episodes = 840 ×6.5 calls×2.2k tokens ≈**12.0M tokens**；70/30 输入输出、$3/$12 → ≈**$68.5≈¥493**；+30% 冗余 → **¥641**；**硬顶 ¥700**（¥600 预警时砍剂量 10pp 档与锚点，主检验不动）。

### 5.9 机制指标与失败归因

- **分配**：regret（相对 oracle Neyman）、waste（分给已掌握/不可学桶比例）、starvation（连续 200 更新权重 <1/(4M)）、churn（TV）、min-skill 成功率。
- **校准**：age–误差斜率、allocation share–误差斜率、σ̂ 与 |误差| 相关性、R²；head vs EMA 的 ECE 对照。
- **效率**：达到 uniform 最终库级成功率的墙钟比；每 GPU-h 的库级成功率增量。
- **失败归因**：4 类（分配/执行/校准/恢复），双人标注 120 条轨迹，κ≥0.8；allocation-swap 后"分配型"比例变化 ≥10pp。

### 5.10 负结果、降级与"最小充分证据集"

| 分支 | 情形 | 论文形态 | 去处 |
|---|---|---|---|
| N1 | H1 不成立、H2 成立 | "分配主要影响校准而非库级成功率" | CoRL（质量 ~35%） |
| N2 | H2 不成立 | "success-band 何时足够"边界研究（margin + TP2） | D&B / CoRL 边缘 |
| N3 | S7 不可行 | 手内版 + 臂式失败分析 | CoRL |
| D1/D2/D3 | N=8192 OOM / LLM 超支 / 主曲线或 agent 降级 | 降 N=4096 / 砍剂量档 / 收为"分配→校准" | 预注册规则；**D3 并入 §8 D10，W24 一处执行** |

**最小充分证据集（MSE，若排期或预算恶化，保命用）**：4 技能（S1/S2/S7/S8）+ 2 任务 + 4 方法（uniform/SGS-band/LGA/PLR-style）×8 seeds×N=8192（32 runs、304.3h、¥2,130；oracle 用日志离线重加权，0 GPU）+ CRN 512k rollouts（32h、¥224）+ H3 200 配对（LLM ≈¥180）≈ **¥2,800、W20 冻结**。**可砍项（按顺序）**：剂量档位（240→120 eps，−¥91）→ G1b N=2048 降 4 seeds（−¥172）→ G2 8→4 变体（−¥198）→ oracle-LP（−¥0，本地）→ 6 技能 gate（不加 S3/S4）。

**加分实验优先级（vs MSE）**：

| 项 | 成本 | 预期 Δp | 决策 |
|---|---|---|---|
| 配对 CRN（M3） | ¥224 | +4–5pp | **必做（MSE 内）** |
| learned head + age 机制（M4） | 0 GPU | +3–4pp | **必做（MSE 内）** |
| 剂量-反应（M6） | ¥91 | +2–3pp | **必做（MSE 内）** |
| Neyman 正文（M2） | 0 | +2–3pp | **必做** |
| H2c 置换检验 | 0 | +1–2pp | 必做 |
| +2 seeds（8→10） | ¥530 | +2pp | 不做（先破预算） |
| 6 技能（S3/S4） | 16 人日 | +1pp | 仅 W12 gate 通过 |
| P2P/OmniReset 外部基线 | ¥361 | +1pp | 移 ICLR |
| 8 技能/Aero/第 3 planner | ¥400+ | +0–1.5pp | 不做 |

### 5.11 Go/No-Go（写死）

- **W5 G0（自 W4 顺延；W4 只做 Aloha peg 代理 sanity + arm 可跑）**：本地 ≥3,000 eff-steps/s；arm+LEAP N=512 可跑；Aloha peg 3 seeds ≥20%；S7 三容差探针完成（1/3/5 mm，记录非零成功率曲线，**不设 10% 阈值**——阈值统一为 Aloha peg 的 20%）。
- **W8 G1**：≥4 技能可训（最简单桶 ≥50% @≤150M 步）；A100 N=2048 ≥10k eff-steps/s；否则 D1/N3。
- **W12 G2**：pilot 中 LGA−uniform ≥3pp；sd 决定 MDE 与是否补 seeds；是否加 S3/S4。
- **W16 G3**：N=2048 主格完成；LGA 优势 <3pp → N1；agent 管线 dry-run 100 episodes。
- **W20 G4**：agent 2×2 跑通；否则 agent 降验证（D3）。
- **W24 G5**：分析冻结（不再改主分析）；**W26 论文数字冻结**。

---

## 6. 预算（≤¥5,000，逐行可复算；附录 C 脚本实跑）

### 6.1 单价与假设
- A100 80GB：**¥7/h**（假设区间 ¥6.5–8；W4 比价 3 家后锁定）。
- 吞吐：§5.1；每 run 固定开销 **+0.25h**（启动/编译/checkpoint）。
- 缓冲：GPU 小时 **+12%**（失败重跑/超时/排队）；杂项：电费 **¥220**（本地 132h×0.35kW×¥0.6/kWh + 空调，向上取整）、存储 **¥80**、**LLM ¥641**（v3.2：840 episodes×6.5 calls×2.2k tokens ≈12.0M tokens，$3/$12 → $68.4≈¥492，+30% 冗余 → **¥641**；硬顶 **¥700**）。
- 汇率 ¥/$=7.2（W4 锁定；波动 >5% 触发削减链）。
- **v3.2 修正说明**：v3.1 的 G6 漏乘方法因子（128k→**512k rollouts**），成本 8h→**32h/¥224**；LLM 由 ¥549→**¥641**（旧口径把 8.58M 当总 token）。基准随之 4,528→**¥4,809**。**口径注**：lead 消息里的 ¥4,788（余 212）按"G6 增量不计 12% 缓冲"估算；本文统一按 §6.1 公式（所有租卡 GPU-h 计 12% 缓冲）得 **¥4,809（余 191）**，差 21 元，附录 C 脚本可复算——以脚本为准。

### 6.2 租卡行表（83 runs + 1 评测批）

| 组 | runs | h/run | GPU-h | 费用 ¥ |
|---|---|---|---|---|
| G1a N=8192（4 方法×8 seeds） | 32 | 9.51 | 304.3 | 2,130 |
| G1b N=2048（4 方法×8 seeds） | 32 | 3.14 | 100.5 | 703 |
| G2 消融（9 变体×2 seeds @2048） | 18 | 3.14 | 56.5 | 396 |
| G6 CRN 评测批（4 方法×16 cell×1,000×8 seeds = 512k rollouts） | 1 批 | 32.0 | 32.0 | 224 |
| **小计** | **83** | — | **493.3** | **3,453** |

h/run 由 `h = B(N)/thru(N)/3600 + 0.25` 直接算出（B=K·N·L，K=1221，L=50；thru：N=8192 15,000、N=2048 12,000）；G6 = 512k rollouts ÷ 16k rollouts/h。

**含缓冲与杂项**：`493.3×1.12 = 552.5 GPU-h → ¥3,868`；`+ 电 220 + 存储 80 + LLM 641 = ¥4,809`；**余 ¥191**（≤5,000 ✓）。若 LLM 用满 ¥700：¥4,868，余 ¥132（仍安全）。

### 6.3 本地免费算力

| 项 | runs | GPU-h |
|---|---|---|
| Tier-0 Aloha peg sanity | 3 | 13.3 |
| Tier-1 协议 B 子集（13 runs：3 方法×4 技能 + LGA 第 2 seed @N=1024） | 13 | 68.7 |
| pilot（吞吐/显存/校准） | 10 | 30.0 |
| 校准 smoke + CRN 随机化验证（5 runs） | 5 | 20.0 |
| **合计** | **31** | **132.0**（≈4.4 h/周，30 周；含无 GPU 的文档/预注册工作） |

### 6.4 情景、削减链、FLOOR 与硬上限

| 情景 | GPU-h | 费用 | 说明 |
|---|---|---|---|
| 基准 | 552.5 | **¥4,809** | 余 191 |
| 单价 +50%（¥10.5/h） | 552.5 | ¥6,742 | 触发 C1–C4 |
| 时长 +50% | 828.8 | ¥6,742 | 同上（两因素对租金同倍数） |
| 双因素（×2.25） | 1,243.2 | ¥9,643 | 触发 C1–C5 → FLOOR |
| FLOOR（双因素） | 561.6 | **¥4,872** | 仅价 +50% 时 ¥3,562 |

**削减链（单价 +50% 或时长 +50%，逐级执行直到 ≤5,000）**：
- **C1** 砍 G2 消融（18 runs，56.5h）→ **¥6,078**
- **C2** G1a seeds 8→6（−8 runs，76.1h）→ **¥5,183**
- **C3** CRN 32→16h（1,000→500 rollouts/cell；H2 power 下降，预注册写明）→ **¥4,995** ✓
- **C4**（安全余量，通常不需要）G1b seeds 8→6（−25.1h）→ **¥4,699**

**双因素（×2.25）逐级**：基准 ¥9,643 → C1 ¥8,646 → C2 ¥7,304 → C3 ¥7,022 → C4 ¥6,579 → C5 FLOOR **¥4,872** ✓。

**FLOOR（双因素触发）**：3 方法（uniform/SGS/LGA）×5 seeds×2 N = 30 runs（189.8h）+ G2 8 runs（25.1h）+ **4-cell 小 CRN（8h，H2 降级为方向性证据）** = 222.9h；×2.25 = 501.5h；×1.12 = 561.6h；×¥7 = ¥3,931；+杂项 ¥941 = **¥4,872**。**注意**：若 FLOOR 仍保留 32h CRN，则 246.9h→≈**¥5,296 破线**，因此 **FLOOR 必须把 H2 降为 4-cell CRN**；FLOOR 仍支撑 H1（Δ 阈值 7pp）、H2（方向性）、H3（2×2 + 剂量 2 档），H4–H9 降附录。

**硬上限规则（预注册）**：①¥5,000 为硬上限，任何情景不得突破；W4/W10/W16 各做一次预算审计（附录 C 脚本）。②余 <¥100 时冻结非主实验（G2/G6 之外的加分项）。③LLM 达 ¥600 时砍剂量 10pp 档与锚点；达 ¥700 停 LLM，用已完成 episodes 分析并报实际 n。④不可预见项（A100 涨价 >20%、API 停服）走 Plan C：2×4090 24GB（¥2.5/h，N=2048 主格）或开源权重模型（租 A100 8h ≈¥56，但需重跑 agent）。⑤所有削减记录在 `budget_log.md`。

### 6.5 LLM 成本明细（v3.2 重算）

840 episodes（600 base + 240 剂量）×6.5 calls×2.2k tokens = **12.0M tokens**；70/30 输入输出、$3/$12 → **$68.4≈¥492**；+30% 冗余（版本漂移/重试/缓存未命中）→ **¥641**；硬顶 **¥700**（预警 ¥600）。

### 6.6 人力与资源声明
- 1 名研究生（作者），45 周全额投入；工程 W1–16（68–72 人日，含缓冲），实验 W10–W24，写作 W24–W31；双人标注由同实验室 1 名同学协助 ≤5 人日（致谢，不计预算）。
- 全部代码、配置、预注册、prompt、模型快照、CRN 状态集、预算日志随论文开源（MIT）；artifact 含一键复现与 3 个 seed 的完整 checkpoint。

### 6.7 Compute accounting（每个 ¥ 可追溯）
`每 run GPU-h = B/thru/3600 + 0.25`；`租卡总费用 = Σ_runs h × 1.12 × ¥7`；`杂项 = 电 ¥220 + 存储 ¥80 + LLM ¥641`；`G6 = 方法数 × cell 数 × seeds × 1,000 / 16k`。本地 run 费用记 ¥0（电费已并入杂项），本地/租卡分界 = 是否 N≥2048 或含臂式接触技能。

---

## 7. 时间线与 Go/No-Go（45 周，1 人）

> 起点 W1 = 2026-11-02。CoRL 2027 截稿预计 2027-06 中旬（**待核实**，按 W31≈2027-06-07 排）；ICLR 2028 截稿预计 2027-09 下旬（**待核实**，按 W46–W47）。所有日期留 1 周机动。
> **主实验期（W1–W24）GPU-h 列和 = 租卡 493.3 / 本地 132**，与 §6.2/§6.3 一致；W32–W45 的 ICLR 加分项另计（≤60 租卡 h / 20 本地 h），不计入 v3.2 主预算。

| 周 | 工作 | 交付/门 | 本周 GPU-h（租/本地） |
|---|---|---|---|
| W1–W2 | 基建：Franka+LEAP MJCF、PPO 训练栈、日志/checkpoint、CRN 评测 harness 骨架、held-out 划分与 SHA256 | 仓库可跑 | 0 / 6 |
| W3–W4 | Tier-0 前半：S1/S2 手内 MVP、S7 arm+LEAP 集成、S8 弹性盖 MVP；**W4 只做 Aloha peg 代理 sanity + arm 可跑**（G0 顺延 W5，见 §5.11） | — | 0 / 30 |
| W5–W6 | **W5 G0**（本地有效吞吐、arm+LEAP N=512、Aloha peg 3 seeds ≥20%、S7 三容差探针）；4 技能流水线打通、calibration head 原型、agent harness；**W6 预注册冻结** | G0 + 预注册 v1 | 0 / 30 |
| W7–W9 | CRN 状态集冻结（16 cell×1,000）；head 拟合脚本；agent dry-run 100 episodes；**W8 G1**：4 技能本地可训、A100 有效吞吐 ≥10k steps/s、S7 3 mm 有学习曲线 | G1（W8） | 0 / 20 |
| W9–W12 | 主格前半 G1b（N=2048，32 runs，50.3h）；W12 预读 LGA−uniform 差值（不做正式检验） | **G2（W12）**：若差值 <3pp 且 head 无改善 → 转 N1/N2；若超前 ≥2 周 → 追加 S3/S4（6 技能）并扩 CRN 到 24 cell | 50.3 / 20 |
| W13–W16 | 主格后半（G1b 剩余 50.2h）+ **G6 CRN 首轮（32h）**；机制回归（EMA age–误差、分配份额–绝对误差）；**W16 G3 主格冻结**（复核 paired SD：≤2.5pp 用 20% 阈值；>2.5pp → 降 15% 或补 2 seeds） | G3 | 82.2 / 20 |
| W17–W20 | G1a（N=8192，32 runs，304.3h）+ G2 消融（18 runs，56.5h）；**W20 中检 + agent 2×2/剂量 840 episodes** | **G4（W20）**：agent 跑不完 → 主贡献收为"分配→校准"，agent 降验证（不砍主表） | 360.8 / 2 |
| W21–W24 | 统计（混合模型、Holm、H2c 置换、H2d 回归、剂量斜率）、held-out 组合评测、失败归因；**W24 分析冻结 + D10 六档决策**（decision_log.md，3 日内） | W24 决策 | 0 / 4 |
| W25–W26 | 图表定稿、用附录 C 脚本复核每个 ¥、摘要/引言定稿 | **W26 数字冻结** | 0 / 0 |
| W27–W30 | 写作（8 页）、附录、artifact 打包、90s 视频、Anonymous.4open 仓库、arXiv 预印（W30） | 投稿包 | 0 / 0 |
| W31 | **CoRL 2027 投稿**（日期待核实；按 2027-06-07 排） | 提交 | 0 / 0 |
| W32–W45 | ICLR 2028 兜底：6 技能扩展（S3/S4/S5/S6）、G3 P2P 外部对照、G4 OmniReset、G5 鲁棒性、Aero 附录、第 3 planner；W38 arXiv v2；W46–W47 ICLR 投稿 | ICLR 包 | 0–60 / 20 |

**门与阈值（统一口径，§5.11 为准）**：G0（W5）本地 ≥3,000 eff-steps/s、arm+LEAP N=512 可跑、Aloha peg 3 seeds ≥20%、S7 三容差探针完成；G1（W8）A100 ≥10k steps/s、S7 3 mm 有非零学习曲线；G2（W12）LGA−uniform ≥3pp 或 head 校准可改善，否则转 N1/N2；G3（W16）主格数据冻结、paired SD 复核；G4（W20）agent 2×2 跑通。任一 gate 失败按 §8 规则降级，不延排期。

**人力核对**：W1–W16 工程与运维 ≈68–72 人日（4 技能 + arm 集成 + harness + 预注册），折 13.6–14.4 周；W17–W26 分析/写作 + 12–14 人日 agent 支持。若 W12 追加 S3/S4，+16 人日，用 W20 后 2 周与 W27–W28 吸收（牺牲 1 周缓冲）。ICLR 阶段（W32–W45）只做加分项，主表不动。

---

## 8. 风险表（R1–R17）与 Plan B

| # | 风险 | 概率 | 影响 | 缓解 / 触发即执行 |
|---|---|---|---|---|
| R1 | 4 技能覆盖面被审稿人质疑"技能库太小" | 高 | 中 | 定位写"受控研究 + 开源基准"；W12 gate 达标则加 S3/S4（6 技能）；否则在 limitation 明示并给扩展清单 |
| R2 | arm+LEAP 集成失败/延期（自建 MJCF 接触不稳） | 中 | 高 | W3–W4 只做最小集成 + 三容差探针；失败则 **接受降级：主表 5 mm + S7 改用 Aloha peg insertion（playground 现成、1 mm）作臂式代理**，S8 保留手内，**不砍 S7**（Lead 已拍板） |
| R3 | S7 3 mm 物理上不可学（400 Hz 下接触抖动） | 中 | 中 | 主表 3 mm；G0 失败→**接受 5 mm**（1 mm 主张**删除**）；limitation 写"3 mm 未达成"；结论写"容差-成功率曲线"（Lead 已拍板） |
| R4 | 手内技能（S1/S2）过于简单，LGA 无差异可测 | 中 | 中 | 用 S4 Finger-Gait 式难度档（如 W12 追加）；并在 H2 报告"简单技能上无差异"作为边界证据 |
| R5 | 桶间 a_g 不可分辨（如无目标 site 的技能） | 中 | 中 | 桶定义用 **spawn 区域 × DR**（同 v1 教训）；M=4 兜底；H1d 只报趋势 |
| R6 | A100 上 N=8192 显存/吞吐不达标（有效 <10k steps/s） | 中 | 高 | 保底 N=4096 并把 B 按 K·N·L 重算（K 不变，B 减半）；或用 2×A100 数据并行（预算 +¥700，触发削减链） |
| R7 | SGS-band / SER 复现不忠实 → baseline 不公平指控 | 中 | 高 | 预注册 sanity：SGS 至少 1 个设置上 >uniform ≥3pp；否则 W20 前排查/联系作者；附录给复现代码与日志 |
| R8 | PLR-style 在技能库级不适配（其原语境是 level/room） | 中 | 低 | 明示"技能级改写的近似"，作为下界基线之一，不作主要对比对象 |
| R9 | 500M×64 runs 排期超支（吞吐低于假设） | 中 | 高 | 每周一核对 GPU-h 消耗；W16 冻结硬门；超 15% 即启用 C1（砍 G2）→C3（G1a 8→6 seeds） |
| R10 | calibration head 过拟合（1,000 状态太少） | 中 | 中 | 拟合/测试按初始状态划分（非按 rollout）；嵌套 5 折 + 报告 CI；与 isotonic/EMA 三条 baseline 对比 |
| R11 | LLM agent 方差大、API 成本超支 | 中 | 中 | 200 配对 + 固定 temperature/seed + 3 次重复取多数；硬顶 ¥700（预警 ¥600）；超支即砍锚点组（−100 episodes） |
| R12 | 纯仿真被质疑"无真机意义" | 高 | 中 | 引言接 P2P/OmniReset 的真机数字做外部锚；limitation 段单列；不声称迁移 |
| R13 | 理论被指"装定理"或界太松（24ε） | 中 | 中 | 明写"方向/斜率预测，不是幅度"；正文只放 3a/3b 与 TP 表，推导进附录；不写"保证" |
| R14 | 与 SGS/OmniReset/ADEPT 的 novelty 重叠被质疑 | 中 | 高 | §9 逐行差异表 + "不声称"清单；把"校准 + 技能 API + 组合评测"作为主差异；ADEPT 若已录 CoRL 2026 则明确其为大规模预训练系统，我们是受控研究 |
| R15 | 噪声注入被视为"人造"（只在该注入下成立） | 中 | 中 | 从现有日志估**自然 per-bucket 方差 σ_a**（0 GPU）与注入曲线并列；主张只限"注入下"，自然情形列为开放问题 |
| R16 | CoRL 2027 日期/政策变化；ICLR 兜底来不及 | 低 | 高 | W1 与 W20 两次核实官网；排期留 W32–W45 全段；若 CoRL 不中，W31 后按 ICLR 计划执行，不回炉主表 |
| R17 | 6 技能 gate 提前触发导致 CRN/预算重排 | 中 | 中 | W12 gate 前先算 24 cell 成本；若加，CRN 从 8h→12h（+¥28），从缓冲出；否则不加 |
**Plan B 的三条已备选路径**（触发条件写在 §8 表里，这里给操作细节）：①**S7 不可行**：改用 playground 现成的 Aloha peg insertion 作臂式代理（1 mm 容差、双臂平行夹爪），代价是"灵巧手"叙事弱化——此时论文标题与摘要改为"hand-centric"，S7 从主表移到附录，H1 的臂式技能单元由 2 个降为 1 个（S8），并在 limitation 说明 embodiment 差异；②**N=8192 不可行**（显存/吞吐不足）：主格降 N=4096，K 不变、B 减半，H1 的 N 维度只保留 {2048,4096}，并把省下的 GPU-h 用于 6 技能或更多 seeds（二选一，W8 决定）；③**CoRL 延误/不中**：主表冻结后直接转 ICLR 版（§11.2），不重跑主格；若 ICLR 也错过，则以 arXiv + workshop 形式发布负结果/方法，并在 W45 前把 benchmark 作为独立 artifact 投稿（期刊/数据 track），保证工作不被"烂在手里"。



**W24 决策规则（D10，六档；写入预注册与 §13）**：(a) **H1a ≥6pp** → CoRL 完整论文（当前主叙事）；(b) **3–6pp** → 仍投 CoRL，H1 降 secondary、H2/H3 为主，allocation 表述降级（"modest"）；(c) **<3pp 但 H2 配对 ΔECE ≥10% 且显著** → CoRL 改投"校准/benchmark"版（贡献②③ 主打，retitle 为 *Calibration and Composition in Dexterous Skill Libraries*）；(d) **<3pp 且 H2 不达标** → 转 ICLR 理论版（W32–W45 扩 6 技能 + 负结果 + 传播界强化）；(e) **agent 2×2 未跑通（原 D3）** → 主贡献收为"分配→校准"，agent 层降为验证章节，仍可投 CoRL（不因此转 ICLR）；(f) **技能数 <4（Plan C 触发）** → hand-only 版（S1/S2 手内 + S7 降 Aloha 代理），CoRL framing 改写，ICLR 兜底不变。**优先级**：H1∈[3,6) 且 H2≥10% → 优先 (b)；(e) 与 (c) 同时满足 → 先执行 (e)（收窄 agent 叙事），再按 H2 结果定 (b)/(c)/(d)；(f) 一旦触发则 (a)–(d) 全部按 hand-only 重评。(e) 与 (d) 的"原 D3"口径合并到 W24 一处执行，不在 W16 单独触发。决策在 W24 分析冻结后 3 日内完成，写入 `decision_log.md`，不追溯（对应 §13 的 D10）。

**Plan C（最坏）**：若 W8 G1 失败（arm+LEAP 不可行且 Aloha 代理也不达标）→ 退化为"LEAP 手内 4 技能 + 手内序列长程任务"的纯手内版本（仍满足 Brief 的 agent 组合叙事），论文标题与摘要改写为 hand-only，CoRL 期望降 5pp（记为条件分支，不隐瞒）。

---

## 9. 相关工作与差异表

**定位句**：大规模并行 RL 的采样分配（SGS/SER/OmniReset）与 LLM agent 的技能调用（SayCan/ReAct/CaP/RoboCodeX）各自成熟，但**"技能库级分配 → 每技能校准 → 长程组合"这条链没有被受控测量**；我们不做更大的系统，而是把这条链拆成可干预、可证伪的实验。

| 工作 | 做什么 | 与我们重叠 | 差异（一句话） | 威胁 |
|---|---|---|---|---|
| SGS 2610.12465 | 2^20 envs、success-band 自适应采样、四足/装配、RGB 蒸馏零样本 | 并行 env 级分配信号 | 按 success 频段加权 reset；我们是**技能库级**分配 + 校准输出 + agent 组合评测，且明确"三档单调近似而非特例" | 高（必须逐行对比） |
| Play2Perfect 2606.26428 | 22-DoF 手 + 双臂、Isaac Sim、24,576 envs play/12,288 finetune、SAPG、33× 样本效率、真机 0.5 mm 60% | 大规模灵巧 RL + 长程装配 | 我们无真机、规模小两个数量级；差异在**受控分配/校准测量**与开源 benchmark，不与其比规模 | 高（定位必须诚实） |
| OmniReset 2603.15789 | reset 多样性驱动的可扩展 RL，单奖励无课程无示范 | "多样性即课程" | 我们按 LP×不确定性**主动分配**，并测校准；OmniReset 作为外部对照（G4，移 ICLR） | 中 |
| ADEPT 2608.19182 | 大规模 RL 预训练+后训练、高 DoF、视觉触觉（judge 称已录 CoRL 2026，**待官方确认**） | 大规模灵巧 RL | 系统级 vs 受控研究；我们测"分配→校准→组合"因果链 | 中高 |
| SER 2511.21011 | staggered resets 改善 massively parallel on-policy RL | env 重置调度 | 我们不改 reset 调度，改**经验分配**；SER 作为基线之一（reset 层面） | 中 |
| PLR 2010.03934 / RED 2110.02439 | 优先级 level replay / 对抗环境设计（单任务） | 学习进度驱动采样 | 单任务 level → 多技能库；我们加校准与 agent 层 | 中 |
| SAPG 2407.20230 | agent 生成 RL 策略/奖励 | LLM 与 RL 结合 | 我们用 LLM 作**调用方**（planner），不生成奖励；SAPG 作 related | 低 |
| Being-0 2503.12533 | 分层 VLA + 技能库 + 导航操作 | 技能库 + agent | 真机 VLA 系统 vs 我们的受控技能库与校准 API | 中 |
| 2606.22471 | 语言条件双臂灵巧 RL 数据生成 | 语言→灵巧技能 | 我们不做数据生成，做分配与校准 | 低 |
| DexSim2Real 2605.05241 | 灵巧仿真到真机 | sim2real | 我们纯仿真；引其作为"仿真-真机差距"证据 | 低 |
| Text2Touch 2509.07445 | 语言-触觉表征（sim） | 语言+接触 | 我们不用触觉（limitation） | 低 |
| MuJoCo Playground 2502.08844 / ManiSkill3 2410.00425 / Isaac Lab 2511.04831 | 仿真栈 | 平台 | 我们用 Playground(MJX+Warp)；Isaac Lab 仅作对照说明 | 低 |
| SayCan 2204.01691 / ReAct 2210.03629 / CaP 2209.07753 / RoboCodeX 2402.16117 | LLM 规划/代码策略/机器人代码生成 | agent 调用 | 我们提供**带校准置信度的技能 API**，并测置信度对规划的作用；planner 用 ReAct 模板（P1）作基线 | 低 |
| Guo 1706.04599 / Angelopoulos & Bates 2107.07511 | 校准与共形预测 | ECE/覆盖 | 我们把校准接到"技能选择"决策上（decision-weighted ECE） | 低 |

**与 SGS 的三句话**（摘要 §1.5 已写）：①SGS 在 env 层按 success 频段加权，我们在技能库层按 LP×估计量不确定性分配并输出校准；②我们的权重映射在"成功概率可测、阈值固定、重采样比固定"三条件下是对 SGS 的单调近似，不覆盖其阈值/调度；③我们把"分配是否值得"变成可证伪的边界判据（σ_obs<Δ/2、σ_a<Δ/2），SGS 没有这条判据。

### 9.2 为什么"调度"不是本题的主变量

一个自然的问题：既然 SER 已经说明 reset 调度能改善大规模并行 on-policy RL，为什么不把"调度"也纳入？答案是**变量控制**：我们要测的是"经验预算在技能间的分配"，若同时改 reset 调度，LGA 的效应与 SER 的效应会在同一实验里纠缠，2×2 的设计会变成 3×2×2，在 45 周/¥5,000 下不可行。因此我们把 SER 放在 baseline 位（它改调度、不改分配），并在 §5.3 的公平性规则里要求"所有方法共用同一 reset 调度（uniform staggered）"，把调度差异作为固定因素。这也是与 SGS 的又一处差别：SGS 的 success-band 已经内嵌了"按成功频段重采样"的调度成分，我们只取它的"分配"部分做基线，并在附录说明这一改写（不声称忠实复现其完整系统）。

## 10. 论文结构（CoRL 8 页）、图表清单、视频与开源

**题目（主选 A）**：*Calibration, Not Scale: Library-Level Allocation Shapes Dexterous Skill Reliability and LLM Agent Composition*
备选 B：*DexSkillBench: A Controlled Study of Allocation, Calibration and Composition in Dexterous Skill Libraries*；C（更保守）：*When Does Allocation Matter? Calibration and Composition in a Dexterous Skill Library*。

**8 页分配**（v2 的 77–91 人日/8 页方案压缩为 P1 包）：

| 节 | 页 | 必须回答 |
|---|---|---|
| 1 Intro | 1.0 | 为什么"技能库级分配→校准→组合"是机器人学问题；三条贡献；不声称 |
| 2 Related | 0.75 | 与 SGS/P2P/OmniReset/ADEPT/PLR-RED 的逐行差异；校准文献 |
| 3 Method | 1.5 | LGA 两级、compute 归一、guards、calibration head、技能 API、planner |
| 4 Theory | 1.0 | Prop 1（含 (R+1)）方向界 + Prop 2 Neyman 26–40% + TP1–TP6 表；推导进附录 |
| 5 Experiments | 2.5 | 主表（fixed-K）、校准（CRN）、agent（剂量-反应）、消融、机制 |
| 6 Limitations/Impact/Artifact | 0.5 | sim-only、4 技能、LLM 层增量、开源与复现 |

**图表（每图对 RQ）**：
- **Fig.1** 系统总览：4 技能 → LGA 两级分配 → 校准 head/API → LLM planner → 2 长程任务（对 §1/§3）。
- **Fig.2** benchmark：4 技能 + 2 长程任务 + API JSON 示例 + bucket 可视化（对 §1/§3）。
- **Fig.3** 主结果：fixed-K 下 4 方法 × 2 N 的最终成功率与学习曲线（95% CI，8 seeds）；内嵌 Δ(LGA−uniform)（对 **RQ1/H1a–c**）。
- **Fig.4** 校准：decision-weighted reliability diagram（4 方法）、配对 CRN 的 ΔECE 分布、head vs EMA（对 **RQ2/H2/H2d**）。
- **Fig.5** agent：剂量-反应（0/2/5/10 pp 注入 → 组合成功率）、成功率 vs 预测 p̂ 散点、2×2 干预（对 **RQ3/H3**）。
- **Fig.6** 机制：EMA age–|误差| 回归、分配份额–|误差| 散点、天然 σ_a 与注入并列（对 **H2c/H9**）。
- **Table 1** 协议与基线；**Table 2** 主结果（成功率、Δ、CI、p、效应量）；**Table 3** 校准指标（ECE/Brier/覆盖率）；**Table 4** 消融 A1–A10（附录扩展到全表）。

**90s 视频脚本**：0–10s 问题（灵巧操作长程失败多因"技能成功率不可信"）；10–25s 4 技能与 2 长程任务实景（仿真渲染）；25–45s 分配动态（LGA vs uniform 的份额/校准变化）；45–65s agent 调用技能（成功/重试/回退）与剂量-反应曲线；65–90s 结论与局限（sim-only、4 技能、开源）。

**开源计划（W30 前）**：`dexs-bench/`（env 包装、4 技能配置、长程任务、CRN 状态集 SHA256）、`dexs-train/`（PPO+LGA、config、seed 脚本）、`dexs-calib/`（head 拟合、ECE/Brier、配对 CRN 脚本）、`dexs-agent/`（API schema、planner、prompt、episode 日志）、`repro/`（一键表格复现、3 个 seed 的 checkpoint）、`PREREGISTRATION.md`、`budget_log.md`。许可证：代码 MIT；资产遵守 MuJoCo Playground/LEAP 原许可；不发布 LLM 厂商的原始响应（只发可复算的解析结果）。**不做**：不发布真机数据（无）；不承诺长期维护（写"best-effort"）。

---

### 10.2 我们主动写进 limitation 的四件事

①**规模**：4 技能、N≤8192、B≤500M，比 P2P（24,576 envs、7 天 play）小两个数量级；我们不比吞吐，只比 fixed-K 下的样本效率与校准。②**仿真**：无真机，所有"成功"都是仿真定义（容差 3 mm、2 转等），迁移性未测；引 P2P/OmniReset 的真机数字只作外部锚。③**LLM 层**：planner 是 ReAct 模板 + 校准 API，不是新的规划算法；我们测的是"校准置信度是否被用上"，不是"LLM 更聪明"。④**理论**：Prop 3 是方向界（24ε 在 ε=1pp 时远大于实际），不能用来预测幅度；Prop 2 的 26–40% 是阈值依据而非保证。把这四点写在正文而不是藏进附录，是 CoRL 审稿人最看重的"自知"。

### 10.3 图表制作与"审稿人 10 分钟"检查

每张图在 W26 前做三项检查：①**坐标轴与 CI 定义**写清（95% CI 是 seed 级 bootstrap 还是混合模型预测区间，图注必须一致）；②**每图只回答一个问题**（Fig.3 只回答"分配是否改变最终性能"，Fig.4 只回答"校准是否更好"，Fig.5 只回答"校准是否被 agent 用上"）；③**无图表的数字必须能由附录 C 脚本或开源日志复算**（论文中出现的每个 ¥、每个 GPU-h 都要能在 `budget_log.md` 找到对应行）。此外做一次"审稿人 10 分钟"演练：把 Fig.1–5 与 Table 1–3 单独打印，不看正文，判断能否在 10 分钟内复述三条贡献；若不能，说明图表没有承担叙事，需要在 W27 前重排（例如把 Fig.4 的 reliability diagram 移到首行）。

## 11. 自评中稿概率（judge 三值 + 修复后估计）

| 口径 | CoRL 2027 | ICLR 2028 | 级联（任一） |
|---|---|---|---|
| judge 终审 v3.1 的**期望**（含执行折价 0.65–0.70） | **35%**（80% CI 26–44%） | **31%** | **42%** |
| judge 终审 v3.1 的**质量**（条件于完美执行） | **47%**（80% CI 43–50%） | **42%** | **59%** |
| **v3.2 补 G1–G3 后估计** | **≈38%**（80% CI 28–46；质量 ≈47%） | **≈33%**（质量 ≈42%） | **≈45%**（质量 ≈60%） |

**判词与缺口**：judge 终审的结论是"这是一篇 **35–38%** 的 CoRL 论文"，并指出三个必须补的洞——G1（H2 的 −30% 阈值不可达，v3.1 的 26–40% 是全-M 异质上界）、G2（G6 CRN 漏方法因子、LLM 少算，基准应 4,809 而非 4,528）、G3（W4 排期过载）——本 v3.2 已全部落地（见文首改动清单 ⑧ 与 §6.4/§7/§2.3）。补后 judge 估计 **CoRL ≈38%、级联 ≈45%**；**单会算术仍 <50%**，>50% 只在"6 技能 + 真机代理 + 当期无强竞争"三者同时成立时才可能，本文不承诺。

**为什么单会到不了 >50%**（诚实陈述）：①理论只占 1 页、且界是方向性的（24ε），审稿人可合理认为"理论贡献小"；②技能数 4（6 为加分），规模比 P2P 小两个数量级；③纯仿真、无真机；④LLM 层是"用校准置信度调技能"，增量而非突破。**>50% 需要的条件**（记录、不承诺）：H3 剂量-反应斜率显著且强（例如 10pp 注入使成功率降 >20pp）、head 比 EMA 校准误差降 >15%、S7 在 2 mm 可行（3 mm 为主表、5 mm 为降级）、6 技能全部跑完、CoRL 当期无同向强竞争论文。

**评审维度对照**（CoRL 六维）：

| 维度 | 我们的证据 | 补强动作 |
|---|---|---|
| Novelty | 技能库级分配 + 校准 API + 组合评测的**受控链条** | §9 逐行差异表 + "不声称"清单；W30 近邻检索 |
| Technical | LGA 两级 + guards + β/isotonic head + fixed-K 协议 | 消融 A1–A10；附录给实现细节与 100–200 行钩子路径 |
| Experimental | 8 seeds、配对 CRN、MDE、剂量-反应、H2c 置换 | 主表 + 图 3–6；负结果预注册 |
| Clarity | 8 页结构、每图对 RQ、一句话 | W27 前写"评审 FAQ"10 条 |
| Robotics relevance | 4 个真实灵巧任务 + 2 长程装配 + SO-101 无关（纯仿真） | 引言接 P2P/OmniReset 真机数字做外部锚；limitation 单列 |
| Reproducibility | 开源 + 预注册 + 预算日志 + CRN 状态集 | Artifact 附录；一键复现表 2/3 |

**Rebuttal 素材（预写 6 条）**：①"和 SGS 的区别？"→ 技能库级 + 校准 + 组合，且判据不同（三档单调近似而非特例）；②"纯仿真意义？"→ 受控因果链 + 开源基准，不声称迁移；③"理论松？"→ 明写方向/斜率，给 TP 表与失败含义；④"技能太少？"→ 4 技能是受控取舍，W12 gate 扩 6，扩展清单开源；⑤"LLM 层增量？"→ 测的是**校准是否被用上**（噪声注入/重试/剂量），不是 planner 本身；⑥"N=8192 vs 2^20？"→ fixed-K 与 fixed-B 双协议，规模差异写进 limitation，不比吞吐。

**会议日历**：CoRL 2027 摘要/全文截止（**待核实**，预计 2027-05/06）；ICLR 2028 截止（**待核实**，预计 2027-09 下旬）；NeurIPS 2027 D&B（机会项，W20 决策，不投 main）。

---

### 11.2 会期选择与级联逻辑

CoRL 2027 是首选，因为它的评审口味与本文最匹配：机器人学问题 + 受控实验 + 开源基准，且 8 页篇幅容得下方法/理论/实验三段；ICLR 2028 是兜底，好处是 ML 评审更接受"校准/分配"的抽象表述、坏处是对"机器人相关性"不敏感（我们的 4 技能 + 2 长程任务反而是加分）。级联不是"两边都投"（双投违规），而是**同一份主表、两套 framing**：CoRL 版强调"分配 → 校准 → 组合"的机器人链条；ICLR 版把 §3 的两级分配与 §4 的传播界提到前面，把任务细节压到附录。W24 不按"强/弱"一刀切，而按 **D10 六档规则**切换叙事：(a) H1a≥6pp→CoRL 完整论文；(b) 3–6pp→CoRL 但 H1 降 secondary、H2/H3 为主，allocation 表述降级；(c) <3pp 但 H2 配对 ΔECE≥10% 且显著→CoRL 改投"校准/benchmark"版（贡献②③，retitle）；(d) <3pp 且 H2 也不达标→转 ICLR 理论版；(e) agent 未跑通→收为分配→校准、agent 降验证；(f) 技能数<4→hand-only 版，用 W32–W45 扩 6 技能与负结果。这是把 45 周排成"31+14"的原因，也是唯一不双投的降级路径。

## 12. Run 矩阵、命令与产物

**租卡（83 runs + 1 评测批 / 493.3 GPU-h / ¥3,453）**：

| 组 | 方法 | 技能 | N | seeds | runs | B | h/run | GPU-h | 周次 | 优先级 |
|---|---|---|---|---|---|---|---|---|---|---|
| G1a | uniform/SER/SGS/LGA | 4 | 8,192 | 8 | 32 | 500.1M | 9.51 | 304.3 | W17–W20 | P0（H1a 主） |
| G1b | uniform/SER/SGS/LGA | 4 | 2,048 | 8 | 32 | 125.0M | 3.14 | 100.5 | W9–W16 | P0 |
| G2 | 9 变体消融 | 4 | 2,048 | 2 | 18 | 125.0M | 3.14 | 56.5 | W17–W20 | P1 |
| G6 | CRN 评测（非训练；4 方法×16 cell×1,000×8 seeds=512k rollouts） | — | — | 8 | 1 批 | — | 32.0 | 32.0 | W13–W16 | P0 |

**本地（31 runs / 132 GPU-h / ¥0）**：L1 Aloha peg sanity 3 runs；L2 协议 B 子集 13 runs；L3 pilot 10 runs；L4 校准 smoke 5 runs。

**命令样例**（实现见 §3.9 的钩子；`dexs` 为仓库包名）：

```bash
# 主格：技能库级训练（fixed-K，K=1221）
python -m dexs.train --skills S1 S2 S7 S8 --method lga --N 8192 --K 1221 \
  --seed 0 --gpu-h-cap 9.51 --out runs/G1a/lga/N8192/s0

# 校准 + CRN（所有方法/种子共用同一状态集）
python -m dexs.calib --ckpt runs/G1a/lga/N8192/s0/ckpt/last.pt \
  --crn eval/crn_states_16x1000.pt --cells 16 --rollouts 1000 --split heldout \
  --head beta --out runs/G1a/lga/N8192/s0/calib

# agent 剂量-反应（注入 0/2/5/10 pp；硬顶 ¥700）
python -m dexs.agent --planner P1 --skills-api runs/G1a/lga/N8192/s0/calib/api.json \
  --tasks L1 L2 --episodes 120 --noise-pp 5 --budget-cap 700
```

**目录与日志组织**：`runs/<group>/<method>/<N>/<seed>/` 下 `config.yaml`、`metrics.jsonl`（每 1k 步：episode、success、p̂、份额、σ̂、age）、`ckpt/`、`calib/{reliability.csv, ece.json, api.json}`、`logs/`；全局 `runs/index.csv` 汇总 `group,method,skill,N,seed,B,gpu_h,cost,status`；每个组结束写 `runs/<group>/REPORT.md`。所有随机种子、CUDA/驱动版本、提交哈希入库。

**产物清单**：预注册（W6）、冻结协议（W16）、主表数据（W24）、图表（W26）、checkpoints ×3 seeds、CRN 状态集、LLM episode 日志（脱敏）、video、artifact 归档。

---

## 13. 关键决策点（D1–D12，需用户/Lead 拍板或已定）

| # | 决策 | 现状 | 触发/备选 |
|---|---|---|---|
| D1 | 技能数：4（核心）vs 6（目标） | **已定 4 + W12 gate 追加 S3/S4** | gate：W12 前 4 技能主曲线完成且剩余 ≥2 周 |
| D2 | S7 容差：1/3/5 mm | **主表 3 mm**；W4 三容差探针，若 3 mm 无学习曲线 → **接受 5 mm + Aloha 代理降级，不砍 S7**；**1 mm 主张删除**，limitation 写明"3 mm 未达成" | 物理 400 Hz 已确认可行；Lead 已拍板 |
| D3 | 主协议：fixed-K K=1221 vs fixed-B 125M | **fixed-K 主、fixed-B 次**；两者都报 | 审稿人问"是否更新饥饿"→附录 K 扫描（可选，§5.2 保留） |
| D4 | CRN 规模：16 cell×1,000（4 技能）vs 24 cell（6 技能） | **16 cell（4 方法×16×8×1,000=512k rollouts，32h/¥224）**；6 技能时扩 24 cell（768k/48h/¥336），从缓冲出 | W12 gate 后重算 |
| D5 | 预测器：EMA vs learned head | 两者都做，**主报告 head，EMA 为 baseline**；H2d 比较 | 若 head 无改善→退回 EMA 并降级 H2 机制主张 |
| D6 | H3 剂量：0/2/5/10 pp + 240 episodes | **已定**（主图）；二元 12pp 移附录 | 预算紧→砍 10pp 档 |
| D7 | Planner：P0/P1/P2/P3/P4 + O1/S | **已定**；主对比 P1 vs P3 vs P4 | 预算紧→砍 P2 |
| D8 | LLM：模型与温度 | 默认 GPT-4o-mini/等价开源，temp=0.2，3 次取多数 | 硬顶 ¥700（预警 ¥600）；超支砍锚点 |
| D9 | 预算触发线 | **按现写死规则执行（不改）**：单价 ≥¥8.4/h 或时长 +50% → C1–C4（C3 闭合）；双因素 → C1–C5（C5=FLOOR） | 不可破 5,000（Lead 已拍板） |
| D10 | 主投会与 W24 叙事切换 | **六档规则（Lead 已拍板，见 §8）**：(a) H1a≥6pp→CoRL 完整论文；(b) 3–6pp→CoRL，H1 降 secondary、H2/H3 主；(c) <3pp 但 H2 配对 ΔECE≥10% 且显著→CoRL 改投"校准/benchmark"版（retitle）；(d) <3pp 且 H2 不达标→转 ICLR 理论版；(e) agent 2×2 未跑通（原 D3）→主贡献收为分配→校准、agent 降验证；(f) 技能数<4（Plan C）→hand-only 版。优先级：H1∈[3,6) 且 H2≥10%→优先 (b)；(e)/(c) 同满足先 (e)。W24 分析冻结后 3 日内执行并写 decision_log.md | 不直接一刀切转 ICLR；D3 合并到此处 |
| D11 | 开源时点 | W30 arXiv + 代码；评审期匿名仓库 | 不早于 W24（避免半成品） |
| D12 | 未核实项 | CoRL/ICLR 日期、S7 3 mm 可行性、SGS 代码发布、ADEPT 录用状态 | W1/W4/W20 三次核实并更新 |

---

## 14. 预答辩：12 个最可能的审稿问题（附回应素材）

1. **"这和 SGS 有什么本质区别？"** 三处：分配单元（技能库 vs env 成功频段）、输出（校准区间 vs 采样权重）、判据（σ_obs/σ_a 边界 vs 无判据）。且我们明确"SGS 是我们在三条件下的单调近似，不是特例"。
2. **"4 个技能也叫技能库？"** 承认规模小；4 技能是"受控"的取舍，W12 gate 可扩 6；扩展清单与代码开源，审稿人可自行加技能。
3. **"为什么不做真机？"** 课题约束（1 人 45 周、¥5,000、纯仿真），且本文的因果链（分配→校准→组合）在真机上无法在预算内做受控干预；limitation 明示。
4. **"理论界太松，24ε 无用。"** 同意；所以只作方向/斜率预测，并给 TP1–TP6 的失败含义；真正可测的是 H2c 置换与 H3 剂量斜率。
5. **"Neyman 最优与你的在线算法无关。"** 对；Prop 2 只用来给 H2 的阈值提供上界依据，并给引理 4a 的"同向单调"；我们不声称 LGA 达到最优。
6. **"校准评测 1,000 个状态够吗？"** 配对 CRN 下方法间差异的方差远小于绝对 ECE 的方差；另外给绝对 ECE 的 CI（bootstrap over seeds），并报告 250/cell 的 old protocol 作为对照。
7. **"奖励/成功定义是否偏向 LGA？"** 所有方法共用同一成功判定与 reset 调度；LGA 只改分配，不改 reward；代码与日志开源可查。
8. **"SGS/SER 复现忠实吗？"** 预注册 sanity（SGS 至少 1 个设置 >uniform ≥3pp），附录给改写细节与复现代码；不忠实之处显式标注。
9. **"LLM 层是增量，为什么算贡献？"** 我们不声称 planner 新；贡献是"校准是否被用上"的测量（2×2 + 剂量 + 重试），以及"把技能成功率的不确定性暴露给 planner"的 API 设计。
10. **"为什么用 fixed-K？是不是故意让 baseline 少更新？"** 相反：fixed-K 保证所有方法更新次数相同；fixed-B 也报（N=8192 时 K=305 的版本），并说明 update-hunger 的文献依据。
11. **"500M 步训练是否足够收敛？"** 不声称收敛；所有结论是"同预算下的相对比较"，且附学习曲线（Fig.3）与 K 的影响（附录）。
12. **"开源会不会泄露答案？"** 代码/配置/预注册/预算日志全开，结果表用脚本一键复现；checkpoint 只发 3 个 seed，其余按需。

**最小充分证据集（若时间被压缩，只保这些）**：主格 4 方法 × 8 seeds × 2 N（fixed-K）→ H1；CRN 校准（16 cell×1,000）→ H2；剂量-反应 4 档 → H3；消融 A1/A2/A5 → 机制。其余（6 技能、G3/G4/G5、第 3 planner、Aero）都是加分项，可整段删除而不影响三条贡献。

---

**可复现性检查清单（W30 前逐项打勾）**：①预注册文件与 W16 冻结版本的 diff 为空；②`repro/` 一键脚本在干净环境（无外网、固定依赖锁文件）跑出 Table 2/3 的 3 个 seed 版本；③CRN 状态集的 SHA256 与论文附录一致；④所有图的原始数据（csv/parquet）与绘图脚本入库；⑤LLM episode 日志脱敏后含 prompt_hash 与 token 计数，可复算成本；⑥预算日志含每次审计的时间戳与触发条件；⑦许可证文件覆盖所有第三方资产；⑧artifact 的 README 能在 30 分钟内让新用户跑通 smoke test（本地 1 个 run）。任何一项未过，不提交（宁可延后到 ICLR）。

## 附录 A：参考文献（全部用 arXiv API 核实标题+ID，2026-11-02）

| # | ID/来源 | 作者（核实） | 标题 | 用途 |
|---|---|---|---|---|
| 1 | 2606.26428 | Lum, Kedia, Liu, Bohg | *Play2Perfect: …*（大规模灵巧操作 sim play+finetune） | 最近邻系统；外部锚 |
| 2 | 2603.15789 | Yin, Westenbroek, Zhang, Tran, Dagnino, Shilamkar | *OmniReset: …*（reset 多样性、单奖励） | 外部对照（G4，ICLR） |
| 3 | 2610.12465 | Zhang, Guaman Castro, Yin, Dagnino, Gupta, Scalise, Boots | *A Balanced Data Diet: …*（success-band 采样、2^20 envs） | 最近邻方法；SGS-band 基线 |
| 4 | 2511.21011 | Bharthulwar, Tao, Su | *Staggered Environment Resets Improve Massively Parallel On-Policy RL* | 基线（reset 层） |
| 5 | 2608.19182 | Lee, Yin, Rana, Blauch, Zhang, Mady, Bhardwaj, Fazeli | *ADEPT: …*（大规模 RL 预训练/后训练、高 DoF） | 最近邻系统（录用状态待确认） |
| 6 | 2503.12533 | Yuan, Bai, Fu, Zhou, Feng, Xu | *Being-0: A Humanoid Robotic Agent with Vision-Language Models and Modular Skills* | 技能库+agent 的近邻 |
| 7 | 2502.08844 | Zakka, Tabanpour, Liao, Haiderbhai, Holt, Luo, et al. | *MuJoCo Playground*（MJX/Warp，RSS 2025） | 平台 |
| 8 | 2410.00425 | Tao, et al. | *ManiSkill3: GPU Parallelized Robotics Simulation and Rendering* | 对照平台（MuJoCo 专注） |
| 9 | 2511.04831 | Mittal, Roth, Tigue, Richard, et al. | *Isaac Lab: A GPU-Accelerated Simulation Framework…* | 对照平台（无预算） |
| 10 | 2010.03934 | Jiang, Grefenstette, Rocktäschel | *Prioritized Level Replay*（PLR） | 基线思想（单任务） |
| 11 | 2110.02439 | Jiang, Dennis, Parker-Holder, Foerster, Grefenstette, Rocktäschel | *Replay-Guided Adversarial Environment Design*（RED） | 基线思想（单任务） |
| 12 | 2407.20230 | Singla, Agarwal, Pathak | *SAPG: Split and Aggregate Policy Gradients* | 相关（P2P 的 RL 机制） |
| 13 | 2204.01691 | Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao | *Do As I Can, Not As I Say*（SayCan） | agent 规划近邻 |
| 14 | 2210.03629 | Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao | *ReAct* | planner P1 模板 |
| 15 | 2209.07753 | Liang, et al. | *Code as Policies* | 技能/代码 API 近邻 |
| 16 | 2402.16117 | Mu, Chen, Zhang, et al. | *RoboCodeX: Multimodal Code Generation for Robotic Behavior Synthesis* | 技能 API 近邻 |
| 17 | 2411.19309 | —（仅标题/ID 核实） | *GRAPE: Generalizing Robot Policy via Preference Alignment* | 相关（对齐，非必需） |
| 18 | 2509.07445 | Field, Yang, Lin, Psomopoulou, Barton, Lepora | *Text2Touch: Tactile In-hand Manipulation via LLM* | 相关（触觉；limitation） |
| 19 | 2605.05241 | Zeng, Ding, Yang, Li, Liao | *DexSim2Real: …* | sim2real 差距证据 |
| 20 | 2606.22471 | Li, Jin, Liu, Peters, Chalvatzaki | *Language-Conditioned Dual-Arm Dexterous RL Data Generation* | 相关（数据生成） |
| 21 | 2602.19260 | Duggan, Lorang, Lu, Scheutz | *The Price Is Not Right: …*（sim-only 基准先例） | 方法论先例 |
| 22 | 1706.04599 | Guo, Pleiss, Sun, Weinberger | *On Calibration of Modern Neural Networks* | 校准（ECE） |
| 23 | 2107.07511 | Angelopoulos, Bates | *A Gentle Introduction to Conformal Prediction…* | 覆盖率/区间 |
| 24 | 书/经典 | Cochran (1977)；Auer et al. (2002)；Freund & Schapire (1997) | 抽样调查；UCB/EXP3；Hedge | Neyman/bandit 理论 |
| — | — | — | RPLR（"Robust Prioritized Level Replay"）**无已核实 arXiv 条目** | 不引用 |

## 附录 B：Brief §5 十四项清单对照

| # | 要求 | v3 位置 |
|---|---|---|
| 1 | 英文题/摘/关键词 + 中文 | §1 |
| 2 | RQ1–4 + 可证伪假设 | §2（H1a–H9） |
| 3 | 方法形式化、伪代码、复杂度、钩子 | §3（3.1–3.9；3.10 工程量） |
| 4 | 理论 toy 分析、预测、假设与局限 | §4（Prop 1–3、引理 4a/4b、TP1–TP6、4.7 边界） |
| 5 | 实验矩阵（三协议/fixed-K/fixed-B/基线/消融/统计/预注册） | §5 |
| 6 | 真机 | **不适用**（纯仿真课题；§6/§10 limitation 说明；无真机预算） |
| 7 | 预算表、±50%、砍单顺序 | §6（6.1–6.7、附录 C） |
| 8 | 45 周时间线 + Go/No-Go | §7、§5.11 |
| 9 | 风险登记表 + Plan B | §8（R1–R17、Plan C） |
| 10 | 近邻差异表 | §9（15 行） |
| 11 | 论文结构/图表/视频/开源 | §10 |
| 12 | 中稿率提升策略 | §11、§5.10 |
| 13 | run 级实验清单（含 ¥/优先级） | §12、§6.2 |
| 14 | 开放问题与需确认决策 | §13（D1–D12） |

## 附录 C：预算脚本（实跑输出，v3.2）

脚本 `budget_v3.py`（与 §6 逐行一致；`python3 budget_v3.py` 输出）：

```
=== rent rows (v3.2) ===
G1a N=8192 (4 methods x 8 seeds)                     runs= 32 h/run= 9.51 GPU-h=  304.3 cost=Y   2130
G1b N=2048 (4 methods x 8 seeds)                     runs= 32 h/run= 3.14 GPU-h=  100.5 cost=Y    703
G2  ablations (9 variants x 2 seeds)                 runs= 18 h/run= 3.14 GPU-h=   56.5 cost=Y    396
G6  CRN batch (4 methods x 16 cells x 8 seeds x 1000) runs=  1 h/run=32.00 GPU-h=   32.0 cost=Y    224
subtotal: GPU-h=493.3 cost=Y3453
+12%: GPU-h=552.5 cost=Y3868
misc: electricity 220 + storage 80 + LLM 641 = Y941
BASE TOTAL = Y4809  (cap 5000, margin Y191)

=== scenarios ===
price+50% (Y10.5/h)   GPU-h=  552.5 total=Y   6742
duration+50%          GPU-h=  828.8 total=Y   6742
  C1 cut G2                    GPU-h=  489.2 total=Y   6078 OVER
  C2 G1a seeds 8->6            GPU-h=  404.0 total=Y   5183 OVER
  C3 CRN 32->16h               GPU-h=  386.1 total=Y   4995 OK
  C4 G1b seeds 8->6            GPU-h=  358.0 total=Y   4699 OK

dual C1 cut G2                    GPU-h= 1100.7 total=Y   8646 OVER
dual C2 G1a 8->6                  GPU-h=  909.0 total=Y   7304 OVER
dual C3 CRN 32->16h               GPU-h=  868.7 total=Y   7022 OVER
dual C4 G1b 8->6                  GPU-h=  805.4 total=Y   6579 OVER
FLOOR GPU-h=222.9: dual Y4872, price+50% Y3562
FLOOR with 32h CRN: Y5296  (must NOT be used)

=== consistency checks ===
N=8192: B=500.1M  h=9.51
N=2048: B=125.0M  h=3.14
LLM: 840 eps*6.5 calls*2200 tok = 12.0M tok
     input 70%@$3 + output 30%@$12 = $68.5 = Y493; +30% buffer = Y641 -> quote Y641
local GPU-h = 13.3+68.7+30.0+20.0 = 132.0 (31 runs, Y0)
```

**读法**：`BASE TOTAL = ¥4,809`（余 ¥191）为基准；单价 +50% 或时长 +50% 时从 **C1→C4** 逐级削减（C3 即闭合于 ¥4,995）；双因素从 **C1→C5**（C5=FLOOR，¥4,872）。**FLOOR 必须用 4-cell CRN（8h）**；若保 32h CRN 则为 ¥5,296，破 5,000 硬线，故不允许。

## 附录 D：LLM 配置模板

```yaml
provider: openai-compatible
model: gpt-4o-mini            # 或等价开源权重，W4 锁定并记录快照
temperature: 0.2
max_tokens: 600
retry: {max: 2, backoff_s: 2}
budget: {cap_cny: 650, alert_cny: 520, kill_at_cny: 650}
prompts:
  planner_react: prompts/react_skill_api.md      # 技能 API + 校准 p̂/CI
  planner_plan_execute: prompts/plan_execute.md  # 先计划后执行
  planner_ablated: prompts/react_no_p.md         # 隐藏 p̂（2×2 干预）
logging: episodes.jsonl（prompt_hash、tokens、latency、parsed_call、success）
```

## 附录 E：命题证明要点与常数修订

1. **Prop 1**（uniform 每桶噪声）：单桶 n=B/M 次 Bernoulli，Hoeffding 联合界给式 (1)；`E|ε|` 的正态近似给式 (2a)，Lipschitz 严格界给 (2b)；`E[ECE]≤0.5√(M/B)` 由 `E[ECE]≤max_b E|ε_b|` 与 √(M/B) 常数整理（v2 的 0.399 只作渐近值）。Prop 1b 由 EMA 的偏差-方差分解 `MSE=(τṗ)²+p(1−p)/τ` 直接得到，最优 τ* 见式 (5)。
2. **Prop 2**（Neyman）：L2 目标 `Σ w_b v_b/n_b` 的拉格朗日解给式 (6)；L1 目标 `Σ w_b√(v_b/n_b)` 给式 (8)；比值式 (7)/(9) 的最坏情形在 `w_b v_b` 集中时取到 O(M)。例 2（M=8、v 集中）比值 0.74，即相对改善 26%；上界扫描给 26–40%，这就是 H2 用 −30% 的依据（**是阈值依据，不是保证**）。引理 4a 把 LGA 分数 `s_b≈LP_b/σ̂_b∝√(v_b n_b)` 与 α*_b∝(w_b√v_b)^{2/3} 对比：两者对 v_b 单调同向，故 LGA 是"方向正确"的在线近似；引理 4b 的局部灵敏度 `ECE(α)/ECE*≤1+(3/8)max δ_b²` 用于用实测 n_b 预测 ECE，ρ>0.6 时只作弱代理。
3. **Prop 3**（组合损失）：planner 规则 q=1−(1−p)^{R+1}；引理 3.1 给 |q̂−q|≤(R+1)ε；引理 3.2 给乘积扰动界；Prop 3a 得 `S(π*)−S(π̂) ≤ 2H(R+1)ε`（H=4,R=2,ε=1pp → 24pp，**方向/斜率预测**）；v2 漏 (R+1)，v3 修正。Prop 3b 用独立误差与 union bound 得 `O(ε√(H log N_π))`；δ-method 一阶 `(q')²Var`、严格 Lipschitz `≤2(R+1)²σ²`（×√2 不改 O）。
4. **常数修订表**：v2→v3：`2Hε→2H(R+1)ε`；`n≳2H²log(2S/δ)/m² → 2H²(R+1)²log(2S/δ)/m²`（H=4,R=2,S=6,δ=.05,m=.12 → ≈1.1e5/skill，在 B=500M 下不 binding）；`E[ECE]` 严格 0.5√(M/B) vs 渐近 0.399√(M/B)。
5. **边界**：A1 指数学习曲线只是近似；A3 误差界是规划值而非保证；A4 加性预算忽略迭代间依赖；Prop 3 的最坏情形在实际 `m` 下仍可能 vacuous（见 §4.7 表）。

## 附录 F：版本记录

| 版本 | 日期 | 主要变化 |
|---|---|---|
| v1 | 2026-11-02 | 初稿：8 技能（6 手内+2 臂式）、¥4,849、45 周 ICRA 口径 |
| v2 | 2026-11-02 | 6 技能（4 核心+2 stretch）、fixed-K、CRN 思想、Prop 1/3、预算 ¥4.8k（见 v2 文件）、CoRL/ICLR 口径 |
| v3 | 2026-11-02 | 终稿（P1 包）：4 技能核心+W12 gate、Prop 1 加 (R+1)、Neyman 进正文、CRN 1,000/cell、head/age 机制、H3 剂量-反应、砍 stretch/G3/G4/G5、¥4,528（余 472）、judge 三值概率 |
| **v3.1** | 2026-11-02 | 小修：CRN 口径统一 16（4 技能）/24（6 技能）、文首预算改 ¥4,528（余 472）；D10 四级决策（v3.2 扩六档，§8/§13）；S7 接受 5 mm + Aloha 代理降级；A7 附录条件项、A9/A10 按计划；无预算/排期变化 |
| **v3.2** | 2026-11-02 | judge 终审补丁：H2 阈值 −30%→**20% 相对**（26–40% 标为全-M 上界，30% 不可达、上界 29.3%）；CRN 512k/32h/¥224、LLM ¥641 → 基准 **¥4,809（余 191）**、削减链加 C4、FLOOR 用 4-cell CRN；G0 顺延 W5 并统一阈值；审计小修 6 项；概率改 judge 终审 35/31/42→补洞后 ≈38/33/45 |
