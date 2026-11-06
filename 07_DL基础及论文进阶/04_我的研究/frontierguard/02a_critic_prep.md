# 02a · Critic 前置功课（Phase A）

> 独立核查（2026-11-02）：用 arXiv API 逐个核实了 Brief 4.1 表中 15 个 ID，**全部真实存在**，标题与 Brief 描述基本吻合。唯一事实错误：**2110.02439 的标题是 "Replay-Guided Adversarial Environment Design"（RED），不是 RPLR**；RPLR（Robust PLR）的 arXiv ID 我未能核实，v1/v2 中若出现"RPLR (2110.02439)"必须改为"PLR 2010.03934 + RED 2110.02439"，或对 RPLR 标注"待核实"。另：SO-101 Bench 为**单作者**（Truman Hickok）且是**语言条件 VLA 评测基准**（4 任务 5 族 / 138 物体 / 916 评测 episode / 3629 演示 / Spearman ρ=1.00、MMRV=0.000），不是 RL-from-scratch 基准。

## 1. ICRA 评审检查清单（逐维度过线标准）

| 维度 | 审稿人实际看什么 | 过线门槛（本课题） |
|---|---|---|
| Novelty & Contribution | 与 curriculum / UED / task-sampling 的**机制**差异，不是措辞差异 | 必须有一句话能说清"分配对象、信号、约束"三点都与 PLR/RED/DRATS/PGB 不同 |
| Technical Quality | 形式化自洽、伪代码可复现、复杂度与超参数量 | LP/不确定性/guard 三项各有定义式 + 复杂度 + 超参数表 |
| Experimental Evaluation | baseline 强度、预算匹配、seed/CI、跨任务、真机 | ≥6 个强 baseline、≥5 seeds、配对检验、**固定交互与固定 wall-clock 双预算** |
| Robotics Relevance | 是否有真机、是否只借机器人当"环境" | 必须独立真机章节；纯 RL 方法易被判"not robotics" |
| Clarity & Presentation | 6 页塞得下吗、图是否自解释 | 主张 × 证据一对一；每图回答一个 RQ |
| Reproducibility | 代码、配置、日志、算力声明 | 开源 + 每 run 的 N/B/seed/GPU-h 表 |

## 2. 最可能被拒的 10 种理由（含审稿人原话式提问）

1. **增量性**：「PLR 的优先级 + ensemble 不确定性 + bandit 采样，这三样拼起来为什么是 ICRA 论文而不是 workshop？」→ 缺"为什么必须三者同时存在"的消融证据。
2. **归因不干净**：「N 变大时你同时拿到了更大 batch 和更多 GPU，凭什么说提升来自'分配'？」→ 缺 compute-matched 与 batch-matched 双控。
3. **scalability 定义漂移**：「固定 env-step 预算下赢，固定 wall-clock 下还赢吗？你主张哪一个？」→ 两套预算结论若不一致而无解释 = 致命。
4. **baseline 不公**：「DRATS 是 MTRL 专用、UPER 是 off-policy，你硬改成 on-policy env 级，baseline 天生残废。」→ 每个 baseline 必须有改编说明 + 调参预算相同。
5. **统计**：「5 seeds 的 CI 重叠就宣称 SOTA？主分析预注册了吗？多重比较校正了吗？」→ 需声明主终点（primary endpoint）+ Holm 校正 + 配对 bootstrap。
6. **真机太弱**：「2 个任务、20 trials、SO-101 这种 100 美元臂，结论能外推吗？」→ 需 Wilson CI、固定初始条件、双臂并行、评分协议盲评。
7. **sim-to-real 借功劳**：「SO-101 Bench 的排名一致性是原作者的结论，不是你实验的贡献。」→ 只能作为"平台可信度依据"引用，不能当自己的结果。
8. **理论太薄**：「toy MAB 的 regret 界对 PPO + 非平稳 on-policy 数据没有任何预测力。」→ 理论必须给出**可被实验证伪的定量预测**（如增益随 N 的曲线形状）。
9. **超参不可扩展**：「β、EMA 窗口、group 粒度、ensemble 规模每个任务都要重调，这还叫 scalable？」→ 需固定超参跨 N、跨任务的证据（或明确的可迁移规则）。
10. **写作/体量**：「6 页里塞 3 层实验 + 理论 + 真机 + 15 个 baseline，图小到看不清。」→ 主文只留 2 个 benchmark + 真机，其余进 appendix/arXiv 版。
11. （隐性）#**机器人相关性弱**：若真机只占 1 页且任务是 pick-place，容易被判"RL 方法论文投错会"。

## 3. 近邻工作：强项 / 弱项 / 威胁度

| 工作 | 强项 | 弱项（我们的切口） | 威胁 |
|---|---|---|---|
| SER 2511.21011 | 直接研究并行 env 扩展；toy+机器人双验证；即插即用 | 只调 **reset 时序**，不分配经验、无 LP/不确定性 | **高**（同一 scaling 问题的最近邻；必须既是 baseline 又是可组合模块） |
| DRATS 2605.14350 | 自适应任务采样 + minimax 可行性理论 + MTBench 50 任务 | 面向**固定任务集 MTRL**，非 N-scaling；无真机；信号是"离解决最远"而非 LP×不确定性 | **高**（概念最近，差异表要写机制级） |
| PGB 2610.25704 | bandit + 时序预测器 + 梯度聚类，形态最像 | LLM reasoning 域；无并行环境；无机器人 | **中高**（会被引为"bandit 数据分配已有人做"） |
| UPER 2506.09270 | epistemic 不确定性排序 replay | off-policy replay 池，非 on-policy 环境分配 | 中低 |
| GMC 2605.05856 | 参数空间 LP 代理，比 prediction-error 稳 | 只提 LP 代理，不做分配策略 | 中（做 LP 信号消融/鲁棒性对照） |
| LPM 2509.25438 | noisy-TV 抗噪 LP 监控 | 探索范式，非并行分配 | 中（guard 消融必须引） |
| PLR 2010.03934 / RED 2110.02439 | 经典 level 优先级 + 对抗式 UED | 小规模 ProcGen，非并行 scaling、非机器人 | 中（必做 baseline，须正确引） |
| 2506.03404 | 并行 actor 数 × rollout × UTD 的 bias-variance 分析 | 只分析不改算法 | 低（机制解释引用） |
| MTBench 2507.23172 | 50 任务大规模并行 MTRL 基准 | 是 benchmark 论文，无分配方法 | 中（可借协议；也是竞争 venue 论文） |
| 2610.33420 | 可预测的 RL scaling law（Elo 指标、两个数量级外推） | StarCraft/Craftax，非机器人 | 低（方法论引用，可照抄其拟合/外推验证流程） |
| 2603.01741 | 数万 env 下 ensemble PG 的多样性 | 关注策略多样性，非经验分配 | 低中 |
| Squint 2602.21203 | 视觉 SAC 15 min/任务（3090）+ SO-101 Task Set 8 任务 + sim2real 已验证 | **off-policy**，我们是 on-policy；任务为 SAC 调优 | 中（平台依赖；审稿人会问"为何不用 Squint 的 SAC"） |
| SO-101 Bench 2610.36420 | Isaac Lab 数字孪生、真机排名一致、物体/演示规模大 | **VLA/语言条件**基准，演示微调范式，RL 从头训练难；单作者 | 中（Tier-3 只能 pilot 后选 1–2 任务） |
| LeRobot 2602.22818 | 真机软件栈 + HIL-SERL 生态 | 是库不是方法 | 低 |

## 4. 审稿人 attack 面清单（v1 必须逐条有防守）

- 机制证据：per-group 采样比例随训练曲线、LP/不确定性分布、熵（证明"分配真的变了"）。
- 归因：uniform 拿到**相同 batch size / 相同 UTD / 相同 wall-clock** 的三组对照。
- N 与 M 的关系：N > M、N ≫ M、随机映射 vs 固定映射，是否退化。
- 同构任务负对照：环境同分布时方法应无效（报负结果反而加分）。
- 不确定性项单独消融：LP-only vs 不确定性-only vs 二者；ensemble 规模 1/3/5 的成本-收益。
- 超参跨 N 固定：证明不是 per-N 调参。
- 统计：主终点预注册、5 seeds、95% CI（bootstrap/Wilson）、配对检验、Holm 校正、失败 run 不剔除。
- 复现：代码 + 所有 yaml + 每 run 结果表 + 随机种子清单。
- 真机：20–30 trials、Wilson CI、固定初始位姿、评分规则、双臂并行、与仿真同条件。
- 视频：必须展示 N 增大时行为差异，而不只是成功片段。

## 5. 我的预判：最大 3 个风险

1. **归因风险（CRITICAL）**：N 增大天然带来更大 batch/更多样本，"分配"效应极易与"规模"效应混淆；若无 batch-matched 与 compute-matched 对照，RQ3 直接崩。
2. **真机章节撑不起 ICRA（CRITICAL）**：1 人 45 周内既要跑三层仿真矩阵又要 2 台 SO-101 的 20–30 trials×多条件×双臂，人工小时极可能爆炸；真机若只剩"装饰性 2 任务"会被判 robotics relevance 不足。
3. **novelty 被 SER+DRATS 夹击（CRITICAL）**：SER 已占"并行 env 扩展"、DRATS 已占"自适应任务采样"、PGB 已占"bandit 分配"，三者组合的缝隙很窄；必须把卖点从"新采样器"升级为"**随 N 增长的分配理论/可预测 scaling 律**"，否则是 workshop 级贡献。
