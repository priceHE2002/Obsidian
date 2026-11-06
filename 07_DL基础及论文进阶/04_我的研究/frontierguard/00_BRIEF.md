# FrontierGuard-RL 团队 Brief（2026-11-02 版）

> 本文件由 Team Lead 撰写，是 proposal-writer / proposal-critic / acceptance-judge 三个 subagent 的共同上下文。所有产出必须与本 Brief 的硬约束一致。只读本文件的副本可以，不要修改本文件。

## 0. 团队角色与交付物

| 角色 | 文件（绝对路径在 `/Users/heyuhang/Desktop/frontierguard/`） | 说明 |
|---|---|---|
| proposal-writer | `01_proposal_v1.md`；后续 `05_proposal_v3.md` | 主方案撰写与修订 |
| proposal-critic | `02a_critic_prep.md`（前置功课）；`02_critique_v1.md`；`03_proposal_v2.md` | 批改 + 修订版全文 |
| acceptance-judge | `04a_judge_prep.md`（前置功课）；`04_acceptance_judgment_v1.md`；后续最终评估 | 中稿概率 + 优化建议 |
| Lead | `00_BRIEF.md`；`99_FINAL_ICRA_EXPERIMENT_PLAN.md` | 统筹、审计、最终整合 |

共享任务板：创建于本会话，任务 id 由 Lead 在 spawn 消息中给出；执行前先 `team_task_get` 拿 revision，再 `team_task_update action=claim`，完成后 `action=complete`。

## 1. 用户原始课题（原文）

> FrontierGuard-RL
> Scalable Reinforcement Learning via Uncertainty-Aware Learning-Progress Sampling
> 中文：基于不确定性感知学习进度采样的可规模化强化学习
>
> 核心研究领域：Scalable RL · Exploration · Curriculum Learning · Sample Efficiency
>
> 核心科研问题：
> Can uncertainty-aware learning-progress sampling improve the scalability of reinforcement learning by allocating experience more effectively as the number of parallel environments increases?
> 即：当并行训练环境数量持续增加时，不确定性感知的学习进度采样能否通过更合理的经验分配，提高强化学习的可扩展性？
>
> 这里的 Scalability 不应只用训练吞吐量来衡量，还应考察随着环境数量增加，在固定交互预算和固定计算预算下，策略性能是否能够持续改善。

用户要求：以投稿 ICRA 为目标撰写**实验方案**；3 个 subagent 分别负责写方案、批改方案、判断中稿概率并给优化建议；目标是尽可能让中稿率高于 50%。

## 2. 硬约束（不可违反）

1. **租卡预算 ≤ 5000 元人民币（硬上限）**。实验矩阵的 GPU 小时 × 单价必须能算平，并给出敏感性分析（±50%）。
2. 本地已有服务器：**RTX 4070 Ti Super 16GB VRAM / 32GB 内存 / 2TB SSD / Ryzen 7 7800X3D**（免费，但算力有限，Isaac Sim 5.x 在 32GB 内存下不保证流畅）。
3. **两套 LeRobot SO-101 真机**（已有，不占租卡预算；可买少量耗材/相机，建议 ≤500 元）。
4. 人力：按 **1 名研究生** 全职投入估算（所有实验、真机、写作都要一个人能完成）。
5. 语言：方案正文用中文；论文题目、摘要、关键词用英文（附中文对照）。
6. 不允许编造论文、作者、实验结果、录取率数据；引用必须给出真实 arXiv ID 或官方 URL。
7. "预期结果" 与 "已获得结果" 必须严格区分；方案阶段所有效果均为**预期/待验证**。
8. 方案必须可直接执行：给出 benchmark 仓库/环境、实验矩阵、命令级或目录级建议、数据与日志组织、代码结构。

## 3. 时间现实（重要，必须在方案中正面处理）

- **ICRA 2027 投稿已截止**：官网 https://2027.ieee-icra.org/ 的 Call for Papers 页面标注 "Submission Closed"，论文提交约在 2026 年 9 月中旬，伴随视频提交期 2026-09-18～22（官网公告 2026-09-17），会议在首尔。
- 因此实际可投的下一届 ICRA 是 **ICRA 2028**，预计 **2027 年 9 月中旬截稿**（以官网为准；ICRA 2026 与 2027 的截稿均在 9 月中旬）。
- 方案主排期按 **2026-11-02 → 2027-09-15（约 45 周）** 设计，目标 ICRA 2028。
- 可选风险对冲（在方案中作为 optional，不得干扰主排期）：IROS 2027（约 2027-03 截稿）、CoRL 2027（约 2027-06 截稿）。若用户其实想投 ICRA 2027，方案开头需明确说明已不可能并给出替代路径。
- 最终回复中 Lead 会请用户确认目标；方案本身要能在"ICRA 2028 主目标"下自洽。

## 4. 前期调研结果（2026-11-02，供三个角色直接使用；可再补充但不得冲突）

### 4.1 最接近的竞争工作（必须做差异表，必须引用真实 arXiv ID）

| 工作 | arXiv | 核心 | 与本课题的差异/威胁 |
|---|---|---|---|
| **SGS (Success-Guided Sampling)** | **2610.12465** | **S0 级威胁（Lead 已核实：CoRL 2026，UW+NVIDIA；项目页 sgs-rl.github.io；代码"coming soon"未开源）**。核心主张：均匀采样 task configuration 会把越来越多经验浪费在"已掌握/还做不到"的配置上，导致 N 增大看不到 scaling 收益；解法 SGS = 固定 N 个 task configuration、按 success rate 加权在 frontier（不太易不太难）上 reset，标准 PPO 外循环；用到 2^20 并行环境；ANYmal 地形 locomotion + Franka/UR5e 接触装配；蒸馏 RGB 策略零样本真机。**差异机会：SGS 是单任务内 config 级、二值 success 信号、无不确定性、无 MTRL/任务级分配；我们是任务级（M 个任务/组）层级分配 + LP×epistemic uncertainty + 非平稳 guard + 对 N 的可检验 scaling 理论，并把 SGS 作为 config 级 baseline/可组合模块。必须正面引用，禁止回避。** | **S0（最高）** |
| Staggered Environment Resets (SER) | 2511.21011 | 大规模并行 on-policy RL 中同步 reset 造成非平稳；错峰 reset 提升样本效率，且随并行环境数增加扩展更好 | 最直接威胁：也研究"并行环境数扩展"。**但 SER 调的是 reset 时序，不是经验/任务/层级的分配采样**；我们的方法是正交的，且应把 SER 作为 baseline 与可组合模块 |
| DRATS | 2605.14350 | 多任务 RL 的自适应任务采样，按"离解决最远"分配数据；MetaWorld MT10/MT50 | 概念最近的任务采样工作，但：非 learning-progress + epistemic uncertainty 框架；非大规模并行 scaling 研究；无真实机器人。必须对比并说明差异 |
| PGB Curriculum | 2610.25704 | 梯度聚类发现样本组 + 轻量预测器 + Boltzmann 采样，用于 LLM RL 数据分配 | 方法形态接近（bandit 数据分配），但领域、信号（梯度聚类）与目标（LLM reasoning）不同；要引用并说明我们在机器人连续控制/并行环境下的 LP+不确定性设计 |
| Uncertainty Prioritized Experience Replay (UPER) | 2506.09270 | 用 epistemic uncertainty 对 replay transition 排序，抑制噪声样本 | 不确定性用于**回放池优先级**（off-policy），我们用于**并行环境/任务实例的采样分配**（on-policy 大规模）；需说明机制差异 |
| GMC | 2605.05856 | 参数空间学习进度代理（梯度×动量耦合），替代 prediction-error 型 LP | 可作为 LP 信号的对照/消融；我们在并行环境分配层做 LP，可用多个 LP 代理做鲁棒性实验 |
| LPM | 2509.25438 | learning progress monitoring 抗 noisy-TV 探索 | 与我们的 "noise guard" 直接相关：noisy/不可学实例的识别与降权；引用并在 guard 消融中讨论 |
| PLR / RED | 2010.03934 / 2110.02439 | PLR=Prioritized Level Replay；2110.02439 实为 **RED (Replay-Guided Adversarial Environment Design)**，不是 RPLR（Lead 已核实）。经典 level replay / UED。 | 经典 curriculum/level sampling，但面向小规模单机、非并行环境 scaling、非机器人；作为 baseline/思想来源。**RPLR 的 arXiv ID 未核实，引用时须标注"ICLR 2022 无 arXiv 或待核实"** |
| On-policy parallel data collection | 2506.03404 | 并行 actor 数 × rollout 长度的 bias-variance/UTD 权衡；更大数据量可提升最终性能 | 提供 scaling 机制解释与超参依据（UTD、rollout 长度、N 的交互） |
| Massively parallel MTRL benchmark | 2507.23172 | 大规模并行多任务 RL 的 benchmark 与 on-policy 方法 | 可借鉴实验协议/任务选择；证明"大规模并行 MTRL"是活跃方向 |
| Empirical RL scaling laws | 2610.33420 | actor-critic 在 Starpilot/Craftax 的经验 scaling law | 我们可在采样策略维度拟合 scaling law（success(N) 曲线），引用其方法论 |
| Ensemble policy gradient at scale | 2603.01741 | 数万并行环境下 ensemble 策略梯度与多样性约束（SAPG/PBT/PPO） | 说明"大规模并行下的探索/多样性"是热点；我们是数据分配视角，需对比或引用 |
| Squint | 2602.21203 | 视觉 SAC + 并行仿真 + 分布 critic + 分辨率 squinting；SO-101 Task Set（ManiSkill3，8 任务，重域随机化），15 分钟/任务 RTX 3090；真机 sim-to-real | **首选实验平台之一**：SO-101 真机迁移已验证；我们必须说明用 on-policy PPO/本文方法（Squint 是 off-policy 参照） |
| SO-101 Bench | 2610.36420 | Isaac Lab SO-101 数字孪生；4 任务/5 族、138 物体、3629 演示；sim 与真机策略排名一致（Spearman 1.00） | **首选实验平台之二**：真机对齐好；但其任务偏 VLA/语言条件，RL 从头训练难度较高，需 pilot 筛选 |
| LeRobot | 2602.22818 | 开源端到端机器人学习库（含 RL 模块 `src/lerobot/rl`、HIL-SERL 社区实现） | 真机软件栈基础；可用 HIL-SERL 做真机样本高效微调（stretch） |

### 4.2 可用平台/仓库（已验证存在，2026-11-02）

- **Squint**：https://github.com/aalmuzairee/squint（103★，ManiSkill3 + SO-101 Task Set + sim-to-real）。本地 4070 Ti Super 可跑；适合主 scaling 实验与真机迁移。
- **SO-101 Bench**：https://github.com/5hadytru/so101_bench（Isaac Lab 数字孪生）；Docker 一键版 https://github.com/kabilankb/so101_bench_quickstart 。Isaac Sim 5.1 + Isaac Lab 2.3，建议租 A100/4090 跑；本地 32GB 内存可能不够。
- **Isaac Lab SO-101 外部工程**：https://github.com/MuammerBay/isaac_so_arm101（318★）；NVIDIA 官方 workshop：https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop 。注意官方测试卡为 RTX 6000/5090，4070 Ti Super 16GB 属"可试但无保证"。
- **LeRobot**：`src/lerobot/rl` 存在（RL 训练模块）；HIL-SERL 指南：https://github.com/michel-aractingi/lerobot-hilserl-guide 。真机：2×SO-101 + 相机 + 遥操作。
- **MuJoCo Playground / Brax**：本地 4070 Ti Super 可高吞吐跑，适合做 N 很大的 scaling 曲线与 toy/理论验证（低成本）。
- 备选：ManiSkill3 自带任务（非 SO-101）可做方法 breadth；Isaac Lab 官方任务（如 lift/stack）作为 reviewer 熟悉的 sanity check。

### 4.3 ICRA 评审现实（供 critic/judge 使用）

- ICRA 近几届录取率大致在 **35%–42%** 区间（请以官方统计为准；方案中若引用需注明"约"与来源年份）。机器人真机实验、系统贡献、可复现性权重高。
- 纯 RL/ML 方法论文投 ICRA 的常见拒稿理由：机器人相关性弱、只有仿真、baseline 不公、统计不严谨、novelty 与已有 curriculum/sampling 工作重叠、写作/图表质量差。
- ICRA 论文一般 6 页正文 + 引用（以当年 author kit 为准），**视频**对录取影响明显；代码/数据开源加分。
- 评审维度（供 rubric）：Novelty & Contributions、Technical Quality、Experimental Evaluation、Clarity & Presentation、Robotics Relevance & Impact、Reproducibility。

## 5. 方案必须覆盖的清单（critic 用它逐项检查，judge 用它打分）

1. 英文题目/摘要/关键词 + 中文题目/摘要；一句话卖点（one-sentence pitch）。
2. 研究问题 RQ1–RQ4 与**可证伪假设**：
   - RQ1：固定交互预算下，均匀/随机经验分配的 sample efficiency 是否随 N 增大而饱和/退化？（诊断实验 + 机制指标）
   - RQ2：FrontierGuard 能否让 scaling 曲线在更大 N 上继续改善？增益随 N 如何变化？
   - RQ3：增益是否来自"分配"而非"更多探索/更大 batch"？需要哪些控制和消融？
   - RQ4：在 SO-101 真机上，sim 训练的采样优势是否迁移（零样本 + 少量真机数据）？
3. 方法 **FrontierGuard**：形式化定义
   - 实例/层级/任务的划分（M 个 group；N 个并行 env 如何映射到 group）；
   - learning progress（LP）估计（success/return 的 EMA 差分；窗口；归一化）；
   - epistemic uncertainty（ensemble/双头/贝叶斯最后层等轻量方案）与 aleatoric/noise 的区分；
   - 优先级分数、bandit/EXP3/UCB 分配、采样分布约束（熵/KL 下界）、"frontier"与"guard"机制（noisy-TV guard、collapse guard、估计置信度 guard）；
   - on-policy PPO 集成（数据管线、UTD、rollout 长度、reset 协调；与 SER 的组合）；
   - 伪代码 + 复杂度 + 实现要点（约 100–200 行 Python 钩子）。
4. 理论/分析：toy MAB 或简化模型分析，给出 LP 估计噪声随 N 的缩放、均匀分配与自适应分配的 regret/样本复杂度对比、可检验的 scaling 预测；明确假设与局限。
5. 实验矩阵：
   - Tier-1 低成本高吞吐（MuJoCo Playground/Brax，本地）：N ∈ {64, 256, 1024, 4096, 16384}，验证 scaling 与理论预测；
   - Tier-2 机器人视觉（Squint/SO-101 Task Set，ManiSkill3）：N ∈ {64, 256, 1024, 4096}，4 个任务；
   - Tier-3 真机对齐（SO-101 Bench / Isaac Lab）：2–4 任务，N ∈ {256, 1024, 4096}；
   - 每个 run 固定 env-step 预算 B（按 pilot 设定），报告不同 N 下的学习曲线；另做固定 wall-clock/GPU-hour 预算对比；
   - baselines ≥6：PPO-uniform、PLR/RPLR、DRATS、SER、SER+ours、UPER（改造成 env 级）、max-return/难度采样、random；可选 ensemble/RND 探索基线；
   - ablations ≥5：去掉 uncertainty、去掉 LP、去掉 guard、改变 group 粒度、改变 N/M、EMA 窗口、β、熵约束；
   - 统计：≥5 seeds（Tier-1/2 核心设置），95% CI（bootstrap/Wilson），配对检验；预注册主分析；计算量匹配。
6. 真实机器人实验：2×SO-101；任务选择（从 Squint/SO-101 Bench 中 pilot 后定 2–3 个）；训练→sim-to-real 零样本评估；最少 20–30 trials/条件，Wilson CI；固定初始条件与评分协议；可选 HIL-SERL 真机微调（stretch，标注降级方案）；两臂并行评测以提高效率。
7. 计算预算表：RMB 明细（≤5000）、GPU 型号与单价假设（如 4090 24GB ≈2–2.5 元/h；A100 80GB ≈6–8 元/h；H100 ≈12–18 元/h；以实际平台为准）、GPU 小时、存储、缓冲；给出"如果超支 30%/50%"的砍实验顺序。
8. 时间线（45 周）与里程碑/Go-No-Go 门（按周或双周粒度），与预算消耗对齐。
9. 风险登记表与 Plan B：方法无效、真机 gap、预算超支、benchmark 跑不起来（Isaac Lab 本地不兼容）、审稿人认为 novelty 不足、人力/时间冲突。
10. 与最近工作差异表（至少覆盖 4.1 表中的 SER、DRATS、PGB、UPER、PLR/RPLR、GMC、LPM）。
11. 论文结构、图表清单（每张图/表要说明放什么数据、回答哪个 RQ）、视频脚本、开源计划。
12. 中稿率提升策略：针对 ICRA 各评审维度的具体动作；"最小充分证据集"与"加分项"。
13. 实验运行清单（每个 run：benchmark/task/N/method/seed/steps/GPU/h/¥/优先级/负责人）。
14. 开放问题与需要用户确认的决策点（如：目标会议确认、真机任务偏好、是否投 CoRL/IROS 对冲）。

## 6. 质量与事实要求

- 严禁编造：论文、作者、实验数字、录取率、硬件价格都要有依据或明确标注"假设/待核实"。
- 预算算术自洽：实验矩阵 → 每 run GPU 小时 → 总 GPU 小时 → 总费用 ≤5000；给敏感性分析。
- 吞吐/收敛时间必须给假设（如 ManiSkill3 视觉 PPO 在 4090 上约 X 千 steps/s），并说明如何用 pilot 校准；不要说"取决于"就结束。
- 可行性优先：1 个人 45 周能完成；本地机器角色 vs 租卡角色分工明确；真机实验时间要算人工小时。
- 区分 hypothesis / expected result / actual result；方案阶段只能是前两者。
- 写作：中文、结构化、表格化；不写空话；每个实验都要回答一个 RQ。
- 不得修改他人文件；每个角色只写自己的文件。

## 7. 提交与沟通

- 完成每个阶段后 `send_message` 给 `lead`，格式：状态 / 文件路径 / ≤500 字要点 / 待决问题。
- 发现问题（如预算不可行、benchmark 不可用）立即上报，不要自行降低硬约束。
- 所有文件写在 `/Users/heyuhang/Desktop/frontierguard/` 下，UTF-8。

## 8. 2026-11-02 Lead 补充核查（新增，优先于上文冲突处；所有角色必须读）

### 8.1 SGS 细节（已核实，必须处理）
- 论文：*A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control*，arXiv **2610.12465**（2026-10-08），**CoRL 2026**；作者 Octi Zhang, Mateo Guaman Castro, Patrick Yin, Ignacio Dagnino, Abhishek Gupta, Rosario Scalise, Byron Boots（UW + NVIDIA）；项目页 https://sgs-rl.github.io/（代码 "coming soon"，截至 2026-11-02 未开源）。
- 算法：固定 N 个 task configuration（s0, goal, environment），每个配置带 success 历史 buffer；每个 episode 结束后按 success rate 加权选择 reset 配置（权重在 "never solved" 低、"getting better but not perfect" 高、"mastered" 低）；外层循环接标准 PPO；不需要额外网络/梯度信号。
- 结果：2^20 并行环境；ANYmal C/D locomotion 与 Franka/UR5e 装配；success 随 N 持续上升，PLR/uniform/linear curriculum 在 N 大时停滞/崩塌；蒸馏 RGB 策略零样本真机。
- **硬约束**：v1 若未引用/对照 SGS，novelty 维度会被判 2/5。方案必须：（a）差异表首行 SGS；（b）机制对照消融：success-band（SGS 式）vs LP（learning progress）vs uncertainty-aware LP，同任务同预算；（c）至少一个设置直接复现 SGS（config 级 success-band 采样器；代码未开源→按论文/项目页描述自实现，注明 "reimplementation"）；（d）把 "何时 success-band 足够、何时必须 LP×uncertainty" 作为 RQ/理论预测的一部分。

### 8.2 建议定位（writer/critic/judge 必须正面评估，不要机械照抄）
- 核心差异应落在**任务级（task/group）经验分配 + 随 N 的分配理论**，而不是 "再加一个不确定性的 SGS"。SGS 做单任务内 config 级 success 阈值采样；DRATS 做任务级但小规模、无 scaling、无真机；两者之间是空白：**M 个异构任务、N 个并行环境、固定交互/计算预算下，任务级分配如何随 N 变化**。
- 建议方法层次：两层分配（任务级 bandit/EXP3 分配 N 个 env 槽位；config 级 frontier 采样可调用 SGS 式 success-band 作为特例/组件），理论给出任务级分配的 scaling 预测（何时增益随 N 增大）。
- 机器人相关性前置：SO-101 8 任务（Squint Task Set）是天然 M=8 异构任务集；真机用其中 2–3 个任务做零样本 + 少量数据验证。
- 若 v1 只做 config 级、不涉及任务级分配，需按上面改写 framing 与实验矩阵。

### 8.3 其他已核实事实
- ICRA 2027：论文截稿 2026-09-15（已闭）、通知 2027-01-31、会期 2027-05-24~28 首尔。
- IROS 2026 论文截稿 2026-03-02 → 推得 IROS 2027 约 2027-03 上旬；CoRL 2026 会期 2026-11-09~12 → 推得 CoRL 2027 约 2027-06。
- Squint 是 RA-L 2026（doi 10.1109/LRA.2026.3730387）；SO-101 Bench 暂无正式 venue。
- ICRA 录取率、ICRA 2026 录用 RL 论文数等统计**无可靠公开来源**，只能标 "估计"，禁止写成精确事实。
- PLR=2010.03934；2110.02439 是 **RED**，不是 RPLR；RPLR 的 arXiv ID 未核实。
