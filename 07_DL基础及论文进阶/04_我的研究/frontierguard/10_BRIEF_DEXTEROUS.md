# DexSkill-Agent 团队 Brief（2026-11-02 修订版，取代 00_BRIEF.md 的课题设定）

> 本文件由 Team Lead 撰写，是 proposal-writer / proposal-critic / acceptance-judge 新一轮工作的共同上下文。硬约束与事实以本文件为准；00_BRIEF.md 只保留上一课题（FrontierGuard-RL 单任务采样）的历史记录，不再作为本轮依据。

## 0. 本轮任务与交付物

用户已调整课题方向（2026-11-02 晚）：

> "我的初步想法是做灵巧手大量精细操作的大规模强化学习，然后把每个精细操作封装成能让 LLM agent 调用的函数，实现一个 agent 驱动的灵巧手精细操作的机器人系统，可以以 Play2Perfect 为参考。"

用户已确认两条关键决策：
1. **纯仿真，不用 SO-101 真机**（用户原话："SO101 真机就不用了，感觉放在一起有点不伦不类，做纯仿真就行"）。
2. 贡献**并重**：大规模 RL 采样方法 + 可调用技能库 + Agent 组合评测。

目标会议仍是 **ICRA 2028**（预计 2027-09 中旬截稿；ICRA 2027 已截稿）。人力和预算约束不变：1 人全职 45 周；租卡预算 ≤ ¥5,000；本地 RTX 4070 Ti Super 16GB / 32GB 内存 / 2TB SSD。

| 角色 | 文件（均在 `/Users/heyuhang/Desktop/frontierguard/`） | 说明 |
|---|---|---|
| proposal-writer | `11_proposal_dexterous_v1.md`；后续 `15_proposal_dexterous_v3.md` | 新课题主方案 v1、终稿 v3 |
| proposal-critic | `12a_critic_dexterous_prep.md`；`12_critique_dexterous_v1.md`；`13_proposal_dexterous_v2.md` | 前置功课、批改、修订版 v2 |
| acceptance-judge | `14a_judge_dexterous_prep.md`；`14_acceptance_dexterous_v2.md`；后续 `16_acceptance_dexterous_final.md` | 前置功课、v2 中稿概率、v3 终审 |
| Lead | `10_BRIEF_DEXTEROUS.md`；`99B_FINAL_DEXTEROUS_ICRA_PLAN.md` | 统筹、审计、最终整合 |

共享任务板：task-7（writer v1）、task-8（critic v2）、task-9（judge v2）、task-10（writer v3）、task-11（judge 终审）。执行前先 `team_task_get` 拿 revision，再 `team_task_update action=claim`，完成后 `action=complete`，并 `send_message` 给 lead（≤500 字）。

## 1. 课题设定（需要 writer 具体化）

工作标题（可再改，但必须体现三件事：大规模/技能库/agent）：
- 候选 A：**DexSkill-Agent: Scaling Dexterous Skill Libraries for Agent-Operated Fine Manipulation**
- 候选 B：**From Play to Skills to Agents: Calibrated Dexterous Skill Libraries for LLM-Operated Manipulation**

核心科学问题（暂定，writer 可精炼）：
> 在固定交互预算与固定算力预算下，不确定性感知的学习进度采样能否提升"大量灵巧手精细技能"的训练效率与**技能可靠性校准**，并进一步转化为 LLM agent 在长程精细操作任务上的组合成功率？

四个 RQ（暂定）：
- RQ1（scaling）：随着技能数 S 与并行环境数 N 增大，FG 采样相对 uniform / SER / SGS-band / OmniReset-style / Play2Perfect-style 的样本效率与最终成功率是否更好？是否出现"uniform 在大规模下饱和"的现象？
- RQ2（calibration）：FG 训练出的技能是否给出更校准的成功概率/不确定性估计（ECE、可靠性图、风险-覆盖曲线），而这是 agent 规划所需的？
- RQ3（composition）：把技能封装为带前置条件/效果/校准成功率的 API 后，LLM agent 的长程组合成功率是否显著高于"无校准 LLM / 脚本编排 / 单体策略 / 随机技能选择"？
- RQ4（morphology）：技能库能否跨灵巧手形态迁移（LEAP Hand ↔ Aero Hand，或至少跨物体/目标分布），迁移的瓶颈在哪？

可证伪假设（H1–H8，writer 给数值阈值）：
- H1：在固定 B 下，FG 在 S≥8、N≥1024 时比 uniform 高 ≥5pp 最终成功率（主格）。
- H2：FG 的 per-skill success 估计 ECE 比 uniform 低 ≥30%（相对）。
- H3：calibrated-API LLM agent 在长程任务上比 uncalibrated LLM 高 ≥15pp。
- H4：技能数从 4→12 时，uniform 的"边际收益/新增技能"显著下降，FG 下降更慢。
- H5：跨手形态迁移（LEAP→Aero）时 FG 预训练先验的收益大于 scratch。
- H6：agent 失败中"技能执行失败"与"规划错误"可分离，校准信息主要改善前者。
- H7：API 粒度（粗/中/细）存在最优，过细增加规划负担、过粗增加执行失败。
- H8：sim-only 条件下，上述结论对渲染/状态噪声与物体分布偏移稳健。

## 2. 必须参考/对比的真实工作（已核实，不得改写数字）

### 2.1 Play2Perfect（核心参考，必须逐维对比）
- arXiv **2606.26428v4**（2026-06-24 首发，2026-10-15 更新），*Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?*；作者 Tyler Ga Wei Lum, Kushal Kedia, C. Karen Liu, Jeannette Bohg（Stanford / Cornell）。
- 硬件：**22-DoF Sharpa 五指手 + 7-DoF KUKA iiwa 14 机械臂**（真机实验）；仿真为 **Isaac Sim / Isaac Lab**。
- 方法：两阶段 RL —— Stage-1 task-agnostic "play" 预训练（随机物体 + 随机目标，学 grasp / in-hand reorientation / 6D pose control）；Stage-2 在 CAD 导出的稀疏奖励装配环境上 finetune。算法 **SAPG**（Split and Aggregate Policy Gradients）。
- 算力（论文 Appendix H）：所有策略在**单张 RTX A6000** 上训练；play 预训练 **24,576 并行环境、7 天**；每个下游装配策略 finetune **12,288 环境、1 天**；物理 120Hz、策略 60Hz。
- 结果：比 from-scratch RL **33× 样本效率**；真机 zero-shot sim2real：0.5mm 间隙插入 60% 成功率（6/10），长程多部件装配与拧螺丝 >50%。
- 开源：`github.com/kushal2000/play2perfect`（133★，2026-10 代码发布）：Isaac Sim 训练代码、四个装配任务 + 两个日常物体插入任务、checkpoints、`assets/urdf/kuka_sharpa_description`、Fabrica/FurnitureBench CAD 资产、FoundationPose 部署参考、`rl_games` 内置 PPO+SAPG。**仓库只支持 Isaac Sim**。
- 对我们的含义：**不能直接复现**（Sharpa 手 + KUKA 硬件与 Isaac Sim 都不可得/不可负担）；把它作为"play 预训练 + 稀疏装配 finetune"的强基线/参照系，在**开源灵巧手 + MuJoCo 系**上做可复现的对照与扩展。

### 2.2 其他必须进差异表的工作
- **OmniReset**（arXiv 2603.15789v3，2026-03-16，*Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning*）：用程序化 reset 系统暴露多样交互，单一奖励、无课程、无示范，随算力扩展；dexterous 多任务。与 FG 的"分配采样"是不同轴（reset 多样性 vs 经验分配），必须做 head-to-head 或正交组合。
- **SGS**（arXiv 2610.12465，CoRL 2026，*A Balanced Data Diet...*）：success-band 三档加权 reset，2^20 环境；上一轮的 S0 威胁。必须在 Tier-2 作一等基线（重实现保真度判据）。
- **SER**（arXiv 2511.21011）：staggered environment resets，提升大规模并行 on-policy 数据多样性。
- **Being-0**（arXiv 2503.12533v2）：FM + 模块化技能库 + VLM Connector 的人形 agent；证明"LLM/基础模型 + 技能库"已有系统先例，我们的差异必须落在**灵巧手精细技能 + 校准成功概率 + 大规模训练方法**，而不是"LLM 调技能"本身。
- **2606.22471v2**：RL 合成数据 + 语言条件双臂灵巧操作（数据生成视角）。
- **DexSim2Real**（arXiv 2605.05241）、**Text2Touch**（arXiv 2509.07445，LLM 设计奖励）等作为 LLM/基础模型用于灵巧操作的近邻。
- 仿真平台：**MuJoCo Playground**（arXiv 2502.08844，MJX GPU 加速；任务含 `LeapCubeReorient`、`LeapCubeRotateZAxis`、`AeroCubeRotateZAxis`、`AlohaSinglePegInsertion`、`PandaPickCube` 等；LEAP 手 16-DoF 开源；Aero Hand Open 由 Chestnut Robotics（原 TetherIA）开源、腱驱动）；**Isaac Lab**（2511.04831，Allegro/Shadow 手任务，需 32–64GB 内存/大显存，本地 16GB 显存不友好）；**ManiSkill3**（2410.00425）。Adroit（24-DoF MoCap 手）可作第三形态备选。
- LLM agent / 技能库：SayCan、Code as Policies、RoboCodeX、Being-0 等（writer 引用前必须核实 arXiv/会议信息，禁止编造）。

### 2.3 事实核查规则
- 所有引用必须给 arXiv ID 或 DOI；无法核实的标"待核实"，不得写成已核实。
- 不得编造实验结果、成功率、算力、价格；价格用区间并注明"假设，以平台实际为准"。
- 论文中的数字（如 Play2Perfect 的 24,576 envs / 7 天 / A6000 / 33× / 60% / 0.5mm）必须与论文一致。

## 3. 技术方案骨架（writer 需给出可执行版本）

### 3.1 仿真与灵巧手
- 主平台：**MuJoCo Playground（MJX）**，因为本地 4070 Ti Super 可跑、GPU 并行、开源、社区活跃；本地做开发与小 N，租 A100 做大 N。
- 手：**LEAP Hand（16-DoF）** 为主，**Aero Hand Open（腱驱动，欠驱动）** 为跨形态第二手；可选 Adroit 手作第三形态（若工程量允许）。
- 备选/交叉验证：Isaac Lab 的 Allegro/Shadow 任务（若时间允许，只做小规模对照，不承诺）。
- 明确说明为什么不用 Play2Perfect 的 Sharpa+KUKA 仿真栈：硬件/许可证/显存不可得；我们在开源手上重实现其"play→specialize"范式。

### 3.2 技能库（"大量精细操作"要具体化）
建议 10–14 个技能，分三类（writer 可调整，但必须给每个技能：观测、动作、目标分布、成功判据、终止条件、训练预算）：
- **手内技能**：目标姿态重定向（任意四元数）、绕轴连续旋转、手内平移/定位、再抓取/手指步态（finger gaiting）。
- **接触丰富技能**：紧配合轴孔插入（clearance 0.5–2mm）、拧螺丝/拧瓶盖、卡扣装配、精密放置。
- **工具/功能技能**：抓取工具（螺丝刀/勺子）、工具定向、按压/推动、交接（handover，若单臂仿真可省略）。
- 技能实现方式二选一或混合：① 每个技能一个 goal-conditioned 策略；② 一个统一 goal-conditioned 策略 + 技能条件（skill ID / goal spec）。必须讨论两者在"大量技能"下的 scaling 差异（这是 RQ1 的一部分）。

### 3.3 训练方法（FG 的落点）
- 沿用上一轮的 FG 核心：归一化 learning-progress 导数 × 估计量不确定性（bootstrap EMA 离散度），Hedge 更新 bucket 权重；bucket = 技能 × 目标难度/物体/初始状态配置。
- 关键改动：bucket 空间从"单任务 config"扩展到"技能库 × 目标分布 × 物体"；LP 需在技能间归一化（上一轮已确认跨任务不可比问题）；不确定性信号用于 **per-skill 成功概率校准**（与 agent 的 API 对接）。
- 训练协议：fixed-B 与 fixed-K 两种；fixed-K 主终点（避免"大 N=少更新"伪影）；8 seeds（Tier-1 可 4 seeds）；预注册、TOST、负结果分支。
- 基线：uniform、SER、SGS-band（重实现）、OmniReset-style、Play2Perfect-style（play 预训练 + finetune，在我们的开源手上重实现）、FG；消融：去归一化、去导数、去不确定性、bucket 粒度、Hedge vs EXP3、warm-up、信任域。

### 3.4 技能 API 与 LLM agent
- API 规范（JSON schema）：`name`、`params`（目标位姿/物体 ID/力/速度）、`preconditions`、`effects`、`success_prob`（校准）、`uncertainty`、`timeout`、`failure_modes`、`recovery`。
- 技能执行器：策略推理 + 成功检测器（状态/视觉）+ 超时/失败分类（执行失败 vs 规划失败）。
- Agent：LLM planner（可用 API 调 GPT-4 级模型，成本小；也可本地开源 LLM 作对照）；输入任务描述 + 当前状态 + 技能 API 文档；输出技能调用序列；支持重规划与失败恢复。评测时固定 LLM 版本与温度，记录 token 成本。
- 组合任务（4–5 个长程任务）：多部件装配（2–3 步）、插入+拧螺丝、工具使用序列、物体分拣/重定向序列、带扰动的恢复任务。
- 对照：单体端到端策略（若有）、脚本编排、无校准 LLM、随机技能选择、oracle planner；消融：API 粒度、是否给校准、是否给视觉反馈、失败恢复开关。
- 评测指标：任务成功率、完成时间、技能调用次数、失败归因、校准指标（ECE/Brier/风险-覆盖）、LLM token/延迟成本。

### 3.5 论文贡献（writer 必须写成 3 条，审稿人一眼能看懂）
1. 方法：把不确定性感知的学习进度分配从单任务扩展到**技能库级大规模灵巧操作训练**，并给出 per-skill 校准信号（不是又一个 PPO 变体）。
2. 系统/基准：开源灵巧手 + MuJoCo 的**技能库 + 可调用 API + LLM agent** 闭环，含 4–5 个长程精细操作任务与评测协议（sim-only，明确 limitation）。
3. 实证：大规模 scaling 曲线 + 校准-组合的因果链证据（训练分配 → 技能可靠性校准 → agent 组合成功率），并与 Play2Perfect-style / OmniReset / SGS / SER 对比。

## 4. 预算与排期（硬约束，writer 必须给逐行算术）

- 租卡 ≤ **¥5,000**；本地 4070 Ti Super 免费但 16GB 显存/32GB 内存限制 N。
- 假设单价（必须标注"假设"并做 ±50% 敏感性）：A100 80GB ¥6–8/h；RTX 4090 24GB ¥1.6–2.5/h；A6000 48GB ¥3–4/h。
- 估算方向（writer 重算）：大规模 play 预训练 1–2 次 × 4–7 天（A100/A6000）≈ ¥1,500–2,500；技能 finetune/多技能训练 2–3 次 × 2–4 天 ≈ ¥1,000–1,500；消融与复跑本地为主；总租卡 ¥3,000–4,500，留 ≥10% 缓冲。**禁止超过 ¥5,000**，并给 FLOOR（只做最小充分证据集）方案。
- 45 周排期（W1 = 2026-11-02 起）：W1–6 环境/技能 suite MVP + 基线跑通；W7–12 小规模 pilot（LEAP 手，N=256–1024，本地）；W13–20 主 scaling（租卡，S×N 网格）；W21–26 技能库 + 校准 + API；W27–32 agent 组合评测 + 消融；W33–38 补充实验/写作；W39–41 投稿（目标 2027-08-09，W41）；W42–45 缓冲。W8/W20/W30 设 Go/No-Go。
- 人力：1 人；把工程拆成"环境/训练/agent/评测"四块，给每周工时估算，避免零缓冲。

## 5. 方案必须覆盖的检查清单（14 项）
1. 英文题目/摘要/关键词（≤250 词摘要）。
2. RQ1–RQ4 + 可证伪假设 H1–H8（数值阈值）。
3. 方法形式化 + 伪代码（FG 分配 + 校准 + API）。
4. 技能库定义表（每个技能：obs/action/goal/success/预算）。
5. 三层实验矩阵（本地小 N → 租卡大 N → 跨形态）。
6. 训练协议（fixed-B / fixed-K、seeds、统计检验、预注册）。
7. ≥6 基线（uniform、SER、SGS-band、OmniReset、Play2Perfect-style、FG）+ 消融 ≥8。
8. 校准评测（ECE/Brier/风险-覆盖）与 agent 组合评测协议。
9. LLM agent 设计（planner、API、失败恢复、成本控制）。
10. 预算表（逐行 GPU 小时 × 单价 = ¥，±50% 敏感性，FLOOR）。
11. 45 周时间线 + Go/No-Go + 风险登记表（≥10 条）。
12. 与 Play2Perfect / OmniReset / SGS / SER / Being-0 的逐维差异表。
13. 论文结构（6 页 ICRA：Method/System/Experiments 分配）、图表清单、视频脚本、开源计划。
14. 中稿率提升策略（针对 sim-only 的弱点：benchmark 贡献、开源、视频、negative results、统计严谨性、rebuttal bank）。

## 6. ICRA 现实与质量要求
- 纯仿真 + LLM agent 系统：审稿人会问"real robot 呢""科学问题是什么""和 Play2Perfect/Being-0 有何本质区别"。方案必须正面回答，不能回避；把 sim-only 写成明确 limitation，并用 benchmark/开源/可复现性换分。
- 所有数字可追溯；实验矩阵与预算算术必须自洽；统计功效要算（不要只写 8 seeds 就完事）。
- 不得编造引用、结果、平台价格；不确定处标"待核实/假设"。
- 写作语言：中文正文（英文摘要/题目可给），术语统一（FG、skill、bucket、calibration、API、agent）。

## 7. 给 writer 的 Phase A 功课（先做，再写 v1）
1. 读本 Brief；用 arXiv API 核实 Play2Perfect、OmniReset、SGS、SER、Being-0、MuJoCo Playground、Isaac Lab、ManiSkill3、DexSim2Real、2606.22471 的标题/作者/ID。
2. 读 `github.com/kushal2000/play2perfect` 的 README/论文 Appendix H（算力）与 MuJoCo Playground 的 LEAP/Aero 环境说明；确认哪些能复用、哪些必须重实现。
3. 设计技能库 v0（10–14 技能）与 4–5 个长程任务；给每个技能的成功判据与预算。
4. 给出 FG 在新设定下的 bucket 定义与 LP 归一化方案；给出校准信号定义。
5. 给出 agent API schema 与 4 种 planner 对照设计。
6. 然后写 `11_proposal_dexterous_v1.md`（中文 12,000–18,000 字，含表格/伪代码/预算/时间线/风险/差异表/论文结构/视频脚本）。
7. 完成后 complete task-7 并 message lead（≤500 字：题目、3 条贡献、技能数、主实验网格、预算、自评中稿率）。

## 8. 参考文件
- 上一轮成果（仅作方法/流程参考，不沿用课题）：`00_BRIEF.md`、`05_proposal_v3.md`（FG 方法、fixed-K 协议、统计与预算审计经验）、`06_acceptance_final.md`（judge 的严格标准）。
- 本 Brief 是唯一现行任务书；与 00_BRIEF.md 冲突时以本文件为准。
