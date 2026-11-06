# 12a 灵巧手大规模技能库 + LLM Agent：前置评审功课

proposal-critic，2026-11-02。已核实：Play2Perfect 2606.26428v4（全文＋仓库）、**ADEPT 2608.19182v2（已录 CoRL 2026，11 作者，NVIDIA 系，2026-08-19 首发）**、OmniReset 2603.15789v3（全文）、SGS 2610.12465、SER 2511.21011、Being-0 2503.12533v2、MuJoCo Playground registry 与 leap_hand/aero_hand 源码。

## 0. 结论：四个最大风险

1. **技能库工程量被严重低估，真正的缺口是"机械臂+灵巧手"。** Playground manipulation 共 10 个：灵巧手 3 个（`LeapCubeReorient`、`LeapCubeRotateZAxis`、`AeroCubeRotateZAxis`，手固定基座的手内旋转）＋臂-夹爪装配/交接 2 个（`AlohaSinglePegInsertion` 双臂平行夹爪 peg-in-socket、`AlohaHandOver` 双臂交接）＋Panda 系列。**装配/交接在 MJX 可行**（Aloha peg `ctrl_dt=0.0025`、`impl="warp"`、插入对齐阈值 5mm、容差 1e-3 级），我此前"MJX 做不了装配"的说法错误，已更正。但**没有任何"机械臂+灵巧手"的现成 env**：8 技能要在同一 embodiment 上完成"手内＋装配"，必须自建 arm+LEAP（2–3 周）或降级为双 track（手内 LEAP track＋Aloha 夹爪 track）。技能工程量 ≈55–67 人日（11–13.5 周）＋集成 2–3 周；1 人 45 周同时做环境＋FG＋校准＋API＋agent＋5 长程任务，仍然超。
2. **novelty 已被双占位。** Play2Perfect 占住"play 预训练→稀疏装配 finetune"；ADEPT 占住"大规模灵巧预训练→稳定 post-training＋两形态＋visual-tactile＋零样本 sim2real＋长程任务"。任何"play→specialize pipeline"主线都会被判 incremental。
3. **校准的必要性会被一句问倒。** "一个技能成功率跑 100 次数一下不就行了，要 ECE 和 epistemic uncertainty 干什么？" 若 API 只给边际成功率，RQ2/RQ3 会塌成 demo。校准必须定义为**条件成功率** P(s|params, 状态摘要, 物体配置)，在 held-out context 上评 ECE/Brier，并证明 agent 用条件估计比边际估计规划得更好。
4. **sim-only 的"系统论文"路线先验很低。** judge 给 24%（80% CI 15–34%）；ICRA 2026 dexterous 22 篇中纯仿真 RL 仅 1–2 篇，能中的是分析/benchmark 类（先例 *The Price Is Not Right*）。"系统集成＋无真机"是双重扣分。

## 1. sim-only 下怎么才可能中

- 只认两类：**受控分析研究**（同预算、单变量、可证伪）或 **benchmark/工具**（可复用、开源、有基线）。系统集成本身不构成贡献。
- 通过要素：受控实验（同 N/B/环境，只变分配与校准）、统计严谨（≥5 seeds、CI、预注册、功效计算）、可复现（代码/资产/harness/配置/MJX 版本）、负结果（"uniform 何时已经够"要敢写）、开源基准（8 技能 suite＋5 长程任务＋API schema＋评测协议）。
- 系统贡献不能当主贡献：①审稿人把"集成"当工程而非研究；②无真机则系统价值不可验证；③方法＋基准＋系统三件套每件都薄。**主贡献应是一个可证伪发现（校准的可靠性→组合成功率），系统只是承载物。** 题目与摘要不能出现 "we build a system"，要 "we show that, at fixed budget, …"。

## 2. 最可能被拒的 14 条（审稿人原话式，附防线）

1. **"和 Play2Perfect 的区别？"** 他们 play→稀疏装配 finetune＋真机 60% 插入；我们若换开源手重做即 incremental。防线：不碰 play 预训练；P2P-style 只作参照系，明写"MJX/LEAP 适配版，非原系统复现"。
2. **"和 ADEPT 的区别？ADEPT 有 23-DoF Kuka-Allegro / 29-DoF Flexiv-Sharpa、触觉、零样本 sim2real、长程任务。"** 防线：他们解决"**怎样**预训练并把先验稳定迁移到下游"；我们解决"**哪个技能值得练、它有多可靠、agent 该怎么用它**"。不同问题，且他们无技能 API、无 per-skill 校准、无组合评测。
3. **"OmniReset 说多样 reset＋扩 N 就够了，为什么还要 FG？"** 最锋利。必须 head-to-head（同 N/B/reward）＋正交组合（OmniReset-style reset × FG）。
4. **"SGS 已在 2^20 环境做 success-band 加权。"** 必须技能库级 bucket（技能×目标×物体）＋给出 SGS 在**技能间**失效证据（成功率不可比、aleatoric 噪声使 band 误判）。
5. **"三个贡献，每个都只做半个实验。"** 6 页装不下 RQ1–RQ4 全部严格证据；选一个 headline（建议 RQ2→RQ3 因果链），其余降级。
6. **"没有真机，凭什么投 ICRA？"** DR/held-out 物理参数鲁棒性＋失败分析＋开源 harness＋明确 limitation＋sim2real-gap 代理实验（摩擦/间隙/延迟扰动保持率）。
7. **"8–14 技能 45 周建得出来？"** 一旦怀疑可信度归零。必须给 8 技能表＋人日；且 v1 要承认"臂+LEAP"要自建（2–3 周），不能假装 Playground 有现成组合。
8. **"N 只有 1k–16k，P2P 24,576、SGS 2^20。"** 用 scaling 斜率＋硬件上限说明，诚实定义"本硬件可达规模"，pilot 实测。
9. **"跨技能共享策略怎么学？"** 上轮 C9 翻版；动作空间/目标类型/成功判据异构，必须 skill ID＋goal spec 条件化，并把"统一 vs 每技能策略"做成消融。
10. **"校准没必要。"** 见 §0.3；条件校准＋规划增益。
11. **"agent 增益来自技能更强还是校准更准？"** 混淆；必须 2×2：{FG, uniform}×{给/不给校准元数据}，固定 planner 与 prompt。
12. **"LLM 调技能不新（Being-0/SayCan/Code as Policies）。"** agent 只算 benchmark＋协议贡献；给失败归因（执行 vs 规划）与 oracle planner 上界。
13. **"基线是稻草人。"** 声明 P2P-style 为适配；最强可比基线定为 SGS-band＋OmniReset-style＋uniform。
14. **"统计/预算算不平。"** 上轮栽点（524 runs 实为 538、缓冲错、30M 步按 15M 计价）；逐行 CSV 可复算＋FLOOR。

## 3. 威胁排序与四强差异矩阵

威胁：**ADEPT 极高（9.6）≈ Play2Perfect 极高（9.5）** > OmniReset 高（8.5）> SGS 中高（7）> Being-0 中（5.5）> SER 中（5）。理由：ADEPT 与 P2P 都占"预训练→专门化"且都有真机零样本；OmniReset 占"大规模＋多样 reset"竞争叙事；SGS 占"success-band 采样"；Being-0 只占"LLM＋技能库"壳，可引用为系统先例。

| 维度 | Play2Perfect | ADEPT | OmniReset | SGS | 我们（必须占住） |
|---|---|---|---|---|---|
| 平台/硬件 | Isaac Sim；KUKA+22-DoF Sharpa | 大规模 RL；23-DoF Kuka-Allegro、29-DoF Flexiv-Sharpa；RGB＋视觉触觉 | IsaacLab；单臂＋灵巧手 | 四足＋装配 | **MuJoCo MJX；LEAP 16-DoF＋Aero 腱驱动；全开源无专有硬件** |
| 范式 | play 预训练→per-task finetune | 预训练→BC 蒸馏/critic warm-up/保守 on-policy 的 post-training 配方 | reset 多样性＋扩 N，单 reward 无课程 | task-config 三档 success-band | **技能库级分配采样（FG）＋条件校准＋API** |
| 规模 | 24,576 env×7d（单 A6000） | 大规模（具体 env 数待核实） | 大规模（8–24h/任务，4×L40S） | 2^20 | **N 1k–16k（A100），诚实报上限** |
| 技能库/API | 无（per-task） | 无 | 无 | 无 | **8 技能 JSON API（前置/效果/成功概率/不确定性/失败模式）** |
| 校准 | 无 | 无 | 无 | 无 | **条件成功率 ECE/Brier/风险-覆盖** |
| 组合评测 | 无（原文明确留给 future work） | 有长程任务但无技能库组合 | 有 emergent 多阶段但无 API | 无 | **5 长程 × 4 planner × 2×2 归因** |
| 开源 | 是（Isaac Sim-only） | 待核实 | 待核实 | 否（coming soon） | **是：env＋训练＋harness＋资产清单** |

定位句（v1 必须逐字落实）：**"At a fixed interaction budget, per-skill calibrated reliability—not raw scale—determines whether an LLM agent can compose dexterous fine-manipulation skills; we show this with a controlled 8-skill study and a reproducible benchmark."**

## 3.5 Embodiment 选项与工程成本（Lead 更正后）

事实：Playground 有 3 个灵巧手 env（手固定基座）＋2 个臂-夹爪装配/交接 env（Aloha 双臂平行夹爪）＋Panda 系列；**装配/交接在 MJX 可行，缺的是"机械臂+灵巧手"组合**。8 技能要同一 embodiment 完成手内＋装配，只有四条路：

| 方案 | 组成 | 工程成本 | 优点 | 缺点 | 定位 |
|---|---|---|---|---|---|
| A 固定基座 LEAP only | 6 手内 | 22 人日 | 最省最稳 | 无装配/工具，叙事弱 | FLOOR |
| B 单臂 LEAP（Panda 臂换手） | 6 手内＋2 装配 | 臂+LEAP MJCF/IK 10–15 人日＋55 人日技能 | 与 Panda/Aloha 资产兼容，单臂省一半 | 手内技能要适配到臂端，接触调试难 | **主推** |
| C 双臂两 LEAP | 6 手内＋2 装配＋交接 | B＋另手 8–10 人日＋同步 | 叙事最完整（bimanual 交接） | 工程/风险最高 | 时间富余才做 |
| D 双 track | track1 手内 LEAP（5 现成/轻改）；track2 Aloha 夹爪装配/交接（现成） | 22＋10 人日 | 风险最低、benchmark 叙事成立 | 两 track 不同 embodiment，FG 不可跨 track，agent 只能单 track | Plan B（W8 失败时） |

推荐：**B 为主线（6 手内＋2 装配，env 总工期 14–17 周）**，A 为 FLOOR，D 为 W8 失败兜底；C 只在 W20 前时间富余时做。无论哪条，v1 必须写明"臂+LEAP 是本项目最大的单点工程风险"。

## 4. 必须补的证据（v1 逐条可验收）

1. 技能库表：8 技能 = 6 个固定基座 LEAP 手内技能（5 个基于现成脚手架或轻改：cube reorient、绕轴旋转、Aero 旋转、掌内平移、手指步态；1 个新槽内放置）＋2 个 LEAP+臂装配（peg-in-hole 3–5mm、工具抓取/放置）。每行给 obs/action/goal/成功判据/终止/训练预算/资产来源/人日/风险；**不承诺 0.5mm**。
2. 缩放曲线：S∈{4,8}×N∈{256,1024,4096}，fixed-K 为主（K=B/(N·L) 显式）、fixed-B 为辅，≥5 seeds，CI，uniform 饱和证据。
3. 校准：条件成功率、held-out context ≥200/技能、ECE/Brier/可靠性图、条件 vs 边际、风险–覆盖。
4. agent 2×2＋4–5 长程任务 × 4 planner（校准 LLM/无校准 LLM/脚本/随机）× ≥50 trials；失败归因双标 κ≥0.8；oracle 上界；固定 LLM 版本/温度/token 成本。
5. 消融 ≥8：去归一化/去导数/去不确定性/bucket 粒度/Hedge vs EXP3/warm-up/统一 vs 每技能策略/校准头开关；外加 OmniReset-style×FG 正交。
6. 鲁棒性：摩擦 ±30%、间隙偏移、延迟 0–40ms、held-out 物体集；性能保持率。
7. W7–12 pilot 门槛：单技能 steps/s、OOM 边界、env 建成时间；不达标砍技能数不砍统计。
8. 可复现：seeds/配置 CSV/checkpoints/资产许可证/MJX 版本/代码与 harness。

## 5. 审计要点（v1 我会逐条重算）

- **技能库工程量（逐项）**：6 手内技能 22 人日（S1–S5 复用/轻改＋S6 新槽放置 8 人日）＋2 臂式技能 33 人日（S7 peg 18、S8 工具 15）＋臂/手 MJCF＋IK 集成 10–15 人日＋槽/夹具资产 3–5 人日 ≈ **13–15 周**，占 45 周的 29–33%，是 Brief 完全没算的隐藏成本；若 W8 判定臂集成失败，降双 track（手内 LEAP＋Aloha 夹爪现成 env）可省 3–4 周。
- **规模上限**：本地 16GB 状态输入 LEAP 估计 N≤2048、RGB N≤128–256；A100 80GB 才谈 N≥4096，合理上限 N≈8k–16k。与 P2P 24,576 差 2–3 倍可接受，但必须 pilot 实测并写清。
- **预算（初算，需逐行）**：A6000 ¥3–4/h、A100 ¥6–8/h（假设）。play 1–2×168h＝¥504–1,344；8 技能 finetune 192h≈¥672（A6000）–¥1,344（A100）；消融/复跑 ¥600–1,000；合计约 ¥1,800–3,700，留 15% 缓冲仍≤5,000；**FLOOR：1 次预训练＋6 技能＋单任务 agent ≈¥2,000**。
- **人力/排期**：env 12–16 周＋FG 4 周＋校准/API 3 周＋agent harness 3–4 周＋长程与评测 4 周＋写作/视频 5–6 周＝31–37 周，无调试余量已逼近 45 周。必须 W8/W20/W30 硬 Go/No-Go（技能<6→降级为方法论文）。
- **2–3 人项目的证据与砍法**：P2P（4 作者＋Sharpa 硬件）、ADEPT（11 作者，NVIDIA 系）、OmniReset（多人＋4×L40S）——我们的有效人力/算力低 1–2 个数量级。因此只做一个受控研究：1 平台＋2 形态（跨形态进附录）＋8 技能（6 主 2 附录）＋1 方法轴（分配×校准）＋3 长程任务；明确不做触觉、不做 RGB 策略（只留 1 个 RGB demo）、不做真机、不做 14 技能全表。

## 6. 上一轮的坑会不会重演

1. 近邻晚出现击穿 novelty（SGS）→ 现在 P2P＋ADEPT 双占位，必须全文读、差异表首行。
2. 方法级硬伤（C9 跨任务共享 PPO）→ 翻版"跨技能共享策略"；显式条件化＋消融。
3. fixed-B 下"N 大＝更新少"伪饱和 → fixed-K 为主，K 显式。
4. 功效不足（真机 30 trials→p≈0.11）→ agent ≥50 trials/任务、校准 ≥200 context。
5. 预算算术错（524 vs 538、缓冲、单价）→ 逐行 CSV＋独立脚本。
6. 非法 env 名/版本（Ant/Quadruped、-v0）→ 已核实 `LeapCubeReorient`、`LeapCubeRotateZAxis`、`AeroCubeRotateZAxis`、`AlohaSinglePegInsertion`、`AlohaHandOver`；**LEAP 无臂、Aloha 有臂但平行夹爪**，两者都要写进方案。
7. 摘要超词/图表超页 → 摘要 ≤250、≤4 图＋3 表。
8. 中稿率/价格写成事实 → 一律"估计/假设，以平台为准"。
