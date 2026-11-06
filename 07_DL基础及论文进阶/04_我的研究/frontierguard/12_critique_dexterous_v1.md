# 12 · 灵巧手 v1 批改（proposal-critic，CoRL/ICLR 口径）

**批改对象**：`11_proposal_dexterous_v1.md`（844 行，中文正文 14,988 字）
**批改口径**：CoRL 2027 主投（W31≈2027-06，8 页）、ICLR 2028 兜底（W46–47）；纯仿真不再是硬伤；ML rubric：Novelty 0.30 / 技术正确 0.20 / 实验严谨 0.30 / 洞察与泛化 0.10 / 清晰 0.05 / 可复现 0.05。
**我的独立复核**：arXiv API 核实 10 个 ID（含 ADEPT 2608.19182v2、P2P 2606.26428v4、OmniReset 2603.15789v3、SGS 2610.12465、SER 2511.21011、Being-0 2503.12533v2）；GitHub 源码核实 MuJoCo Playground 的 `aloha/single_peg_insertion.py`、`aloha/xmls/mjx_single_peg_insertion.xml`、`leap_hand/reorient.py`、`manipulation/__init__.py`；预算、run 数、GPU-h、±50%、FLOOR、H1/H2/H3 功效全部用脚本重算（§5）。v1 的数字我一个都没采信。

---

## 0. 判决（先说结论）

**当前版本不能投。** 不是"写得不够好"，而是**主假设对应的实验在预算里没有被真正跑过**（C1）、**预算在一致吞吐下不闭合**（C2）、**主终点统计功效不足**（C3）、**核心"精细装配"技能不可行性被 8mm 间隙的事实证伪**（C4）、**CoRL 30 周窗口装不下 8 技能+4 长程+agent 全链**（C5）。这五条任意一条被审稿人抓住都是 reject；而它们全部出现在同一版里。

| 维度（ML rubric） | 我的分（1–5） | 加权 | 一句话理由 |
|---|---|---|---|
| Novelty | 3.0 | 0.90 | "分配→校准→组合"链条有价值，但与 SGS/OmniReset 同层，差异靠 P3 支撑而 P3 只是直觉 |
| 技术正确 | 2.5 | 0.50 | 协议自相矛盾（700M/500M/450M/112.5M/125M）、混合控制率未处理 |
| 实验严谨 | 2.0 | 0.60 | 主格 5 seeds、H1 5pp 低于 MDE、H3 power≈0.6–0.7、校准在训练分布内评 |
| 洞察与泛化 | 3.0 | 0.30 | Prop 缺失使"校准为何重要"无法证伪；只有 O(εH) 一句话 |
| 清晰 | 3.5 | 0.175 | 结构完整、可读，但 8 页塞 8 技能+4 任务+4 planner+10 消融必爆 |
| 可复现 | 3.0 | 0.15 | 有 artifact 计划，但 LLM 快照/温度/成本、MJX 版本、吞吐假设均未锁死 |
| **合计** | | **2.625/5 = 52.5%** | 低于 CoRL 近年录取线（约 60–65% 质量分） |

**我的两口径概率（v1 原样）**：CoRL 2027 **22%**（15–28%）；ICLR 2028 条件概率 **24%**；级联 **≈41%**。judge 14a 给 CoRL 34%/25%、ICLR 30%/22%、级联 52%/37%；v1 自评 CoRL 40%、级联 60%。**我不支持 v1 的 40%**：v1 的 40% 建立在"H1 主格 700M 会跑、预算 4,849 闭合、5 seeds 够用、S7 能 1mm"四个未验证假设上，任意一个不成立就掉到 25% 以下。**v2 若修掉 C1–C9，我给 CoRL 30%（24–36%）、ICLR 条件 27%、级联 49%**——仍到不了 60%，因为 sim-only + 8 技能规模 + 无真机的组合在 CoRL 属于"可发但不惊艳"，天花板由证据规模决定。

---

## 1. 我独立复算了什么（方法说明）

- 协议：按 `K=B/(N·L)`、`L=50` 重算所有 B；核对 fixed-B 对应的 K；核对 H1 主格与预算表 G1 的实际步数。
- 预算：逐行 `runs×h/run×¥7`，加 12% 缓冲与杂项 550；再按 §7.1 的吞吐（N=2048 6,000 steps/s）与 §5.1 的吞吐（N=2048 18k）分别重算；对 C1–C5 削减逐条重算并检查缓冲处理；FLOOR 从行项重建。
- 统计：H1 用"seed 为重复单元"算 MDE（sd 4/5/6/8pp × seeds 5/6/8/10）；H3 用配对 McNemar 模拟（p1=0.60, p3=0.45, ρ∈{0,0.3,0.5}, n∈{120,200}）；H2 用 ECE 估计量方差给出 MDE 量级。
- 工程量：按 v1 技能表逐项加人日，再按 v1 自己声明的 20% 缓冲复算，并加 harness/API/agent 代码人日。
- 文献/平台：Playground 源码行号、Aloha peg 几何、LEAP 控制频率均已核实（见 §5.1、§6）。

---

## 2. CRITICAL（必须修，否则不成立）

### C1. 协议自相矛盾：H1 主格写的 B=700M 在预算里从没跑过
**问题（逐行）**：
- L99、L134：H1 主格 `S=8, N=8192, B=700M`。
- L252：fixed-K `K=1221`，锚点 `N=8192, B=500M`（1221×8192×50=500.1M，对）。
- L382：写 `N=2048→112.5M，N=8192→450M`——**两个都错**：1221×2048×50=125.0M；1221×8192×50=500.1M；450M 对应 K≈1099。
- L383：fixed-B `B=112.5M`；L374/L386 同；但 L252/L509 写 `N=2048→125M`。
- L384：pilot 用"库级均分值 62.5M"判断是否提高 B——62.5M=500M/8，**又回到 500M**，与 700M 矛盾。
- L509：`N=8192 时 B=500M；N=2048 时 B=125M`（正确），但 L516–531 的 G1 是 `20 runs × 8.58h`：8.58h 只在 B≈450M、15k steps/s 下成立（500M 应为 9.51h）。
**审稿人会怎么问**："Your main hypothesis is stated at B=700M, but Table 2's main runs use 450M and your protocol says 500M. Which experiment tests H1?" —— 这是 desk-reject 级别的不一致，不是笔误。
**修改（v2 §2.2/§2.6/§3.7/§5.2/§5.3/§7.1 全部换成一张表）**：固定 `K=1221`（锚点 N=8192, B=500M, L=50）；`B(N)=K·N·L`（N=2048→125M，N=1024→62.5M，N=256→15.6M）；fixed-B 次协议统一为 `B=125M`（N=8192 时 K=305）；H1 主格改为 `B=500M`；pilot 校准的是 **K**（成功率平台判据），不是 B；删掉 700M/450M/112.5M 三个数字。

### C2. 预算在一致吞吐下不闭合；C4 削减算错；双因素极限 4,987 不成立
**问题**：
1. L506 写 `N=2048 训练吞吐 6,000 steps/s`，L364 写 `N=2048 约 18k`，而 §7.2 的 G2/G5/G8（2.33h/run）**隐含 15,000 steps/s**（我按 B=112.5M、0.25h overhead 反推：15,024）。三套吞吐互斥。
2. 若按 L506 的 6,000 重算：G2 每 run = 112.5M/6000/3600+0.25 = **5.46h**（不是 2.33h），32 runs = 174.7h（+100.1h）；G8 20 runs = 109.2h（+62.6h）。总 GPU-h 从 548.4 → **711.1h**，费用 `711.1×7×1.12+550 = ¥6,125`，**超 5,000**。
3. C4 削减：G1 5→4 seeds = 4 runs×8.58h = 34.32h，×10.5 = **¥360**，不是 L552 的 ¥517；且 L552 说"MDE 升到 6pp"错——sd=6pp、n=4 时 MDE80=8.4pp，power≈0.63，不是 0.85。
4. 削减未计 12% 缓冲：C1–C5 从 6,999 逐减后 `6,999−811−361−294−360−98 = ¥5,075`，**仍 >5,000**；v1 写"≈4,987 仍 ≤5,000"是算术不闭合。若把削减也乘 1.12，则 4,844 能闭合——**两种处理结论相反，v1 没写规则**。
5. FLOOR 126.8h 与行项不自洽：77.2(G1)+24(P0)+18.6(G8)+7=126.8，但 P0 已包含 G1/G2/G8，FLOOR 又把 G2 砍掉，重复计算。
6. 余 ¥151（3%）覆盖不了任何吞吐/时长偏差；H1 若真按 700M 跑，G1 需 13.21h/run×20=264h（+92h，+¥645），直接到 ¥5,494。
**审稿人会怎么问**："Your own throughput table says 6k steps/s at N=2048; why do the rows assume 15k? Recompute the budget." —— 预算不闭合在 ML 会议是硬伤（reproducibility 0.05 + 实验严谨 0.30 都扣）。
**修改**：v2 给一张**统一吞吐表**（W6 实测前用保守值：N=8192 15k、N=2048 12k、arm+LEAP 6k、本地 N=1024 3.5k），所有行按此重算；削减链给"含缓冲/不含缓冲"两种口径并写死一种；FLOOR 从零重建；留 ≥¥250（5%）余量；±50% 与双因素必须闭合（v2 §6/§7）。

### C3. 主终点统计功效不足：5 seeds 撑不起 5pp；64 单元是伪重复
**问题**：
- L423–425 把 `8 seeds × 8 技能 = 64 单元`当独立重复，算 SE=1.2pp、power>0.95。**但一个 seed 只训一个共享策略，8 个技能是固定水平、不是 iid 样本**；正确重复单元是 seed（n=5），或 mixed model 里 skill 作随机效应。
- 以 seed 为单位、sd=6pp：n=5 时 SE=2.68pp，**MDE80=7.5pp**（不是 5pp），5pp 的 power≈0.40。n=8 时 MDE=5.9pp；n=10 时 5.3pp。**H1 的 5pp 阈值在 5 seeds 下不可检验**。
- H1b（ΔN 差 ≥2pp）是交互效应，n=5 下 MDE 至少 2×7.5=15pp，不可能显著；只能描述。
- H1c（uniform 边际 <1pp）是等价性主张，需要 TOST；5 seeds 的 CI 宽度 ≈±7pp，无法证 <1pp。
- H3：30 trials/task×4=120 配对；McNemar 模拟（p1=.60, p3=.45）：ρ=0 → power 0.59；ρ=0.3 → 0.67；ρ=0.5 → 0.76。**15pp 主张在 120 对下不够**；50 trials/task（n=200）→ 0.83/0.90/0.95。且 4 任务做 Bonferroni 会更低。
- H2：ECE −30% 没给 MDE；n=300 rollouts/skill 时 ECE 估计量 SE 约 0.01–0.02，若基线 ECE=0.10、目标 0.07，差 0.03 尚可，但若基线 0.05 则差 0.015 与噪声同量级。
**审稿人会怎么问**："Your main effect is 5pp with 5 seeds and sd 6pp—what is your MDE? Your 64 units are 8 skills from one policy; that's pseudoreplication." —— ML 会议最恨这个。
**修改**：v2 主格 **8 seeds at N=8192**（MDE 5.9pp；或 6 seeds + 阈值 7pp，二选一写死）；H1 阈值改 **≥6pp**（8 seeds, sd≤6pp）；H1c 用 TOST margin 2pp 且只作次要；H1b 降为描述性；agent **50 trials/task（n=200）**、H3 阈值 **≥12pp** 或 15pp 但 power≥0.85 并预注册；H2 给 MDE 表、n=500、ECE 主指标定为 held-out frontier 分层；预注册里写清"seed 为重复单元、skill 为固定水平、mixed model `success ~ method + (1|seed) + (1|skill)`"。

### C4. S7/S8 可行性被平台事实证伪：Aloha peg 是 8mm 间隙，不是 1mm
**问题**：
- L287、L319、L677、L702 反复写 Aloha peg "1 mm 级容差/1mm 容差"，并据此声称"与 Aloha peg-insertion 对齐"。**源码事实**：`aloha/xmls/mjx_single_peg_insertion.xml` L17–35，peg 截面 20×20mm，socket 内腔 36×36mm（半宽 0.018）→ **单边间隙 8mm**；`single_peg_insertion.py` L143 的 5mm 是奖励门限，L246–249 的 `(0,0.001)` 是 reward tolerance margin，不是几何间隙。所以 Tier-0 "验证 1mm 可达性"验不了任何 mm 级能力，只验证了 8mm 松配合流水线。
- S7 写"2mm（1mm 参数化）、深度≥20mm、横向误差<0.5mm"，同时 LEAP 手内 env 是 `ctrl_dt=0.05`（20Hz）、Aloha peg 是 `ctrl_dt=0.0025`（400Hz）；v1 没处理混合控制率（见 C8），S7 在 20Hz 控制下做亚毫米插入不现实。
- S8 "螺纹用螺旋副+摩擦近似"：MJX 接触模型无法稳定表达螺纹，审稿人会直接判 toy。
**审稿人会怎么问**："You claim 1mm alignment with the Aloha peg task, but its socket is 36mm for a 20mm peg. Where does the 1mm come from?" —— 一句事实错误会毁掉整个"精细操作"叙事。
**修改**：v2 主容差改 **2–3mm**（peg 20mm、孔 26mm，单边 3mm），1mm 仅附录/stretch 且必须给失败分析；Tier-0 只声称"验证臂+LEAP+接触流水线"，不声称精度；S8 改为 **Cap-Twist（2 圈 + 2mm 轴向，弹性盖，无螺纹）** 或降为 stretch；R2/R14 与 §4.1/§5.1/§11 同步改。

### C5. CoRL 30 周装不下 v1 的 scope；工程量少算 2–3 周
**问题**：
- 技能表人日实际 **54**（3+2+8+8+6+6+10+10+1；v1 写 53），加 arm+LEAP 10–15 = 64–69；按 v1 自己的 20% 缓冲 = **77–83 人日 = 15.5–16.6 周**，不是 L313 的"68–70 人日≈13.5–14 周"（53+15 未乘 1.2，加法也错）。
- 还没算 harness/API/calibration/planner/agent 代码（v1 §3.9 自己说"工程量 20 周"，与 13.5–14 周矛盾）；这部分至少 15–20 人日。
- 排期：W1–6 env MVP、W5–10 技能库 → **8 个技能+arm 集成塞进 10 周**，比 16 周的实际需求少 6 周；W14 就开始主实验，意味着 S7/S8 最晚 W13 可用（20 人日技能 = 4 周，从 W9 起才可能）；W20 才做 agent G2，但 agent 代码/2×2 设计不在 W1–20 任何阶段。
- CoRL W31 距 W1 只有 30 周，证据必须 W26 冻结；v1 的 W26–30 写作 + W31 投，等于**零缓冲**；W8 Go/No-Go 只测"4 个手内技能可训"，臂式（最大风险）到 W20 才检查。
**审稿人会怎么问**："One person, 45 weeks, 8 new contact-rich skills, a skill API, a calibration suite, an LLM agent study, and 4 long-horizon tasks—how?" —— 计划不可信会直接拉低 technical quality。
**修改**：CoRL 版 **6 个核心技能**（S1–S4 + S7 + S8；S5/S6 降 stretch/appendix），工程 W1–16，Go/No-Go 三个：**W4**（arm+LEAP 稳定 + N≥256 + 本地吞吐）、**W10**（4 手内可训 + 本地 N=1024 实测）、**W16**（6 技能 + FG 主曲线 1 任务）；实验 W17–24；冻结 W24–26；写作 W26–31；ICLR 兜底期（W31–45）再补 8 技能/robustness/ADEPT-style。

### C6. P3 不是界，是直觉：缺"校准误差→组合成功率"的可证伪命题
**问题**：L350 只有一句"若 |p̂−p|≤ε 则长程 H 步损失 O(εH)"；没有假设、常数、证明、可检验预测，也没说明"损失"是期望还是概率、ε 是逐技能还是统一、planner 是贪心还是风险感知。审稿人（尤其 ICLR）会说："This is a tautology; any H-step composition has O(εH) error. What does calibration buy you beyond that?"
**审稿人会怎么问**："State the proposition, its assumptions, and a prediction that could fail."
**修改（v2 §4.4 必写，judge 指为最高 ROI）**：给 **Proposition 1（组合误差传播）**——计划集 Π、计划 π 的真成功率 `S_π=∏p_i`、估计 `Ŝ_π=∏p̂_i`、`|p̂_i−p_i|≤ε`；则对所有 π，`|Ŝ_π−S_π| ≤ Hε(1+ε)^{H−1}`，故 `S_{π̂} ≥ S_{π*} − 2Hε(1+ε)^{H−1}`；**margin 条件**：若计划间隔 `m = S_{π*} − max_{π≠π*} S_π > 2Hε(1+ε)^{H−1}`，则 `π̂=π*`，损失为 0。**Corollary**：要保证选对计划，rollout 数 `n ≳ 2H² log(2|Π|/δ)/m²`（Hoeffding；Beta 区间更紧，实测 n*）。**三个可检验预测**：TP1 组合损失对 H 的斜率≈2ε；TP2 margin 翻转测试（构造 m∈{0.02,0.05,0.1,0.2} 的计划对，预测 m<2Hε 时校准收益最大）；TP3 干预比（ε_uncal/ε_cal 应≈损失比）。这条同时把 RQ4（success-band 何时够）从口号变成推论：**margin 大时 SGS 的 success-band 足够，margin 小时才需要 FG 的 LP+校准**。

### C7. 校准与 agent 评测都在训练分布内，因果链被"分布内"污染
**问题**：L113 说"held-out 目标分布"，但 L480 写初始状态"从训练分布内采样后冻结"；L427 的 30 个固定初始状态也没说与训练目标桶不相交。若校准与 agent 评测都在训练分布内，H2/H3 只是 in-distribution 数字；审稿人会问"agent 的价值恰恰在分布外决策，你的校准在 OOD 还成立吗？"（H8 只测鲁棒性，未与校准/agent 挂钩。）
**修改**：v2 明确 **goal split**：8 个目标桶 → 6 训练 / 2 held-out（held-out 只用于校准与 agent 评测，训练中不采样）；H2 主指标 = held-out frontier ECE；H3 的 agent 初始状态来自 held-out 目标桶；H8 增加"held-out 目标外移 20%"条件；§6.4 写 `eval/goal_split.json` 与 `eval/init_states.json` 的生成规则与哈希。

### C8. 混合控制率未处理：手内 20Hz vs 装配 400Hz，"共享策略 + fixed-K"在物理时间/算力上不可比
**问题**：`leap_hand/reorient.py` `ctrl_dt=0.05`（20Hz）；`aloha/single_peg_insertion.py` `ctrl_dt=0.0025`（400Hz）。v1 用**一个共享 π 覆盖 8 技能**，却没说统一控制率；fixed-K 固定的是"更新次数×每更新 N×L 步"，但 1 个 arm 步与 1 个 hand 步的物理时长（50ms vs 2.5ms）和 GPU 成本差 10–20×。于是 FG 的 LP 按"每步成功率"分配，会系统性偏向**每步便宜**的手内技能；"库级平均成功率"也混合了不同物理时长的技能。
**审稿人会怎么问**："Your allocation maximizes success per environment step, but your budget is GPU-hours. Are arm skills penalized by a factor of 20?" —— 这是公平性硬伤。
**修改**：v2 统一 **决策率 20Hz**（arm 技能 `ctrl_dt=0.05` + 20 个 2.5ms 物理 substep；手内 20Hz + 2 个 10ms substep），所有技能 1 步=50ms；**主 LP 用 compute-normalized**：`u_g = Δp_g / c_g`，`c_g` = W6 实测的每步 GPU 时间（arm 技能约 6–8× 手内），同时报告 step-normalized 作对照；加 **协议 C（fixed wall-clock）** 验证分配结论不随预算单位翻转；§7 预算按 `c_g` 重算（arm 技能行单独列）。

### C9. LLM 成本低估、可复现性不足；本地 vLLM 方案不可行
**问题**：L491–493：360 episodes × 8 calls × 2.5k tokens = 7.2M tokens，按 $2–5/M input、$8–15/M output 估"≤¥400，预算取 ¥250"。按 70/30 输入输出分算：$27–58 → **¥197–415**；加 30% 重试/长 prompt → **¥256–539**；若 8 calls 实际 4k tokens → ¥400–860。**¥250 是乐观下限**，且 v1 自己写"≤¥400"却取 250，自相矛盾。D5 的"开源权重大模型本地 vLLM"在 32GB RAM/16GB VRAM 上只能跑 7–8B Q4，规划质量不足以支撑 4 技能长程任务，不能作为主方案。
**审稿人会怎么问**："What exact model snapshot, temperature, prompt template, and token cost per episode? Can I reproduce the agent experiment for ¥250?" —— ICLR 的 repro 分会扣。
**修改**：v2 给 **input/output 分算表**（70/30、60/40 两情景）、**硬顶 ¥600**（到顶即停并报告覆盖率）、精确模型快照+日期+温度 0+固定 prompt 模板+每 episode token 日志进 artifact、缓存相同前缀；D5 改为"API 主方案 + 本地 8B 仅作离线复现 sanity，不进主实验"；预算杂项改 ¥750（电 220 + 存储 80 + LLM 450）。

---

## 3. MAJOR（尽量修，至少给回应）

**M1. H1b/H1c 阈值无推导、与功效不匹配**（L100–104）。H1b 2pp 交互在 n=5 下 MDE≈15pp；H1c "<1pp" 需要 TOST。改：H1b 降描述性并给 CI；H1c 用 margin=2pp 的 TOST，主格 8 seeds。
**M2. Tier-1 本地 2 seeds 不够调参**（L373）。FG 的 λ/η/T_alloc/guard 阈值需要 ≥3 seeds × 2 技能调，再冻结。改：Tier-1 只作 pilot，调参用 3 seeds，正式结果不引用 Tier-1 的 2-seed 数字。
**M3. ECE 分箱不一致且未给 MDE**（L113 10 等宽 vs L441 15 等宽+等频）。改：主指标 held-out **10 等质量 bin**，敏感性 15 等宽 + 5/20 bin；n=500/skill；给 ECE 的 bootstrap CI 与 MDE；报告 bin-count 敏感性曲线。
**M4. σ̂ 用 3 个 EMA 快照不是 ensemble**（L211）。3 个滞后高度相关，σ̂ 会低估估计量方差；L211 已正确声明"不是 model epistemic"，但 guard G-1 的 `σ_max/n_min=5 eps` 无依据。改：σ̂ 用 **lag-diversity**（H={5,20,100} episodes）+ 验证 σ̂ vs |p̂−p| 校准图（F4b）；G-1 阈值由 W8 pilot 的 σ̂ 分布定（如 90 分位），不拍 5 eps。
**M5. S8 螺纹不可行**（L310）。改：Cap-Twist（弹性盖、2 圈、2mm 轴向、无螺纹），或降 stretch；成功判据用"盖到位 + 无滑脱"状态量。
**M6. 长程任务规格太虚**（L331–334）。L1 "销到位+盖拧紧（状态判定）"没给几何容差、步数上限、失败定义。改：每任务给 `success predicate`（几何量+阈值）、最大步数、重试规则；加 **scripted planner**（无 LLM，按 p̂ 贪心）作"技能质量上界"，把 planning 与 skill 分离。
**M7. 失败归因是相关不是因果**（L487–489）。改：加 **allocation-swap 干预**（固定策略、只换分配），归因到"分配型失败"的比例作为中介变量；H6 改为"归因一致性 + 干预后失败率变化"双指标。
**M8. 基线保真度不足**（L390–401）。SER/SGS-band 的超参、reset 调度、桶映射未给；OmniReset-style "风格化"无定义；P2P-style 用完整版计价但只跑等步数版。改：给 **fidelity 表**（每个基线的原论文设置→我们的适配→差异→验证判据）；SGS 重实现须先在 `LeapCubeReorient` 复现"success-weighted > uniform"趋势（G-S1 已有，补数字）；P2P-style 主表用等总步数 600M，完整版只作附录参考。
**M9. 缺少 PLR/RED 与 ADEPT-style 基线**。差异表提了 PLR/RED 但主实验没有；ADEPT 是 post-training 范式，可讨论为何不适配（需示范/真机）。改：加 PLR/RED-style（regret 权重）作第 5 基线（预算允许则 3 seeds），ADEPT 在 related work 明确"不同范式，不作为 baseline"。
**M10. 吞吐假设未验证且直接决定预算**（L364/L506）。改：W4/W6 实测三档（N=256/1024/2048/8192）写入 v2 §5.1，预算按实测重算；若实测 <假设 80%，触发 §7 削减链。
**M11. H5 Aero 迁移证据太弱**（L121，附录 6 runs/1 技能）。改：H5 降 **exploratory**，只报效应量与 CI，不作主张；摘要不提 Aero 为第二手。
**M12. 8 页装不下**（L641–651）。8 技能+4 长程+4 planner+10 消融+理论+校准套件。改：正文 6 技能、3 planner（P1/P3/O1）、8 消融（A1–A8）、2 个 N；robustness/Aero/长程 L3–L4 进附录；Fig 减到 5 张、Table 3 张。
**M13. 自评概率过于乐观**（L663）。40% CoRL 与 judge 34/25、我的 22% 差距大。改：v2 写 **30%（24–36%）** 并给"为什么不是 40%"的三条理由（规模、sim-only、无真机）。
**M14. "首次"与"two open hands"过度**（L39、L65）。改：删"首次"，改"据我们所知，首个把……作为受控变量隔离的研究（待 related work 复核）"；摘要改 "LEAP hand as the main platform, with Aero as an appendix probe"。
**M15. 平台事实错误**：R5 写 "ManiSkill3 env"（L584），实际是 MuJoCo Playground；L287 `rotate_z.py` 路径需确认（Playground 为 `leap_hand/reorient.py` 与 `rotate_z.py`）；`aloha/handover.py` 应为 `AlohaHandOver` env。改：全稿统一 Playground 名称与路径，R5 改为"改 Playground 的 domain randomizer/桶注入"。
**M16. H2 主指标前后不一致**：L109 总体 ECE −30%，L222 又说主指标是 frontier 分层 ECE。改：预注册写死 **frontier 分层 ECE 为主、总体 ECE 为次**，H2 表述同步。
**M17. fixed-B 定义与 K 不自洽**（L383 vs L252）：112.5M 在 N=2048 对应 K=1099，不是 1221；在 N=8192 对应 K=275。改：fixed-B 统一 125M，并写明对应 K(N)。
**M18. 可复现性清单缺失**：MJX/Playground 版本、`impl="warp"`、`naconmax`、seed 列表、env config hash、LLM 快照、评测脚本、Docker/uv lock。改：v2 加"artifact 附录"：`uv.lock`、`configs/*.yaml`、`eval/`、`scripts/reproduce.sh`、`results/` 与 LICENSE(MIT)。
**M19. 8 技能共享策略的负迁移未讨论**：S7/S8 的臂动作与 S1–S4 的手内动作冲突（手内时臂 mask），共享 trunk 可能互相拖累；v1 只提"共享 goal-conditioned π"，没给 per-skill 头/适配器消融。改：加消融 A11（shared trunk vs per-skill adapter），并在 R 风险里写负迁移。
**M20. 时间线没有并行/等待分析**：租 A100 4 周跑 64 runs 需要 8 卡并行，但 1 人无法同时盯 4 个实验 + 写代码 + 标定；W17–22 "主实验二"与 W20 agent 重叠。改：给"每周 GPU 卡数×任务"的甘特，标出单人瓶颈；实验批次串行化（每批 ≤3 天），agent 与长程任务由脚本自动跑、人只做监控。

---

## 4. MINOR（快速修）

1. L99 "fixed-K 主格 B=700M"与 L252 冲突（见 C1）。
2. L252 公式 `K=B/(N·L)` 与 L509 一致，但 L382 的 450M/112.5M 破坏一致性。
3. L313 "53 人日"应为 54（逐项加和）。
4. L441 "15 等宽箱 + 等频箱"与 L113 "10 等宽 bin"统一（见 M3）。
5. L384 pilot 的 `{25M,50M,100M}` 与"平台 80%"判据：应改为对 K 的校准（如 K∈{600,900,1221,1600}）。
6. L581 R2 写"1–3mm 插入"但 S7 是 2mm；改 2–3mm。
7. L584 R5 "ManiSkill3 env"→Playground。
8. L677/L702 "1–3mm 容差"与 C4 冲突，统一 2–3mm。
9. L715 D3 "1mm/2mm"与主容差统一（主 2–3mm，1mm 仅 stretch）。
10. L65 摘要 213 词偏长（CoRL 摘要一般 ≤200），压到 190 词内并改 Aero 表述。
11. L131 "预注册 W8 冻结"与 W8 才做 Go/No-Go 冲突：预注册应在 W6（协议/假设/分析计划）冻结，W8 只冻结实现细节。
12. L824 预算脚本输出 `+50%: ¥6999` 与正文一致，但未含 C4 修正；脚本要加"吞吐情景"参数。
13. L591 R12 "novelty 与 SGS 重叠"只给"强调校准与组合"一句话，缺"若审稿人仍说重叠"的备选（把 SGS-band 作为机制对照、把贡献移到 Prop 1）。
14. 附录 A 引用格式不统一（有的给 arXiv、有的给 venue）；RPLR 已明确不引，但差异表仍出现 "PLR/RED" 需给全名。
15. L836 `eval/init_states.json` 需加"生成 seed 与哈希"，否则不可复现。

---

## 5. 独立复算（v1 的数字我逐项重算）

### 5.1 协议复算：K=1221 对应的 B 与 fixed-B 对应的 K

`K=1221, L=50`：

| N | B=K·N·L | fixed-B=125M 时的 K | v1 的 fixed-B=112.5M 时 K |
|---|---|---|---|
| 256 | 15.6M | 9,766 | 8,789 |
| 1024 | 62.5M | 2,441 | 2,197 |
| 2048 | **125.0M** | 1,221 | 1,099 |
| 8192 | **500.1M** | 305 | 275 |
| 16384 | 1,000.2M | 153 | 137 |

结论：v1 的 `N=2048→112.5M`、`N=8192→450M`、H1 的 `700M`、fixed-B 的 `112.5M` 全部与 `K=1221` 不相容；只有 L252/L509 的 500M/125M 是自洽的。**H1 主格必须改成 500M**，否则"主假设"没有对应 run。

### 5.2 预算逐行复算

v1 §7.2 行项（单价 ¥7/h，overhead 0.25h/run，12% 缓冲，杂项 550）：

| 行 | runs | h/run | GPU-h | ¥ | 我反推的吞吐 |
|---|---|---|---|---|---|
| P0 | 1 | 24.00 | 24.0 | 168 | — |
| G1 | 20 | 8.58 | 171.6 | 1,201 | 15,006 steps/s |
| G2 | 32 | 2.33 | 74.6 | 522 | 15,024 steps/s |
| G3 | 9 | 8.58 | 77.2 | 541 | 15,006 |
| G4 | 3 | 8.58 | 25.7 | 180 | 15,006 |
| G5 | 12 | 2.33 | 28.0 | 196 | 15,024 |
| G6 | 3 | 17.20 | 51.6 | 361 | 14,995 |
| G7 | 3 | 9.20 | 27.6 | 193 | 14,991 |
| G8 | 20 | 2.33 | 46.6 | 326 | 15,024 |
| G9 | 6 | 0.92 | 5.5 | 39 | 24,876 |
| G10 | 1 | 10.00 | 10.0 | 70 | — |
| G11 | 1 | 6.00 | 6.0 | 42 | — |
| **合计** | **111** | | **548.4** | **3,839** | |

`548.4×7=3,838.8`；`×1.12=4,299`；`+550=4,849`；余 ¥151。**行项加总正确**，但：
1. **所有行隐含 15,000 steps/s**，与 §7.1 的"N=2048 6,000 steps/s"矛盾（G2/G5/G8 若按 6k 应为 5.46h/run）。
2. 若按 6k：G2=174.7h、G8=109.2h，总计 **711.1h → ¥6,125**，超预算 ¥1,125。
3. 若按折中 12k：G2=100.5h、G8=62.8h，总计 **590.5h → ¥5,180**，仍超。
4. 若把 H1 的 700M 当真：G1 每 run=700M/15k/3600+0.25=13.21h，20 runs=264.2h（+92.6h），总 **641.0h；`641.0×7×1.12+550=¥5,575`**。
**所以：v1 的 ¥4,849 只在"15k steps/s + H1 实际跑 450M（不是 700M）"两个未声明假设同时成立时才对。**

### 5.3 ±50%、削减链、双因素、FLOOR 复算

- 基准 4,849；价格 +50%：`548.4×10.5×1.12+550=6,999` ✓（v1 对）。
- 时长 +50%：`548.4×1.5×7×1.12+550=6,999` ✓。
- 双因素（价×1.5、时长×1.5）：`548.4×2.25×7×1.12+550=10,222` ✓（v1 对）。
- 削减（v1 的 C1–C5，**未乘缓冲**）：C1 −811 ✓、C2 −361 ✓、C3 −294 ✓、**C4 −517 ✗（应为 −360）**、C5 −98 ✓；累计 `6,999−1,924=5,075`，**不闭合**；v1 写 4,987 是 C4 高估 ¥157 的结果。
- 若削减也乘 1.12：−2,155 → 4,844，能闭合。**两种口径结论相反，v1 没写用哪种**；必须写死"削减按含缓冲成本计"。
- C4 的 power 说明也错：5→4 seeds、sd=6pp，MDE80=8.4pp、power(5pp)≈0.63，不是"0.85 / MDE 6pp"。
- FLOOR 126.8h = G1 77.2 + G2 21.0 + G8 18.6 + P0 6 + 评测 4（我反推），但 v1 没写 G2 是 9 runs（3 方法×3 seeds）而不是主格 32 runs；且 P0 已含 G1/G2/G8，FLOOR 又列 P0，口径重复。费用 `126.8×7=888+107+350=1,345` ✓ 算术对、构成不对。

### 5.4 统计功效复算

**H1（seed 为重复单元，sd=6pp）**：

| seeds | SE | MDE80（5% 双侧） | power(Δ=5pp) |
|---|---|---|---|
| 5 | 2.68pp | **7.52pp** | **0.36** |
| 6 | 2.45pp | 6.86pp | 0.44 |
| 8 | 2.12pp | **5.94pp** | 0.60 |
| 10 | 1.90pp | 5.32pp | 0.72 |

若 sd=4pp：5 seeds MDE 5.01pp、8 seeds 3.96pp；若 sd=8pp：8 seeds 7.92pp。**v1 的"SE 1.2pp / power>0.95"是把 8 个技能当 8 个独立重复（√8 虚增），不能这么算。** 结论：H1 要 5pp 成立，必须 8 seeds 且实测 sd≤5pp；否则阈值改 6pp（8 seeds）或 7pp（6 seeds）。H1b/H1c 见 C3。

**H3（McNemar，p1=.60, p3=.45）**：

| n（配对 episode） | ρ=0 | ρ=0.3 | ρ=0.5 |
|---|---|---|---|
| 120（30 trials/task×4） | 0.59 | 0.67 | 0.76 |
| 200（50 trials/task×4） | 0.83 | 0.90 | 0.95 |

**H3 的 15pp 在 120 对下 power 只有 0.59–0.76**，且 4 任务分别检验会再降；必须 50 trials/task 或把阈值降到 12pp（并报告 MDE）。

**H2（ECE）**：n=300/skill 时 ECE 估计量 bootstrap 半宽约 0.015–0.02（10 bin）；相对 −30% 若基线 ECE=0.10 → 差 0.03 可检；若基线 0.05 → 差 0.015 与噪声同量级。v1 没给基线 ECE 预期，也没给 MDE——必须补，并把 n 提到 500。

### 5.5 技能人日与工程量复算

v1 技能表逐项：S1 3 + S2 2 + S3 8 + S4 8 + S5 6 + S6 6 + S7 10 + S8 10 + A1 1 = **54 人日**（v1 写 53）。加 arm+LEAP 10–15 = 64–69；×1.2 缓冲 = **77–83 人日**；再加 harness/API/calibration/planner/agent 代码 15–20 = **92–103 人日 = 18.4–20.6 周**（5 人日/周）。v1 §3.9 说"工程量 20 周"其实对；§4.2 的"13.5–14 周"少算 5–6 周。**CoRL 30 周窗口里，工程 18–20 周 + 实验 8 周 + 写作 5 周 = 31–33 周，超窗口 1–3 周**，所以 v1 的排期是负缓冲。

### 5.6 LLM 成本复算（7.2M tokens）

| 输入/输出占比 | $2/M in, $8/M out | $3/M in, $12/M out | $5/M in, $15/M out |
|---|---|---|---|
| 70/30 | ¥197 | ¥295 | ¥415 |
| 60/40 | ¥228 | ¥342 | ¥467 |
| +30% 重试/长 prompt（70/30） | ¥256 | ¥384 | ¥539 |

**¥250 只在"70/30 + 最低价 + 无重试"时成立**，是乐观下限。若 8 calls 实际 4k tokens/次，成本翻 1.6×。**必须给 ¥600 硬顶 + 逐 episode token 日志**。

### 5.7 平台可行性
- 本地 4070 Ti Super 16GB：8 技能共享策略（23-DoF 动作、N=1024、state-based）显存约 6–9GB，**可能**跑得动，但未实测；arm+LEAP 接触（N=512）显存与稳定性未知。v1 把"本地 N≤1024/arm N≤512"写成假设（L364），没有 W8 实测前的保底。
- Playground 的 `aloha/single_peg_insertion.py` 是 `impl="warp"`、`naconmax=24*1024`、400Hz；如果 arm+LEAP 也用 400Hz 物理 + 20Hz 决策（20 substeps），吞吐会显著低于手内任务——v1 的 15k 混合吞吐是乐观值（见 C8）。
- W8 Go/No-Go 只测 4 个手内技能，**没有测"混合库（含 arm）"的显存/吞吐**，这是预算与排期的单点风险。

---

## 6. 公平性与基线审计（逐条）

| 基线 | v1 设置 | 公平性问题 | 修法 |
|---|---|---|---|
| uniform | 主格 4 方法之一，5/8 seeds | 与 FG 同 seeds ✓ | 保持 |
| SER（2511.21011） | 重实现，桶级 staggered reset | 未给 reset 周期/相位超参；不同 reset 策略可能改变样本效率 | 给参数表 + 在 `LeapCubeReorient` 复现原趋势 |
| SGS-band（2610.12465） | 成功档位加权 reset | 代码未开源，重实现保真度未验证；v1 有 G-S1 判据 ✓ | 加"与原文一致的 task-config 语义"说明；G-S1 给数字 |
| PLR/RED | 只在差异表出现 | **主实验缺席**；审稿人会问"为什么不比 regret 采样" | 加 PLR-style（regret 权重）3 seeds，或明确"不同问题设定" |
| OmniReset-style（2603.15789） | "风格化" | 无定义；OmniReset 是单一 reward+reset 设计，不是分配方法 | 定义为"uniform 权重 + 多样化 reset 区域"，说明这是 reset 多样性而非分配基线 |
| P2P-style（2606.26428） | 等总步数 600M + 完整版 1.08B 计价 | **主表只跑等步数版，却按完整版计价**；两阶段 play+finetune 的"分配"语义不同 | 主表用 600M 等预算、3 seeds；完整版只作附录 1 seed；说明两阶段不是同一分配问题 |
| ADEPT-style（2608.19182） | 未提 | post-training 范式，需示范/真机 | related work 明确"不同范式，不比较" |
| oracle-LP / random-skills | 有 ✓ | oracle 需知道真 p，只能近似 | 用"训练后 held-out 评测"的 p 作 oracle，说明是上界 |

**固定 K 的公平性**：同 N 下所有方法 K 相同 ✓；但 **不同技能的控制率/每步成本不同**（C8），所以"每步成功率"不是算力归一；主 LP 应改 `Δp/c_g`。**fixed-B 次协议**：B 必须统一为 125M（不是 112.5M），并写明 N=8192 时 K=305（少更新是设计意图，不是 bug）。**seeds**：主格 4 方法必须同 seeds（8），基线 3 seeds 只能进附录；v1 的"5 vs 3"会被直接质疑。

---

## 7. Novelty 与近邻复核（CoRL/ICLR 口径）

**威胁排序（高→低）**：SGS 2610.12465（高：同层 success-rate 分配，2026-10，且明确指出"uniform 浪费在已掌握/学不会的配置上"——这正是 FG 的动机）→ OmniReset 2603.15789（高：diverse resets + 大规模，真机零样本；它把"多样性"做成 scaling 故事）→ P2P 2606.26428（高：灵巧手精密装配的 play→finetune，且 limitation 明确留了"sequencing/composition"缺口）→ ADEPT 2608.19182（中高：post-training + 23-DoF 真机，规模碾压）→ Being-0 2503.12533（中：系统/agent 组合，但非技能库级分配）→ SER 2511.21011（中：reset 调度，与分配同层但更窄）→ 2606.22471（中：RL 多任务数据生成，偏数据侧）→ DexSim2Real / Text2Touch（低）。

**v1 的差异化是否够？** 部分够：P2P 的 limitation 原文明确说"learns short-horizon skills; task sequencing... specified externally; future work could combine these skills with sequencing"——这是 v1 最硬的 gap 证据（应引用原话，L? 附录 A 引了 P2P 但正文没用这句话）。**不够的部分**：SGS 已经做了"success-rate 加权 reset + 标准 PPO"，v1 的 Level-1 `(p_fast−p_slow)(1−p_slow)+λσ̂` 与 SGS 的三档加权在机制上非常近；若审稿人认定"FG = SGS + 导数 + guards"，novelty 只剩 calibration + composition。**所以 Prop 1 与 held-out 校准是必须的差异化载体**，否则 CoRL novelty 分会掉到 2。v1 的 H5（Aero 迁移）和 H7（API 粒度）都不能承担 novelty。

**v2 必须补的差异化陈述**：①SGS 是"按成功率加权、单任务配置级"，FG 是"按学习进度导数 × 估计量不确定性、技能库级、且输出 per-skill 校准区间"；②SGS 的成功率在 p≈0.5 高 aleatoric 噪声桶上会把噪声当进步（可证伪预测：SGS 在噪声桶的权重方差大于 FG）；③**Prop 1 给出"何时 success-band 足够（margin 大）、何时必须校准（margin 小）"**——这把 SGS 与 FG 的关系从"竞争"变成"适用域划分"，是最强的 related-work 回应。

---

## 8. Brief 14 项清单勾选（v1 状态）

| # | 项 | 状态 | 备注 |
|---|---|---|---|
| 1 | 真实机器人相关性/任务 | ✓ | 但 Aloha peg 精度表述错误（C4） |
| 2 | 技能库 + 可调用 API | ✓ | schema 完整 |
| 3 | LLM agent 组合 | ✓ | 2×2 设计好，但功效不足（C3）、成本低估（C9） |
| 4 | 大规模并行 RL | ✓ | N=8192 但 H1 预算不一致（C1） |
| 5 | 分配/采样方法 | ✓ | 与 SGS 差异需强化（§7） |
| 6 | 校准 | ✓ | bin 不一致、分布内（C7/M3） |
| 7 | 理论 | △ | P3 只有直觉（C6） |
| 8 | 统计规范 | △ | seeds/功效/MDE 不足（C3） |
| 9 | 消融 | ✓ | 10 个，但 2 seeds |
| 10 | 预算逐行 + ±50% | △ | 吞吐矛盾、C4 算错（C2） |
| 11 | 排期 | ✗ | 少算 5–6 周、零缓冲（C5） |
| 12 | 风险与 Plan B | ✓ | R1–R15 全，但 R5 平台写错 |
| 13 | 近邻差异表 | ✓ | 10 行，SGS 差异不够硬（§7） |
| 14 | 复现/artifact | △ | 缺版本/哈希/LLM 快照（M18） |

---

## 9. v2 修改对照（critique → v2 位置）

| 编号 | v2 必须落点 |
|---|---|
| C1 | §2.2/§2.6/§3.7/§5.2/§5.3/§7.1 统一协议表；H1 改 B=500M；pilot 校准 K |
| C2 | §6 预算全表重算；吞吐表统一；削减链含缓冲口径；FLOOR 重建；留 ≥¥250 |
| C3 | §2.6 预注册 MDE 表；主格 8 seeds；H1≥6pp；agent 50 trials/task；H3≥12pp；H2 MDE |
| C4 | §4.1/§4.2/§5.1/§9/§11 改 2–3mm；Tier-0 只验证流水线；S8 改 Cap-Twist |
| C5 | §8 排期重写（W4/W10/W16 Go/No-Go；6 核心技能；W24–26 冻结；W26–31 写作） |
| C6 | §4.4 新增 Prop 1 + Corollary + TP1–TP3 + 附录证明 |
| C7 | §5.5/§6.4/§6.5 加 goal split；held-out 校准；agent 用 held-out 目标 |
| C8 | §3.1/§3.2/§3.6 统一 20Hz 决策；compute-normalized LP；协议 C |
| C9 | §6.7/§7.3 LLM 分算表 + ¥600 硬顶 + 快照/温度/日志；D5 改 API 主方案 |
| C7+（après Lead 复审） | §2.3 校准机制声明（B=500M 采样噪声 <0.1pp，机制只能是转移偏差/非平稳）；§3.5 预测器只用训练期统计（EMA/value head），250/250 split，(skill×bucket) 24 cell ECE + per-seed 方差；§2.2 H1 改共同主终点 H1a 平均/H1b min-skill 非劣 δ=5pp/H1c utility-weighted/H1d scaling/H1e 等价；§2.6 pilot 时点改 W4–W6 + 阈值规则（sd≤5→6pp，5–7→7pp，>7→降 N=8192） |
| M1–M20 | 见 v2 §2.4/§3.8/§5.4/§5.6/§6/§8/§11/§13 与附录 A/C |

---

## 10. 结论

v1 的**骨架是对的**：定位清楚、贡献分层、有负结果分支、有 artifact 意识。它的问题不是"想法不行"，而是**证据链在四个关键处断裂**：协议（C1）、预算（C2）、功效（C3）、可行性（C4/C5）；再加理论缺口（C6）与分布内校准（C7）。这六条正好对应 ML rubric 的"实验严谨 0.30 + 技术正确 0.20"——修不掉就必然 reject，修掉之后 CoRL 有机会到 30%。

**v2 的最小可行形态**（我已按此写 `13_proposal_dexterous_v2.md`）：
- 主格：fixed-K `K=1221`、N∈{2048,8192}、4 方法、**8 seeds**（64 runs，404.8h，¥2,834）；
- 主终点：H1 Δ≥6pp（MDE 5.9pp），H1b 描述性，H1c TOST margin 2pp；
- 预算：**¥4,738**（508.6h + 12% + 杂项 750），余 ¥262；价格 +50% 经 4 级削减到 ¥4,934；双因素触发 FLOOR ¥4,742；
- 理论：Prop 1 + Corollary + 3 个可检验预测（这是把 CoRL 从 22% 推到 30% 的唯一高 ROI 杠杆）；
- 排期：6 核心技能（S5/S6 stretch）、W4/W10/W16 三个 Go/No-Go、W24–26 冻结、W26–31 写作；
- 校准：held-out 目标 split + 10 等质量 bin + n=500 + 分布偏移；
- agent：50 trials/task、2×2（校准×分配）+ scripted planner、¥600 硬顶；
- 公平性：compute-normalized LP + fixed-K/fixed-B/固定墙钟三协议、基线 fidelity 表。
