# Lead 种子候选（不替代 lens 输出，供汇总对照）

## 候选 A：Compositional Calibration & Allocation（CCA）——组合校准的形式化 + 分配定理

**一句话**：把"技能库"重新定义为一个**面向组合任务的统计估计对象**；证明 per-skill 平均校准最优 ≠ 组合风险最优，给出组合校准误差（CCE）的上下界与"规划敏感度加权"的最优分配，并用 4–6 个 LEAP 技能 + 规划器验证。

**形式化**：
- 技能 k 在上下文 x 上成功率 p_k(x)，训练后估计 p̂_k(x)；分配预算 B_k。
- 组合任务 = 技能序列 π（含重试）；真实成功率 q(π)=E[∏ p_k]、规划器用 q̂(π)=E[∏ p̂_k]。
- 定义 CCE = sup_π |q(π) − q̂(π)|；以及 planner regret = q(π*) − q(π̂*)。
- 命题 1（上界）：CCE ≤ Σ_k w_k ε_k（w_k 为技能在最优计划中的边际使用权重，ε_k 为 TV 校准误差）；朴素 union bound H·max ε 是松的。
- 命题 2（下界/不可能性）：存在技能库使任何"只保证平均 ECE ≤ ε"的估计器在组合 regret 上达 Ω(√H·ε) 或 Ω(Hε)，即平均校准不能控制组合风险。
- 命题 3（最优分配）：min Σ w_k ε_k(B_k) s.t. Σ B_k = B 的解 n_k* ∝ √(w_k v_k / c_k)（Neyman 型，但 w_k 是**规划敏感度**而非 uniform 技能权重）；并证明 uniform/技能级 Neyman 在 w_k 异质时次优 O(√K) 或 O(K)。
- 算法 CCA-LGA：在线估计 w_k（技能在规划中被选中/成为瓶颈的频率），按 w_k 加权分配；校准 head 输出区间；guard 防饿死。
- 可证伪预测：P1 分配按 w_k 后组合成功率提升 > 按 per-skill ECE 的提升（解耦）；P2 在 w_k 均匀时两者重合（阴性对照）；P3 composition regret 随 K 的 scaling 斜率由 w_k 分布决定；P4 风险厌恶规划器（CVaR）下校准误差权重更大。

**实验**：现有 4 技能（S1/S2/S7/S8）+ 2 长程任务；加"pivotal skill"日志（几乎零成本）；主格 4 方法（uniform/SGS/LGA-per-skill/CCA）× 8 seeds × 2 N；加合成 Bernoulli 环境（CPU，秒级）验证下界与分配相变；LLM planner 只作一个实例（另有 scripted planner），避免"prompt 工程"质疑。
**成本**：与现计划同量级（≤90 租卡 runs，≤¥4,500），工程 +2–3 周（w_k 估计 + 合成实验），无需新技能。
**S 增量估计**：N +0.5（新对象/不可能性）、TQ +0.35、Insight +0.55（解耦/相变）、Exp +0.05、C +0.05、Rep +0.05 → S≈4.0–4.1。
**venue**：ICLR 2028 主投（Sep 2027）/ NeurIPS 2027 main（May 2027，紧）；CoRL 2027 可作机器人版。
**风险**：定理若太弱会被说"trivial union bound"；需真给出紧界或相变。w_k 的可识别性是关键。

## 候选 B：The Compositional Calibration Gap——负结果 + 测量协议

**一句话**：首个系统性证据表明"per-skill 校准误差下降"并不自动转化为"LLM/规划器组合成功率上升"，并给出何时必须转化的判据（planner sensitivity）与评测协议（decision-weighted CCE + CRN）。
**形式化**：定义 CCE 与 planner sensitivity；证明两个 regime（planner 用 p̂ vs 不用）的分界；给出可检验判据。
**实验**：4 技能 × 4 分配方法 × 8 seeds；agent 2×2（用/不用 p̂）× 剂量；CRN 16 cell×1,000。
**成本**：与现计划相同（负结果版可砍部分技能）。
**S 增量**：N +0.35、Insight +0.5、TQ +0.2、Exp +0.1 → S≈3.9–4.0；但负结果有"故事不够 positive"的风险。
**venue**：CoRL 2027（测量版）/ NeurIPS D&B。

## 候选 C：Skill-Library Scaling Laws（补充）
用 6–8 个技能、每技能 5 个预算点拟合 p_k(B) 饱和曲线，推导 compute-optimal allocation，holding out 1 技能做外推检验；证明 uniform 在固定 B 下次优 X%。成本 +20–30 租卡 runs；S≈3.85–3.95；适合作为 A 的补充实验而非主贡献（K 太小）。

## Lead 倾向
A 为主线（新对象 + 不可能性 + 分配定理 + 解耦实验），B 作为 A 的"如果 H2/H3 不如预期"的叙事备份，C 作为 A 的一个实验节（scaling）。三者共享现有 4 技能与预算，边际成本低。
