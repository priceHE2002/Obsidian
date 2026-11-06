# 理论备忘录：per-skill 校准误差 → 长程组合成功率的传播界

> 用途：作为 CoRL/ICLR 版论文的正式理论小节（正文放命题 3 + 表 1；命题 1/1b/2 的完整推导放附录）。所有界都写成"可验证的弱命题"，不声称收敛性或新算法；常数取保守值，宁可松但正确。
> 与 `11_proposal_dexterous_v1.md` §4.5 的关系：本文修正了那里的 P1 直觉（见 §2.3），并以本备忘录为准。

## 1. 设定与记号

- 技能库 $\mathcal{G}=\{1,\dots,S\}$；技能 $g$ 的桶（bucket）集合 $\mathcal{B}_g$，$|\mathcal{B}_g|=M_g$，总桶数 $M=\sum_g M_g$。桶 = 目标区域 × 物体/DR 配置（见 v1 §4.2）。
- 桶 $b$ 的真实成功率 $p_b\in[0,1]$；训练期从桶 $b$ 采到 $n_b$ 个 episode，估计
  $$\hat p_b=\frac{1}{n_b}\sum_{i=1}^{n_b}X_{b,i},\qquad X_{b,i}\sim\mathrm{Bern}(p_b)\ \text{i.i.d.}$$
- 评测分布权重 $w_b\ge0,\ \sum_{b\in\mathcal{B}_g}w_b=1$（held-out 评测时桶出现频率，**不是**训练分配频率）。技能级真值与估计：
  $$p_g=\sum_{b}w_b\,p_b,\qquad \hat p_g=\sum_b w_b\,\hat p_b.$$
- 校准误差（per skill）：$\varepsilon_g=|\hat p_g-p_g|$；ECE 版本 $\mathrm{ECE}_g=\sum_b w_b|\hat p_b-p_b|$。
- 决策相关误差：$\varepsilon_{\mathrm{dec}}=\sum_b w_b^{\mathrm{dec}}|\hat p_b-p_b|$，其中 $w^{\mathrm{dec}}$ 是"规划时会被查询的桶"的权重（前沿桶）。
- 总训练预算 $B$（每个技能的环境步数；跨技能按 v1 §3.2 归一化后再分配）。

**假设 A1（桶内 i.i.d. 与平稳）**：同一桶内的 episode 独立同分布，且训练窗口内 $p_b$ 不变。A1 在真实训练中**不严格成立**（策略在变），所以 §2.2 给出非平稳修正；命题 1 的作用是给出"采样噪声本身有多小"，从而说明观测到的校准差主要来自哪里。

## 2. 命题 1：uniform 分配的 per-bucket 校准误差

### 2.1 采样噪声界（在 A1 下）

uniform 分配 $n_b=B/M$。对任意 $\delta\in(0,1)$，Hoeffding + union bound：

$$\Pr\Big[\max_{b}|\hat p_b-p_b|\ge \epsilon\Big]\le 2M\exp\!\Big(-\frac{2B\epsilon^2}{M}\Big).$$

令右端 $=\delta$，得**高概率界**：以概率 $\ge1-\delta$，

$$\boxed{\ \max_b|\hat p_b-p_b|\le \sqrt{\frac{M\log(2M/\delta)}{2B}}\ } \tag{1}$$

均值版本：由 $E|X-EX|\le\sqrt{\mathrm{Var}(X)}$ 与 $p_b(1-p_b)\le1/4$，

$$\mathbb{E}[\mathrm{ECE}_g]\le \sqrt{\frac{2}{\pi}}\cdot\frac12\sqrt{\frac{M}{B}}\approx0.399\sqrt{\frac{M}{B}}. \tag{2}$$

技能级加权估计的方差（比逐桶更松）：

$$\mathrm{Var}(\hat p_g)=\frac{M}{B}\sum_b w_b^2p_b(1-p_b)\le\frac{M}{4B}\sum_b w_b^2. \tag{3}$$

若评测在 $M$ 个桶上等权（$w_b=1/M$），则 $\sum w_b^2=1/M$，$\mathrm{SD}(\hat p_g)\le\frac{1}{2\sqrt B}$——**与 $M$ 无关**。

### 2.2 数字与推论（关键）

取 v1 的主格 $B=112.5\text{M}$、$\delta=0.05$，式 (1)(2) 的数值如下：

| $M$（每技能桶数） | 式 (1) 高概率界 $\max_b|\hat p_b-p_b|$ | 式 (2) 期望 ECE 上界 |
|---|---|---|
| 4 | $3.0\times10^{-4}$ | $8.0\times10^{-5}$ |
| 8 | $4.5\times10^{-4}$ | $1.1\times10^{-4}$ |
| 16 | $6.8\times10^{-4}$ | $1.6\times10^{-4}$ |

即：**在这些预算下，纯采样噪声比我们在意的 pp 级校准差小两个数量级**。因此：

- **观测到的 ECE 差（若有）不来自"uniform 采样太少"，而来自**（a）评测分布与训练分布不一致（$w_b$ 用错），（b）非平稳（$p_b$ 在训练中变化），（c）用训练期成功率冒充 held-out 成功率（bias），（d）FG 主动"饿死"低优先级桶造成的**分配偏斜**。这四点必须分开报告，否则"FG 更校准"可能是伪影。
- 这也是 v1 §5.8（域内自然噪声审计）与 A9（后处理校准消融）的理论依据。

### 2.3 对 P1 直觉的修正

v1 §4.5 曾写"uniform ECE 随 $M$ 增长 $\propto c_1\sqrt{M_{\mathrm{front}}/B}$"。按上式，正确形式是

$$\text{per-bucket: }\ \mathbb{E}[\mathrm{ECE}]\le0.399\sqrt{M/B},\qquad \text{aggregate skill: }\ \mathrm{SD}(\hat p_g)\le\frac{1}{2\sqrt B}\sqrt{M\textstyle\sum_bw_b^2},$$

前者随**总桶数 $M$** 增长（稀释），后者在等权时与 $M$ 无关。$M_{\mathrm{front}}$ 只在"预算只花在前沿桶、每个前沿桶分到 $B/M_{\mathrm{front}}$"的**另一种分配**下才出现，不是 uniform 的行为。v1 正文与图注统一用本节的 $\sqrt{M/B}$ 表述。

### 2.4 非平稳修正（命题 1b）：为什么 LP 导数与不确定性两个信号都要

去掉 A1 的平稳假设，设桶 $b$ 的成功率以速率 $\dot p_b$ 线性漂移。用最近 $\tau_b$ 个样本（窗口长度）估计，则

$$\mathrm{MSE}_b(\tau_b)\le\underbrace{(\tau_b\dot p_b)^2}_{\text{滞后偏差}}+\underbrace{\frac{p_b(1-p_b)}{\tau_b}}_{\text{采样方差}}. \tag{4}$$

对 $\tau_b$ 最小化：$\tau_b^*=\big(p_b(1-p_b)/(2\dot p_b^2)\big)^{1/3}$，且

$$\mathrm{MSE}_b^*\le1.89\,\big(p_b(1-p_b)\dot p_b^2\big)^{1/3}. \tag{5}$$

**含义**：最优窗口由"漂移速度 $\dot p_b$"与"噪声水平 $p_b(1-p_b)$"共同决定——这正是 FG 两个信号的经验对应物：LP 导数 $\widehat{\dot p}_b$ 估计趋势（偏差项），bootstrap EMA 离散度 $\hat\sigma_b$ 估计噪声（方差项）。因此命题 1b 不是"FG 最优"的证明，而是解释**为什么两个信号都不可省**（v1 消融 A1 去导数、A4 去不确定性分别对应式 (4) 的两项）。可检验预测：漂移快的桶用更短窗口更准；固定窗口的消融应在异质桶上出现系统性偏差。

## 3. 命题 2：最小化校准误差的最优分配与 FG 的近似

### 3.1 L2 准则（加权方差）：Neyman 型

$$\min_{\{n_b\}}\ \sum_b w_b\frac{p_b(1-p_b)}{n_b}\quad\text{s.t.}\ \sum_b n_b=B.$$

拉格朗日条件 $-\,w_bp_b(1-p_b)/n_b^2+\lambda=0$ 给出

$$n_b^\*=\frac{B\sqrt{w_bp_b(1-p_b)}}{\sum_{b'}\sqrt{w_{b'}p_{b'}(1-p_{b'})}}. \tag{6}$$

最优方差 $\frac1B\big(\sum_b\sqrt{w_bp_b(1-p_b)}\big)^2$。由 Cauchy–Schwarz，uniform（$n_b=B/M$）的方差满足

$$\frac{\text{uniform}}{\text{optimal}}=\frac{M\sum_bw_bp_b(1-p_b)}{\big(\sum_b\sqrt{w_bp_b(1-p_b)}\big)^2}\ \ge1, \tag{7}$$

等号当且仅当 $w_bp_b(1-p_b)$ 为常数；异质时 uniform 可差至 $O(M)$ 倍。

### 3.2 L1 准则（ECE）：$2/3$ 次幂

规划真正用的是"每个桶/技能的绝对误差"，对应 L1。$\mathbb{E}|\hat p_b-p_b|\le\sqrt{2p_b(1-p_b)/(\pi n_b)}$，于是

$$\min_{\{n_b\}}\ \sum_b w_b\sqrt{\frac{2p_b(1-p_b)}{\pi n_b}}\quad\Rightarrow\quad n_b^\*\propto\big(w_b\sqrt{p_b(1-p_b)}\big)^{2/3}=w_b^{2/3}\,p_b(1-p_b)^{1/3}. \tag{8}$$

最优 ECE 为 $\sqrt{2/\pi}\,B^{-1/2}Z^{3/2}$，其中 $Z=\sum_b(w_b\sqrt{p_b(1-p_b)})^{2/3}$；uniform 为 $\sqrt{2/\pi}\,B^{-1/2}\sqrt{M}\,U$，$U=\sum_bw_b\sqrt{p_b(1-p_b)}$。两者之比

$$\frac{\mathrm{ECE}^\*}{\mathrm{ECE}_{\text{uniform}}}=\frac{Z^{3/2}}{\sqrt M\,U}\le1, \tag{9}$$

且当 $v_b=p_b(1-p_b)$ 全相等时取等号（此时"怎么分都一样"）。**这给出 H2c（FG≈SGS 在低 S/低 N 下等价）的理论解释**：桶越同质、权重越均匀，最优分配越接近均匀，任何"聪明分配"的收益都趋近于 0；FG 的收益应随 $v_b$ 与 $w_b$ 的异质性增长（可检验预测 P2）。

### 3.3 FG 是"非平稳、未知 p"下的自适应近似（不声称等价）

式 (6)/(8) 需要真实 $p_b$ 与 $w_b$，二者都未知且随训练变化。FG 用 $\hat p_b$（EMA）、$\widehat{\dot p}_b$（LP 导数）、$\hat\sigma_b$（bootstrap EMA 离散度）构造分数 $a_b=\widehat{\dot p}_b/\hat\sigma_b$（或 v1 的 $z$ 分数），再经 Hedge 更新权重。它与最优分配的对应关系是**定性**的：$\hat\sigma_b$ 近似 $\sqrt{p_b(1-p_b)/n_b}$（式 (8) 的分子），$\widehat{\dot p}_b$ 近似前沿/非平稳修正（式 (4)）。**不声称** FG 达到式 (6)/(8) 的最优值，只声称"它把样本从低 $v_b$/低 $w_b$ 桶移向前沿桶，方向与最优解一致"，并用 oracle 分配（用真实 $p_b$ 算式 (8)）测量 FG 离最优有多远（v1 §5.4 的 oracle-LP 参照）。

### 3.3b 一个必须写进论文的 caveat：FG 可能让"低权重桶"更差

式 (8) 的最优分配会**主动饿死**低 $w_b$ 桶。若评测 ECE 在全部桶上等权（$w_b=1/M$ 而非评测分布权重），FG 的 ECE 可能**比 uniform 更差**（被饿死桶的 $n_b$ 小、$\varepsilon_b$ 大），尽管它在"决策相关桶"上更好。这不是反例，而是度量口径问题：

- 论文主表报告 **decision-weighted ECE**（$w_b^{\mathrm{dec}}$ = 评测时被查询的桶频率）与 **frontier ECE**（只在前沿桶上平均），并把 uniform-weight ECE 放附录作为"最坏口径"；
- 可检验预测 **P2b**：FG 在 $w^{\mathrm{dec}}$ 口径下 ECE 更低，但在 uniform 口径下可能持平或略差；若两个口径方向一致，说明 FG 没有饿死重要桶（guard G-1/G-3 生效）；
- 这也是 v1 §5.8 噪声审计要区分"评测分布"与"训练分布"的原因：用错权重会让结论完全反转。

### 3.4 命题 2b：Hedge 的对抗性 regret（标准结论，不新）

设 $M$ 个桶、$T$ 轮，每轮损失 $\ell_{b,t}\in[0,1]$。Hedge/EXP3 的静态 regret 为 $O(\sqrt{T\log M})$；非平稳情形用 EXP3.S / 折扣 Hedge 得 $O(\sqrt{TM\log M})$（标准结果，见 Auer et al. 2002；Cesa-Bianchi & Lugosi 2006）。**引用即可，不作为本文贡献**；它的作用是说明"分配器在桶间切换的代价是有界的"，从而把 §2–§3 的统计误差与分配器本身的 regret 解耦。

## 4. 命题 3（核心）：校准误差 → 长程组合成功率

### 4.1 规划规则（明确、可复现）

长程任务计划 $\pi=(s_1,\dots,s_H)$，$H$ 次技能调用；每次调用技能 $s_i$ 的真实成功率 $p_i$，估计 $\hat p_i$，校准误差 $\varepsilon_i=|\hat p_i-p_i|\le\varepsilon$。允许每技能最多 $R$ 次重试（v1 取 $R=2$）。定义

$$q_i=1-(1-p_i)^{R+1},\qquad \hat q_i=1-(1-\hat p_i)^{R+1}. \tag{10}$$

planner 在有限计划集 $\Pi$（$|\Pi|=N_\pi$，由前置条件过滤）上选

$$\hat\pi=\arg\max_{\pi\in\Pi}\ \hat S(\pi),\qquad \hat S(\pi)=\prod_{i=1}^{H}\hat q_i(\pi), \tag{11}$$

真值 $S(\pi)=\prod_i q_i(\pi)$。若没有计划满足阈值 $\tau$，退化为 $\arg\max\hat S$（v1 §6.3 的规则）。

### 4.2 引理

**引理 3.1（重试成功率对 $p$ 的 Lipschitz）**：$|q_i-\hat q_i|\le(R+1)\varepsilon_i$。
证明：$q(p)=1-(1-p)^{R+1}$，$q'(p)=(R+1)(1-p)^R\le R+1$，中值定理即得。∎

**引理 3.2（乘积扰动）**：若 $a_i,b_i\in[0,1]$，则 $|\prod_i a_i-\prod_i b_i|\le\sum_i|a_i-b_i|$。
证明：对 $H$ 归纳；$|\prod_{i\le H}a_i-\prod_{i\le H}b_i|\le|\prod_{i<H}a_i||a_H-b_H|+|b_H||\prod_{i<H}a_i-\prod_{i<H}b_i|\le|a_H-b_H|+\sum_{i<H}|a_i-b_i|$。∎

### 4.3 命题 3a（最坏情况：线性累积）

在 4.1 的规则与"每次调用误差 $\le\varepsilon$"下，对任意 $\pi\in\Pi$：

$$|S(\pi)-\hat S(\pi)|\le H(R+1)\varepsilon, \tag{12}$$

从而相对 oracle（$\pi^*=\arg\max_\pi S(\pi)$）的损失

$$\boxed{\ S(\pi^*)-S(\hat\pi)\le2H(R+1)\varepsilon\ } \tag{13}$$

证明：式 (12) 由引理 3.1 + 3.2；式 (13) 由 $S(\pi^*)-S(\hat\pi)\le[S(\pi^*)-\hat S(\pi^*)]+[\hat S(\pi^*)-\hat S(\hat\pi)]+[\hat S(\hat\pi)-S(\hat\pi)]$，中间项 $\le0$（$\hat\pi$ 最大化 $\hat S$），两端各 $\le H(R+1)\varepsilon$。∎

**数值**：$H=4,R=2$ 时损失 $\le12\varepsilon$；$\varepsilon=1\text{pp}\Rightarrow\le0.12$（12pp 绝对成功率），$\varepsilon=0.5\text{pp}\Rightarrow\le6\text{pp}$。界是**最坏情况**且偏松（当 $H(R+1)\varepsilon>1$ 时退化），但它给出两个可操作结论：（i）组合误差随 $H$ **线性**放大，故长程任务是校准误差的"放大器"；（ii）重试越多（$R$ 大），同样的校准误差被放大得越多——这解释了为什么"重试机制"既是鲁棒性来源，也是校准敏感性的来源。

### 4.4 命题 3b（独立误差：$\sqrt H$ 更紧）

设 $\hat p_i$ 相互独立、$\mathbb{E}[\hat p_i]=p_i$、$\mathrm{Var}(\hat p_i)\le\sigma_p^2$，且各技能估计独立（v1 的每技能独立评测满足；共享 trunk 不满足，见 4.5）。则

$$\mathrm{Var}(\hat S(\pi))\le S(\pi)^2\sum_i\Big(\frac{\partial S/\partial q_i}{S}\Big)^2\mathrm{Var}(\hat q_i)\le S^2\,\frac{H(R+1)^2\sigma_p^2}{q_{\min}^2}, \tag{14}$$

其中 $q_{\min}=\min_iq_i$。对 $N_\pi$ 个计划取 union bound，以概率 $\ge1-\delta$：

$$\max_{\pi}|\hat S(\pi)-S(\pi)|\le\sigma_S\sqrt{2\log(2N_\pi/\delta)},\qquad \sigma_S\le\frac{S(R+1)\sigma_p\sqrt H}{q_{\min}}, \tag{15}$$

故 $S(\pi^*)-S(\hat\pi)\le O\!\big(\varepsilon\sqrt{H\log N_\pi}\big)$，其中 $\varepsilon$ 取 $\sigma_p$ 量级。**与 3a 的差别**：最坏情况线性，独立误差时 $\sqrt H$；两者都随 $H$ 单调增，但斜率不同，可用"改变 $H$ 的消融"（v1 的 L1–L4 不同调用数）区分。

### 4.5 假设边界与"界不成立时数据里会看到什么"

| 假设 | 若不成立 | 数据中的信号 | 应对 |
|---|---|---|---|
| 失败可检测（成功判定无假阳性/假阴性） | 观测成功率偏离 $q_i$ | 记录的成功率与评测成功率系统性偏差 | 用独立评测器复核；报告检测器混淆矩阵 |
| 重试独立且不改变 $p_i$ | 重试成功率随次数下降 | 第 2/3 次重试成功率显著低于第 1 次 | 用衰减模型 $q=1-(1-p)(1-\rho)^{k}$ 修正；报告 $\rho$ |
| 计划集有限、前置条件准确 | planner 选了非法计划 | 前置条件违反率 >0 | 记录 violation rate；v1 §6.5 的失败归因 |
| 技能间误差独立 | 共享 trunk 导致相关误差 | 跨技能 $\varepsilon$ 相关系数高 | 用 3a 的线性界；报告相关系数 |
| $p_i$ 在评测期平稳 | 技能在评测中自适应 | 同一技能前后半段成功率漂移 | 分段估计；用 per-episode 估计 |
| planner 使用 $\hat p$ | P3（无校准）与 P1 无差异 | **注入噪声后成功率不变** | 这正是 RQ3 的负结果判据（见 §5 的 P4） |

### 4.6 重试衰减的修正模型（若假设不成立）

若重试成功率随次数衰减（第 $k$ 次尝试成功率为 $p\rho^{k-1}$，$\rho\in(0,1]$），则

$$q_i=1-(1-p_i)\prod_{k=2}^{R+1}\big(1-p_i\rho^{k-1}\big),$$

对 $p_i$ 的导数仍 $\le R+1$（每项 $(1-p\rho^{k-1})$ 的导数绝对值 $\le1$，乘积展开后总系数 $\le R+1$），故引理 3.1、命题 3a 的常数不变（仍 $\le H(R+1)\varepsilon$），只是数值更保守。若实测 $\rho<0.5$，说明"重试"不是独立重复，论文应改报"有效重试次数"$R_{\mathrm{eff}}=\sum_{k=1}^{R+1}\rho^{k-1}$，并用 $R_{\mathrm{eff}}$ 替换式 (13) 中的 $R+1$。这给出一个可检验量（P6）：从日志里估计 $\rho$ 并检查 $R_{\mathrm{eff}}$ 是否显著小于 $R+1$。

## 5. 可检验预测表（理论 → 实验 → 失败含义）

**识别性问题（写论文时必须回答）**：观测到的 ECE 差 $\Delta\mathrm{ECE}$ 可能来自四种来源——采样噪声、分布偏移、非平稳、分配偏斜。命题 1 说明第一种在 $B=112.5$M 下可忽略（$\lesssim0.1\%$），因此 $\Delta\mathrm{ECE}$ 的**主效应必须归因于后三种**。可操作的识别策略：(i) 用**固定评测集**（同一组 held-out 初始状态）消除评测噪声；(ii) 用**训练中期快照**（同一策略、不同训练步）估计非平稳项；(iii) 用**重加权评测**（把评测桶频率改成训练频率）分离分布偏移；(iv) 用 **oracle 分配**（真实 $p_b$）给出"分配偏斜可解释的上界"。四者缺一，审稿人就有理由说"FG 更校准"是度量伪影。

| 预测 | 内容 | 对应实验/假设 | 若预测失败说明什么 |
|---|---|---|---|
| **P1** | 采样噪声在 $B=112.5$M 下 $\lesssim0.1\%$；观测 ECE 差主要来自非平稳/分布偏移/分配偏斜 | H2、A9、v1 §5.8 噪声审计 | 若增加样本仍不能降低 ECE 差 → 确认非采样来源；若 ECE 差随样本消失 → 之前是噪声，H2 降级 |
| **P2** | FG 的分配比例与式 (8) 的 $(w_b\sqrt{v_b})^{2/3}$ 正相关（Spearman $\rho>0.6$），uniform 为常数 | A2/A3/A4、oracle 分配对照 | 若相关弱 → FG 的分数不是好的分配代理；A4（去不确定性）应更差，否则不确定性项无必要 |
| **P3** | 组合损失 $\le2H(R+1)\varepsilon$；$\varepsilon$ 减半，损失约减半（线性） | H6（P1 vs P3）、L1–L4 不同 $H$ | 若斜率远小于 $2(R+1)$ → planner 未用 $\hat p$（RQ3 负结果）；若远大于 → 存在未建模的失败传播 |
| **P4** | 对 $\hat p$ 注入 $\delta$ 噪声后，组合成功率下降约 $H(R+1)\delta$（最坏）或 $\sqrt H$ 型（独立） | H3c 噪声注入 | 成功率**不降** → planner 忽略 $\hat p$（直接反驳 H3）；下降过快 → 校准误差之外的因素主导 |
| **P5** | 桶越同质（$v_b$ 方差小）、权重越均匀，最优分配越接近 uniform，FG 收益趋 0 | H2c（FG≈SGS，S=4 vs 8） | 若 S=4 时 FG 仍显著优于 SGS → 存在与异质性无关的机制（需排查） |
| **P6** | 重试成功率随次数近似几何衰减（$\rho$ 可估） | H6 失败归因 | 若强衰减 → 式 (10) 需换模型，3a/3b 的常数变大（更保守） |

## 6. 与文献的关系（校准用于决策）

- **校准基础**：Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017（arXiv:1706.04599）——ECE/可靠性图的现代用法与"校准≠准确率"的区分。
- **决策校准 / 个体校准**：Zhao et al., *Individual Calibration with Randomized Forecasting*, ICML 2020（venue/年份已核实，**arXiv ID 待核实**）——"预测要按决策粒度校准"的思想，支撑本文"per-skill 校准而非 pooled 校准"的选择。
- **Conformal prediction（分布无关覆盖）**：Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, 2021/2023（arXiv:2107.07511）——给出"校准保证需要可交换性"的边界，是 §4.5 假设表的来源；本文不声称 conformal 覆盖，只借用其"假设必须显式"的写法。
- **采样分配**：Neyman allocation 的标准结论（Cochran, *Sampling Techniques*, 1977）；bandit regret 用 Auer et al. 2002（EXP3）与 Freund & Schapire 1997（Hedge），均为标准结果，本文只引用不重证。

> 引用规则：arXiv ID 未核实的一律标"待核实"；不引用无法核实的 RPLR 等条目。

## 7. 写作建议（放进 CoRL/ICLR 稿的方式）

- **正文（约 0.75 页）**：只放命题 3a/3b（式 (12)(13)(15)）+ 一张"理论→实验"表（本文表 1 的精简版）+ 一段"为什么两个信号都要"（命题 1b 的一句话版）。
- **附录（约 2 页）**：命题 1/1b/2 的完整推导、式 (7)(9) 的不等式证明、数值表、假设边界表（§4.5）。
- **审稿人最可能的 3 个攻击 + 回应**：
  1. **"界太松/可能 vacuous"** → 承认最坏界在 $H(R+1)\varepsilon>1$ 时退化；给出独立误差版（3b）与数值区间（$\varepsilon=0.5$–$1$pp、$H=4$、$R=2$），并说明界的作用是**方向性**（随 $H$ 线性、随 $\varepsilon$ 线性），不是紧常数；实验只检验斜率符号与量级。
  2. **"独立/平稳假设不现实"** → §4.5 假设表逐条给"不成立时数据里看到什么"，并报告实测的 $\rho$、误差相关、漂移；最坏界（3a）不依赖独立性，仍然成立（只是更松）。
  3. **"为什么不直接用 Neyman 分配？"** → 因为 $p_b$ 与 $w_b$ 未知且非平稳（式 (6)(8) 需要真值）；FG 是自适应近似，oracle 分配实验量化差距；这本身是可检验的实证问题（P2）。
- **篇幅核算**：正文 0.75 页 + 附录 2 页 ≈ 2.75 页当量；若 CoRL 8 页紧张，正文压到 0.5 页（只留式 (13) + 表），附录保留完整推导。

## 8. 与 v1 的一致性检查清单（写完回填）

- [ ] v1 §4.5 的 P1 表述改为 $\sqrt{M/B}$（per-bucket）与 $1/\sqrt B$（aggregate），删除 $M_{\mathrm{front}}$ 的错误直觉。
- [ ] v1 §2.3 H2 的阈值（相对 ECE 降 30%）需与命题 1 的"采样噪声 $\lesssim0.1\%$"对齐：说明该阈值只有在非平稳/偏移存在时才有意义。
- [ ] v1 §6.3 planner 规则与式 (11) 一致（阈值 $\tau$ + 重试 $R=2$）。
- [ ] v1 §5.6 的 TOST $\Delta=2.5$pp 与命题 3 的 $\varepsilon$ 口径一致（用 per-skill $\varepsilon$，不是 pooled ECE）。
- [ ] 若 critic 指出"理论太弱"，优先补式 (15) 的常数与数值表，而不是加新定理。

## 附录 A：完整推导

**A.1 式 (1)（Hoeffding + union bound）**
对固定 $b$，$X_{b,i}\in[0,1]$ i.i.d.，Hoeffding 不等式给出 $\Pr[|\hat p_b-p_b|\ge\epsilon]\le2e^{-2n_b\epsilon^2}$。取 $n_b=B/M$，对 $M$ 个桶 union bound：$\Pr[\max_b|\hat p_b-p_b|\ge\epsilon]\le2M e^{-2B\epsilon^2/M}$。令其等于 $\delta$：$2B\epsilon^2/M=\log(2M/\delta)$，即式 (1)。∎

**A.2 式 (2)（期望 ECE）**
$\mathbb{E}|\hat p_b-p_b|\le\sqrt{\mathbb{E}[(\hat p_b-p_b)^2]}=\sqrt{p_b(1-p_b)/n_b}\le\frac12\sqrt{M/B}$。乘以 $w_b$ 求和即得式 (2)。∎

**A.3 式 (3)（技能级方差）**
$\hat p_g=\sum_bw_b\hat p_b$，各桶独立：$\mathrm{Var}(\hat p_g)=\sum_bw_b^2\frac{p_b(1-p_b)}{n_b}$。代入 $n_b=B/M$ 并用 $p(1-p)\le1/4$ 得式 (3)。∎

**A.4 式 (6)（Neyman 分配的 Lagrange 推导）**
拉格朗日函数 $\mathcal{L}=\sum_b\frac{w_bv_b}{n_b}+\lambda(\sum_bn_b-B)$，$v_b=p_b(1-p_b)$。$\partial\mathcal{L}/\partial n_b=-w_bv_b/n_b^2+\lambda=0$ 给出 $n_b=\sqrt{w_bv_b/\lambda}$；代入约束得 $\sqrt{\lambda}=\frac{1}{B}\sum_b\sqrt{w_bv_b}$，即式 (6)。最优值 $\sum_b\frac{w_bv_b}{n_b^*}=\frac{1}{B}(\sum_b\sqrt{w_bv_b})^2$。∎

**A.5 式 (7)（uniform 的相对劣化）**
uniform：$\sum_bw_bv_b/(B/M)=\frac{M}{B}\sum_bw_bv_b$。与最优值之比为 $\frac{M\sum_bw_bv_b}{(\sum_b\sqrt{w_bv_b})^2}$。对 $u_b=\sqrt{w_bv_b}$ 用 Cauchy–Schwarz：$(\sum_bu_b)^2\le M\sum_bu_b^2=M\sum_bw_bv_b$，故比值 $\ge1$。∎

**A.6 式 (8)(9)（L1 最优分配的 $2/3$ 次幂）**
目标 $\sum_bw_b\sqrt{2v_b/(\pi n_b)}$，令 $\alpha_b=n_b/B$：$\propto B^{-1/2}\sum_bw_b\sqrt{2v_b/(\pi\alpha_b)}$。Lagrange：$-\frac{w_b}{2}\sqrt{\frac{2v_b}{\pi}}\alpha_b^{-3/2}+\mu=0$，故 $\alpha_b\propto(w_b\sqrt{v_b})^{2/3}$，即式 (8)。代回：令 $u_b=w_b\sqrt{v_b}$、$Z=\sum_bu_b^{2/3}$、$\alpha_b=u_b^{2/3}/Z$，则
$$\sum_b\frac{u_b}{\sqrt{\alpha_b}}=Z^{1/2}\sum_bu_b^{2/3}=Z^{3/2},$$
故 $\mathrm{ECE}^*=\sqrt{2/\pi}\,B^{-1/2}Z^{3/2}$。uniform：$\mathrm{ECE}_{\rm unif}=\sqrt{2/\pi}\,B^{-1/2}\sqrt{M}\,U$，$U=\sum_bu_b$。比值即式 (9)；由 $(\sum u_b^{2/3})^{3/2}\le\sqrt M\sum u_b$（幂平均不等式，或对 $u_b^{2/3}$ 用 Cauchy–Schwarz）得 $\le1$，且 $v_b$ 全相等时取等。∎

**A.7 式 (14)(15)（组合方差）**
delta 方法：$\hat S=\prod_i\hat q_i$，$\partial\hat S/\partial\hat q_i=\prod_{j\ne i}\hat q_j=S/\hat q_i$。独立近似下 $\mathrm{Var}(\hat S)\approx S^2\sum_i\mathrm{Var}(\hat q_i)/q_i^2$。由引理 3.1，$\mathrm{Var}(\hat q_i)\le(R+1)^2\mathrm{Var}(\hat p_i)\le(R+1)^2\sigma_p^2$；又 $q_i\ge q_{\min}$，故 $\mathrm{Var}(\hat S)\le S^2H(R+1)^2\sigma_p^2/q_{\min}^2$，即式 (14)。对 $N_\pi$ 个计划用 union bound + 次高斯尾界得式 (15)。∎

## 附录 B：两个数值例子（写进正文的一句话版本）

**例 1（组合放大）**：$H=4$、$R=2$、$\varepsilon=1$pp → 式 (13) 给损失 $\le12$pp；若实际观测到 P1−P3 差为 10–15pp，与"校准误差 1pp 放大 12 倍"的尺度一致，说明长程任务确实在放大校准误差。若观测差远小于 12pp，说明 planner 只部分依赖 $\hat p$（或误差被重试吸收），需要在 H6 的失败归因里区分。

**例 2（最优分配收益上限，已修正计算）**：$M=8$，权重 $w=(0.4,0.2,0.2,0.1,0.05,0.03,0.01,0.01)$（评测时桶被查询的频率，非均匀），$v_b=(0.25,0.25,0.16,0.09,0.04,0.01,0.0025,0.0025)$。则 $u_b=w_b\sqrt{v_b}=(0.200,0.100,0.080,0.030,0.010,0.003,0.0005,0.0005)$，$U=\sum u_b=0.4235$，$Z=\sum u_b^{2/3}=0.919$。于是

$$\frac{\mathrm{ECE}^*}{\mathrm{ECE}_{\rm uniform}}=\frac{Z^{3/2}}{\sqrt M\,U}=\frac{0.881}{2.828\times0.4235}=\frac{0.881}{1.198}\approx0.74,$$

即最优分配把 ECE 降到 uniform 的 **74%（约降 26%）**。若权重更集中（例如最大桶 $w=0.9$、其余 7 桶共 0.1），比值可到 **≈0.35–0.45**（降 55–65%）。**结论**：在"权重非均匀 + 方差异质"的现实中，分配带来的 ECE 改善上限约为 **25–40%**；H2 的"相对降 30%"接近该上限的中间值，是一个**有挑战但可达**的阈值；若 $w_b$ 很均匀或 $v_b$ 很同质，收益会逼近 0（与 P5/H2c 一致）。这也说明 H2 不应被当作"必然成立"的理论推论，而是需要实验检验的**经验命题**。

## 附录 C：与 v1 阈值的一致性（回填用）

- H2 的"相对 ECE 降 30%"：由附录 B 例 2，异质度足够时最优可降 26%（极端集中可达 55–65%），所以 30% 是"接近上限的合理目标"；若 H2 未达 30% 但方向正确且 $w^{dec}$ 口径显著，可按 N3（部分负结果）报告，不硬凑。
- H2c 的 TOST $\Delta=2.5$pp：命题 3 的 $\varepsilon$ 用 per-skill 绝对误差，2.5pp 对应式 (13) 的 $2H(R+1)\varepsilon\approx60$pp（$H=4,R=2$）——这在最坏界下超过 1，说明 TOST 的等价界不能直接用最坏界解释；应写成"$2.5$pp 是实验分辨率的等价界，不是理论可容许上界"，并在正文明确这一点（避免审稿人抓"你自己说的 60pp 都不显著"）。
- 图 Fig.4（校准）的右轴建议画"理论预测的 $\sqrt{M/B}$ 曲线"（式 2），显示采样噪声远低于观测 ECE 差，从而把差异归因于非平稳/偏移/分配偏斜。
