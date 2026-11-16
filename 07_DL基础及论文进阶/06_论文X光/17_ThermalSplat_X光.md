---
tags:
  - 论文
  - X光
  - 3DGS
  - 热红外
  - 无位姿重建
  - 前馈重建
created: 2026-11-16
paper_title: "ThermalSplat: Generalizable Pose-Free Gaussian Splatting for Sparse-View Thermal 3D Reconstruction"
paper_authors: "Prasanna S, Gopi Raju Matta, Divya Madhuri Vemunuri, Kaushik Mitra"
paper_year: 2026
paper_venue: "WACV 2027"
paper_url: "https://arxiv.org/abs/2611.15435"
arxiv_id: "2611.15435"
method: "ThermalSplat"
code_url: ""
---

# ThermalSplat_X光

**ThermalSplat: Generalizable Pose-Free Gaussian Splatting for Sparse-View Thermal 3D Reconstruction**
*Prasanna S, Gopi Raju Matta, Divya Madhuri Vemunuri, Kaushik Mitra（IIT Madras） | WACV 2027*

## 论文说了什么

**问题**：热红外是搜救、夜间导航的关键模态，但转成 3D 表示一直卡住。三个具体病灶：(1) 热图低纹理、对比度低，SfM 提不出稳定特征——Table 1 显示稠密输入还能部分注册成功，**两视图下几乎提不出 3D 点**；(2) 现有热 3DGS 全是逐场景优化，要稠密采集 + 15–25 分钟训练；(3) 它们大多靠**对齐的 RGB-热数据**拿位姿与初始点云，纯热、无位姿场景直接不可用。

**视角**：已经有「无位姿、前馈」的多视图几何 backbone（NAS3R，VGGT 底座），它在 RGB 上学会了跨视图推理；把这个先验迁到热域，缺的不是结构，而是**模态特有的结构监督**——热图的几何线索几乎全在温度不连续的边界上，且强度会随发射率与自动增益漂移。所以不改架构，只加两项自监督损失。

**结果**：六个留出热场景均值 PSNR 25.55 / SSIM 0.828 / LPIPS 0.181，单次前向 80 ms；零样本 NAS3R 23.44 / 0.768 / 0.240、热微调 NAS3R+ 24.34 / 0.785 / 0.220。只给两张无位姿图，感知质量已追平用稠密输入逐场景优化的热 3DGS。

## 方法拆解

1. **预处理**：热图是单通道灰度、辐射范围随场景变；先复制成三通道接 RGB 预训练 backbone，再用全语料统一的鲁棒百分位（1st/99th，$\alpha=0.01$）做全局强度归一化并裁到 $[0,1]$：
$$x'=\mathrm{clip}\!\left(\frac{x-p_\alpha}{p_{1-\alpha}-p_\alpha},0,1\right)$$
2. **Backbone**：NAS3R（VGGT 底座，RE10K 预训练）。每张视图 patch 化后进 ViT，聚合器交替做**帧内自注意力**（单视图外观/结构）与**全局自注意力**（跨视图对应与几何上下文）——不做显式特征匹配或 cost volume，正好绕开热域低纹理匹配不可靠的问题。
3. **三个头**：camera head 直接回归每视图相对位姿（可选内参）；depth head 出稠密深度；Gaussian head 在反投影点上出每像素高斯的 opacity、协方差、SH 外观。所有视图的高斯拼起来做可微 tile 光栅化，一次前向渲染新视角。
4. **损失**：
$$\mathcal{L}=\mathcal{L}_{photo}+\lambda_b\mathcal{L}_{boundary}+\lambda_c\mathcal{L}_{census}$$
$\mathcal{L}_{photo}=\mathcal{L}_{MSE}+0.5\,\mathcal{L}_{LPIPS}$；
$$\mathcal{L}_{boundary}=\frac{1}{|T|}\sum_{i\in T}\big|S(\hat I_i)-S(I_i)\big|$$
其中 $S(\cdot)$ 是**颜色/亮度不变**的梯度响应（用局部强度归一化，避免把发射率造成的强度跳变当成结构边）；
soft census 用软阈值比较局部强度关系
$$\xi(I)_{p,q}=\frac{|L_q-L_p|}{\sqrt{0.81+(L_q-L_p)^2}}$$
使监督只关心「邻域谁比谁亮」，不关心绝对值——对抗视角相关发射率与自动增益。
5. **数据**：聚合 5 个公开多视图热数据集（MVTV、ThermalGaussian、Thermal3DGS、ThermalMix、ThermoNeRF），覆盖室内物体、室外、建筑立面、工业设备、低光、UAV 航拍。
6. **训练**：AdamW，lr $10^{-5}$，weight decay 0.05，10,000 步，224×224，batch 4，$\lambda_b=\lambda_c=0.075$，单卡 RTX Ada 6000。
7. **评测**：六个留出场景（Hand / Dark Scenes / Fire3Raw / BuildingA Winter / Transmission Tower / UAV6），只给两张无位姿图；指标 PSNR / SSIM / LPIPS，位姿用 AUC@5°/10°/20°。对照分三组：热域逐场景优化方法（稠密 + 稀疏两档）、通用 RGB 前馈方法（零样本与热微调）、自身消融。

## 结果

Table 2（稀疏两视图 vs 稠密逐场景优化，六个场景均值）：

| 设定 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| 稠密输入 | Thermal3DGS | 25.140 | 0.8383 | 0.2522 |
| 稠密输入 | ThermalGaussian | 26.020 | 0.8694 | 0.2411 |
| 稠密输入 | PhysIR | 26.766 | 0.8622 | 0.2773 |
| 稀疏两视图 | Thermal3DGS | 11.889 | 0.4887 | 0.5469 |
| 稀疏两视图 | ThermalGaussian | 15.161 | 0.6420 | 0.4583 |
| 稀疏两视图 | PhysIR | 19.025 | 0.7149 | 0.4190 |
| **两视图无位姿** | **ThermalSplat** | **24.964** | **0.8194** | **0.2092** |

Table 3（与通用 RGB 前馈方法比，六场景均值）：MVSplat 8.15 / 0.148 / 0.747，MVSplat+ 16.97 / 0.602 / 0.456，SPFSplat 18.46 / 0.651 / 0.405，SPFSplat+ 19.30 / 0.700 / 0.426，NAS3R 23.44 / 0.768 / 0.240，NAS3R+ 24.34 / 0.785 / 0.220，**Ours 25.55 / 0.828 / 0.181**。

Table 4（两视图位姿，overall AUC@5°/10°/20°）：NAS3R 9.5 / 21.3 / 35.8，NAS3R+ 7.7 / 21.5 / 38.6，**Ours 10.4 / 23.8 / 39.7**。

Table 5（时间）：Thermal3DGS ~15 min、ThermalGaussian OMMG ~25 min、PhysIR ~15 min（逐场景优化），**Ours 80 ms**（前馈）。

Table 6（消融，六场景均值）：Naive transfer 23.44 / 0.768 / 0.240 → 仅光度 24.34 / 0.785 / 0.220 → +Census 25.15 / 0.810 / 0.192 → +Boundary 25.46 / 0.822 / 0.183 → 两者 25.55 / 0.828 / 0.181。

## 锐评

- **对照一：NAS3R 零样本（同 backbone、RGB 预训练）**。它的 PSNR 已达 23.44，离本文 25.55 只差 2.1 dB，但 LPIPS 差 0.059（0.240 vs 0.181）。这说明**通用几何先验在热域迁移得比预期好**——「热图无法重建」的瓶颈不在几何推理，而在感知质量：低对比、边界模糊让渲染看起来糊。本文的贡献因此更准确地说是「补感知」，而不是「补几何」。
- **对照二：NAS3R+（同 backbone、热数据微调）**。它 24.34 / 0.220，仍低于本文，而且恰好等于本文 Table 6 里「仅光度损失」那一档（24.34 / 0.785 / 0.220）。这是一个很干净的对照：**增益全部来自两项结构损失，不是来自热数据本身**。两项损失各贡献约 0.8 dB 与 0.3 dB，组合再补 0.09 dB。
- **对照三：稠密 + 逐场景优化（Thermal3DGS / ThermalGaussian / PhysIR）**。稠密下 25.14–26.77 dB，比本文的 24.96 略高；但同一批方法只给两视图就崩到 11.889–19.025 dB。作者的表述是「稀疏设定严格优于、稠密设定追平」，这个声明是诚实的——**这是不同输入预算的对照，不能读成「前馈优于优化」**。真正的结论是：优化的收益高度依赖采集密度，前馈把下限抬高了 6–13 dB。
- **真实局限**：作者自述假设**静态场景**，动目标（热源移动）要等后续加时序注意力或形变场；宽基线、少视图下未观测区域存在几何歧义；六个评测场景是自选的、规模不大，没有公开热域 sparse-view 基准；依赖 RGB 预训练 backbone 的可迁移性，换成别的底座是否仍成立未验证。Dark Scenes 上只有 20.23 dB，说明极低对比场景仍是短板。另外它仍需 GT 位姿做评测（训练不需要），跨数据集泛化只在补充材料的 SEAR 上给了结果。

## 借鉴意义

- **模态适配的最小改动**：不换架构、不加大模型，只加两项与模态物理特性对齐的自监督损失 + 换训练语料。任何「新传感器 + 已有几何 backbone」的组合都可以先照这个套路试一遍。
- **监督信号要对着失效模式设计**：热域的两个失效是「强度漂移」与「边界糊」，于是有 census（亮度不变）+ boundary（结构边）——不是通用的 edge loss，而是把亮度归一化写进损失里。这个「先列失效模式，再设计损失」的顺序值得抄。
- **软阈值代替硬比较**：$|x|/\sqrt{0.81+x^2}$ 让 census 在强边缘处饱和、在平坦区保持线性，避免硬阈值对噪声敏感；可用于事件相机、SAR、深度图等任何强度不稳的模态。
- **前馈替代逐场景优化**：80 ms vs 15–25 min 是三个数量级的差距，代价是 1–2 dB 的峰值画质——对在线机器人建图是明确划算的交换。

## 关联

- [[2026-11-16-论文推荐]]：当日日报条目。
- [[06_RF3DGS_X光|RF3DGS]]：那条线补的是「反射/透射」这一物理维度，这条补的是「不可见光模态」——都是给 3DGS 加它默认不建模的物理量。
- [[14_Topo-4DGS_X光|Topo-4DGS]]：同样在质疑 3DGS 的隐含假设（一个是「拓扑不变」，一个是「可见光 + 有位姿 + 稠密」），且都用「加一项结构监督」而非换表示来解决。
- [[08_ObjectWorld_X光|ObjectWorld]]：可对照看「物体级几何先验」与「传感器级几何先验」两种注入方式——前者给生成器，后者给重建器。
