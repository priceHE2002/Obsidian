---
tags:
  - 论文
  - WAM
  - 视频协同训练
  - 清华MARS
created: 2026-06-30
paper_title: "Fast-WAM: Do World Action Models Need Test-time Future Imagination?"
paper_authors: "袁天远 et al. (清华MARS Lab + 星海图Galaxea AI)"
paper_year: 2026
paper_venue: "arXiv 2026.3"
paper_url: "https://arxiv.org/abs/2603.16666"
github: "https://github.com/yuantianyuan01/FastWAM"
---

# Fast-WAM

**Fast-WAM: Do World Action Models Need Test-time Future Imagination?**
*清华 MARS Lab（赵行教授）+ 星海图 Galaxea AI | 2026.3 | arXiv 2603.16666*

> **WAM 的性能到底来自哪里？是推理时对未来的"想象"（生成未来视频帧），还是训练时学到的更好的世界表征？** Fast-WAM 用一圈精心设计的对照实验给出了答案：**训练时做视频联合训练（学到好的表征）是关键；推理时生成未来视频帧几乎没有贡献，可以安全跳过。** 这意味着 WAM 可以跑得和 VLA 一样快——完全推翻了"WAM 必须慢"的认知。

---

## 一、核心研究问题

### 1.1 所有 WAM 的共同设计

WAM（世界动作模型）的核心特征是**同时做两件事**：
1. 预测动作
2. 预测"如果我执行这个动作，世界会变成什么样"（视频预测）

在推理时，典型的 WAM 会先"想象"未来的视频帧，再基于这个想象做动作预测。这个过程被称为 **"imagine-then-execute"（先想象再执行）**。

### 1.2 解耦的难题

WAM 将"训练时视频建模"和"推理时未来想象"**耦合在一起**——你无法从现有 WAM 的消融中区分这两个因素的各自贡献。Fast-WAM 的目的就是解耦它们。

---

## 二、实验设计：一套精巧的对照条件

Fast-WAM 的核心贡献不是架构创新，而是**实验设计**。论文构建了多个精心设计的变体：

| 变体 | 训练时视频 Co-training | 推理时生成未来帧 | 目的 |
|------|---------------------|---------------|------|
| **Fast-WAM**（主推）| ✅ | ❌ 跳过 | 测试"推理时想象是否必要"|
| Imagine-then-Execute WAM | ✅ | ✅ | 标准 WAM 基线 |
| No-Video WAM | ❌ | ❌ | 纯 VLA 等效——测试"训练时视频 co-training 是否必要"|
| Video-Only Pretrain | ✅ 但冻住 | ❌ | 测试"视频 co-training 是否需要端到端"|

### 2.1 关键发现 1：推理时的未来想象几乎不贡献性能

| 设置 | RoboTwin 2.0 平均 |
|------|-----------------|
| Imagine-then-Execute WAM | 91.8% |
| **Fast-WAM**（跳过推理时想象）| **与完整 WAM 持平** |

两者在 LIBERO 和 RoboTwin 2.0 上都没有统计显著的性能差异。

### 2.2 关键发现 2：训练时的视频 co-training 是关键

| 设置 | RoboTwin 2.0 平均 |
|------|-----------------|
| Fast-WAM（训练时 co-training）| 91.8% |
| No-Video WAM（移除训练时 co-training）| **83.8%（大幅骤降！）** |

**移除训练时的视频 co-training → 性能崩溃。** 这说明 WAM 的性能增益来自训练阶段通过视频预测学到的更好的世界表征，而非推理时的显式想象。

---

## 三、极简架构

基于以上发现，Fast-WAM 设计了一个极简架构，只为速度优化：

**训练时：**
- Video DiT (Wan2.2-5B) + Action DiT → 联合优化动作预测和视频预测
- 结构化 attention mask：action token 不能看到未来 video token（防止信息泄露）

**推理时：**
- Video DiT 做**单次前向传播**提取 latent world representation
- Action DiT 直接从表征解码动作
- **完全跳过未来视频生成和迭代去噪**
- 推理延迟：**190ms** → 优化后 <**90ms**（与 π0.5 VLA 持平）

---

## 四、为什么训练时视频 co-training 如此重要？

论文分析了几个可能的原因：

1. **稠密的物理监督信号**：视频预测迫使模型学习世界是如何运作的——物体不会凭空消失，运动会保持连续性
2. **丰富的视觉表征**：视频 co-training 让模型学到更好的视觉特征（不仅识别"这是什么"，还理解"它怎么动"）
3. **隐式数据增强**：预测未来帧要求模型理解物体的 3D 结构和物理属性，这是一种强大的正则化

---

## 五、学术和产业影响

- **谢赛宁**（DiT 核心作者，NYU 助理教授）将 Fast-WAM 与图灵奖得主 **Yann LeCun** 的 LeWorldModel 并列推荐："最好一起看"
- **星海图 Galaxea AI** 累计融资近 **30 亿元**，估值百亿级别
- Fast-WAM 的结论改变了 WAM 的研究方向——从"让想象更逼真"转向"让训练视频 co-training 更有效"

---

## 六、与 Motus 的对比

| | Motus (CVPR 2026) | Fast-WAM |
|---|---|---|
| 核心贡献 | 大一统 MoT 架构 | 解耦分析（训练视频 co-training vs 推理想象）|
| 推理时是否生成视频 | 是 | **否（关键发现）** |
| 推理速度 | 较慢 | **190ms → <90ms（与 VLA 持平）** |
| 关键结论 | MoT 架构有效 | 视频 co-training 是核心，推理时想象可有可无 |

---

## 七、对你的启示

1. **如果你想做世界模型方向**：Fast-WAM 告诉你**不需要在推理时生成视频**——这大大简化了工程实现
2. **视频 co-training 是一个成本低的"免费午餐"**：在你的 VLA 训练中加入视频预测作为辅助损失，可能带来显著提升
3. **极简架构也可以很强**：不需要复杂的"想象-执行"管道

## 八、硬件适配

⚠️ **4070 Ti Super 16GB**：推理可能可行（190ms 延迟在高端 GPU 上测得，16GB 会慢一些）。但 Wan2.2-5B 的视频 DiT 对 16GB 来说训练太重。

## PDF

[[FastWAM.pdf]]
