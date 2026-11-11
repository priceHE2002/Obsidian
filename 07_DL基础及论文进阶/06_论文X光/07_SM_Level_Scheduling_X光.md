---
tags:
  - 论文
  - X光
  - 3DGS
  - GPU调度
  - 渲染系统
created: 2026-11-11
paper_title: "Mitigating SM-Level Load Imbalance in 3D Gaussian Splatting on GPUs"
paper_authors: "Junha Ryu, Hoi-Jun Yoo"
paper_year: 2026
paper_venue: "arXiv 2611.12319"
paper_url: "https://arxiv.org/abs/2611.12319"
arxiv_id: "2611.12319"
method: "SM-Level Scheduling"
code_url: ""
---

# SM_Level_Scheduling_X光

**Mitigating SM-Level Load Imbalance in 3D Gaussian Splatting on GPUs**
*Junha Ryu, Hoi-Jun Yoo | arXiv 2611.12319*

## 论文说了什么

**问题**：3DGS 渲染的耗时在屏幕 tile 之间极不均匀——少数 tile 叠了几百个高斯，多数 tile 很轻。GPU 按 tile 分给 SM 后，轻 tile 早早做完，重 tile 拖出一条长尾，大部分 SM 在尾部空转。城市场景（高斯密度极不均）尤其严重。

**视角**：不改算法、不换硬件，只在调度层解决。现有优化要么动 3DGS 本身（剪枝/压缩，要重训），要么上专用加速器（不通用）；作者选择用软件定义的 GPU 调度吃掉长尾。

**结果**：在大尺度城市场景上，逐场景平均 alpha blending 最高加速 **2.93×**；最不均衡的帧最高 **5.6×**；且保持原始 alpha blending 公式不变。

## 方法拆解

- **persistent kernel**：常驻 kernel 从全局队列取 tile 任务，先取预估最重的 tile，避免「重活最后才开工」。
- **离群 tile 分块**：对特别重的 tile，按 CTA 切成多块并行处理，把单点长尾摊平。
- **每帧轻量策略**：用渲染管线里已有的 tile 统计（高斯数/预估耗时）决定这一帧切不切、切多少，策略本身开销可忽略。
- **正交性**：不依赖特定 3DGS 变体，也不改光栅化/混合公式，可叠加在已有优化之上。

## 锐评

- 与两类工作对比：算法层优化（压缩高斯数、剪枝、LOD）会改变重建质量且常需重训；专用加速器（3DGS ASIC/FPGA）灵活性和通用性差。这篇走第三条——纯调度，零重训、零硬件改动。注意：摘要未列具体对比系统，数字只对 alpha blending 阶段，端到端渲染帧率 `[摘要未给]`。
- 真实局限：收益完全取决于负载偏斜程度，均匀场景可能接近 1×；分块会带来额外调度与合并开销，摘要没给这部分的账；只在城市场景验证，室内/小物体场景的 tile 分布不同。另外「alpha blending 加速」不等于整条渲染管线加速，前后处理（排序、投影）仍是瓶颈时收益会被稀释。

## 借鉴意义

- 对 3DGS 数据生成：world model / 自动驾驶仿真要靠 3DGS 批量出图，渲染吞吐直接决定数据成本；这种零重训优化可以立刻挂到现有管线上。
- 对 GPU 系统研究：把「tile 级负载不均」抽象成通用调度问题，思路可迁移到 NeRF、点云渲染等不规则负载。

## 关联

- [[2026-11-11-论文推荐]]：当日日报条目。
- [[2026-11-08-论文推荐]]：那天的 CrossPressor / BudgetSplat 从表示层降成本，这篇从调度层降成本，两者可叠加。
