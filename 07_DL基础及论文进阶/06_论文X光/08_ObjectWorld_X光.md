---
tags:
  - 论文
  - X光
  - 具身智能
  - 世界模型
  - object-centric
created: 2026-11-11
paper_title: "ObjectWorld: Scaling Object-Centric World Models with A Large-Scale 3D Full-Object Flow Dataset"
paper_authors: "Haonan Chen, Sicheng Isabella Lu, Haitian Li, Jiajin Wu, Hexiang Wei, Jiawei Zhang 等"
paper_year: 2026
paper_venue: "arXiv 2611.11085"
paper_url: "https://arxiv.org/abs/2611.11085"
arxiv_id: "2611.11085"
method: "ObjectWorld"
code_url: "https://nus-lins-lab.github.io/ObjectWorld/"
---

# ObjectWorld_X光

**ObjectWorld: Scaling Object-Centric World Models with A Large-Scale 3D Full-Object Flow Dataset**
*Haonan Chen, Sicheng Isabella Lu, Haitian Li, Jiajin Wu, Hexiang Wei, Jiawei Zhang 等（NUS LIN S Lab） | arXiv 2611.11085*

## 论文说了什么

**问题**：从大规模视频里学世界模型，需要一个「能大规模监督、又能直接指导控制」的中间表示。现有的 object flow 只覆盖**可见表面**——物体转过去、被挡住的部分没有监督；而且刚体和可变形物体通常要两套表示。

**视角**：把「可见表面流」扩成 **3D full-object flow**：给目标物体**完整表面（含被遮挡区域）**的每个点分配稠密 3D 轨迹。这样物体的完整运动被显式写出来，刚体和可变形物体用同一套表示。

**结果**：构建 **1210 小时**全物体流数据集（真实视频 + 物理仿真），配可扩展标注管线和质量指标；世界模型输入图像 + 任务描述，预测重建几何上的稠密流；实验显示标注质量可靠、流预测随数据规模提升、在新场景与新本体上的下游操作有效。具体指标 `[摘要未给]`。

## 方法拆解

- **表示**：3D full-object flow——物体表面每点一条 3D 轨迹，包含不可见部分；对刚体是刚体变换的采样，对可变形物体是逐点运动场。
- **数据**：1210 小时，真实视频 + 仿真混合；标注管线从真实观测恢复完整物体运动（可见部分靠跟踪/重建，遮挡部分靠先验补全）。
- **质量指标**：给标注本身打分，用来筛数据、也用来解释「数据规模↑→性能↑」的曲线。
- **世界模型**：图像 + 任务描述 → 在重建物体几何上预测稠密全物体流；下游把流接到操作策略。

## 锐评

- 与两条路线对比：一类是可见表面的 object flow / point tracking（如Track2Act 一类，遮挡处无监督）；一类是 per-object 的 3D 重建 + 物理仿真（贵、不可扩展）。ObjectWorld 用「全表面 + 大规模标注」把两者折中。注意：摘要未点名具体基线，对比数字 `[摘要未给]`。
- 真实局限：1210 小时里相当一部分来自仿真，sim-to-real 的流分布差异会直接影响下游；遮挡部分的轨迹本质是「补全」，标注质量指标能否抓住系统性偏差存疑；世界模型预测的是流而非像素，视觉保真度不在评测范围内。

## 借鉴意义

- 对 object-centric world model：把「表示」和「数据」一起 scale，是比换架构更稳的路线；1210 小时的量级说明这条赛道开始拼数据基建。
- 对 VLA：全物体流可以作为动作的中间监督——比末端位姿更稠密，比像素预测更贴近控制。
- 对评测：自带标注质量指标的做法值得抄，避免「数据变多但噪声变大」的假 scaling。

## 关联

- [[2026-11-11-论文推荐]]：当日日报条目。
- [[04_ReactWorld_X光|ReactWorld]]：同属「显式中间表示」——ReactWorld 建模交互响应（时间维），ObjectWorld 建模物体完整运动（几何维）。
- [[01_World Models Survey]]：综述指出世界模型的瓶颈之一是缺少可大规模监督的中间表示，这篇是正面回应。
