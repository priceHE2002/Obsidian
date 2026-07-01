---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# MolmoAct - 代码实现

> 本文档包含 PyTorch 教学实现。

```python
"""
MolmoAct: 结构化视觉推理到动作执行
=====================================
论文: "Molmo and MolmoAct" (Deitke et al., 2024)
核心贡献: 将点追踪(point tracking)引入 VLA，实现结构化的
         视觉推理 → 轨迹规划 → 动作执行 三步流程。

关键设计:
  1. 点追踪: 在图像中标记关键点（物体位置、抓取点等）
  2. 轨迹规划: 基于点标记生成末端执行器轨迹
  3. 动作执行: 将轨迹转化为关节角度指令

与 [[../03_RT-2/RT-2|RT-2]] 的关系: 继承"VLM 做机器人"范式，但引入结构化推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 点追踪模块 — 在图像中检测和追踪关键点
# ==============================================================================
class PointTracker(nn.Module):
    """
    点追踪模块: 给定图像和文本提示，输出关键点的像素坐标。

    例如: "抓取点" → (x, y) 坐标
         "目标位置" → (x, y) 坐标

    为什么用点追踪？
    - 比 bounding box 更精确（机器人需要毫米级精度）
    - 比 segmentation mask 更轻量
    - 可以直接映射到末端执行器的笛卡尔空间
    """

    def __init__(self, embed_dim=256, num_points=4):
        super().__init__()
        self.num_points = num_points

        # 简化的视觉编码器
        self.vis_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
        )

        # 文本提示编码（如 "抓取点", "目标位置"）
        self.text_proj = nn.Linear(embed_dim, embed_dim)

        # 点热力图预测头
        # 对每个关键点输出一个热力图
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256 + embed_dim, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, num_points, 1),  # 每个点一个通道
        )

        self.grid_size = 8  # 输出热力图大小

    def forward(self, image, text_embed, point_prompts):
        """
        image: (B, 3, H, W) — 输入图像
        text_embed: (B, embed_dim) — 任务语义嵌入
        point_prompts: list of str — 要追踪的点名称

        返回: (B, num_points, 2) — 每张图的关键点像素坐标
        """
        B, _, H, W = image.shape

        # 视觉特征
        vis_feat = self.vis_encoder(image)  # (B, 256, 8, 8)

        # 文本条件注入
        text_feat = self.text_proj(text_embed)  # (B, embed_dim)
        text_feat = text_feat[:, :, None, None].expand(-1, -1, self.grid_size, self.grid_size)

        # 融合视觉和文本
        fused = torch.cat([vis_feat, text_feat], dim=1)  # (B, 256+embed_dim, 8, 8)

        # 预测每个关键点的热力图
        heatmaps = self.heatmap_head(fused)  # (B, num_points, 8, 8)

        # 从热力图中提取峰值位置（用 soft-argmax 以保证可导）
        coords = soft_argmax_2d(heatmaps, H, W)

        return coords, heatmaps


def soft_argmax_2d(heatmaps, orig_h, orig_w):
    """
    二维 soft-argmax: 对热力图做空间 softmax 后求期望位置。
    完全可导，适合端到端训练。

    公式: x̂ = Σ x·p(x), ŷ = Σ y·p(y),  p = softmax(heatmap)
    """
    B, N, H, W = heatmaps.shape
    heatmaps = heatmaps.reshape(B, N, -1)
    probs = F.softmax(heatmaps * 100, dim=-1)  # ×100 使分布更尖锐

    # 生成坐标网格
    y_coords = torch.arange(H, device=heatmaps.device).float()
    x_coords = torch.arange(W, device=heatmaps.device).float()
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    yy, xx = yy.flatten(), xx.flatten()

    # 期望坐标（归一化到 [0,1]）
    y_expected = (probs * yy).sum(-1) / H
    x_expected = (probs * xx).sum(-1) / W

    # 映射回原始图像坐标
    y_expected = y_expected * orig_h
    x_expected = x_expected * orig_w

    coords = torch.stack([x_expected, y_expected], dim=-1)  # (B, N, 2)
    return coords


# ==============================================================================
# 2. 轨迹规划器 — 从点到轨迹
# ==============================================================================
class TrajectoryPlanner(nn.Module):
    """
    给定关键点（如抓取点、放置点），生成完整的末端执行器轨迹。

    使用贝塞尔曲线或五次多项式进行轨迹平滑。
    """

    def __init__(self, num_waypoints=20, action_dim=7):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.action_dim = action_dim

        # 从关键点生成轨迹的 MLP
        self.waypoint_generator = nn.Sequential(
            nn.Linear(8 + 3, 128),  # 4 个关键点(x,y) + 当前状态
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints * action_dim),
        )

    def forward(self, keypoints, current_state):
        """
        keypoints: (B, num_points, 2) — 关键点坐标
        current_state: (B, action_dim) — 当前末端执行器状态
        """
        B = keypoints.shape[0]
        x = keypoints.flatten(1)  # (B, num_points * 2)
        x = torch.cat([x, current_state[:, :3]], dim=-1)  # 加当前位置

        traj = self.waypoint_generator(x)
        traj = traj.view(B, self.num_waypoints, self.action_dim)

        # 平滑轨迹（可选：应用多项式插值）
        traj = smooth_trajectory(traj)

        return traj


def smooth_trajectory(traj, window=3):
    """简单的移动平均平滑"""
    kernel = torch.ones(1, 1, window) / window
    traj = traj.transpose(1, 2)  # (B, dim, steps)
    traj = F.conv1d(F.pad(traj, (window // 2, window // 2), mode='replicate'),
                    kernel.to(traj.device))
    return traj.transpose(1, 2)  # (B, steps, dim)


# ==============================================================================
# 3. 动作执行器 — 轨迹 → 关节角度
# ==============================================================================
class ActionExecutor(nn.Module):
    """
    将笛卡尔空间轨迹转化为关节角度指令（逆运动学近似）。
    """

    def __init__(self, joint_dim=7, cartesian_dim=3):
        super().__init__()
        self.ik_net = nn.Sequential(
            nn.Linear(cartesian_dim + joint_dim, 128),  # 目标位姿 + 当前关节角
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, joint_dim),
        )

    def forward(self, target_pose, current_joints):
        """target_pose: (B, 3), current_joints: (B, 7) → target_joints: (B, 7)"""
        x = torch.cat([target_pose, current_joints], dim=-1)
        delta_joints = self.ik_net(x)
        return current_joints + delta_joints


# ==============================================================================
# 4. 完整 MolmoAct 推理流程
# ==============================================================================
class MolmoActPipeline(nn.Module):
    """
    结构化推理流水线:
      Step 1: 点追踪 — 从图像+文本找到关键点
      Step 2: 轨迹规划 — 关键点→完整轨迹
      Step 3: 动作执行 — 笛卡尔轨迹→关节角度
    """

    def __init__(self):
        super().__init__()
        self.point_tracker = PointTracker(num_points=4)
        self.traj_planner = TrajectoryPlanner()
        self.executor = ActionExecutor()

    def forward(self, image, text_embed, point_prompts, current_state, current_joints):
        """
        完整推理流程

        返回:
          keypoints: 检测到的关键点
          trajectory: 规划的末端执行器轨迹
          joint_actions: 最终关节角度指令
        """
        # Step 1: 点追踪
        keypoints, _ = self.point_tracker(image, text_embed, point_prompts)

        # Step 2: 轨迹规划
        trajectory = self.traj_planner(keypoints, current_state)

        # Step 3: 动作执行（对轨迹上的每一步）
        B = trajectory.shape[0]
        joint_actions = []
        j = current_joints
        for step in range(trajectory.shape[1]):
            target_pose = trajectory[:, step, :3]
            j = self.executor(target_pose, j)
            joint_actions.append(j)
        joint_actions = torch.stack(joint_actions, dim=1)

        return keypoints, trajectory, joint_actions


# ==============================================================================
# 演示
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MolmoAct 流水线演示")
    print("=" * 60)

    pipeline = MolmoActPipeline()

    B = 2
    image = torch.randn(B, 3, 224, 224)
    text_embed = torch.randn(B, 256)
    current_state = torch.randn(B, 7)
    current_joints = torch.randn(B, 7)

    keypoints, trajectory, joint_actions = pipeline(
        image, text_embed, ["grasp_point", "target_point"],
        current_state, current_joints
    )

    print(f"图像: {image.shape}")
    print(f"关键点: {keypoints.shape}  (x,y坐标)")
    print(f"轨迹: {trajectory.shape}  (20步 × 7维)")
    print(f"关节指令: {joint_actions.shape}")
    print(f"\nMolmoAct 核心流程:")
    print(f"  1. 点追踪 → 关键点坐标")
    print(f"  2. 轨迹规划 → 笛卡尔空间轨迹")
    print(f"  3. 动作执行 → 关节角度指令")
    print(f"\n优势: 结构化推理使得每一步都是可解释的")
```
