---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# SimpleVLA-RL - 代码实现

> 本文档包含 [[SimpleVLA-RL]] 的 PyTorch/NumPy 教学实现，涵盖 VLA + GRPO 强化学习微调循环、动态采样、自适应 Clipping 和温度退火探索增强。

```python
"""
SimpleVLA-RL 教学实现 — VLA + GRPO 强化学习微调
- GRPO (Group Relative Policy Optimization): 无需价值网络
- VLA 专属探索增强: 动态采样、自适应 Clipping、温度退火
- 纯结果奖励 (0/1): 简单但有效
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import List, Tuple


# ============================================================
# 一、VLA 策略基类（简化版）
# ============================================================

class VLAPolicy(nn.Module):
    """VLA 策略的简化表示（实际基于 OpenVLA 或类似架构）
    
    WHY 离散动作 token: OpenVLA 把连续动作量化为 256 bins，
    每个 bin 对应一个离散 token。这天然支持随机采样——
    只需从 token 分布中采样即可产生多样化轨迹。
    这正是 GRPO 需要的"同一状态多条不同 rollout"。
    参考 [[SimpleVLA-RL]] Section 2.3。
    """
    def __init__(self, obs_dim=1024, action_dim=7, num_bins=256, hidden_dim=512):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        
        # 简化的 VLA 骨干（实际是 7B LLaMA/Prism）
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 每个动作维度独立预测 256-bin 分布
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_bins) for _ in range(action_dim)
        ])
        # 温度参数（训练时可退火）
        self.log_temperature = nn.Parameter(torch.zeros(1))
    
    def forward(self, obs):
        """返回每个动作维度的 logits"""
        h = self.encoder(obs)
        logits = [head(h) for head in self.action_heads]  # list of (B, num_bins)
        return torch.stack(logits, dim=1)  # (B, action_dim, num_bins)
    
    def sample_action(self, obs, temperature=None, deterministic=False):
        """从离散分布中采样动作
        
        WHY 随机采样: GRPO 需要从同一个初始状态采样多条
        不同轨迹才能计算组内相对优势。确定性解码（argmax）
        只能产生一条轨迹 → 无法做组间对比。
        """
        logits = self.forward(obs)  # (B, action_dim, num_bins)
        temp = temperature if temperature is not None else self.log_temperature.exp()
        probs = F.softmax(logits / temp, dim=-1)
        
        if deterministic:
            action_tokens = probs.argmax(dim=-1)
        else:
            action_tokens = torch.multinomial(
                probs.view(-1, self.num_bins), 1
            ).view(-1, self.action_dim)
        # 量化 token → 连续值（简化映射）
        actions = (action_tokens.float() / self.num_bins) * 2 - 1  # 映射到 [-1, 1]
        return actions, probs  # 返回 probs 用于计算 KL 散度
    
    def log_prob(self, obs, action_tokens):
        """计算给定动作 token 的对数概率（用于 PPO/GRPO 的 ratio）"""
        logits = self.forward(obs)  # (B, action_dim, num_bins)
        log_probs = F.log_softmax(logits, dim=-1)
        # 收集对应 token 的 log prob
        selected = log_probs.gather(-1, action_tokens.unsqueeze(-1)).squeeze(-1)
        return selected.sum(dim=-1)  # (B,) — 各维度独立概率之和


# ============================================================
# 二、GRPO 核心算法
# ============================================================

class GRPO:
    """Group Relative Policy Optimization
    
    WHY GRPO vs PPO: PPO 需要一个独立的价值网络来估计
    优势函数。对于 7B+ 的 VLA，这等于再训练一个同样大的模型
    ——昂贵且不稳定。
    
    GRPO 通过组间相对归一化计算优势：
    A_i = (R_i - mean({R})) / std({R})
    
    对于同一个初始状态，采样 G 条不同轨迹（不同随机采样），
    用组内奖励的相对排名代替绝对优势值。
    参考 [[SimpleVLA-RL]] Section 2.2，灵感来自 [[DeepSeek-R1]]。
    """
    def __init__(self, clip_epsilon=0.2, kl_beta=0.01, group_size=8):
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.group_size = group_size  # G: 每组采样数
    
    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """组内相对归一化计算优势
        
        WHY 相对归一化: 不需要价值网络来估计 baseline。
        只要同一组内有人做得更好，其他人就知道自己差了。
        这本质上是"相对比较"而非"绝对评估"。
        """
        rewards = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards.mean()
        std_r = rewards.std()
        if std_r < 1e-8:
            # 所有轨迹奖励相同→无信号
            return torch.zeros_like(rewards)
        advantages = (rewards - mean_r) / std_r
        return advantages
    
    def compute_loss(self, policy, ref_policy, obs, action_tokens, 
                     advantages, temperature=None):
        """GRPO 损失函数（简化版）
        
        L_GRPO = min(r*A, clip(r, 1-ε, 1+ε)*A) - β*KL(π||π_ref)
        
        WHY clipping: 和 PPO 一样防止策略更新太大。
        WHY KL 正则: 防止策略偏离原始预训练模型太远
        （否则 RL 可能破坏预训练的通用能力）。
        """
        new_log_prob = policy.log_prob(obs, action_tokens)
        with torch.no_grad():
            old_log_prob = ref_policy.log_prob(obs, action_tokens)
        # 概率比率
        ratio = torch.exp(new_log_prob - old_log_prob)
        # PPO-style clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL 正则化（简化：用 log_prob 差近似）
        kl_approx = (old_log_prob - new_log_prob).mean()
        total_loss = policy_loss + self.kl_beta * kl_approx
        
        return total_loss, policy_loss.item(), kl_approx.item()


# ============================================================
# 三、探索增强策略
# ============================================================

class ExplorationScheduler:
    """三项探索增强策略的统一调度器
    
    SimpleVLA-RL 发现 GRPO 本身对探索不友好（偏向保守更新），
    提出三项增强。参考 [[SimpleVLA-RL]] Section 2.4。
    """
    def __init__(self, total_steps=10000, initial_temp=1.0, final_temp=0.1,
                 min_epsilon=0.2, max_epsilon=0.28):
        self.total_steps = total_steps
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.current_step = 0
    
    def get_temperature(self):
        """温度退火: 早期高温→多探索，后期低温→精细化
        
        WHY 退火: 类似于模拟退火在 RL 中的应用。
        训练初期策略很不确定，需要多探索来发现好轨迹；
        后期策略基本正确，需要精细化来避免随机噪声。
        """
        progress = min(self.current_step / self.total_steps, 1.0)
        # 余弦退火
        temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * \
               (1 + np.cos(progress * np.pi))
        return temp
    
    def get_clip_epsilon(self):
        """自适应 Clipping 扩展
        
        WHY 扩展 clipping 范围: 标准 PPO clipping [0.8, 1.2]
        在训练初期过于保守。扩展到 [0.8, 1.28] 允许更大的
        策略更新步长，鼓励更多探索。这在策略还很不确定时
        尤其重要。
        """
        progress = min(self.current_step / self.total_steps, 1.0)
        # 训练初期用 larger clip，后期恢复标准值
        return self.max_epsilon - progress * (self.max_epsilon - self.min_epsilon)
    
    def get_dynamic_samples(self, confidence: float) -> int:
        """动态采样: 根据策略自信度动态调整采样数量
        
        WHY 动态采样: 对不自信的状态多采几条轨迹，
        对已经学会的少采——把采样预算用在刀刃上。
        confidence 越低 → 采样越多。
        """
        if confidence < 0.3:
            return 16  # 很不自信：多采
        elif confidence < 0.6:
            return 8   # 中等自信
        else:
            return 4   # 很有自信：少采
    
    def step(self):
        self.current_step += 1


# ============================================================
# 四、VLA RL 训练循环
# ============================================================

class VLARLTrainer:
    """VLA + GRPO 强化学习训练循环
    
    完整流程:
    1. 从数据集中采样初始状态 s_0
    2. 对每个 s_0 采样 G 条轨迹（不同随机 seed）
    3. 执行轨迹，收集 0/1 成功奖励
    4. 用 GRPO 更新策略
    5. 重复
    
    关键差异 vs LLM RL:
    - 每个 rollout 步骤需要与环境交互（仿真或真实）
    - VLA 用离散动作 token 实现随机采样
    - 需要并行多环境渲染加速
    参考 [[SimpleVLA-RL]] Section 2.3。
    """
    def __init__(self, policy, ref_policy, grpo, explorer, lr=1e-5):
        self.policy = policy
        self.ref_policy = ref_policy  # 冻结的参考模型
        self.grpo = grpo
        self.explorer = explorer
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=1000)
    
    def collect_trajectory(self, env, obs, max_steps=100):
        """收集一条轨迹
        
        在真实系统中，env.step(action) 会控制真实机器人
        （通过仿真或真机）。这里简化为随机环境。
        """
        obs_list, action_list, done = [], [], False
        for _ in range(max_steps):
            # 温度随训练退火
            temp = self.explorer.get_temperature()
            action, _ = self.policy.sample_action(
                obs.unsqueeze(0), temperature=temp
            )
            action = action.squeeze(0)
            # 与环境交互（此处简化）
            next_obs, reward, done = env.step(action.numpy())
            obs_list.append(obs)
            action_list.append(action)
            if done:
                break
            obs = torch.from_numpy(next_obs).float()
        return obs_list, action_list, reward
    
    def train_step(self, env, obs_batch):
        """单步 GRPO 训练
        
        obs_batch: (B, obs_dim) — 多个初始状态
        """
        B = obs_batch.shape[0]
        G = self.grpo.group_size
        device = obs_batch.device
        clip_eps = self.explorer.get_clip_epsilon()
        self.grpo.clip_epsilon = clip_eps
        
        all_obs = []
        all_actions = []
        all_advantages = []
        
        for i in range(B):
            s0 = obs_batch[i]
            # 对同一初始状态采样 G 条轨迹
            group_rewards = []
            group_obs = []
            group_actions = []
            
            for g in range(G):
                obs_seq, act_seq, reward = self.collect_trajectory(env, s0)
                group_rewards.append(reward)
                group_obs.append(torch.stack(obs_seq))
                group_actions.append(torch.stack(act_seq))
            
            # 计算组内优势
            advantages = self.grpo.compute_advantages(group_rewards)
            
            for g in range(G):
                all_obs.append(group_obs[g])
                all_actions.append(group_actions[g])
                all_advantages.append(advantages[g].expand(group_obs[g].shape[0]))
        
        # 合并所有数据
        all_obs = torch.cat([o.to(device) for o in all_obs], dim=0)
        all_actions = torch.cat([a.to(device) for a in all_actions], dim=0)
        all_advantages = torch.cat([adv.to(device) for adv in all_advantages], dim=0)
        
        # 计算 GRPO 损失并更新
        loss, policy_loss, kl_loss = self.grpo.compute_loss(
            self.policy, self.ref_policy, all_obs, all_actions, all_advantages
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（稳定训练）
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        self.explorer.step()
        return loss.item(), policy_loss, kl_loss
    
    def update_ref_policy(self):
        """定期用当前策略更新参考模型（EMA 或直接复制）"""
        self.ref_policy.load_state_dict(self.policy.state_dict())


# ============================================================
# 五、仿真环境和演示
# ============================================================

class DummyEnv:
    """简化的仿真环境（替代 Isaac Sim / LIBERO）"""
    def __init__(self, success_distance=0.1):
        self.success_distance = success_distance
    
    def step(self, action):
        # 简化的动力学：动作越接近 [0.5]*7 越好
        target = np.ones_like(action) * 0.5
        distance = np.linalg.norm(action - target)
        done = distance < self.success_distance
        reward = 1.0 if done else 0.0  # WHY: 纯结果奖励 (0/1)
        next_obs = np.random.randn(1024).astype(np.float32) + action.mean() * 0.1
        return next_obs, reward, done


# ============================================================
# 六、演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("[SimpleVLA-RL] 代码演示 — VLA + GRPO 强化学习微调")
    print("=" * 60)
    
    obs_dim = 1024
    action_dim = 7
    
    # 初始化策略和参考模型
    policy = VLAPolicy(obs_dim=obs_dim, action_dim=action_dim)
    ref_policy = VLAPolicy(obs_dim=obs_dim, action_dim=action_dim)
    ref_policy.load_state_dict(policy.state_dict())
    # 冻结参考模型
    for p in ref_policy.parameters():
        p.requires_grad = False
    
    grpo = GRPO(clip_epsilon=0.2, kl_beta=0.01, group_size=8)
    explorer = ExplorationScheduler(total_steps=200)
    trainer = VLARLTrainer(policy, ref_policy, grpo, explorer)
    env = DummyEnv()
    
    print(f"\n策略参数量: {sum(p.numel() for p in policy.parameters())/1e6:.2f}M")
    print(f"GRPO 组大小: G={grpo.group_size}")
    
    # --- 演示：单步训练 ---
    print("\n1. 单步 GRPO 训练")
    obs_batch = torch.randn(4, obs_dim)
    loss, policy_loss, kl_loss = trainer.train_step(env, obs_batch)
    print(f"   Total Loss:   {loss:.4f}")
    print(f"   Policy Loss:  {policy_loss:.4f}")
    print(f"   KL Penalty:   {kl_loss:.4f}")
    print(f"   Temperature:  {explorer.get_temperature():.3f}")
    print(f"   Clip Epsilon: {explorer.get_clip_epsilon():.3f}")
    
    # --- 演示：动态采样 ---
    print("\n2. 动态采样策略")
    for conf in [0.1, 0.4, 0.8]:
        n = explorer.get_dynamic_samples(conf)
        print(f"   confidence={conf} → 采样 {n} 条轨迹")
    
    # --- 演示：采样多样性 ---
    print("\n3. 动作采样多样性（高温度 vs 低温度）")
    obs = torch.randn(1, obs_dim)
    for temp in [1.0, 0.5, 0.1]:
        actions = []
        for _ in range(20):
            act, _ = policy.sample_action(obs, temperature=temp)
            actions.append(act.squeeze(0))
        actions = torch.stack(actions)
        std = actions.std(dim=0).mean().item()
        print(f"   temperature={temp}: 动作标准差={std:.4f}")
    
    print("\n关键设计要点:")
    print("  - GRPO: 组内相对归一化 → 无需价值网络")
    print("  - 离散动作 token: 天然支持随机采样（GRPO 必要）")
    print("  - 温度退火: 早期高温探索 → 后期低温精细化")
    print("  - 自适应 Clipping: 训练初期允许更大更新步长")
    print("  - 纯结果奖励(0/1): 极简设计，不需要中间奖励")
    print("  - Pushcut 现象: RL 发现演示数据中没有的新行为模式")
    print("\n参考: [[SimpleVLA-RL]] Section 2, 灵感来自 [[DeepSeek-R1]]")
```

## 设计说明

- **GRPO**：组内相对归一化计算优势，无需价值网络。当 G=8 时，不需要独立训练 7B 价值模型
- **离散动作 token**：从 256-bin 分布中采样是 GRPO 多样性的基础
- **三项探索增强**：动态采样 + 自适应 Clipping + 温度退火
- **Pushcut**：RL 发现的新行为模式——此实现不直接展示但架构支持
- 对照 [[SimpleVLA-RL]] Section 2，[[DeepSeek-R1]] 的思想复现
- 注意：实际训练需 8xA800 GPU，本代码为教学实现框架
```
