---
tags: [代码, PyTorch]
created: 2026-11-08
---
# SuperMimic - 代码实现

> 本文档是**教学实现**（非官方代码），按论文 §3 的结构复刻训练循环：采数据 → 拟合可微世界模型 → 在世界模型上 BPTT 更新策略与 critic。
> 论文：[[01_SuperMimic]] | CoRL 2026 Oral | UC Berkeley | arXiv 2610.28480

```python
"""
SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model

核心三段循环（论文 Figure 2 / Algorithm A.1）：
  1) 当前策略在仿真器里采 transition（带动作噪声），进固定大小 replay buffer
  2) 从 buffer 采 k=8 步窗口，拟合确定性世界模型（状态/奖励/无噪观测/终止概率四个头）
  3) 从 buffer 状态出发跑 H=32 步 imagined rollout，BPTT 反传更新策略；critic bootstrap H 之后的 return
关键点：世界模型冻结后再反传；仿真器本身不可微，梯度全部来自学到的代理模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 世界模型：f(s, a, ξ) -> s'，外加 reward / observation / termination 三个头
# ═══════════════════════════════════════════════════════════════════════════════

class WorldModel(nn.Module):
    """确定性动力学代理。条件在 sim context ξ（摩擦、标定偏移、质心）上。"""

    def __init__(self, state_dim, action_dim, ctx_dim, goal_dim, hidden=512, depth=3):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim

        layers, last = [], state_dim + action_dim + ctx_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.SiLU()]
            last = hidden
        self.trunk = nn.Sequential(*layers)

        self.next_state = nn.Linear(hidden, state_dim)          # f_ω
        self.reward = nn.Linear(hidden, 1)                      # r_ω
        self.obs = nn.Linear(hidden, state_dim)                 # g_ω：预测无噪观测
        self.termination = nn.Linear(hidden, 1)                 # d_ω：终止概率（可微）

    def forward(self, s, a, ctx):
        h = self.trunk(torch.cat([s, a, ctx], dim=-1))
        return {
            "s_next": s + self.next_state(h),                   # 残差：学“变化量”更稳
            "reward": self.reward(h).squeeze(-1),
            "obs": self.obs(h),
            "term_logit": self.termination(h).squeeze(-1),
        }


def world_model_loss(model, batch):
    """论文 Eq.1：状态 MSE + 奖励 MSE + 观测 MSE + 终止 BCE，k=8 步自回归。"""
    s, a, ctx, r, obs, done = batch["s"], batch["a"], batch["ctx"], batch["r"], batch["obs"], batch["done"]
    pred = model(s, a, ctx)
    loss = (
        F.mse_loss(pred["s_next"], batch["s_next"])
        + F.mse_loss(pred["reward"], r)
        + F.mse_loss(pred["obs"], obs)
        + F.binary_cross_entropy_with_logits(pred["term_logit"], done)
    )
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 策略与 critic
# ═══════════════════════════════════════════════════════════════════════════════

class Policy(nn.Module):
    """确定性策略：a_t = π(o_t, g_t)，a 是归一化关节设定点。"""

    def __init__(self, obs_dim, goal_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs, goal):
        return torch.tanh(self.net(torch.cat([obs, goal], dim=-1)))


class Critic(nn.Module):
    """V_ψ(s, g, ξ)：估计 H 步之后的 return，用来 bootstrap。"""

    def __init__(self, state_dim, goal_dim, ctx_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, g, ctx):
        return self.net(torch.cat([s, g, ctx], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. imagined rollout + BPTT（论文 Eq.2）
# ═══════════════════════════════════════════════════════════════════════════════

def imagine_return(world_model, policy, critic, s0, goal, ctx, obs_noise_std,
                   horizon=32, gamma=0.99):
    """在世界模型上滚 H 步，返回可微的目标 Ĵ(θ)。"""
    s, alive, total = s0, torch.ones(s0.shape[0], device=s0.device), 0.0
    for t in range(horizon):
        pred = world_model(s, policy(pred_obs_prev if t else s, goal), ctx)
        # 论文细节：策略吃的是“无噪观测 + 重新加的传感器噪声”
        obs = pred["obs"] + torch.randn_like(pred["obs"]) * obs_noise_std
        action = policy(obs, goal)
        pred = world_model(s, action, ctx)
        alive = alive * (1.0 - torch.sigmoid(pred["term_logit"]))   # 软存活权重
        total = total + (gamma ** t) * alive * pred["reward"]
        s, pred_obs_prev = pred["s_next"], pred["obs"]
    bootstrap = (gamma ** horizon) * alive * critic(s, goal, ctx)
    return (total + bootstrap).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 训练循环（采数据 → 拟合 WM → BPTT 更新策略）
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg, sim, buffer, world_model, policy, critic):
    opt_wm = torch.optim.Adam(world_model.parameters(), lr=cfg.wm_lr)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.pi_lr)
    opt_v = torch.optim.Adam(critic.parameters(), lr=cfg.v_lr)

    for step in range(cfg.total_steps):
        # ① 采数据：执行加噪动作，只把带噪 transition 存进 buffer
        for _ in range(cfg.rollouts_per_step):
            s = sim.reset()
            while not sim.done:
                a = policy(sim.obs(), sim.goal())
                a_noisy = a + torch.randn_like(a) * cfg.action_noise
                s_next, r, done, info = sim.step(a_noisy)
                buffer.add(s, a_noisy, info["ctx"], r, sim.obs(), done, s_next, sim.goal())
                s = s_next

        # ② 拟合世界模型：k=8 步自回归窗口
        for _ in range(cfg.wm_updates_per_step):
            batch = buffer.sample_windows(k=8, n=cfg.batch_size)
            loss_wm = world_model_loss(world_model, batch)
            opt_wm.zero_grad(); loss_wm.backward(); opt_wm.step()

        # ③ 更新策略：世界模型冻结，BPTT 过 H=32 步
        world_model.eval()
        for p in world_model.parameters():
            p.requires_grad_(False)
        for _ in range(cfg.pi_updates_per_step):
            batch = buffer.sample_states(n=cfg.batch_size)          # s0, g, ctx
            J = imagine_return(world_model, policy, critic,
                               batch["s"], batch["g"], batch["ctx"],
                               obs_noise_std=cfg.obs_noise, horizon=cfg.horizon)
            opt_pi.zero_grad(); (-J).backward(); opt_pi.step()

            # critic：在 imagined transitions 上做 GAE / TD 回归
            with torch.no_grad():
                target = imagine_return(world_model, policy, critic,
                                        batch["s"], batch["g"], batch["ctx"],
                                        obs_noise_std=cfg.obs_noise, horizon=cfg.horizon)
            v = critic(batch["s"], batch["g"], batch["ctx"])
            loss_v = F.mse_loss(v, target)
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()
        for p in world_model.parameters():
            p.requires_grad_(True)

    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 关键工程点（论文强调的坑）
# ═══════════════════════════════════════════════════════════════════════════════

"""
1. 世界模型是确定性的：随机性来自已知的 sim context ξ 和观测噪声 η，把 η 放在模型外重加。
2. buffer 里的旧数据一直可用：动力学不依赖采数据的策略，所以不需要 on-policy 丢弃。
3. 短窗口拟合（k=8）+ 短 horizon 想象（H=32）+ critic bootstrap：
   模型只需在这 32 步内足够准，超过的部分交给 V_ψ，避免长程误差爆炸。
4. 软终止很重要：硬终止不可微，用 sigmoid 概率乘存活权重，梯度才能一路反传。
5. 部署时策略只吃带噪本体感知 + goal，不需要 sim context ξ（ξ 只服务训练期的世界模型）。
"""
```
