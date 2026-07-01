---
tags:
  - 代码
  - PyTorch
  - 世界模型
  - RSSM
  - MBRL
created: 2026-07-01
---

# 世界模型核心组件 PyTorch 实现

本文件基于综述 [World Models Survey](../World%20Models%20Survey.md) 中讨论的核心架构，实现三个关键组件的教学代码：

1. **RSSM（Recurrent State-Space Model）** — Dreamer 系列的核心，确定性 + 随机性潜状态分解
2. **Dreamer 风格的潜空间想象训练** — 在学到的世界模型中进行 policy rollout
3. **MuZero 风格的价值等价模型** — 不重建观测，只预测规划相关的量

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# 第一部分：RSSM（Recurrent State-Space Model）
# DreamerV1-V3 的核心组件
# 将潜状态分解为确定性部分 h_t（GRU）和随机性部分 z_t（VAE 采样）
# ============================================================================

@dataclass
class RSSMState:
    """RSSM 的状态结构"""
    deter: torch.Tensor   # 确定性潜状态 h_t, shape: (B, D_deter)
    stoch: torch.Tensor   # 随机性潜状态 z_t, shape: (B, D_stoch)

class RSSM(nn.Module):
    """
    Recurrent State-Space Model

    核心思想（来自综述 Section 2.2, 4.1）：
    - 确定性部分 h_t = f_θ(h_{t-1}, z_{t-1}, a_{t-1})  通过 GRU 传递长程记忆
    - 随机性部分 z_t ~ q_ϕ(z_t | h_t, o_t)              捕捉环境不确定性
    - 先验 p(z_t | h_t) 用于想象 rollout（此时没有观测 o_t）
    - 后验 q(z_t | h_t, o_t) 用于训练（有观测）

    DreamerV3 的 ELBO 目标：
    L_WM = Σ_t [ln p(o_t|h_t,z_t) + ln p(r_t|h_t,z_t) - β·KL(q(z_t|h_t,o_t)||p(z_t|h_t))]
    """

    def __init__(
        self,
        obs_dim: int = 1024,         # 编码后的观测维度
        action_dim: int = 6,         # 动作维度
        deter_dim: int = 512,        # 确定性状态维度 h_t
        stoch_dim: int = 32,         # 随机性状态维度 z_t（每个类别）
        stoch_classes: int = 32,     # 随机性状态的类别数（DreamerV2/V3 用离散）
        hidden_dim: int = 256,       # GRU 隐藏层维度
        embed_dim: int = 256,        # 嵌入维度
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes

        # 编码器：将观测 + 动作投影到嵌入空间
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.action_embed = nn.Linear(action_dim, embed_dim)

        # 确定性路径：GRU（综述中提到的"确定性 RNN 组件"）
        self.rnn = nn.GRUCell(
            input_size=stoch_dim * stoch_classes + embed_dim,
            hidden_size=deter_dim,
        )

        # 先验网络 p(z_t | h_t)：在想象 rollout 时使用（没有观测）
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

        # 后验网络 q(z_t | h_t, o_t)：训练时使用（有观测）
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

        # 解码器：从潜状态重建观测和预测奖励
        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim * stoch_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim * stoch_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _sample_stochastic(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 categorical 分布中采样随机性状态 z_t

        DreamerV2/V3 将 z_t 建模为 stoch_classes 个独立的 categorical 分布，
        每个有 stoch_dim 个类别。这与 VAE 中常见的 Gaussian 不同，
        离散化有助于更稳定的训练（避免 posterior collapse）。

        返回 (采样的 z_t one-hot, 采样前的 logits)
        """
        # logits shape: (B, stoch_dim * stoch_classes)
        B = logits.shape[0]
        logits = logits.view(B, self.stoch_classes, self.stoch_dim)
        dist = td.OneHotCategoricalStraightThrough(logits=logits)
        sample = dist.rsample()  # straight-through 梯度
        return sample.view(B, -1), logits.view(B, -1)

    def forward(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        obs_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[RSSMState, Dict[str, torch.Tensor]]:
        """
        单步前向传播

        Args:
            prev_state: 上一步的 RSSM 状态 (h_{t-1}, z_{t-1})
            action: 当前动作 a_{t-1}
            obs_embed: 当前观测 o_t 的嵌入（训练时有，想象时无）

        Returns:
            (新状态 (h_t, z_t), 包含 prior/post/sample 的字典)
        """
        # 将动作和上一个随机状态拼接作为 RNN 输入
        rnn_input = torch.cat([prev_state.stoch, self.action_embed(action)], dim=-1)
        deter = self.rnn(rnn_input, prev_state.deter)  # h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

        # 计算先验分布（仅依赖确定性状态）
        prior_logits = self.prior_net(deter)

        if obs_embed is not None:
            # 训练模式：用后验（有观测）
            post_logits = self.post_net(torch.cat([deter, obs_embed], dim=-1))
            stoch, _ = self._sample_stochastic(post_logits)
        else:
            # 想象模式：用先验（没有观测）
            stoch, _ = self._sample_stochastic(prior_logits)
            post_logits = prior_logits  # 想象时没有后验

        new_state = RSSMState(deter=deter, stoch=stoch)

        info = {
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "deter": deter,
        }
        return new_state, info

    def decode_obs_reward(
        self, state: RSSMState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从潜状态解码观测和奖励"""
        features = torch.cat([state.deter, state.stoch], dim=-1)
        obs_pred = self.obs_decoder(features)
        reward_pred = self.reward_decoder(features)
        return obs_pred, reward_pred

    def compute_loss(
        self,
        obs_embed: torch.Tensor,       # (T, B, obs_dim)
        actions: torch.Tensor,          # (T, B, action_dim)
        rewards: torch.Tensor,          # (T, B, 1)
        beta: float = 1.0,             # KL 正则化系数
    ) -> Dict[str, torch.Tensor]:
        """
        在完整序列上计算 DreamerV3 的 ELBO 损失

        遍历整个序列，从初始状态开始逐步前向传播。
        这是综述 Eq. 21 的实现。
        """
        T, B, _ = obs_embed.shape
        device = obs_embed.device

        # 初始化状态（全零）
        prev_state = RSSMState(
            deter=torch.zeros(B, self.deter_dim, device=device),
            stoch=torch.zeros(B, self.stoch_dim * self.stoch_classes, device=device),
        )

        obs_losses = []
        reward_losses = []
        kl_losses = []

        for t in range(T):
            # 前向传播：计算先验和后验
            new_state, info = self.forward(prev_state, actions[t], obs_embed[t])

            # 解码预测
            obs_pred, reward_pred = self.decode_obs_reward(new_state)

            # 重建损失（MSE，因为是连续嵌入空间）
            obs_loss = F.mse_loss(obs_pred, obs_embed[t])
            reward_loss = F.mse_loss(reward_pred, rewards[t])

            # KL 散度：后验 || 先验
            prior_dist = td.OneHotCategoricalStraightThrough(
                logits=info["prior_logits"].view(
                    B, self.stoch_classes, self.stoch_dim
                )
            )
            post_dist = td.OneHotCategoricalStraightThrough(
                logits=info["post_logits"].view(
                    B, self.stoch_classes, self.stoch_dim
                )
            )
            kl_loss = td.kl_divergence(post_dist, prior_dist).sum(dim=-1).mean()

            obs_losses.append(obs_loss)
            reward_losses.append(reward_loss)
            kl_losses.append(kl_loss)

            prev_state = new_state

        # 总损失
        obs_loss = torch.stack(obs_losses).mean()
        reward_loss = torch.stack(reward_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()

        total_loss = obs_loss + reward_loss + beta * kl_loss

        return {
            "total": total_loss,
            "obs": obs_loss,
            "reward": reward_loss,
            "kl": kl_loss,
        }


# ============================================================================
# 第二部分：Dreamer 风格的潜空间想象训练
# 在学到的世界模型中进行 policy rollout，优化 actor-critic
# ============================================================================

class DreamerAgent(nn.Module):
    """
    在 RSSM 潜空间中进行想象训练的 Agent

    核心思想（来自综述 Section 5.1, 5.2, 6.5）：
    - 在世界模型的潜空间中 rollout H 步
    - 使用 λ-return 传播 credit：G^λ_t = r_t + γ[(1-λ)v(s_{t+1}) + λ G^λ_{t+1}]
    - 直接从想象的 rollout 中优化 actor 和 critic
    - 这与 model-free RL 的关键区别：credit 通过潜空间前向传播而非时序差分
    """

    def __init__(
        self,
        rssm: RSSM,
        feat_dim: int = 512 + 32 * 32,   # deter_dim + stoch_dim * classes
        action_dim: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.rssm = rssm
        self.feat_dim = feat_dim
        self.action_dim = action_dim

        # Actor：从潜状态输出动作分布
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * action_dim),  # mean + std
        )

        # Critic：从潜状态输出价值估计
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_feature(self, state: RSSMState) -> torch.Tensor:
        """拼接确定性和随机性状态作为 actor-critic 的输入"""
        return torch.cat([state.deter, state.stoch], dim=-1)

    def act(self, state: RSSMState, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从当前潜状态采样动作

        Returns:
            (采样的动作, 动作的对数概率)
        """
        feat = self.get_feature(state)
        out = self.actor(feat)
        mean, std = out.chunk(2, dim=-1)
        std = F.softplus(std) + 1e-4  # 确保正的

        dist = td.Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def imagine_rollout(
        self,
        start_state: RSSMState,
        horizon: int = 15,
    ) -> Dict[str, torch.Tensor]:
        """
        在潜空间中想象 rollout（综述 Eq. 3, 4）

        从 start_state 开始，循环 H 步：
        1. 用 actor 采样动作
        2. 用 RSSM 先验预测下一个潜状态
        3. 用 critic 估计价值
        4. 用 reward decoder 预测奖励

        这是 Dreamer 的核心循环——所有 actor-critic 训练都在想象中进行。
        """
        B = start_state.deter.shape[0]
        device = start_state.deter.device

        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []

        state = start_state
        for _ in range(horizon):
            # Step 1: 采样动作
            action, log_prob = self.act(state)
            actions.append(action)
            log_probs.append(log_prob)

            # Step 2: RSSM 前向（用先验，没有观测）
            state, _ = self.rssm(state, action, obs_embed=None)
            states.append(state)

            # Step 3: 预测价值和奖励
            feat = self.get_feature(state)
            value = self.critic(feat)
            _, reward = self.rssm.decode_obs_reward(state)

            values.append(value)
            rewards.append(reward)

        return {
            "states": states,
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "rewards": torch.stack(rewards),
            "values": torch.stack(values),
        }

    def compute_imagination_loss(
        self,
        start_state: RSSMState,
        horizon: int = 15,
        gamma: float = 0.997,
        lambda_: float = 0.95,
        entropy_weight: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        在想象 rollout 上计算 actor-critic 损失

        λ-return（综述提到的核心公式）：
        G_t^λ = r_t + γ[(1 - λ) v(s_{t+1}) + λ G_{t+1}^λ]

        这与 model-free TD 的关键区别：
        - Model-free TD：需要实际环境交互 → r_{t+1}
        - Dreamer 想象：通过 RSSM.dynamics 内推 → r̂_{t+1}, ŝ_{t+1}

        这意味着 credit 通过潜空间前向传播，而非通过真实环境的反向传播。
        """
        rollout = self.imagine_rollout(start_state, horizon)

        rewards = rollout["rewards"]       # (H, B, 1)
        values = rollout["values"]         # (H, B, 1)
        log_probs = rollout["log_probs"]   # (H, B)

        # 计算 λ-return（从后往前递推）
        returns = []
        gae = torch.zeros_like(rewards[0])
        for t in reversed(range(horizon)):
            if t == horizon - 1:
                # 最后一步用 value bootstrap
                return_t = rewards[t] + gamma * values[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * lambda_ * gae
                return_t = gae + values[t]
            returns.append(return_t)

        returns = torch.stack(list(reversed(returns)))  # (H, B, 1)
        advantages = returns - values

        # Actor 损失：PPO-style clipping
        ratio = torch.exp(log_probs - log_probs.detach())
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        actor_loss = -torch.min(ratio * advantages.detach(), clip_ratio * advantages.detach()).mean()

        # Critic 损失：MSE 回归到 λ-return
        critic_loss = F.mse_loss(values, returns.detach())

        # 熵奖励（鼓励探索）
        entropy = td.Normal(
            torch.zeros_like(rollout["actions"]),
            torch.ones_like(rollout["actions"]),
        ).entropy().mean()

        total = actor_loss + 0.5 * critic_loss - entropy_weight * entropy

        return {
            "total": total,
            "actor": actor_loss,
            "critic": critic_loss,
            "entropy": entropy,
        }


# ============================================================================
# 第三部分：MuZero 风格的价值等价模型
# 不重建观测，只预测规划相关的量（policy, value, reward）
# ============================================================================

class MuZeroModel(nn.Module):
    """
    价值等价模型 —— MuZero 的核心组件

    核心思想（来自综述 Section 5.1, 6.5）：
    - 不像 Dreamer 那样重建观测 o_t
    - 只学习三个函数：
      1. h_θ(o_t) → s^0_t：表征函数（编码当前观测到潜状态）
      2. g_θ(s^k_t, a^k_t) → s^{k+1}_t, r^{k+1}_t：动力学函数（潜状态转移）
      3. f_θ(s^k_t) → p^k_t, v^k_t：预测函数（策略和价值）
    - 训练目标：最小化规划相关量的预测误差，而非重建误差
    - 这是 MuZero 在没有显式环境规则时实现超人类表现的关键
    """

    def __init__(
        self,
        obs_dim: int = 1024,
        action_dim: int = 6,
        latent_dim: int = 256,
        num_actions: int = 10,      # 离散动作空间大小（MuZero 通常用离散动作）
        hidden_dim: int = 512,
        num_simulations: int = 50,  # MCTS 模拟次数
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.num_simulations = num_simulations

        # h_θ: 表征函数
        self.representation = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # g_θ: 动力学函数（潜状态转移 + 奖励预测）
        # 输入: 潜状态 + 动作 one-hot → 输出: 下一个潜状态 + 即时奖励
        self.dynamics_state = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.dynamics_reward = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # f_θ: 预测函数（policy + value）
        # 输入: 潜状态 → 输出: 策略 logits + 标量价值
        self.prediction_policy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.prediction_value = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def initial_inference(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        h_θ + f_θ：从观测到初始潜状态 + 初始策略/价值

        这是 MCTS 根节点的输入。
        """
        # h_θ(o_t) → s^0
        latent = self.representation(obs)

        # f_θ(s^0) → policy, value
        policy = self.prediction_policy(latent)
        value = self.prediction_value(latent)

        return {
            "latent": latent,
            "policy": policy,
            "value": value,
        }

    def recurrent_inference(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        g_θ + f_θ：从潜状态和动作 → 下一个潜状态 + 奖励 + 策略 + 价值

        这是 MCTS 展开节点时的核心操作。
        """
        B = latent.shape[0]

        # g_θ(s, a) → s', r
        latent_onehot = latent  # 已经在 latent space
        action_onehot = F.one_hot(action.squeeze(-1).long(), self.num_actions).float()
        dynamics_input = torch.cat([latent_onehot, action_onehot], dim=-1)

        next_latent = self.dynamics_state(dynamics_input)
        reward = self.dynamics_reward(next_latent)

        # f_θ(s') → policy', value'
        policy = self.prediction_policy(next_latent)
        value = self.prediction_value(next_latent)

        return {
            "latent": next_latent,
            "reward": reward,
            "policy": policy,
            "value": value,
        }

    def rollout(self, obs: torch.Tensor, num_steps: int) -> Dict[str, torch.Tensor]:
        """
        在潜空间中 rollout 多步（无 MCTS，简单版）

        与 Dreamer 的想象 rollout 对比：
        - Dreamer：在 RSSM 潜空间中用 actor 采样动作
        - MuZero：在价值等价潜空间中，可以用 policy 网络的 argmax 选动作
        - 关键区别：MuZero 不预测观测，只预测 (reward, value, policy)
        """
        B = obs.shape[0]
        device = obs.device

        latents = []
        rewards = []
        values = []
        policies = []

        # 初始推理
        result = self.initial_inference(obs)
        latent = result["latent"]
        latents.append(latent)
        values.append(result["value"])

        for step in range(num_steps):
            # 用当前 policy 选动作（在 MCTS 中这里会有搜索）
            policy = self.prediction_policy(latent)
            action = torch.argmax(policy, dim=-1, keepdim=True)  # (B, 1)
            policies.append(policy)

            # 递归推理
            result = self.recurrent_inference(latent, action)
            latent = result["latent"]
            latents.append(latent)
            rewards.append(result["reward"])
            values.append(result["value"])

        return {
            "latents": torch.stack(latents),      # (T+1, B, D)
            "rewards": torch.stack(rewards),       # (T, B, 1)
            "values": torch.stack(values),         # (T+1, B, 1)
            "policies": torch.stack(policies),     # (T, B, num_actions)
        }

    def compute_loss(
        self,
        obs: torch.Tensor,
        target_values: torch.Tensor,      # MCTS 搜索得到的目标价值
        target_rewards: torch.Tensor,     # 真实环境奖励
        target_policies: torch.Tensor,    # MCTS 搜索得到的目标策略
        num_unroll_steps: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        MuZero 训练损失

        三个损失项（全部不涉及观测重建）：
        1. Value loss: 预测价值 vs MCTS 目标价值
        2. Reward loss: 预测奖励 vs 真实奖励
        3. Policy loss: 预测策略 vs MCTS 目标策略（交叉熵）
        """
        B = obs.shape[0]
        device = obs.device

        # 初始推理
        result = self.initial_inference(obs)
        latent = result["latent"]

        # 初始预测的价值误差
        value_loss = F.mse_loss(result["value"], target_values[:, 0:1])
        policy_loss = F.cross_entropy(result["policy"], target_policies[:, 0])

        reward_loss = torch.tensor(0.0, device=device)

        # 展开 K 步
        for k in range(num_unroll_steps):
            action = target_policies[:, k].unsqueeze(-1)  # 用真实动作
            result = self.recurrent_inference(latent, action)

            latent = result["latent"]
            reward_loss += F.mse_loss(result["reward"], target_rewards[:, k:k+1])
            value_loss += F.mse_loss(result["value"], target_values[:, k+1:k+2])
            policy_loss += F.cross_entropy(result["policy"], target_policies[:, k+1])

        # 平均化
        K = num_unroll_steps + 1
        value_loss = value_loss / K
        reward_loss = reward_loss / num_unroll_steps
        policy_loss = policy_loss / K

        # 总损失：价值 + 奖励 + 策略（无观测重建项！）
        total_loss = value_loss + reward_loss + policy_loss

        return {
            "total": total_loss,
            "value": value_loss,
            "reward": reward_loss,
            "policy": policy_loss,
        }


# ============================================================================
# 使用示例
# ============================================================================

def demo():
    """演示三个核心组件的用法"""
    B, T, obs_dim, action_dim = 2, 8, 1024, 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("世界模型核心组件演示")
    print("=" * 60)

    # ---- 1. RSSM 演示 ----
    print("\n--- 1. RSSM (DreamerV3 核心) ---")
    rssm = RSSM(obs_dim=obs_dim, action_dim=action_dim).to(device)

    obs = torch.randn(T, B, obs_dim, device=device)
    actions = torch.randn(T, B, action_dim, device=device)
    rewards = torch.randn(T, B, 1, device=device)

    loss = rssm.compute_loss(obs, actions, rewards)
    print(f"  Total loss: {loss['total'].item():.4f}")
    print(f"  Obs loss:   {loss['obs'].item():.4f}")
    print(f"  KL loss:    {loss['kl'].item():.4f}")

    # ---- 2. Dreamer 想象训练演示 ----
    print("\n--- 2. Dreamer 潜空间想象训练 ---")
    agent = DreamerAgent(rssm).to(device)

    start_state = RSSMState(
        deter=torch.zeros(B, rssm.deter_dim, device=device),
        stoch=torch.zeros(B, rssm.stoch_dim * rssm.stoch_classes, device=device),
    )

    imag_loss = agent.compute_imagination_loss(start_state, horizon=15)
    print(f"  Total loss:  {imag_loss['total'].item():.4f}")
    print(f"  Actor loss:  {imag_loss['actor'].item():.4f}")
    print(f"  Critic loss: {imag_loss['critic'].item():.4f}")
    print(f"  Entropy:     {imag_loss['entropy'].item():.4f}")

    # ---- 3. MuZero 演示 ----
    print("\n--- 3. MuZero 价值等价模型 ---")
    muzero = MuZeroModel(obs_dim=obs_dim, latent_dim=256).to(device)

    obs_single = torch.randn(B, obs_dim, device=device)
    rollout = muzero.rollout(obs_single, num_steps=5)
    print(f"  Latents shape:  {rollout['latents'].shape}")   # (6, B, 256)
    print(f"  Rewards shape:  {rollout['rewards'].shape}")   # (5, B, 1)
    print(f"  Values shape:   {rollout['values'].shape}")    # (6, B, 1)
    print(f"  Policies shape: {rollout['policies'].shape}")  # (5, B, 10)

    # ---- 对比：Dreamer vs MuZero ----
    print("\n--- 核心对比 ---")
    print("Dreamer 路线（RSSM）：")
    print("  ✅ 潜状态 = 确定性(h_t) + 随机性(z_t)")
    print("  ✅ 重建观测 o_t（ELBO 目标）")
    print("  ✅ 想象中训练 actor-critic")
    print("  ✅ 适合从零学习的 RL 问题")
    print()
    print("MuZero 路线（价值等价）：")
    print("  ✅ 不重建观测（只学 (reward, value, policy) 的预测）")
    print("  ✅ 用 MCTS 在潜空间做显式搜索")
    print("  ✅ 潜状态只编码规划相关信息")
    print("  ✅ 适合需要精确规划的确定性环境（如棋类、Atari）")
    print()
    print("两种路线共同体现世界模型的核心洞察（综述中心论点）：")
    print("  → 预测性抽象：压缩经验到足够简单以模拟、足够丰富以支持行动的形式")


if __name__ == "__main__":
    demo()
```
