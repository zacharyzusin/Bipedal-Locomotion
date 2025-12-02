# algorithms/ppo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_iters: int = 80
    batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GAE-Lambda advantage + returns.
    rewards, values, dones: shape [T]
    last_value: scalar bootstrap at t = T
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - float(dones[t])
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


class PPO:
    def __init__(self, actor_critic: nn.Module, cfg: PPOConfig, device: str = "cpu"):
        self.actor_critic = actor_critic.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = Adam(self.actor_critic.parameters(), lr=cfg.lr)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        batch dict has keys:
        'obs' (dict of arrays), 'actions', 'log_probs', 'returns', 'advantages'
        """
        # Convert obs dict to tensors
        obs = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch["obs"].items()
        }
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get num_samples from any obs key
        num_samples = next(iter(obs.values())).shape[0]
        batch_size = self.cfg.batch_size

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
        }

        for _ in range(self.cfg.train_iters):
            idx = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]

                mb_obs = {k: v[mb_idx] for k, v in obs.items()}
                mb_actions = actions[mb_idx]
                mb_old_logp = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                # Recompute log_probs and values
                dist = self._make_dist(mb_obs)
                new_logp = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                values = self.actor_critic.value(mb_obs)

                # Policy loss
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = torch.nn.functional.mse_loss(values, mb_returns)

                # Combined loss
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    kl = (mb_old_logp - new_logp).mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["kl"] += kl

        # Average metrics
        num_updates = self.cfg.train_iters * max(1, num_samples // batch_size)
        for k in metrics:
            metrics[k] /= num_updates
        return metrics

    def _make_dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        dist, _ = self.actor_critic._dist(obs)
        return dist