"""Proximal Policy Optimization (PPO) algorithm implementation.

This module implements PPO, a state-of-the-art on-policy reinforcement learning
algorithm. PPO uses clipped surrogate objectives to prevent large policy updates
and includes GAE-Lambda for advantage estimation.

Reference:
    Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
    https://arxiv.org/abs/1707.06347
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm.
    
    Attributes:
        gamma: Discount factor for future rewards (0 < gamma <= 1).
        lam: GAE-Lambda parameter for bias-variance trade-off (0 < lam <= 1).
            Higher values reduce variance but increase bias.
        clip_ratio: PPO clipping ratio (typically 0.1-0.3). Larger values allow
            bigger policy updates.
        lr: Learning rate for Adam optimizer.
        train_iters: Number of PPO update iterations per batch.
        batch_size: Mini-batch size for each update iteration.
        value_coef: Coefficient for value function loss in total loss.
        entropy_coef: Entropy bonus coefficient to encourage exploration
            (0 = no entropy bonus).
        max_grad_norm: Maximum gradient norm for gradient clipping.
    """
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
    """Compute Generalized Advantage Estimation (GAE-Lambda).
    
    GAE provides a bias-variance trade-off for advantage estimation by
    combining TD(0) and Monte Carlo estimates. When lam=1, it's equivalent
    to Monte Carlo; when lam=0, it's TD(0).
    
    Args:
        rewards: Array of rewards for each timestep, shape [T].
        values: Array of value function estimates, shape [T].
        dones: Array of done flags (True if episode ended), shape [T].
        last_value: Bootstrap value for the state after the last timestep.
        gamma: Discount factor.
        lam: GAE-Lambda parameter (0 = TD(0), 1 = Monte Carlo).
        
    Returns:
        Tuple of (advantages, returns) arrays, both shape [T].
        Returns are computed as advantages + values.
        
    Reference:
        Schulman et al. "High-Dimensional Continuous Control Using
        Generalized Advantage Estimation" (2016)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    
    # Compute advantages backwards through time
    for t in reversed(range(T)):
        # Mask out next value if episode terminated
        next_nonterminal = 1.0 - float(dones[t])
        next_value = last_value if t == T - 1 else values[t + 1]
        
        # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        
        # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    
    # Returns = advantages + values (Q(s,a) = A(s,a) + V(s))
    returns = adv + values
    return adv, returns


class PPO:
    """Proximal Policy Optimization algorithm.
    
    PPO is an on-policy algorithm that uses clipped surrogate objectives
    to update the policy. It alternates between collecting rollouts and
    performing multiple gradient updates on the collected data.
    
    Attributes:
        actor_critic: Neural network policy (must implement _dist() and value()).
        cfg: PPO configuration.
        device: Device to run computations on ("cpu" or "cuda").
        optimizer: Adam optimizer for policy updates.
    """
    def __init__(self, actor_critic: nn.Module, cfg: PPOConfig, device: str = "cpu"):
        """Initialize PPO algorithm.
        
        Args:
            actor_critic: Actor-critic neural network.
            cfg: PPO configuration.
            device: Device to run on ("cpu" or "cuda").
        """
        self.actor_critic = actor_critic.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = Adam(self.actor_critic.parameters(), lr=cfg.lr)

    def update(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform PPO policy update on a batch of rollouts.
        
        Updates the policy using clipped surrogate objective and value
        function learning. Performs multiple iterations over the data
        with random mini-batches.
        
        Args:
            batch: Dictionary containing:
                - 'obs': Observation dictionary with arrays of shape [T, ...]
                - 'actions': Actions array of shape [T, act_dim]
                - 'log_probs': Old log probabilities of shape [T]
                - 'returns': Monte Carlo returns of shape [T]
                - 'advantages': GAE advantages of shape [T]
                
        Returns:
            Dictionary of training metrics:
                - 'policy_loss': Average clipped surrogate loss
                - 'value_loss': Average value function MSE loss
                - 'entropy': Average policy entropy
                - 'kl': Average KL divergence between old and new policy
        """
        # Convert numpy arrays to PyTorch tensors
        obs = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch["obs"].items()
        }
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        # Normalize advantages for stability (reduces variance)
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

        # Perform multiple PPO update iterations
        for _ in range(self.cfg.train_iters):
            # Random permutation for mini-batch sampling
            idx = torch.randperm(num_samples, device=self.device)
            
            # Process data in mini-batches
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]

                # Extract mini-batch
                mb_obs = {k: v[mb_idx] for k, v in obs.items()}
                mb_actions = actions[mb_idx]
                mb_old_logp = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                # Recompute log probabilities and values with current policy
                dist = self._make_dist(mb_obs)
                new_logp = dist.log_prob(mb_actions).sum(-1)  # Sum over action dims
                entropy = dist.entropy().sum(-1).mean()  # Average entropy

                values = self.actor_critic.value(mb_obs)

                # PPO clipped surrogate objective
                # ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv  # Unclipped objective
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                ) * mb_adv  # Clipped objective
                policy_loss = -torch.min(surr1, surr2).mean()  # Minimize negative advantage

                # Value function learning (MSE with returns)
                value_loss = torch.nn.functional.mse_loss(values, mb_returns)

                # Combined loss: policy + value - entropy (entropy encourages exploration)
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                # Compute KL divergence for monitoring (approximate)
                with torch.no_grad():
                    kl = (mb_old_logp - new_logp).mean().item()

                # Accumulate metrics
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
        """Create action distribution from observations.
        
        Args:
            obs: Observation tensor (or dict) for batch of states.
            
        Returns:
            Normal distribution over actions.
        """
        dist, _ = self.actor_critic._dist(obs)
        return dist