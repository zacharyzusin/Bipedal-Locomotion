"""Actor-Critic neural network policy.

This module implements a shared-parameter actor-critic architecture where
the policy (actor) and value function (critic) share a common feature
extractor. The policy outputs a Gaussian distribution over continuous
actions, and the value function estimates expected returns.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from core.specs import EnvSpec


@dataclass
class PolicyOutput:
    """Output from policy forward pass.
    
    Attributes:
        action: Sampled actions, shape [B, act_dim].
        log_prob: Log probability of actions, shape [B].
        value: Value function estimates, shape [B].
        mean: Mean of action distribution, shape [B, act_dim].
        std: Standard deviation of action distribution, shape [act_dim]
            (shared across batch).
    """
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class ActorCritic(nn.Module):
    """Actor-Critic policy network with shared feature extractor.
    
    The network consists of:
    - Shared body: Multi-layer MLP with Tanh activations
    - Policy head: Outputs mean of Gaussian action distribution
    - Value head: Outputs scalar value estimate
    
    The policy uses a learnable log_std parameter (state-independent
    standard deviation) for the action distribution.
    
    Attributes:
        obs_dim: Observation space dimensionality.
        act_dim: Action space dimensionality.
        body: Shared feature extractor (MLP).
        mean_head: Linear layer mapping features to action means.
        log_std: Learnable parameter for action standard deviation.
        value_head: Linear layer mapping features to value estimates.
    """
    def __init__(
        self,
        spec: EnvSpec,
        hidden_sizes=(64, 64),
    ):
        """Initialize Actor-Critic network.
        
        Args:
            spec: Environment specification (defines obs and action dimensions).
            hidden_sizes: Tuple of hidden layer sizes for shared MLP.
                Default (64, 64) creates a 2-layer network.
        """
        super().__init__()
        self.obs_dim = spec.obs.shape[0]
        self.act_dim = spec.act.shape[0]

        # Build shared feature extractor (MLP with Tanh activations)
        layers = []
        last = self.obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)

        # Policy head: outputs mean of action distribution
        self.mean_head = nn.Linear(last, self.act_dim)
        # Learnable log standard deviation (state-independent)
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))

        # Value head: outputs scalar value estimate
        self.value_head = nn.Linear(last, 1)

    def _concat_obs(self, obs: dict) -> torch.Tensor:
        """Concatenate observation dictionary into a single tensor.
        
        Args:
            obs: Dictionary of observation arrays/tensors.
            
        Returns:
            Concatenated observation tensor, shape [B, obs_dim].
        """
        for k, v in obs.items():
            obs[k] = torch.as_tensor(v, dtype=torch.float32)
        # Sort keys for consistent ordering
        return torch.cat([obs[k] for k in sorted(obs.keys())], dim=-1)

    def _forward_body(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared feature extractor.
        
        Args:
            obs: Observation tensor, shape [B, obs_dim].
            
        Returns:
            Feature tensor, shape [B, hidden_size].
        """
        return self.body(obs)

    def _dist(self, obs: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """Create action distribution from observations.
        
        Args:
            obs: Observation tensor or dict, shape [B, obs_dim] or dict.
            
        Returns:
            Tuple of (action distribution, feature tensor).
        """
        # Handle dict observations
        if isinstance(obs, dict):
            obs = self._concat_obs(obs)
        
        x = self._forward_body(obs)
        mean = self.mean_head(x)  # [B, act_dim]
        std = self.log_std.exp()  # [act_dim] -> broadcast to [B, act_dim]
        dist = torch.distributions.Normal(mean, std)
        return dist, x

    def forward(self, obs: torch.Tensor) -> PolicyOutput:
        """Forward pass for training (samples action and computes value).
        
        Used during PPO updates to compute policy loss and value loss.
        
        Args:
            obs: Observation tensor or dict, shape [B, obs_dim] or dict.
            
        Returns:
            PolicyOutput containing action, log_prob, value, mean, and std.
        """
        dist, x = self._dist(obs)
        action = dist.rsample()  # Reparameterized sampling (differentiable)
        log_prob = dist.log_prob(action).sum(-1)  # Sum over action dimensions
        value = self.value_head(x).squeeze(-1)  # Remove singleton dimension
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            value=value,
            mean=dist.loc,
            std=dist.scale,
        )

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action for rollout collection.
        
        Args:
            obs: Observation tensor or dict, shape [B, obs_dim] or dict.
            deterministic: If True, return mean action (no exploration).
                If False, sample from distribution.
                
        Returns:
            Tuple of (action, log_prob) tensors.
            - action: shape [B, act_dim]
            - log_prob: shape [B]
        """
        dist, _ = self._dist(obs)
        if deterministic:
            action = dist.loc  # Use mean for evaluation
        else:
            action = dist.rsample()  # Sample for exploration
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value function estimate.
        
        Args:
            obs: Observation tensor or dict, shape [B, obs_dim] or dict.
            
        Returns:
            Value estimates, shape [B].
        """
        _, x = self._dist(obs)
        return self.value_head(x).squeeze(-1)