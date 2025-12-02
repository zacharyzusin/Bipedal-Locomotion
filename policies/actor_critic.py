# policies/actor_critic.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from core.specs import EnvSpec


@dataclass
class PolicyOutput:
    action: torch.Tensor        # [B, act_dim]
    log_prob: torch.Tensor      # [B]
    value: torch.Tensor         # [B]
    mean: torch.Tensor          # [B, act_dim]
    std: torch.Tensor           # [act_dim]


class ActorCritic(nn.Module):
    def __init__(
        self,
        spec: EnvSpec,
        hidden_sizes=(64, 64),
    ):
        super().__init__()
        self.obs_dim = spec.obs.shape[0]
        self.act_dim = spec.act.shape[0]

        layers = []
        last = self.obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)

        self.mean_head = nn.Linear(last, self.act_dim)
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))

        self.value_head = nn.Linear(last, 1)

    def _concat_obs(self, obs: dict) -> torch.Tensor:
        for k, v in obs.items():
            obs[k] = torch.as_tensor(v, dtype=torch.float32)
        return torch.cat([obs[k] for k in sorted(obs.keys())], dim=-1)

    def _forward_body(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)

    def _dist(self, obs: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        obs = self._concat_obs(obs)
        x = self._forward_body(obs)
        mean = self.mean_head(x)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist, x

    def forward(self, obs: torch.Tensor) -> PolicyOutput:
        """
        For training: sample action, compute log_prob + value.
        obs: [B, obs_dim]
        """
        dist, x = self._dist(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.value_head(x).squeeze(-1)
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            value=value,
            mean=dist.loc,
            std=dist.scale,
        )

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For rollout: returns (action, log_prob)
        obs: [B, obs_dim]
        """
        dist, _ = self._dist(obs)
        if deterministic:
            action = dist.loc
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        _, x = self._dist(obs)
        return self.value_head(x).squeeze(-1)