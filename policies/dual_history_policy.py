# policies/dual_history_policy.py
from __future__ import annotations
from typing import Dict, Tuple, Union
import torch
from torch import nn
import torch.distributions as D
import numpy as np
from core.specs import EnvSpec


class DualHistoryActorCritic(nn.Module):
    """
    https://arxiv.org/abs/2401.16889
    Dual-history policy:
    - Long history encoded with 1D CNN
    - Short history + reference motion + command + CNN(long) go into MLP
    - Output: Gaussian over normalized actions (tanh on mean)
    
    Expects observation dict with keys:
        - "short_history": (short_horizon * pair_dim,) flattened
        - "long_history": (long_horizon * pair_dim,) flattened
        - "ref": (num_lookahead * ref_dim,) reference positions
        - "command": (command_dim,) command vector
    """

    def __init__(
        self,
        spec: EnvSpec,
        pair_dim: int,  # obs_dim + act_dim per timestep
        short_horizon: int,
        long_horizon: int,
        ref_dim: int,
        command_dim: int = 0,
        hidden_size: int = 512,
        act_std: float = 0.1,
    ):
        super().__init__()
        self.spec = spec
        self.pair_dim = pair_dim
        self.short_horizon = short_horizon
        self.long_horizon = long_horizon
        self.ref_dim = ref_dim
        self.command_dim = command_dim

        act_dim = spec.act.shape[0]

        # Dimension calculations
        self.short_dim = short_horizon * pair_dim
        self.long_dim = long_horizon * pair_dim

        # CNN encoder for long history
        # Input shape: (B, pair_dim, long_horizon) - channels first
        self.long_encoder = nn.Sequential(
            nn.Conv1d(pair_dim, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size with a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, pair_dim, long_horizon)
            cnn_out_dim = self.long_encoder(dummy).shape[-1]
        self.cnn_out_dim = cnn_out_dim

        # MLP input: short_hist + ref + command + CNN(long_hist)
        mlp_input_dim = self.short_dim + ref_dim + command_dim + cnn_out_dim

        self.base_net = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.v_head = nn.Linear(hidden_size, 1)

        # State-independent log_std
        self.log_std = nn.Parameter(
            torch.full((act_dim,), float(np.log(act_std)))
        )

    @classmethod
    def from_history_env(
        cls,
        hist_env,  # HistoryEnv instance
        hidden_size: int = 512,
        act_std: float = 0.1,
    ) -> "DualHistoryActorCritic":
        """
        Convenience constructor that extracts dimensions from a HistoryEnv.
        """
        return cls(
            spec=hist_env.spec,
            pair_dim=hist_env.pair_dim,
            short_horizon=hist_env.hist_cfg.short_horizon,
            long_horizon=hist_env.hist_cfg.long_horizon,
            ref_dim=hist_env.obs_dims["ref"],
            command_dim=hist_env.obs_dims["command"],
            hidden_size=hidden_size,
            act_std=act_std,
        )

    def _process_obs(
        self, obs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract components from observation dict or convert numpy arrays.
        Returns: (short_history, long_history, ref, command)
        """
        if isinstance(obs, dict):
            short = obs["short_history"]
            long = obs["long_history"]
            ref = obs["ref"]
            cmd = obs["command"]

            # Convert numpy arrays to tensors if needed
            if isinstance(short, np.ndarray):
                short = torch.from_numpy(short).float()
            if isinstance(long, np.ndarray):
                long = torch.from_numpy(long).float()
            if isinstance(ref, np.ndarray):
                ref = torch.from_numpy(ref).float()
            if isinstance(cmd, np.ndarray):
                cmd = torch.from_numpy(cmd).float()
        else:
            # Assume flattened tensor: [short | long | ref | command]
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float()
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            idx = 0
            short = obs[:, idx:idx + self.short_dim]
            idx += self.short_dim
            long = obs[:, idx:idx + self.long_dim]
            idx += self.long_dim
            ref = obs[:, idx:idx + self.ref_dim]
            idx += self.ref_dim
            cmd = obs[:, idx:idx + self.command_dim]

        # Ensure batch dimension
        if short.dim() == 1:
            short = short.unsqueeze(0)
            long = long.unsqueeze(0)
            ref = ref.unsqueeze(0)
            cmd = cmd.unsqueeze(0)

        return short, long, ref, cmd

    def _encode_long_history(self, long_hist: torch.Tensor) -> torch.Tensor:
        """
        Reshape and encode long history through CNN.
        Input: (B, long_horizon * pair_dim)
        Output: (B, cnn_out_dim)
        """
        B = long_hist.shape[0]
        # Reshape to (B, pair_dim, long_horizon) for Conv1d
        long_reshaped = long_hist.view(B, self.long_horizon, self.pair_dim)
        long_reshaped = long_reshaped.transpose(1, 2)  # (B, pair_dim, long_horizon)
        return self.long_encoder(long_reshaped)

    def _forward_body(
        self, obs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Shared trunk:
        1. Encode long history with CNN
        2. Concatenate: short_hist + ref + command + long_embedding
        3. Pass through MLP
        Returns features h for policy and value heads.
        """
        short, long, ref, cmd = self._process_obs(obs)

        # Encode long history
        long_emb = self._encode_long_history(long)

        # Concatenate all components
        x = torch.cat([short, ref, cmd, long_emb], dim=-1)
        h = self.base_net(x)
        return h

    def _dist(
        self, obs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[D.Normal, torch.Tensor]:
        """
        Construct action distribution.
        Returns: (distribution, features)
        """
        h = self._forward_body(obs)
        mu = torch.tanh(self.mu_head(h))
        std = self.log_std.exp()
        dist = D.Normal(mu, std)
        return dist, h

    def forward(
        self, obs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (mu, value, log_std)
        """
        h = self._forward_body(obs)
        mu = torch.tanh(self.mu_head(h))
        value = self.v_head(h).squeeze(-1)
        return mu, value, self.log_std

    def act(
        self,
        obs: Union[Dict[str, torch.Tensor], torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action for rollout.
        Returns: (action, log_prob)
        """
        dist, _ = self._dist(obs)
        if deterministic:
            action = dist.loc
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def value(
        self, obs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Value function V(s).
        """
        h = self._forward_body(obs)
        return self.v_head(h).squeeze(-1)