# core/history_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from collections import deque

import numpy as np

from .base_env import Env, StepResult
from .specs import EnvSpec, SpaceSpec


@dataclass
class HistoryConfig:
    reference_path: str
    short_horizon: int = 4    # K_short in the paper
    long_horizon: int = 66    # K_long in the paper
    ref_lookahead_steps: Tuple[int, ...] = (1, 4, 7)  # future reference timesteps


class HistoryEnv(Env):
    """
    Wraps a base Env and augments observations with I/O history and an optional
    command vector.

    Observation dict keys:
        - "ref": reference trajectory lookahead positions
        - "short_history": K_short * (obs_dim + act_dim) flattened
        - "long_history": K_long * (obs_dim + act_dim) flattened
        - "command": command vector
    """

    def __init__(
        self,
        base_env: Env,
        hist_cfg: HistoryConfig,
        command_dim: int = 0,
    ):
        self.base_env = base_env
        self.hist_cfg = hist_cfg
        self.command_dim = command_dim

        # History buffers of (flattened_obs, action)
        self._short: deque = deque(maxlen=self.hist_cfg.short_horizon)
        self._long: deque = deque(maxlen=self.hist_cfg.long_horizon)

        # Compute flattened obs dim from base env spec
        # Base env returns dict: euler(3) + lin_vel(3) + joint_pos + joint_vel
        self.base_obs_dim = base_env.spec.obs.shape[0]
        self.act_dim = base_env.spec.act.shape[0]
        self.pair_dim = self.base_obs_dim + self.act_dim

        # Load reference data
        self.r_data = np.load(self.hist_cfg.reference_path)
        self.time = self.r_data["time"]
        self.ref_qpos_dim = self.r_data["qpos"].shape[1]

        # Compute observation space dimensions
        short_dim = self.hist_cfg.short_horizon * self.pair_dim
        long_dim = self.hist_cfg.long_horizon * self.pair_dim
        ref_dim = len(self.hist_cfg.ref_lookahead_steps) * self.ref_qpos_dim

        # Total flattened obs dim for spec
        total_obs_dim = ref_dim + short_dim + long_dim + command_dim

        self.spec = EnvSpec(
            obs=SpaceSpec(shape=(total_obs_dim,), dtype=np.float32),
            act=base_env.spec.act,
        )

        # Store individual dims for reference
        self.obs_dims = {
            "ref": ref_dim,
            "short_history": short_dim,
            "long_history": long_dim,
            "command": command_dim,
        }

        # Current command
        self._command = np.zeros(command_dim, dtype=np.float32)
        self._t = 0

    def index(self, target_time: float) -> int:
        """Find the index of the closest time in the reference data."""
        return int(np.argmin(np.abs(self.time - target_time)))

    @property
    def dt(self) -> float:
        return getattr(self.base_env, "dt", 0.0)

    def set_command(self, command: np.ndarray) -> None:
        assert self.command_dim == command.shape[0]
        self._command = command.astype(np.float32)

    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten the base env observation dict into a single array."""
        # Consistent ordering: euler, lin_vel, joint_pos, joint_vel
        return np.concatenate([
            obs_dict["euler"],
            obs_dict["lin_vel"],
            obs_dict["joint_pos"],
            obs_dict["joint_vel"],
        ], axis=0).astype(np.float32)

    def _get_ref_obs(self, time_sec: float) -> np.ndarray:
        """Get reference positions at future lookahead steps."""
        r_idx = self.index(time_sec)
        max_idx = len(self.time) - 1

        ref_positions = []
        for offset in self.hist_cfg.ref_lookahead_steps:
            future_idx = min(r_idx + offset, max_idx)
            joint_pos = self.r_data["qpos"][future_idx]
            ref_positions.append(joint_pos)

        return np.concatenate(ref_positions, axis=0).astype(np.float32)

    def _flatten_history(self, history: deque, max_len: int) -> np.ndarray:
        """Flatten history buffer, padding with zeros if needed."""
        hist_list = list(history)
        pad_count = max_len - len(hist_list)

        if pad_count > 0:
            zero_pair = np.zeros(self.pair_dim, dtype=np.float32)
            hist_list = [zero_pair] * pad_count + hist_list

        return np.concatenate(hist_list, axis=0).astype(np.float32)

    def _build_obs_dict(self) -> Dict[str, np.ndarray]:
        """Build the augmented observation dictionary."""
        return {
            "ref": self._get_ref_obs(self._t * self.dt),
            "short_history": self._flatten_history(self._short, self.hist_cfg.short_horizon),
            "long_history": self._flatten_history(self._long, self.hist_cfg.long_horizon),
            "command": self._command.copy(),
        }

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        self._short.clear()
        self._long.clear()
        self._t = 0

        self.base_env.reset(seed=seed)
        return self._build_obs_dict()

    def step(self, action: np.ndarray) -> StepResult:
        res = self.base_env.step(action)
        self._t += 1

        # Flatten base obs and concatenate with action for history
        flat_obs = self._flatten_obs(res.obs)
        history_pair = np.concatenate([flat_obs, action.copy()], axis=0)

        # Store in history buffers
        self._short.append(history_pair)
        self._long.append(history_pair)

        obs_dict = self._build_obs_dict()

        return StepResult(
            obs=obs_dict,
            reward=res.reward,
            done=res.done,
            info=res.info,
            frame=getattr(res, "frame", None),
        )

    def close(self) -> None:
        self.base_env.close()

    def set_camera(
        self,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ):
        self.base_env.set_camera(
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            lookat=lookat,
        )