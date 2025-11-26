# core/mujoco_env.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, Union, Mapping

import numpy as np
import mujoco

from .base_env import Env, StepResult
from .specs import EnvSpec, SpaceSpec

RewardReturn = Union[
    float,
    tuple[float, Mapping[str, float]],
]
RewardFn = Callable[[mujoco.MjModel, mujoco.MjData], RewardReturn]

DoneFn = Callable[[mujoco.MjModel, mujoco.MjData, int], bool]


@dataclass
class MujocoEnvConfig:
    xml_path: str | Path
    episode_length: int = 1000
    frame_skip: int = 1
    ctrl_scale: float = 1.0
    reward_fn: Optional[RewardFn] = None
    done_fn: Optional[DoneFn] = None
    reset_noise_scale: float = 0.0
    
    # Rendering-related (for evaluation / streaming)
    render: bool = False
    width: int = 640
    height: int = 480
    camera_id: int | str | None = None  # can use named or numeric camera


class MujocoEnv(Env):
    def __init__(self, cfg: MujocoEnvConfig):
        self.cfg = cfg
        xml_path = str(cfg.xml_path)

        # Core MuJoCo objects
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Make sure we start from the model's default (qpos0, etc.)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Initial state
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Episode time-step counter
        self._t = 0

        # RNG for any noise / randomization
        self._rng = np.random.default_rng()

        # Obs and action specs (default: qpos + qvel, direct control)
        obs_dim = self.model.nq + self.model.nv
        act_dim = self.model.nu

        self.spec = EnvSpec(
            obs=SpaceSpec(shape=(obs_dim,), dtype=np.float32),
            act=SpaceSpec(shape=(act_dim,), dtype=np.float32),
        )

        # Reward / done callbacks
        self._reward_fn: RewardFn = cfg.reward_fn or self._default_reward_fn
        self._done_fn: DoneFn = cfg.done_fn or self._default_done_fn

        # Optional renderer for offscreen rendering
        self._renderer: Optional[mujoco.Renderer] = None
        if cfg.render:
            self._renderer = mujoco.Renderer(
                self.model, height=cfg.height, width=cfg.width
            )
            # optional: set camera here if desired; we’ll do it on each render

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        qpos = np.asarray(self.data.qpos, dtype=np.float32)
        qvel = np.asarray(self.data.qvel, dtype=np.float32)
        return np.concatenate([qpos, qvel], axis=0)

    def _set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        assert qpos.shape == (self.model.nq,)
        assert qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        # Ensure correct shape + dtype
        action = np.asarray(action, dtype=np.float32).reshape(self.spec.act.shape)

        # Optional rescaling
        if self.cfg.ctrl_scale != 1.0:
            action = self.cfg.ctrl_scale * action

        # Optional clipping to actuator control ranges if defined
        if self.model.nu > 0 and self.model.actuator_ctrlrange.size != 0:
            low = self.model.actuator_ctrlrange[:, 0]
            high = self.model.actuator_ctrlrange[:, 1]
            action = np.clip(action, low, high)

        self.data.ctrl[:] = action
        return action

    def _render_frame(self) -> Optional[np.ndarray]:
        if self._renderer is None:
            return None

        if self.cfg.camera_id is not None:
            self._renderer.update_scene(self.data, camera=self.cfg.camera_id)
        else:
            self._renderer.update_scene(self.data)

        rgb = self._renderer.render()  # float32 in [0,1], shape (H,W,3)
        frame = (rgb * 255).astype(np.uint8)
        return frame

    # -------------------------------------------------------------------------
    # Default reward & done
    # -------------------------------------------------------------------------
    def _default_reward_fn(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        # You’ll override this per-task; 0 is safe default.
        return 0.0

    def _default_done_fn(
        self, model: mujoco.MjModel, data: mujoco.MjData, t: int
    ) -> bool:
        # Time-limit termination
        return t >= self.cfg.episode_length

    # -------------------------------------------------------------------------
    # Env interface
    # -------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0

        # Fully reset MuJoCo internal state
        mujoco.mj_resetData(self.model, self.data)

        qpos = self._init_qpos.copy()
        qvel = self._init_qvel.copy()

        s = self.cfg.reset_noise_scale
        if s > 0.0:
            # noise in [-s, s] for each element
            qpos += self._rng.uniform(-s, s, size=qpos.shape)
            qvel += self._rng.uniform(-s, s, size=qvel.shape)
        
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        
        # clear controls
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def step(self, action: np.ndarray) -> StepResult:
        applied_action = self._apply_action(action)

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._t += 1

        obs = self._get_obs()

        reward_raw = self._reward_fn(self.model, self.data)
        if isinstance(reward_raw, tuple):
            reward, reward_components = reward_raw
        else:
            reward = float(reward_raw)
            reward_components = {}

        done = self._done_fn(self.model, self.data, self._t)

        info: Dict[str, Any] = {
            "t": self._t,
            "applied_action": applied_action,
            "reward_components": reward_components,
        }

        if done:
            info["episode_length"] = self._t

        frame = self._render_frame()

        return StepResult(
            obs=obs,
            reward=float(reward),
            done=done,
            info=info,
            frame=frame,
        )


    def close(self) -> None:
        # Renderer holds GL resources; cleanly drop reference
        self._renderer = None