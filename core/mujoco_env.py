# core/mujoco_env.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, Union, Mapping

import numpy as np
import mujoco

from .base_env import Env, StepResult
from .specs import EnvSpec, SpaceSpec
from control.pd import PDConfig, PDController

RewardReturn = Union[
    float,
    tuple[float, Mapping[str, float]],
]
RewardFn = Callable[..., RewardReturn]

DoneFn = Callable[[mujoco.MjModel, mujoco.MjData, int], bool]

@dataclass
class CameraConfig:
    distance: float = 3.0
    azimuth: float = 90.0
    elevation: float = -20.0
    lookat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))

@dataclass
class MujocoEnvConfig:
    xml_path: str | Path
    episode_length: int = 1000
    frame_skip: int = 1
    pd_cfg: Optional[PDConfig] = None
    reward_fn: Optional[RewardFn] = None
    done_fn: Optional[DoneFn] = None
    reset_noise_scale: float = 0.0
    
    # Rendering-related (for evaluation / streaming)
    render: bool = False
    width: int = 640
    height: int = 480
    camera_id: int | str | None = None  # can use named or numeric camera

    # for generic gait/IK calibration
    base_site: Optional[str] = None
    left_foot_site: Optional[str] = None
    right_foot_site: Optional[str] = None

    # recroding options
    path: Optional[Path] = None  # path to save recorded data
    record_joints: bool = False  # whether to record joint positions

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

        # Determine root joint type and set appropriate offsets
        # mjtJoint: mjJNT_FREE=0, mjJNT_BALL=1, mjJNT_SLIDE=2, mjJNT_HINGE=3
        self._root_joint_type = self.model.jnt_type[0] if self.model.njnt > 0 else -1
        
        if self._root_joint_type == mujoco.mjtJoint.mjJNT_FREE:
            # 3D free joint: 7 qpos (x,y,z,qw,qx,qy,qz), 6 qvel
            self._qpos_offset = 7
            self._qvel_offset = 6
            self._is_2d = False
        else:
            # 2D planar (e.g., Walker2D): typically 3 qpos (x,z,angle), 3 qvel
            # Count DOFs for the root body (usually first few joints are the "root")
            # For Walker2D: rootx (slide), rootz (slide), rooty (hinge) = 3 qpos, 3 qvel
            self._qpos_offset = 3
            self._qvel_offset = 3
            self._is_2d = True
        
        # Initial state
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Episode time-step counter
        self._t = 0
        self.dt = float(self.model.opt.timestep * self.cfg.frame_skip)

        # Expose neutral hip/foot info for controllers
        self.hip_height: Optional[float] = None
        self.foot_base_pos: Dict[str, np.ndarray] = {}

        # Expose neutral hip/foot info for controllers
        self.hip_height: Optional[float] = None
        self.foot_base_pos: Dict[str, np.ndarray] = {}

        # setup pd controller
        if self.cfg.pd_cfg is not None:
            self.pd_controller = PDController(self.cfg.pd_cfg)
            
        if (
            self.cfg.base_site is not None
            and self.cfg.left_foot_site is not None
            and self.cfg.right_foot_site is not None
        ):
            base_id = self.model.site(self.cfg.base_site).id
            base_pos = self.data.site_xpos[base_id].copy()
            self.hip_height = float(base_pos[2])

            left_sid = self.model.site(self.cfg.left_foot_site).id
            right_sid = self.model.site(self.cfg.right_foot_site).id

            left_pos = self.data.site_xpos[left_sid].copy()
            right_pos = self.data.site_xpos[right_sid].copy()

            # Hip frame, sagittal (x,z)
            self.foot_base_pos["left"] = np.array(
                [left_pos[0] - base_pos[0],
                 left_pos[2] - base_pos[2]],
                dtype=np.float32,
            )
            self.foot_base_pos["right"] = np.array(
                [right_pos[0] - base_pos[0],
                 right_pos[2] - base_pos[2]],
                dtype=np.float32,
            )

        # RNG for any noise / randomization
        self._rng = np.random.default_rng()

        # Obs and action specs (default: base euler + base lin vel + joint pos + joint vel)
        # Adjust for root joint type
        obs_dim = 3 + 3 + (self.model.nq - self._qpos_offset) + (self.model.nv - self._qvel_offset)
        act_dim = self.model.nu

        self.spec = EnvSpec(
            obs=SpaceSpec(shape=(obs_dim,), dtype=np.float32),
            act=SpaceSpec(shape=(act_dim,), dtype=np.float32),
        )

        self.spec = EnvSpec(
            obs=SpaceSpec(shape=(obs_dim,), dtype=np.float32),
            act=SpaceSpec(shape=(act_dim,), dtype=np.float32),
        )

        # Reward / done callbacks
        self._reward_fn: RewardFn = cfg.reward_fn or self._default_reward_fn
        self._done_fn: DoneFn = cfg.done_fn or self._default_done_fn

        # Camera configuration for rendering
        self._camera_config = CameraConfig()

        # Create a MuJoCo camera object for custom camera control
        self._mjv_camera = mujoco.MjvCamera()
        # Initialize with default values
        self._mjv_camera.distance = self._camera_config.distance
        self._mjv_camera.azimuth = self._camera_config.azimuth
        self._mjv_camera.elevation = self._camera_config.elevation
        self._mjv_camera.lookat[:] = self._camera_config.lookat

        # Optional renderer for offscreen rendering
        self._renderer: Optional[mujoco.Renderer] = None
        if cfg.render:
            self._renderer = mujoco.Renderer(
                self.model, height=cfg.height, width=cfg.width
            )

        # Set up joint recording
        if cfg.record_joints:
            cfg.path = Path(cfg.path) if cfg.path is not None else None
            self._recorded = {"time": [], "qpos": [], "qvel": []}
        else:
            self._recorded = {}

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------
    def _record_obs(self) -> None:
        """Record actuated joint positions and velocities."""
        self._recorded["time"].append(self._t * self.dt)
        # Record only actuated joint positions (skip root DOFs)
        self._recorded["qpos"].append(self.data.qpos[self._qpos_offset:].copy())
        # Record only actuated joint velocities (skip root DOFs)
        self._recorded["qvel"].append(self.data.qvel[self._qvel_offset:].copy())

    def _save_recording(self) -> None:
        if self.cfg.path is None or not self.cfg.record_joints:
            return
        save_path = self.cfg.path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            time=np.array(self._recorded["time"]),
            qpos=np.array(self._recorded["qpos"]),
            qvel=np.array(self._recorded["qvel"]),
        )

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        if self.cfg.record_joints:
            self._record_obs()
        
        if self._is_2d:
            # 2D model (e.g., Walker2D): root is (x, z, angle)
            # Extract angle and convert to "euler-like" format
            root_angle = self.data.qpos[2]  # rooty (pitch)
            base_euler = np.array([0.0, root_angle, 0.0], dtype=np.float32)
            
            # Root velocities: (vx, vz, angular_vel)
            base_lin_vel = np.array([self.data.qvel[0], 0.0, self.data.qvel[1]], dtype=np.float32)

        else:
            # 3D model with free joint: qpos = [x,y,z,qw,qx,qy,qz,...]
            base_quat = self.data.qpos[3:7]
            base_euler = self._quat_to_euler(base_quat)
            base_lin_vel = self.data.qvel[0:3].astype(np.float32)
        
        # Joint positions (skip root DOFs)
        joint_pos = self.data.qpos[self._qpos_offset:]
        
        # Joint velocities (skip root DOFs)
        joint_vel = self.data.qvel[self._qvel_offset:]

        obs = {
            "euler": base_euler,
            "lin_vel": base_lin_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }
        
        return obs

    def _set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        assert qpos.shape == (self.model.nq,)
        assert qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        # Ensure correct shape + dtype
        action = np.asarray(action, dtype=np.float32).reshape(self.spec.act.shape)
        # Apply PD control if configured

        if self.cfg.pd_cfg is not None:
            # Get current joint positions and velocities (skip root DOFs)
            q = self.data.qpos[self._qpos_offset:]
            qd = self.data.qvel[self._qvel_offset:]

            # Compute desired torques using PD controller
            tau = self.pd_controller.compute(
                q=q,
                qd=qd,
                q_des=action,
                qd_des=None,
            )
            action = tau

        self.data.ctrl[:] = action

        return action

    def _render_frame(self) -> Optional[np.ndarray]:
        if self._renderer is None:
            return None

        # Use custom camera settings if no specific camera_id is set
        if self.cfg.camera_id is not None:
            self._renderer.update_scene(self.data, camera=self.cfg.camera_id)
        else:
            # Use our custom mjvCamera
            self._renderer.update_scene(self.data, camera=self._mjv_camera)

        rgb = self._renderer.render()
        # Check if already uint8 or needs conversion
        if rgb.dtype == np.uint8:
            return rgb
        frame = (rgb * 255).astype(np.uint8)
        return frame

    # -------------------------------------------------------------------------
    # Default reward & done
    # -------------------------------------------------------------------------
    def _default_reward_fn(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: int = 0,
        dt: float = 0.0,
        action: Optional[np.ndarray] = None,
    ) -> float:
        # Override this per-task.
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

        # save recording only if there's data, then reset
        if self.cfg.record_joints and any(len(v) > 0 for v in self._recorded.values()):
            self._save_recording()
            self._recorded = {"time": [], "qpos": [], "qvel": []}

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def step(self, action: np.ndarray) -> StepResult:
        applied_action = self._apply_action(action)

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._t += 1

        obs = self._get_obs()
        
        reward_raw = self._reward_fn(self.model, self.data, self._t, self.dt, applied_action)
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

    def set_reward_fn(self, reward_fn: RewardFn) -> None:
        """
        Replace the current reward function at runtime.
        Useful when the reward needs access to the fully
        constructed env (e.g., reference motion, hip height).
        """
        self._reward_fn = reward_fn

    def close(self) -> None:
        # Renderer holds GL resources; cleanly drop reference
        self._renderer = None

    def set_camera(
        self,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
        track_body: str | None = None,
    ) -> None:
        """Update camera parameters for rendering.
        
        Args:
            distance: Distance from the target point
            azimuth: Horizontal rotation in degrees
            elevation: Vertical rotation in degrees  
            lookat: Point to look at (offset from body if tracking)
            track_body: Name of body to track (e.g., "biped", "hips")
        """
        # Update config
        self._camera_config.distance = distance
        self._camera_config.azimuth = azimuth
        self._camera_config.elevation = elevation
        self._camera_config.lookat[:] = lookat
        
        # Update the MuJoCo camera object
        self._mjv_camera.distance = distance
        self._mjv_camera.azimuth = azimuth
        self._mjv_camera.elevation = elevation
        self._mjv_camera.lookat[:] = lookat
        
        # Enable body tracking if specified
        if track_body is not None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, track_body)
            if body_id >= 0:
                self._mjv_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._mjv_camera.trackbodyid = body_id
            else:
                raise ValueError(f"Body '{track_body}' not found in model")
        else:
            self._mjv_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._mjv_camera.trackbodyid = -1