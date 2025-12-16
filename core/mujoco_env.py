"""MuJoCo physics environment wrapper for reinforcement learning.

This module provides a MuJoCo-based environment implementation that supports
both 2D and 3D robot models, optional PD control, custom reward/done functions,
and rendering capabilities for visualization and evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, Union, Mapping

import numpy as np
import mujoco

from .base_env import Env, StepResult
from .specs import EnvSpec, SpaceSpec
from control.pd import PDConfig, PDController

# Type aliases for reward and done functions
RewardReturn = Union[
    float,
    tuple[float, Mapping[str, float]],  # (reward, components_dict)
]
RewardFn = Callable[..., RewardReturn]
DoneFn = Callable[[mujoco.MjModel, mujoco.MjData, int], bool]


@dataclass
class CameraConfig:
    """Camera configuration for rendering.
    
    Attributes:
        distance: Distance from the look-at point in meters.
        azimuth: Horizontal rotation angle in degrees (0-360).
        elevation: Vertical rotation angle in degrees (-90 to 90).
        lookat: 3D point to look at (x, y, z) in world coordinates.
    """
    distance: float = 3.0
    azimuth: float = 90.0
    elevation: float = -20.0
    lookat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))


@dataclass
class MujocoEnvConfig:
    """Configuration for MuJoCo environment.
    
    Attributes:
        xml_path: Path to MuJoCo XML model file.
        episode_length: Maximum number of steps per episode.
        frame_skip: Number of simulation steps per action (action repeat).
        pd_cfg: Optional PD controller configuration for low-level joint control.
        reward_fn: Custom reward function (model, data, t, dt, action) -> reward.
        done_fn: Custom termination function (model, data, t) -> bool.
        reset_noise_scale: Scale of uniform noise added to initial state (0 = no noise).
        render: Whether to enable rendering (required for visualization).
        width: Render width in pixels (if render=True).
        height: Render height in pixels (if render=True).
        camera_id: Optional camera ID or name to use (None = custom camera).
        base_site: Optional site name for base/hip body (for IK controllers).
        left_foot_site: Optional site name for left foot (for IK controllers).
        right_foot_site: Optional site name for right foot (for IK controllers).
    """
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

class MujocoEnv(Env):
    """MuJoCo-based reinforcement learning environment.
    
    This environment wraps MuJoCo physics simulation and provides a standard
    RL interface. It automatically handles:
    - 2D vs 3D model detection (based on root joint type)
    - Observation space construction (base pose/velocity + joint states)
    - Optional PD control for joint-level actuation
    - Custom reward and termination functions
    - Rendering for visualization
    
    The observation space includes:
    - Base orientation (Euler angles for 3D, angle for 2D)
    - Base linear velocity
    - Actuated joint positions (excluding root DOFs)
    - Actuated joint velocities (excluding root DOFs)
    
    Attributes:
        model: MuJoCo model object.
        data: MuJoCo data object (contains current state).
        dt: Effective timestep (model timestep * frame_skip).
        hip_height: Height of base/hip body (if sites configured).
        foot_base_pos: Dictionary with "left" and "right" foot base positions
            relative to hip (if sites configured).
        spec: Environment specification (observation and action spaces).
    """
    def __init__(self, cfg: MujocoEnvConfig):
        """Initialize MuJoCo environment.
        
        Args:
            cfg: Environment configuration.
            
        Raises:
            FileNotFoundError: If XML model file doesn't exist.
            ValueError: If model is invalid or sites are misconfigured.
        """
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

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------
    def _record_obs(self) -> None:
        """Record actuated joint positions and velocities.
        
        Note: This method is currently unused but kept for potential future
        data logging functionality.
        """
        if not hasattr(self, "_recorded"):
            self._recorded = {"time": [], "qpos": [], "qvel": []}
        self._recorded["time"].append(self._t * self.dt)
        # Record only actuated joint positions (skip root DOFs)
        self._recorded["qpos"].append(self.data.qpos[self._qpos_offset:].copy())
        # Record only actuated joint velocities (skip root DOFs)
        self._recorded["qvel"].append(self.data.qvel[self._qvel_offset:].copy())

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles using ZYX convention.
        
        Converts a quaternion in (w, x, y, z) format to Euler angles
        (roll, pitch, yaw) using the standard aerospace convention.
        
        Args:
            quat: Quaternion array [w, x, y, z].
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians.
        """
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
        """Construct observation dictionary from current state.
        
        The observation includes:
        - Base orientation (Euler angles for 3D, angle for 2D)
        - Base linear velocity
        - Actuated joint positions (excluding root DOFs)
        - Actuated joint velocities (excluding root DOFs)
        
        Returns:
            Dictionary with keys: "euler", "lin_vel", "joint_pos", "joint_vel".
        """
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
        """Set the simulation state directly.
        
        Args:
            qpos: Joint positions array of shape (nq,).
            qvel: Joint velocities array of shape (nv,).
            
        Raises:
            AssertionError: If array shapes don't match model dimensions.
        """
        assert qpos.shape == (self.model.nq,)
        assert qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action to the environment.
        
        If PD control is configured, converts desired joint positions to
        torques. Otherwise, applies actions directly as torques.
        
        Args:
            action: Action array (desired joint positions if PD enabled,
                otherwise torques).
                
        Returns:
            Applied action (torques) that were actually set.
        """
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
        """Render current frame as RGB image.
        
        Returns:
            RGB frame array of shape (height, width, 3) with dtype uint8,
            or None if rendering is disabled.
        """
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
        """Default reward function (returns zero).
        
        This should be overridden with a task-specific reward function
        via the config or set_reward_fn().
        
        Args:
            model: MuJoCo model.
            data: MuJoCo data (current state).
            t: Current time step.
            dt: Time step duration.
            action: Applied action.
            
        Returns:
            Reward value (default: 0.0).
        """
        return 0.0

    def _default_done_fn(
        self, model: mujoco.MjModel, data: mujoco.MjData, t: int
    ) -> bool:
        """Default termination function (time limit only).
        
        Args:
            model: MuJoCo model.
            data: MuJoCo data (current state).
            t: Current time step.
            
        Returns:
            True if episode should terminate (time limit reached).
        """
        return t >= self.cfg.episode_length

    # -------------------------------------------------------------------------
    # Env interface
    # -------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment to initial state.
        
        Resets the simulation to the default initial state (from XML),
        optionally adds noise for domain randomization, and returns the
        initial observation.
        
        Args:
            seed: Optional random seed for reproducible resets.
            
        Returns:
            Initial observation dictionary.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0

        # Fully reset MuJoCo internal state to defaults from XML
        mujoco.mj_resetData(self.model, self.data)

        qpos = self._init_qpos.copy()
        qvel = self._init_qvel.copy()

        # Add noise for domain randomization if configured
        s = self.cfg.reset_noise_scale
        if s > 0.0:
            # Uniform noise in [-s, s] for each element
            qpos += self._rng.uniform(-s, s, size=qpos.shape)
            qvel += self._rng.uniform(-s, s, size=qvel.shape)
        
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        
        # Clear controls (zero torques)
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        # Forward kinematics to update all dependent quantities
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def step(self, action: np.ndarray) -> StepResult:
        """Execute one environment step.
        
        Applies the action, advances the simulation by frame_skip steps,
        computes reward and done status, and returns the result.
        
        Args:
            action: Action array (shape must match action space).
            
        Returns:
            StepResult containing:
            - obs: New observation dictionary
            - reward: Scalar reward
            - done: Termination flag
            - info: Additional info (time, applied action, reward components)
            - frame: Optional RGB frame if rendering enabled
        """
        applied_action = self._apply_action(action)

        # Advance simulation by frame_skip steps
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._t += 1

        obs = self._get_obs()
        
        # Compute reward (may return tuple with components)
        reward_raw = self._reward_fn(self.model, self.data, self._t, self.dt, applied_action)
        if isinstance(reward_raw, tuple):
            reward, reward_components = reward_raw
        else:
            reward = float(reward_raw)
            reward_components = {}

        # Check termination condition
        done = self._done_fn(self.model, self.data, self._t)

        info: Dict[str, Any] = {
            "t": self._t,
            "applied_action": applied_action,
            "reward_components": reward_components,
        }

        if done:
            info["episode_length"] = self._t

        # Render frame if enabled
        frame = self._render_frame()

        return StepResult(
            obs=obs,
            reward=float(reward),
            done=done,
            info=info,
            frame=frame,
        )

    def set_reward_fn(self, reward_fn: RewardFn) -> None:
        """Replace the current reward function at runtime.
        
        Useful when the reward function needs access to the fully
        constructed environment (e.g., reference motion, hip height).
        
        Args:
            reward_fn: New reward function with signature
                (model, data, t, dt, action) -> reward or (reward, components).
        """
        self._reward_fn = reward_fn

    def close(self) -> None:
        """Clean up environment resources.
        
        Releases the renderer and any other allocated resources.
        """
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
            # Free camera mode (not tracking any body)
            self._mjv_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._mjv_camera.trackbodyid = -1