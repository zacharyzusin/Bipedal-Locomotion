# tasks/biped/reward.py
from __future__ import annotations
import mujoco
import numpy as np


def _quat_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    Returns Python floats.
    """
    w, x, y, z = [float(q) for q in quat]

    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = np.arcsin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


def reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t: float,
    dt: float,
    action: np.ndarray,

    # ---- walking speed target ----
    v_des: float = 0.35,
    vel_alpha: float = 3.0,

    # ---- penalties & bonuses ----
    upright_weight: float = 2.0,
    bounce_weight: float = 0.5,
    ctrl_cost_weight: float = 0.001,
    alive_bonus: float = 0.5,
    fall_height_threshold: float = 0.10,
) -> tuple[float, dict]:
    """
    Simple reward that produces stable walking:

        R = speed_reward + alive - ctrl_cost - upright_cost - bounce_cost

    - speed_reward encourages walking at target v_des (no running)
    - upright_cost keeps torso vertical
    - bounce_cost reduces vertical hopping (prevents pogo-running)
    - ctrl_cost keeps movements moderate
    """

    # ---------------------------------------------------------
    # 1. Forward speed reward around target walking speed
    # ---------------------------------------------------------
    forward_vel = float(data.qvel[0])
    vel_err = forward_vel - v_des
    speed_reward = float(np.exp(-vel_alpha * vel_err * vel_err))

    # ---------------------------------------------------------
    # 2. Small control cost
    # ---------------------------------------------------------
    ctrl_cost = ctrl_cost_weight * float(np.sum(np.square(action)))

    # ---------------------------------------------------------
    # 3. Upright posture penalty (penalizes torso pitch)
    # ---------------------------------------------------------
    hips_quat = data.body("hips").xquat
    _, pitch, _ = _quat_to_euler(hips_quat)
    upright_cost = upright_weight * float(pitch * pitch)

    # ---------------------------------------------------------
    # 4. Bounce penalty (vertical velocity)
    #     Reduces hopping → produces smooth walking gait
    # ---------------------------------------------------------
    vz = float(data.qvel[2])          # vertical hip velocity
    bounce_cost = bounce_weight * vz * vz

    # ---------------------------------------------------------
    # 5. Alive bonus / fall penalty
    # ---------------------------------------------------------
    height = float(data.body("hips").xpos[2])
    if height < fall_height_threshold:
        alive = -10.0
    else:
        alive = alive_bonus

    # ---------------------------------------------------------
    # 6. Final reward
    # ---------------------------------------------------------
    total = (
        speed_reward
        + alive
        - ctrl_cost
        - upright_cost
        - bounce_cost
    )

    components = {
        "forward_vel": forward_vel,
        "vel_err": vel_err,
        "speed_reward": speed_reward,
        "ctrl_cost": ctrl_cost,
        "pitch": pitch,
        "upright_cost": upright_cost,
        "vz": vz,
        "bounce_cost": bounce_cost,
        "alive_bonus": alive,
        "height": height,
    }

    return float(total), components
