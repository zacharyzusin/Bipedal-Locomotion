# tasks/biped/reward.py
from __future__ import annotations
import mujoco
import numpy as np


def _quat_to_pitch(quat) -> float:
    """
    Extract torso pitch (rotation about the Y-axis) from a wxyz quaternion.
    Positive = leaning forward.
    """
    w, x, y, z = [float(q) for q in quat]
    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    return float(np.arcsin(sinp))


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t: float,
    dt: float,
    action: np.ndarray,
    #
    # Tunable parameters — these are good defaults for early walking.
    #
    v_des: float = 0.25,        # target walking speed
    ctrl_cost_weight: float = 0.001,
    acc_cost_weight: float = 0.01,  # lighter: allows natural COM shifts
    fall_penalty: float = -5.0,
) -> tuple[float, dict]:
    """
    Combined reward promoting stable and natural walking:
      • Standing upright is okay
      • Walking upright is best
      • Falling is bad
      • Small forward velocity helps exploration
      • Torso acceleration penalty prevents rope-pull exploit
    """

    # ------------------------------------------------------------------
    # 1) Base measurements
    # ------------------------------------------------------------------
    forward_vel = float(data.qvel[0])    # x-velocity of root body

    hips_pos = data.body("hips").xpos
    hips_quat = data.body("hips").xquat

    hip_z = float(hips_pos[2])
    pitch = _quat_to_pitch(hips_quat)

    # Rough nominal standing hip height from XML
    h_des = 0.17

    # ------------------------------------------------------------------
    # 2) Upright & height reward (both 0..1)
    # ------------------------------------------------------------------
    upright_clip = 0.7     # rad ≈ 40°
    height_clip = 0.07     # ±7 cm

    upright_raw = 1.0 - abs(pitch) / upright_clip
    upright_reward = _clamp01(upright_raw)

    height_raw = 1.0 - abs(hip_z - h_des) / height_clip
    height_reward = _clamp01(height_raw)

    # ------------------------------------------------------------------
    # 3) Velocity reward (0..1), gated by uprightness
    # ------------------------------------------------------------------
    if v_des > 1e-6:
        vel_raw = 1.0 - abs(forward_vel - v_des) / v_des
    else:
        vel_raw = 0.0
    vel_raw = _clamp01(vel_raw)

    # Allow more torso lean for gait initiation
    if upright_reward < 0.3:
        vel_raw = 0.0

    vel_reward = vel_raw

    # ------------------------------------------------------------------
    # 4) Healthy (alive) bonus
    # ------------------------------------------------------------------
    healthy_bonus = 0.0
    if upright_reward > 0.4 and 0.13 < hip_z < 0.23:
        healthy_bonus = 0.1

    # ------------------------------------------------------------------
    # 5) Control cost
    # ------------------------------------------------------------------
    ctrl_cost = ctrl_cost_weight * float(np.sum(np.square(action)))

    # ------------------------------------------------------------------
    # 6) Torso linear acceleration penalty (prevents rope-pull hack)
    # ------------------------------------------------------------------
    hip_id = model.body("hips").id
    # data.cacc[i][0:3] = angular acc, [3:6] = linear acc in world frame
    hip_lin_acc = np.array(data.cacc[hip_id, 3:6], dtype=float)
    acc_cost = acc_cost_weight * float(np.linalg.norm(hip_lin_acc))

    # ------------------------------------------------------------------
    # 7) Fall penalty
    # ------------------------------------------------------------------
    fall_term = 0.0
    if hip_z < 0.11 or abs(pitch) > 1.0:
        fall_term = float(fall_penalty)

    # ------------------------------------------------------------------
    # 8) Extra forward drift bonus (helps agent discover locomotion)
    # ------------------------------------------------------------------
    forward_bonus = 0.1 * max(forward_vel, 0.0)

    # ------------------------------------------------------------------
    # 9) Combine final reward
    # ------------------------------------------------------------------
    w_vel = 3.0
    w_upright = 0.5
    w_height = 0.5

    total = (
        w_vel * vel_reward +
        w_upright * upright_reward +
        w_height * height_reward +
        healthy_bonus -
        ctrl_cost -
        acc_cost +
        fall_term +
        forward_bonus
    )

    # ------------------------------------------------------------------
    # 10) Components dictionary (all primitive floats)
    # ------------------------------------------------------------------
    components = {
        "forward_vel": forward_vel,
        "vel_reward": vel_reward,
        "upright_reward": upright_reward,
        "height_reward": height_reward,
        "forward_bonus": forward_bonus,
        "healthy_bonus": healthy_bonus,
        "ctrl_cost": ctrl_cost,
        "acc_cost": acc_cost,
        "fall_term": fall_term,
        "pitch": pitch,
        "hip_z": hip_z,
    }

    return float(total), components
