"""Reward function for bipedal locomotion task.

This module defines the reward function that encourages forward locomotion
while maintaining stability. The reward includes components for forward
velocity, staying upright, low control effort, and minimal lateral deviation.
"""
import mujoco
import numpy as np


def _body_forward_velocity_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> float:
    """Compute forward (world-x) linear velocity of a body.
    
    Uses MuJoCo's mj_objectVelocity to get the 6D velocity (rotational
    and linear) in world coordinates, then extracts the x-component of
    the linear velocity.
    
    Args:
        model: MuJoCo model.
        data: MuJoCo data (current state).
        body_name: Name of the body to query.
        
    Returns:
        Forward velocity (world-x component) in m/s.
    """
    body_id = model.body(body_name).id

    vel6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        vel6,
        0,  # 0 = world orientation
    )

    linear_world = vel6[3:]           # [vx, vy, vz]
    forward_vel = float(linear_world[0])  # world-x as "forward"
    return forward_vel


def _body_y_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> float:
    """Get the global y-position (lateral) of a body.
    
    Args:
        model: MuJoCo model.
        data: MuJoCo data (current state).
        body_name: Name of the body to query.
        
    Returns:
        Y-position (lateral position) in meters.
    """
    body_id = model.body(body_name).id
    return float(data.xpos[body_id, 1])  # y-component


def reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t: float,
    dt: float,
    action: np.ndarray,
    vel_weight: float = 1.0,
    alive_bonus: float = 0.05,
    ctrl_cost_weight: float = 0.01,
    lateral_cost_weight: float = 0.1,
) -> tuple[float, dict]:
    """Compute reward for bipedal locomotion task.
    
    The reward encourages:
    - Forward velocity (from feet and base)
    - Staying upright (alive bonus when height > threshold)
    - Low control effort (penalty on squared torques)
    - Minimal lateral deviation (penalty on y-position)
    
    Args:
        model: MuJoCo model.
        data: MuJoCo data (current state).
        t: Current time step.
        dt: Time step duration.
        action: Applied action (torques).
        vel_weight: Weight for forward velocity reward.
        alive_bonus: Bonus given when robot is upright.
        ctrl_cost_weight: Weight for control cost penalty.
        lateral_cost_weight: Weight for lateral deviation penalty.
        
    Returns:
        Tuple of (total_reward, reward_components_dict) where the dict
        contains breakdown of individual reward components for analysis.
    """
    # --- forward velocity from hips + feet ---
    base_forward  = 2.0 * _body_forward_velocity_world(model, data, "hips")
    left_forward  = _body_forward_velocity_world(model, data, "left_foot")
    right_forward = _body_forward_velocity_world(model, data, "right_foot")

    # simple linear combo (tune weights as you like)
    forward_vel = 0.3 * (left_forward + right_forward + 0 * base_forward)

    # linear forward reward (you can revert to Gaussian shaping if you want)
    forward_reward = vel_weight * forward_vel

    # --- lateral deviation cost (penalize y deviation from 0) ---
    hip_y = _body_y_position(model, data, "hips")
    lateral_cost = lateral_cost_weight * hip_y ** 2  # quadratic penalty

    # --- control cost ---
    ctrl_cost = ctrl_cost_weight * float(np.sum(np.square(data.ctrl)))

    # --- alive bonus ---
    z = float(data.qpos[2])
    alive = z > 0.12
    alive_reward = alive_bonus if alive else 0.0

    total = forward_reward + alive_reward - ctrl_cost - lateral_cost

    components = {
        "forward_vel": forward_vel,
        "base_forward_vel": base_forward,
        "left_forward_vel": left_forward,
        "right_forward_vel": right_forward,
        "forward_reward": forward_reward,
        "alive_reward": alive_reward,
        "ctrl_cost": ctrl_cost,
        "lateral_cost": lateral_cost,
        "hip_y": hip_y,
        "is_alive": float(alive),
    }
    return float(total), components