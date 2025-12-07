# tasks/biped/reward.py
import mujoco
import numpy as np


def _body_forward_velocity_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> float:
    """
    Forward (world-x) linear velocity of a body using mj_objectVelocity.

    mj_objectVelocity returns a 6D velocity [rot(3), lin(3)] for the body,
    in an object-centered frame but with world orientation when flg_local=0.
    We just take the linear part and its x-component.
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
    """
    Get the global y-position of a body.
    data.xpos contains [x, y, z] world positions for each body.
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
    lateral_cost_weight: float = 0.1,  # new parameter for y-deviation penalty
) -> tuple[float, dict]:
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