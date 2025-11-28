# tasks/walker_reward.py
import mujoco
import numpy as np

def reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    forward_reward_weight: float = 1.5,
    healthy_reward: float = 0.05,
    ctrl_cost_weight: float = 0.01,
) -> tuple[float, dict]:
    """
    Approximate Gymnasium Walker2d-v4 reward.

    - forward_reward ~ x-velocity of the root body (qvel[0])
    - healthy_reward if within healthy range of height & angle
    - ctrl_cost penalizes squared actions
    """
    # Root joint velocity in x (this is effectively (x_after - x_before)/dt)
    forward_vel = float(data.qvel[0])
    forward_reward = forward_reward_weight * forward_vel

    # Control cost
    ctrl_cost = ctrl_cost_weight * float(np.sum(np.square(data.ctrl)))

    # Health / alive
    # For Walker2d in Gym:
    #   z (height) in [0.8, 2.0]
    #   angle in [-1.0, 1.0]
    z = float(data.qpos[1])      # vertical position of torso
    angle = float(data.qpos[2])  # orientation pitch

    is_healthy = (
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and (0.8 < z < 2.0)
        and (-1.0 < angle < 1.0)
    )

    healthy_bonus = healthy_reward if is_healthy else 0.0

    total = forward_reward + healthy_bonus - ctrl_cost

    components = {
        "forward_vel": forward_vel,
        "forward_reward": forward_reward,
        "ctrl_cost": ctrl_cost,
        "healthy_bonus": healthy_bonus,
        "is_healthy": float(is_healthy),
    }
    return float(total), components