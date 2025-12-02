# tasks/biped/history_reward.py
from __future__ import annotations
from pyexpat import model
from typing import Callable, Mapping, Tuple, Dict
from collections import defaultdict
import numpy as np
import mujoco

from core.mujoco_env import MujocoEnv, RewardReturn

recording_path = "recordings/biped_reference_recording_4096.npz"
r_data = np.load(recording_path)

def index(times: np.ndarray, target_time: float) -> int:
    """Find the index of the closest time in the reference data."""
    return int(np.argmin(np.abs(times - target_time)))

def exp_reward(u: np.ndarray, v: np.ndarray, alpha: float) -> float:
    """
    r(u, v) = exp(-alpha * ||u - v||^2)
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    diff = u - v
    return float(np.exp(-alpha * float(np.dot(diff, diff))))

def make_historic_reward(
    env: MujocoEnv,
    v_des: float = 0.6,   # desired forward speed
) -> Callable[[mujoco.MjModel, mujoco.MjData, int, float, np.ndarray], RewardReturn]:
    """
    Build a Cassie-style reward for your Walker, simplified but structurally similar:

      - Motion tracking:
          * joint positions vs reference: r(q_m, q_m^r(t))
          * pelvis height vs nominal:     r(q_z, q_z^r)
      - Task completion:
          * forward velocity vs target:   r(v_x, v_des)
      - Smoothing:
          * small torques:                r(tau, 0)
          * small action changes:         r(a_t, a_{t-1})

    Combined as a weighted sum and normalized by L1 norm of weights, as in rt = (w / ||w||_1)^T r. 
    """ 
    last_action = np.zeros(env.spec.act.shape[0], dtype=np.float32)
    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    w_motion_p      = 5.0
    w_pelvis_vel = 0.0
    w_torque        = 0.0
    w_action_diff   = 0.0

    w_sum = (
        abs(w_motion_p)
        + abs(w_pelvis_vel)
        + abs(w_torque)
        + abs(w_action_diff)
    )


    def reward_fn(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: int,
        dt: float,
        action: np.ndarray,
    ) -> Tuple[float, Mapping[str, float]]:
        nonlocal last_action

        r_idx = index(r_data["time"], t * dt)
        r_joint_pos = r_data["time"][r_idx]
        joint_pos = data.qpos[env._qpos_offset:]


        # --------------------------------------------------------------
        # 1) Motion tracking: joint positions vs reference
        # --------------------------------------------------------------
        r_motion_p = exp_reward(
            r_joint_pos,
            joint_pos,
            alpha=50.0,
        )

        # --------------------------------------------------------------
        # 2) Task completion: forward velocity vs target
        #    (pelvis vx in world frame)
        # --------------------------------------------------------------
        # MuJoCo cvel: [wx, wy, wz, vx, vy, vz] in world coordinates.
        base_site_id = model.site("base").id
        base_body_id = model.site_bodyid[base_site_id]
        vel_spatial = data.cvel[base_body_id]
        vx = float(vel_spatial[3])
        
        r_pelvis_vel = exp_reward(
            np.array([vx], dtype=np.float32),
            np.array([v_des], dtype=np.float32),
            alpha=50.0,
        )

        # --------------------------------------------------------------
        # 4) Smoothing: torque magnitude r(tau, 0)
        # --------------------------------------------------------------
        a = np.asarray(action, dtype=np.float32)
        r_torque = exp_reward(a, np.zeros_like(a), alpha=0.1)

        # --------------------------------------------------------------
        # 5) Smoothing: change of action r(a_t, a_{t-1})
        # --------------------------------------------------------------
        if t == 0:
            last_action = a.copy()
        delta_a = a - last_action
        r_action_diff = exp_reward(delta_a, np.zeros_like(delta_a), alpha=1.0)
        last_action = a.copy()

        # --------------------------------------------------------------
        # Weighted sum + normalization
        # --------------------------------------------------------------
        components_weighted = {
            "motion_p":      w_motion_p      * r_motion_p,
            "pelvis_vel":    w_pelvis_vel    * r_pelvis_vel,
            "torque":        w_torque        * r_torque,
            "action_diff":   w_action_diff   * r_action_diff,
        }

        total = sum(components_weighted.values()) / w_sum

        # For logging / streaming, expose *un-normalized* weighted terms
        return float(total), {k: float(v) for k, v in components_weighted.items()}

    return reward_fn