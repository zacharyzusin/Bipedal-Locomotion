# scripts/biped/stream_historic_biped.py
from __future__ import annotations

import numpy as np
import uvicorn

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from core.history_env import HistoryEnv, HistoryConfig
from streaming.mjpeg_server import create_app

from policies.dual_history_policy import DualHistoryActorCritic
from policies.reference_policy import (
    GaitParams,
    WalkerJointMap,
    LegJointIndices,
    ReferenceWalkerPolicy,
)
from control.ik_2r import Planar2RLegConfig
from control.pd import PDConfig

from tasks.biped.historic_reward import make_historic_reward
from tasks.biped.done import done


# ---------------------------------------------------------------------
# 1) Base MuJoCo env + reference policy + reward (with rendering)
# ---------------------------------------------------------------------

def make_base_env(render: bool = True) -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=4_096,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
        reset_noise_scale=0.0,
        render=render,
        done_fn=done,
        base_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
        reward_fn=None,   # set below
        width=640,
        height=480,
    )
    env = MujocoEnv(cfg)

    # ---------------------------
    # Reference gait controller
    # ---------------------------
    gait_params = GaitParams(
        step_length=0.02,
        step_height=0.01,
        cycle_duration=1.25,
    )

    joint_map = WalkerJointMap(
        left=LegJointIndices(
            hip=7,
            knee=8,
            ankle=9,
        ),
        right=LegJointIndices(
            hip=10,
            knee=11,
            ankle=12,
        ),
    )

    left_leg_geom = Planar2RLegConfig(
        L1=0.05,   # thigh length [m]
        L2=0.058,   # shank length [m]
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0,
        ankle_offset=-np.pi/2,

    )
    right_leg_geom = Planar2RLegConfig(
        L1=0.05,
        L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0,
        ankle_offset=-np.pi/2,
    )

    pd_cfg = PDConfig(kp=5.0, kd=1.0)

    ref_policy = ReferenceWalkerPolicy(
        env=env,
        gait_params=gait_params,
        joint_map=joint_map,
        left_leg_geom=left_leg_geom,
        right_leg_geom=right_leg_geom,
        pd_config=pd_cfg,
    )

    def ref_q_fn(time_sec: float) -> np.ndarray:
        return ref_policy.compute_q_ref(time_sec)

    reward_fn = make_historic_reward(
        env=env,
        ref_q_fn=ref_q_fn,
        base_site="base",
        v_des=0.02,
    )
    env.set_reward_fn(reward_fn)

    return env


# ---------------------------------------------------------------------
# 2) Wrap with HistoryEnv for dual history + command
# ---------------------------------------------------------------------

def make_env() -> HistoryEnv:
    base = make_base_env(render=True)

    hist_cfg = HistoryConfig(
        short_horizon=4,
        long_horizon=66,
    )

    env = HistoryEnv(
        base_env=base,
        hist_cfg=hist_cfg,
        command_dim=4,  # [qdot_x^d, qdot_y^d, q_z^d, q_psi^d]
    )

    base_h = base.hip_height

    cmd = np.array([0.02, 0.0, base_h, 0.0], dtype=np.float32)
    env.set_command(cmd)

    return env


# ---------------------------------------------------------------------
# 3) Policy factory: DualHistoryActorCritic
# ---------------------------------------------------------------------

def make_policy(env: HistoryEnv):
    """
    Build a dual-history policy that matches HistoryEnv’s obs layout:
      obs = [ base_obs |
              K_short * (obs, act) |
              K_long  * (obs, act) |
              command(4) ]
    """
    short_h = 4
    long_h = 66

    act_dim = env.spec.act.shape[0]
    A = act_dim
    Ks = short_h
    Kl = long_h
    cmd_dim = 4
    total = env.spec.obs.shape[0]

    # total = B + (Ks + Kl) * (B + A) + cmd_dim
    # => total - cmd_dim = B * (1 + Ks + Kl) + (Ks + Kl) * A
    B = (total - cmd_dim - (Ks + Kl) * A) / (1 + Ks + Kl)
    base_obs_dim = int(B)

    return DualHistoryActorCritic(
        spec=env.spec,
        base_obs_dim=base_obs_dim,
        short_hist_len=short_h,
        long_hist_len=long_h,
        command_dim=cmd_dim,
        hidden_size=512,
        act_std=0.2,
    )


# ---------------------------------------------------------------------
# 4) Run MJPEG streaming server
# ---------------------------------------------------------------------

def main():
    checkpoint_path = "checkpoints/biped_historic_ppo_mp.pt"  # historic MP training
    device = "cpu"

    app = create_app(
        env_factory=make_env,
        policy_factory=make_policy,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()