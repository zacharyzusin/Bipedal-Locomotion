# scripts/biped/train_historic_biped_mp.py
from __future__ import annotations

import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from core.history_env import HistoryEnv, HistoryConfig
from training.on_policy import OnPolicyTrainer, TrainConfig
from algorithms.ppo import PPO, PPOConfig

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
# 1) Base MuJoCo env + reference policy + reward
# ---------------------------------------------------------------------

def make_base_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=4096,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
        reset_noise_scale=0.01,
        render=False,
        done_fn=done,
        base_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
        reward_fn=None, # set this below
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
            hip=7,    # e.g. index of left hip in qpos
            knee=8,   # index of left knee in qpos
            ankle=9,  # index of left ankle in qpos
        ),
        right=LegJointIndices(
            hip=10,   # index of right hip
            knee=11,  # index of right knee
            ankle=12, # index of right ankle
        ),
    )

    left_leg_geom = Planar2RLegConfig(L1=0.05, L2=0.058)
    right_leg_geom = Planar2RLegConfig(L1=0.05, L2=0.058)

    pd_cfg = PDConfig(kp=5.0, kd=1.0)

    # This is the same reference controller you already use for streaming
    ref_policy = ReferenceWalkerPolicy(
        env=env,
        gait_params=gait_params,
        joint_map=joint_map,
        left_leg_geom=left_leg_geom,
        right_leg_geom=right_leg_geom,
        pd_config=pd_cfg,
    )

    # We assume ReferenceWalkerPolicy exposes compute_q_ref(time_sec)
    # that returns a full qpos reference (or at least for the motor joints).
    def ref_q_fn(time_sec: float) -> np.ndarray:
        return ref_policy.compute_q_ref(time_sec)

    reward_fn = make_historic_reward(
        env=env,
        ref_q_fn=ref_q_fn,
        torso_body="hips",  # adapt the body name if needed
        v_des=0.6,           # desired forward speed
    )
    env.set_reward_fn(reward_fn)

    return env


# ---------------------------------------------------------------------
# 2) Wrap with HistoryEnv to get dual I/O history + command
# ---------------------------------------------------------------------

def env_factory() -> HistoryEnv:
    base = make_base_env()

    hist_cfg = HistoryConfig(
        short_horizon=4,    # K_short (paper uses 4) 
        long_horizon=66,    # K_long  (paper uses 66)
    )

    env = HistoryEnv(
        base_env=base,
        hist_cfg=hist_cfg,
        command_dim=4,      # [qdot_x^d, qdot_y^d, q_z^d, q_psi^d]
    )

    # For now: fixed command; you can randomize per episode later
    if base.hip_height is None:
        base_h = 1.0
    else:
        base_h = base.hip_height

    cmd = np.array([0.6, 0.0, base_h, 0.0], dtype=np.float32)
    env.set_command(cmd)

    return env

# ---------------------------------------------------------------------
# 3) Policy factory: DualHistoryActorCritic
# ---------------------------------------------------------------------

def policy_factory(env: HistoryEnv):
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
# 4) Algo factory + run
# ---------------------------------------------------------------------

def algo_factory(policy):
    ppo_cfg = PPOConfig()
    return PPO(policy, ppo_cfg, device="cpu")


def main():
    train_cfg = TrainConfig(
        total_steps=2_500_000,
        horizon=2048,
        log_interval=10,
        device="cpu",
        checkpoint_path="checkpoints/walker_paper.pt",
    )

    trainer = OnPolicyTrainer(
        env_factory=env_factory,
        policy_factory=policy_factory,
        algo_factory=algo_factory,
        train_cfg=train_cfg,
    )

    trainer.run()



if __name__ == "__main__":
    main()