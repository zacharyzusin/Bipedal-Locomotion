# scripts/biped/train_historic_biped_mp.py
from __future__ import annotations

import multiprocessing as mp
import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from core.history_env import HistoryEnv, HistoryConfig

from algorithms.ppo import PPO, PPOConfig
from training.mp_on_policy import (
    MultiProcessOnPolicyTrainer,
    MPTrainConfig,
)

from policies.dual_history_policy import DualHistoryActorCritic
from policies.reference_policy import (
    GaitParams,
    WalkerJointMap,
    LegJointIndices,
    ReferenceWalkerPolicy,
)
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
        pd_cfg = PDConfig(kp=5.0, kd=1.0),
        reset_noise_scale=0.01,
        render=False,
        done_fn=done,
        base_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
        reward_fn=None,  # set this below
    )
    env = MujocoEnv(cfg)

    reward_fn = make_historic_reward(
        env=env,
        v_des=0.02,
    )
    env.set_reward_fn(reward_fn)

    return env


# ---------------------------------------------------------------------
# 2) Wrap with HistoryEnv to get dual I/O history + command
# ---------------------------------------------------------------------

def env_factory() -> HistoryEnv:
    base = make_base_env()

    hist_cfg = HistoryConfig(
        short_horizon=4,
        long_horizon=66,
        reference_path="recordings/biped_reference_recording_4096.npz",
    )

    env = HistoryEnv(
        base_env=base,
        hist_cfg=hist_cfg,
        command_dim=4,     # [qdot_x^d, qdot_y^d, q_z^d, q_psi^d]
    )

    base_h = base.hip_height

    cmd = np.array([0.02, 0.0, base_h, 0.0], dtype=np.float32)
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
    pair_dim = env.base_obs_dim + act_dim
    cmd_dim = 4

    ref_dim = 18   # lookahead joint positions only
        
    return DualHistoryActorCritic(
        spec=env.spec,
        pair_dim=pair_dim,
        short_horizon=short_h,
        long_horizon=long_h,
        ref_dim=ref_dim,
        command_dim=cmd_dim,
        hidden_size=512,
        act_std=0.2,
    )


# ---------------------------------------------------------------------
# 4) Algo factory for PPO
# ---------------------------------------------------------------------

def make_ppo(policy):
    # You can tune these separately for historic training, but this is a
    # reasonable starting point for multi-process PPO.
    ppo_cfg = PPOConfig(
        gamma=0.995,
        lam=0.98,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=80,
        batch_size=256,   # larger batch since we aggregate across workers
        value_coef=0.5,
        entropy_coef=0.00,
        max_grad_norm=0.5,
    )
    return PPO(policy, ppo_cfg, device="cpu")


# ---------------------------------------------------------------------
# 5) Multi-process trainer entrypoint
# ---------------------------------------------------------------------

def main():
    # Good practice with multiprocessing + PyTorch
    mp.set_start_method("spawn", force=True)

    train_cfg = MPTrainConfig(
        total_steps=1_000_000,
        horizon=4096,
        num_workers=4,
        log_interval=1,
        device="cpu",
        checkpoint_path="checkpoints/biped_historic_ppo_mp.pt",
    )

    trainer = MultiProcessOnPolicyTrainer(
        env_factory=env_factory,
        policy_factory=policy_factory,
        algo_factory=make_ppo,
        train_cfg=train_cfg,
    )

    trainer.run()


if __name__ == "__main__":
    main()