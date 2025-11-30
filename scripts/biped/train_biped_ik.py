# scripts/biped/train_biped_ik.py

from __future__ import annotations
import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from core.biped_obs_wrapper import BipedSensorWrapper
from core.biped_ik_env import BipedIKEnv, FootActionConfig

from policies.actor_critic import ActorCritic
from policies.reference_policy import WalkerJointMap, LegJointIndices

from control.ik_2r import Planar2RLegConfig
from control.pd import PDConfig

from algorithms.ppo import PPO, PPOConfig
from training.on_policy import OnPolicyTrainer, TrainConfig

from tasks.biped.reward_inplace import reward
from tasks.biped.done import done


# ------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------
def make_env():
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=300,
        frame_skip=5,
        ctrl_scale=1.0,
        reset_noise_scale=0.01,
        render=False,
        reward_fn=reward,
        done_fn=done,
        width=640,
        height=480,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )

    base = MujocoEnv(cfg)
    low_env = BipedSensorWrapper(base)

    joint_map = WalkerJointMap(
        left=LegJointIndices(hip=7, knee=8, ankle=9),
        right=LegJointIndices(hip=10, knee=11, ankle=12),
    )

    leg_geom_left = Planar2RLegConfig(
        L1=0.05, L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi / 2,
        knee_offset=0.0,
        ankle_offset=-np.pi / 2,
    )

    leg_geom_right = Planar2RLegConfig(
        L1=0.05, L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi / 2,
        knee_offset=0.0,
        ankle_offset=-np.pi / 2,
    )

    # Tuned PD gains: strong but not crazy, with a reasonable torque limit
    pd_cfg = PDConfig(
        kp=600.0,     # was 1200.0
        kd=40.0,      # a bit more damping
        torque_limit=150.0,
    )

    # Give the policy a bit more range to actually move the feet
    foot_cfg = FootActionConfig(
        max_dx=0.05,
        max_dz=0.04,
    )

    env = BipedIKEnv(
        base_env=low_env,
        joint_map=joint_map,
        left_leg_geom=leg_geom_left,
        right_leg_geom=leg_geom_right,
        pd_cfg=pd_cfg,
        foot_cfg=foot_cfg,
        desired_foot_angle=0.0,
    )
    return env


# ------------------------------------------------------------
# Policy and PPO config
# ------------------------------------------------------------
def make_policy(env):
    return ActorCritic(env.spec, hidden_sizes=(128, 128))


def make_ppo(policy):
    cfg = PPOConfig(
        gamma=0.995,
        lam=0.98,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=80,
        batch_size=512,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
    return PPO(policy, cfg, device="cpu")


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------
def main():
    train_cfg = TrainConfig(
        total_steps=500_000,
        horizon=1024,
        log_interval=10,
        checkpoint_path="checkpoints/biped_ik.pt",
        device="cpu",
    )

    trainer = OnPolicyTrainer(
        env_factory=make_env,
        policy_factory=make_policy,
        algo_factory=make_ppo,
        train_cfg=train_cfg,
    )
    trainer.run()


if __name__ == "__main__":
    main()
