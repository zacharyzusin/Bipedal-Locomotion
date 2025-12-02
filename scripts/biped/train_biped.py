# scripts/train_walker.py
from __future__ import annotations

import mujoco
import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig
from training.on_policy import OnPolicyTrainer, TrainConfig

from tasks.biped.reward import reward
from tasks.biped.done import done

from control.pd import PDConfig

def make_env():
    # Point to your MuJoCo walker model
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=1000,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
        reward_fn=reward,
        done_fn=done,
        render=False,
    )
    return MujocoEnv(cfg)


def make_policy(env):
    # Hidden sizes are easily configurable per-task here
    return ActorCritic(env.spec, hidden_sizes=(64, 64))


def make_ppo(actor_critic):
    ppo_cfg = PPOConfig(
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=80,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
    )
    return PPO(actor_critic, ppo_cfg, device="cpu")


def main():
    train_cfg = TrainConfig(
        total_steps=200_000,
        horizon=2048,
        log_interval=10,
        device="cpu",
        checkpoint_path="checkpoints/walker_ppo.pt",
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