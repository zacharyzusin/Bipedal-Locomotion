# scripts/train_walker_mp.py
from __future__ import annotations

import multiprocessing as mp

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig
from training.mp_on_policy import (
    MultiProcessOnPolicyTrainer,
    MPTrainConfig,
)

from tasks.biped.reward import reward
from tasks.biped.done import done

from control.pd import PDConfig


def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=4096,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
        reward_fn=reward,
        done_fn=done,
        reset_noise_scale=0.0,
        render=False,
    )
    return MujocoEnv(cfg)


def make_policy(env):
    return ActorCritic(env.spec, hidden_sizes=(64, 64))


def make_ppo(actor_critic):
    ppo_cfg = PPOConfig(
        gamma=0.995,
        lam=0.98,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=10,
        batch_size=512,   # larger batch since we're combining workers
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
    )
    return PPO(actor_critic, ppo_cfg, device="cpu")


def main():
    # Good practice with multiprocessing + PyTorch
    mp.set_start_method("spawn", force=True)

    train_cfg = MPTrainConfig(
        total_steps=20_000_000,
        horizon=4096,
        num_workers=7,
        log_interval=1,
        device="cpu",
        checkpoint_path="checkpoints/biped_ppo_mp.pt",
    )

    trainer = MultiProcessOnPolicyTrainer(
        env_factory=make_env,
        policy_factory=make_policy,
        algo_factory=make_ppo,
        train_cfg=train_cfg,
    )
    trainer.run()


if __name__ == "__main__":
    main()