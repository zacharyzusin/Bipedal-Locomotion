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

from tasks.walker_reward import walker_reward
from tasks.walker_done import walker_done


def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/walker2d/walker2d.xml",
        # xml_path="assets/biped/biped.xml",
        episode_length=5_000,
        frame_skip=5,
        ctrl_scale=0.1,
        reward_fn=walker_reward,
        done_fn=walker_done,
        reset_noise_scale=0.0,
        render=False,  # workers don't need rendering
    )
    return MujocoEnv(cfg)


def make_policy(env_spec):
    return ActorCritic(env_spec, hidden_sizes=(64, 64))


def make_ppo(actor_critic):
    ppo_cfg = PPOConfig(
        gamma=0.995,
        lam=0.98,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=80,
        batch_size=512,   # larger batch since we're combining workers
        value_coef=0.5,
        entropy_coef=0.00,
        max_grad_norm=0.5,
    )
    return PPO(actor_critic, ppo_cfg, device="cpu")


def main():
    # Good practice with multiprocessing + PyTorch
    mp.set_start_method("spawn", force=True)

    train_cfg = MPTrainConfig(
        total_steps=200_000,
        horizon=1024,
        num_workers=7,
        log_interval=10,
        device="cpu",
        checkpoint_path="checkpoints/walker_ppo_mp.pt",
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