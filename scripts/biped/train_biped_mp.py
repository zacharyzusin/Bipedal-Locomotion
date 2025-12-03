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
        episode_length=5_000,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
        reward_fn=reward,
        done_fn=done,
        reset_noise_scale=0.0,
        render=False,  # workers don't need rendering
    )
    return MujocoEnv(cfg)


def make_policy(env):
    return ActorCritic(env.spec, hidden_sizes=(64, 64))


def make_ppo(actor_critic):
    ppo_cfg = PPOConfig(
        gamma=0.99,         # slightly more myopic, more stable for bipeds
        lam=0.90,           # less variance & less overestimation in GAE
        clip_ratio=0.1,     # smaller, gentler policy updates
        lr=1e-4,            # smaller LR to avoid destroying good gaits
        train_iters=80,
        batch_size=512,
        value_coef=0.5,
        entropy_coef=0.0005,  # tiny bit of exploration, no entropy explosion
        max_grad_norm=0.5,
    )
    return PPO(actor_critic, ppo_cfg, device="cpu")



def main():
    # Good practice with multiprocessing + PyTorch
    mp.set_start_method("spawn", force=True)

    train_cfg = MPTrainConfig(
        total_steps=3_000_000,
        horizon=1024,
        num_workers=7,
        log_interval=10,
        device="cpu",
        checkpoint_path="checkpoints/biped_new.pt",
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