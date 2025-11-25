# scripts/train_walker.py
from __future__ import annotations

import mujoco
import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig
from training.on_policy import OnPolicyTrainer, TrainConfig


def walker_reward(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """
    Classic walker reward:
    + forward velocity
    - small control cost
    """

    # Forward velocity of torso (body 1, by convention)
    # You may need to adjust: often torso is body 1 or body named "torso"
    torso_id = 1
    vel = data.cvel[torso_id]   # spatial velocity (6D: angular + linear)
    forward_vel = vel[3]        # linear x velocity

    # Control effort penalty
    ctrl_cost = 0.001 * np.sum(np.square(data.ctrl))

    reward = forward_vel - ctrl_cost
    return float(reward)

def walker_done(model: mujoco.MjModel, data: mujoco.MjData, t: int) -> bool:
    # Termination if torso falls too low or flips
    torso_id = 1
    height = data.xpos[torso_id][2]

    if height < 0.7:   # tune for your model
        return True

    return False


def make_env():
    # Point to your MuJoCo walker model
    cfg = MujocoEnvConfig(
        xml_path="assets/walker2d/walker2d.xml",  # or your custom walker
        episode_length=1000,
        frame_skip=5,
        ctrl_scale=1.0,
        reward_fn=walker_reward,
        done_fn=walker_done,
        render=False,
    )
    return MujocoEnv(cfg)


def make_policy(env_spec):
    # Hidden sizes are easily configurable per-task here
    return ActorCritic(env_spec, hidden_sizes=(64, 64))


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
