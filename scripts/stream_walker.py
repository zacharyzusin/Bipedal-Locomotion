# scripts/stream_walker.py
from __future__ import annotations

import uvicorn

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from training.on_policy import TrainConfig  # if you want same device config
from streaming.mjpeg_server import create_app

from tasks.walker_reward import walker_reward
from tasks.walker_done import walker_done

def make_env() -> MujocoEnv:
    # For streaming we want rendering enabled
    cfg = MujocoEnvConfig(
        xml_path="assets/walker2d/walker2d.xml",  # or your custom walker XML
        episode_length=5_000,
        frame_skip=5,
        ctrl_scale=0.1,
        reset_noise_scale=0.01,
        render=True,
        reward_fn=walker_reward,
        done_fn=walker_done,
        width=640,
        height=480,
    )
    return MujocoEnv(cfg)


def make_policy(env_spec):
    return ActorCritic(env_spec, hidden_sizes=(64, 64))


def main():
    checkpoint_path = "checkpoints/walker_ppo_mp.pt"  # same as in training
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