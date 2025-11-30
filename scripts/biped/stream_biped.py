# scripts/stream_walker.py
from __future__ import annotations

import uvicorn

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from core.biped_obs_wrapper import BipedSensorWrapper
from policies.actor_critic import ActorCritic
from training.on_policy import TrainConfig  # if you want same device config
from streaming.mjpeg_server import create_app

from tasks.biped.reward import reward
from tasks.biped.done import done

def make_env():
    # Make the EXACT SAME base config as training
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=5000,
        frame_skip=5,
        ctrl_scale=0.1,
        reset_noise_scale=0.01,
        reward_fn=reward,
        done_fn=done,
        render=True,          # only difference allowed
        width=640,
        height=480,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )

    base = MujocoEnv(cfg)
    # Critically: wrap it in the SAME observation wrapper
    return BipedSensorWrapper(base)

def make_policy(env):
    return ActorCritic(env.spec, hidden_sizes=(128, 128))


def main():
    checkpoint_path = "checkpoints/biped_ppo_mp.pt"  # same as in training
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