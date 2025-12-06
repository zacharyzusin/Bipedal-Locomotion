# scripts/biped/record_biped.py
from __future__ import annotations

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from policies.actor_critic import ActorCritic
from control.pd import PDConfig

from tasks.biped.reward import reward
from tasks.biped.done import done

from recording import CameraSettings, RecordingConfig, record_video


def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=5_000,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
        reset_noise_scale=0.0,
        render=True,
        reward_fn=reward,
        done_fn=done,
        width=640,
        height=480,
    )
    return MujocoEnv(cfg)


def make_policy(env):
    return ActorCritic(env.spec, hidden_sizes=(64, 64))


def main():
    # -------------------------------------------------------------------------
    # Recording settings - modify these as needed
    # -------------------------------------------------------------------------
    checkpoint_path = "checkpoints/biped_ppo_mp.pt"
    output_path = "recordings/biped_video.mp4"
    num_steps = 2000  # Number of simulation steps to record
    device = "cpu"
    seed = None  # Set to an integer for reproducible recordings

    # Camera settings
    camera = CameraSettings(
        distance=3.0,
        azimuth=90.0,
        elevation=-20.0,
        lookat=(0.0, 0.0, 0.5),
    )
    # -------------------------------------------------------------------------

    config = RecordingConfig(
        output_path=output_path,
        num_steps=num_steps,
        camera=camera,
    )

    print(f"Recording {num_steps} steps to {output_path}...")

    stats = record_video(
        env_factory=make_env,
        policy_factory=make_policy,
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        seed=seed,
    )

    print("\nRecording complete!")
    print(f"  Output: {stats['output_path']}")
    print(f"  Frames: {stats['num_frames']}")
    print(f"  FPS: {stats['fps']:.1f}")
    print(f"  Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Mean reward: {stats['mean_episode_reward']:.2f}")


if __name__ == "__main__":
    main()