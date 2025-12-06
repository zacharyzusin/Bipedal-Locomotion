# recording/video_recorder.py
from __future__ import annotations
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Optional, List
import tempfile
import shutil

import numpy as np
import torch
from PIL import Image

from core.base_env import Env
from policies.actor_critic import ActorCritic


@dataclass
class CameraSettings:
    """Camera configuration for recording."""
    distance: float = 3.0
    azimuth: float = 90.0
    elevation: float = -20.0
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)


@dataclass
class RecordingConfig:
    """Configuration for video recording."""
    output_path: str | Path
    num_steps: int = 1000
    camera: CameraSettings = field(default_factory=CameraSettings)
    
    # Video encoding settings
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    crf: int = 18  # Quality (lower = better, 18-23 is good)
    
    # Optional: override FPS (default uses simulation time)
    fps_override: Optional[float] = None


# Type aliases
EnvFactory = Callable[[], Env]
PolicyFactory = Callable[[Any], ActorCritic]


def record_video(
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    config: RecordingConfig,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    seed: int | None = None,
    deterministic: bool = True,
) -> dict:
    """
    Record a video of the policy running in the environment.
    
    The video is recorded at the simulation's native framerate to ensure
    1 second in the video equals 1 second in the simulation.
    
    Args:
        env_factory: Callable that creates the environment
        policy_factory: Callable that creates the policy given an env
        config: Recording configuration
        checkpoint_path: Path to policy checkpoint (optional)
        device: Device to run policy on
        seed: Random seed for environment reset
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary with recording statistics
    """
    # Create environment and policy
    env = env_factory()
    policy = policy_factory(env)
    
    if hasattr(policy, "to"):
        policy = policy.to(device)
    if hasattr(policy, "eval"):
        policy.eval()
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if hasattr(policy, "load_state_dict"):
            policy.load_state_dict(state_dict)
    
    # Calculate FPS from simulation timestep
    # dt = timestep * frame_skip, so FPS = 1/dt for real-time playback
    sim_dt = env.dt if hasattr(env, "dt") else 0.02  # fallback to 50 FPS
    fps = config.fps_override if config.fps_override is not None else (1.0 / sim_dt)
    
    # Collect frames
    frames: List[np.ndarray] = []
    rewards: List[float] = []
    episode_lengths: List[int] = []
    
    obs = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    
    # Set camera if environment supports it
    if hasattr(env, "set_camera"):
        cam = config.camera
        env.set_camera(
            distance=cam.distance,
            azimuth=cam.azimuth,
            elevation=cam.elevation,
            lookat=cam.lookat,
        )
    
    current_episode_length = 0
    current_episode_reward = 0.0
    
    for step in range(config.num_steps):
        # Get action from policy
        with torch.no_grad():
            action, _ = policy.act(obs, deterministic=deterministic)
        
        action = action.squeeze(0).cpu().numpy()
        step_res = env.step(action)
        
        # Collect frame
        if step_res.frame is not None:
            frames.append(step_res.frame)
        
        # Track rewards
        current_episode_reward += step_res.reward
        current_episode_length += 1
        
        # Handle episode termination
        if step_res.done:
            rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0.0
            current_episode_length = 0
            
            obs = env.reset()
            if hasattr(policy, "reset"):
                policy.reset()
        else:
            obs = step_res.obs
    
    # Add final partial episode if any
    if current_episode_length > 0:
        rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)
    
    env.close()
    
    if len(frames) == 0:
        raise RuntimeError(
            "No frames were captured. Make sure the environment is configured "
            "with render=True"
        )
    
    # Encode video
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    _encode_video_ffmpeg(
        frames=frames,
        output_path=output_path,
        fps=fps,
        codec=config.codec,
        pixel_format=config.pixel_format,
        crf=config.crf,
    )
    
    # Compute stats
    stats = {
        "output_path": str(output_path),
        "num_frames": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps,
        "num_episodes": len(rewards),
        "total_reward": sum(rewards),
        "mean_episode_reward": np.mean(rewards) if rewards else 0.0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
    }
    
    return stats


def _encode_video_ffmpeg(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float,
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
    crf: int = 18,
) -> None:
    """
    Encode frames to video using FFmpeg via pipe.
    
    Args:
        frames: List of RGB uint8 frames (H, W, 3)
        output_path: Path to save video
        fps: Frames per second
        codec: Video codec (default: libx264)
        pixel_format: Pixel format (default: yuv420p for compatibility)
        crf: Constant rate factor for quality (lower = better)
    """
    if len(frames) == 0:
        raise ValueError("No frames to encode")
    
    height, width = frames[0].shape[:2]
    
    # Prepare all frame data as a single bytes object
    frame_data = b"".join(
        frame.astype(np.uint8).tobytes() if frame.dtype != np.uint8 
        else frame.tobytes() 
        for frame in frames
    )
    
    # FFmpeg command to read raw RGB frames from stdin
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",  # Read from stdin
        "-c:v", codec,
        "-pix_fmt", pixel_format,
        "-crf", str(crf),
        str(output_path),
    ]
    
    # Run FFmpeg with all data passed via communicate()
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    _, stderr = process.communicate(input=frame_data)
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")


def _encode_video_pillow(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float,
) -> None:
    """
    Fallback: encode to GIF using Pillow (lower quality, no FFmpeg needed).
    
    Args:
        frames: List of RGB uint8 frames
        output_path: Path to save (will be .gif)
        fps: Frames per second
    """
    if len(frames) == 0:
        raise ValueError("No frames to encode")
    
    # Convert to PIL images
    images = [Image.fromarray(f) for f in frames]
    
    # Calculate duration per frame in milliseconds
    duration_ms = int(1000 / fps)
    
    # Save as GIF
    gif_path = output_path.with_suffix(".gif")
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def record_video_simple(
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    output_path: str | Path,
    num_steps: int = 1000,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    distance: float = 3.0,
    azimuth: float = 90.0,
    elevation: float = -20.0,
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5),
    seed: int | None = None,
    deterministic: bool = True,
) -> dict:
    """
    Simplified interface for recording a video.
    
    Args:
        env_factory: Callable that creates the environment
        policy_factory: Callable that creates the policy given an env
        output_path: Where to save the video
        num_steps: Number of simulation steps to record
        checkpoint_path: Path to policy checkpoint
        device: Device to run policy on
        distance: Camera distance
        azimuth: Camera azimuth angle
        elevation: Camera elevation angle
        lookat: Camera look-at point (x, y, z)
        seed: Random seed
        deterministic: Use deterministic actions
        
    Returns:
        Dictionary with recording statistics
    """
    config = RecordingConfig(
        output_path=output_path,
        num_steps=num_steps,
        camera=CameraSettings(
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            lookat=lookat,
        ),
    )
    
    return record_video(
        env_factory=env_factory,
        policy_factory=policy_factory,
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        seed=seed,
        deterministic=deterministic,
    )