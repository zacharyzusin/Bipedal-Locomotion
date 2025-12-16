# Bipedal Locomotion

A reinforcement learning framework for training and simulating bipedal robots using MuJoCo physics simulation and Proximal Policy Optimization (PPO). This project provides a complete pipeline for training, evaluating, streaming, and recording bipedal locomotion policies.

## Overview

This codebase implements a modular RL training system specifically designed for bipedal robot control. It supports both 2D (Walker2D) and 3D (Biped) robot models, with features including:

- **PPO Algorithm**: Full implementation of Proximal Policy Optimization with GAE-Lambda
- **Multi-process Training**: Parallel rollout collection for faster training
- **PD Control**: Optional proportional-derivative control for joint actuation
- **Reference Policies**: Hand-crafted baseline controllers using inverse kinematics
- **Real-time Streaming**: Web-based visualization with MJPEG streaming
- **Video Recording**: High-quality video export of trained policies

## Features

- **Bipedal Locomotion**: Train policies for bipedal robots to walk forward
- **PPO Training**: State-of-the-art on-policy RL algorithm
- **Multi-process Support**: Parallel environment workers for efficient data collection
- **PD Control**: Low-level joint control with configurable gains
- **Live Streaming**: Real-time visualization via web interface
- **Video Recording**: Export simulation videos with FFmpeg
- **Modular Design**: Easy to extend with new robots, tasks, and algorithms

## Installation

### Prerequisites

- Python 3.10+
- MuJoCo 3.1.2+
- FFmpeg (for video recording)

### Setup

1. **Install system dependencies** (Linux):
```bash
sudo apt update
sudo apt install -y \
    libx11-6 libgl1 libgl1-mesa-dev libglew-dev \
    libosmesa6 libosmesa6-dev patchelf
```

2. **Install MuJoCo**:
```bash
mkdir -p ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/3.1.2/mujoco-3.1.2-linux-x86_64.tar.gz
tar -xvf mujoco-3.1.2-linux-x86_64.tar.gz -C ~/.mujoco/
export MUJOCO_DIR=$HOME/.mujoco/mujoco-3.1.2
export MUJOCO_GL=osmesa
echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc
```

3. **Create conda environment**:
```bash
conda create -n rl-mujoco python=3.10 -y
conda activate rl-mujoco
```

4. **Install Python packages**:
```bash
pip install numpy==1.26.4
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or 'cuda' for GPU
pip install mujoco==3.1.2
pip install fastapi uvicorn pillow
pip install matplotlib
pip install shimmy>=2.0
pip install dm_control>=1.0.35
```

5. **Install project**:
```bash
cd /path/to/simulate-bidped
pip install -e .
```

## Project Structure

```
simulate-bidped/
├── algorithms/          # RL algorithms (PPO)
├── assets/             # MuJoCo XML models and meshes
│   ├── biped/
│   └── walker2d/
├── checkpoints/         # Saved model checkpoints
├── control/            # Control modules (PD, IK)
├── core/               # Core environment abstractions
├── policies/           # Neural network policies
├── recording/          # Video recording utilities
├── sampling/           # Rollout collection workers
├── scripts/            # Training and evaluation scripts
│   ├── biped/
│   └── walker2d/
├── streaming/          # MJPEG streaming server
├── tasks/              # Task definitions (rewards, done conditions)
│   ├── biped/
│   └── walker2d/
└── training/           # Training loops (single/multi-process)
```

## Quick Start

### Training a Biped Policy

**Single-process training:**
```bash
python scripts/biped/train_biped.py
```

**Multi-process training (faster):**
```bash
python scripts/biped/train_biped_mp.py
```

### Streaming Visualization

Start a web server to visualize a trained policy in real-time:
```bash
python scripts/biped/stream_biped.py
```

Then open `http://localhost:8000` in your browser.

### Recording Videos

Record a video of a trained policy:
```bash
python scripts/biped/record_biped.py
```

## Key Components

### Core Environment (`core/`)

- **`base_env.py`**: Abstract base class for environments
- **`mujoco_env.py`**: MuJoCo environment wrapper with support for:
  - 2D and 3D robot models
  - Custom reward and done functions
  - PD control integration
  - Rendering and camera control
  - Observation space: base orientation/velocity + joint positions/velocities

### Algorithms (`algorithms/`)

- **`ppo.py`**: PPO implementation with:
  - GAE-Lambda advantage estimation
  - Clipped surrogate objective
  - Value function learning
  - Gradient clipping

### Policies (`policies/`)

- **`actor_critic.py`**: Actor-Critic neural network with:
  - Shared feature extractor
  - Gaussian action distribution
  - Separate value head
- **`reference_policy.py`**: Hand-crafted baseline using:
  - Gait pattern generation
  - Inverse kinematics (2R leg)
  - PD control

### Control (`control/`)

- **`pd.py`**: Proportional-derivative controller for joint control
- **`ik_2r.py`**: Inverse kinematics solver for 2-link planar legs

### Tasks (`tasks/`)

- **`biped/reward.py`**: Reward function encouraging:
  - Forward velocity
  - Staying upright (alive bonus)
  - Low control effort
  - Minimal lateral deviation
- **`biped/done.py`**: Episode termination when robot falls

### Training (`training/`)

- **`on_policy.py`**: Single-process training loop
- **`mp_on_policy.py`**: Multi-process training with parallel workers

### Streaming (`streaming/`)

- **`mjpeg_server.py`**: FastAPI server for:
  - MJPEG video streaming
  - Real-time reward statistics
  - Interactive camera controls

### Recording (`recording/`)

- **`video_recorder.py`**: Video export with:
  - FFmpeg encoding
  - Configurable camera settings
  - Episode statistics

## Configuration

### Environment Configuration

The `MujocoEnvConfig` allows customization of:
- `xml_path`: Path to MuJoCo XML model
- `episode_length`: Maximum steps per episode
- `frame_skip`: Number of simulation steps per action
- `pd_cfg`: PD controller parameters (kp, kd, torque_limit)
- `reward_fn`: Custom reward function
- `done_fn`: Custom termination condition
- `reset_noise_scale`: Initial state randomization
- `render`: Enable rendering for visualization

### PPO Configuration

The `PPOConfig` includes:
- `gamma`: Discount factor (default: 0.99)
- `lam`: GAE-Lambda parameter (default: 0.95)
- `clip_ratio`: PPO clipping ratio (default: 0.2)
- `lr`: Learning rate (default: 3e-4)
- `train_iters`: PPO update iterations (default: 80)
- `batch_size`: Mini-batch size (default: 64)
- `value_coef`: Value loss coefficient (default: 0.5)
- `entropy_coef`: Entropy bonus coefficient (default: 0.0)
- `max_grad_norm`: Gradient clipping (default: 0.5)

### Training Configuration

- `total_steps`: Total training steps
- `horizon`: Steps per rollout collection
- `log_interval`: Logging frequency
- `device`: "cpu" or "cuda"
- `checkpoint_path`: Where to save models

## Usage Examples

### Creating a Custom Environment

```python
from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from tasks.biped.reward import reward
from tasks.biped.done import done
from control.pd import PDConfig

cfg = MujocoEnvConfig(
    xml_path="assets/biped/biped.xml",
    episode_length=1000,
    frame_skip=5,
    pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
    reward_fn=reward,
    done_fn=done,
    render=False,
)
env = MujocoEnv(cfg)
```

### Training a Policy

```python
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig
from training.on_policy import OnPolicyTrainer, TrainConfig

policy = ActorCritic(env.spec, hidden_sizes=(64, 64))
ppo = PPO(policy, PPOConfig(), device="cpu")

trainer = OnPolicyTrainer(
    env_factory=lambda: env,
    policy_factory=lambda e: ActorCritic(e.spec),
    algo_factory=lambda p: PPO(p, PPOConfig()),
    train_cfg=TrainConfig(total_steps=200_000, horizon=2048),
)
trainer.run()
```

### Evaluating a Policy

```python
import torch

policy = ActorCritic(env.spec)
policy.load_state_dict(torch.load("checkpoints/biped_ppo.pt"))
policy.eval()

obs = env.reset()
for _ in range(1000):
    with torch.no_grad():
        action, _ = policy.act(obs, deterministic=True)
    step_res = env.step(action.squeeze(0).cpu().numpy())
    obs = step_res.obs if not step_res.done else env.reset()
```

## Robot Models

### Biped

3D bipedal robot with:
- 6 actuated joints (hip, knee, ankle per leg)
- Free-floating base (7 DOF: x, y, z, quaternion)
- STL mesh assets for visualization

### Walker2D

2D planar walker with:
- Simplified 2D dynamics
- Planar motion (x, z, rotation)

## Tips

1. **Multi-process Training**: Use `train_biped_mp.py` for faster training with multiple CPU cores
2. **PD Control**: Adjust `kp` and `kd` gains to match your robot's dynamics
3. **Reward Tuning**: Modify `tasks/biped/reward.py` to shape desired behaviors
4. **Episode Length**: Longer episodes (4096+) help with learning stable gaits
5. **Checkpointing**: Models are saved automatically during training

## Troubleshooting

- **MuJoCo errors**: Ensure `MUJOCO_GL=osmesa` is set for headless rendering
- **Import errors**: Make sure you've run `pip install -e .` in the project root
- **Video recording fails**: Install FFmpeg: `sudo apt install ffmpeg`
- **Slow training**: Use multi-process training or reduce `horizon`

## Acknowledgments

Built for robotics research and education. Uses MuJoCo physics engine and PyTorch for deep learning.
