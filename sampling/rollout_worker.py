"""Rollout collection worker for multi-process training.

This module implements worker processes that collect rollouts in parallel.
Workers receive policy parameters via queues, run rollouts, and send results
back to the main training process.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Any, Callable

import numpy as np
import torch

from core.base_env import Env
from policies.actor_critic import ActorCritic

# Type aliases for factories (must be picklable for multiprocessing)
EnvFactory = Callable[[], Env]
PolicyFactory = Callable[[Any], ActorCritic]


@dataclass
class WorkerConfig:
    """Configuration for rollout worker.
    
    Attributes:
        horizon: Number of steps to collect per rollout.
    """
    horizon: int = 1024


@dataclass
class RolloutBatch:
    """Batch of rollout data collected by a worker.
    
    Attributes:
        obs: Observation dictionary with arrays of shape [T, ...].
        actions: Actions array of shape [T, act_dim].
        rewards: Rewards array of shape [T].
        dones: Done flags array of shape [T].
        log_probs: Log probabilities array of shape [T].
        last_obs: Last observation dictionary (for bootstrapping).
    """
    obs: Dict[str, np.ndarray]        # {key: [T, dim]}
    actions: np.ndarray               # [T, act_dim]
    rewards: np.ndarray               # [T]
    dones: np.ndarray                 # [T]
    log_probs: np.ndarray             # [T]
    last_obs: Dict[str, np.ndarray]   # {key: [dim]}

def worker_loop(
    worker_id: int,
    env_factory: EnvFactory,
    policy_factory: PolicyFactory,
    params_queue,
    rollout_queue,
    cfg: WorkerConfig,
    device: str = "cpu",
):
    """Main loop for rollout worker process.
    
    This function runs in a separate process and continuously:
    1. Waits for policy parameters from params_queue
    2. Loads parameters into local policy
    3. Collects a rollout of length cfg.horizon
    4. Sends RolloutBatch to rollout_queue
    5. Repeats until receiving "STOP" message
    
    Args:
        worker_id: Unique identifier for this worker.
        env_factory: Function to create environment (must be picklable).
        policy_factory: Function to create policy (must be picklable).
        params_queue: Queue to receive policy parameters (state_dict).
        rollout_queue: Queue to send RolloutBatch results.
        cfg: Worker configuration.
        device: Device to run policy on.
    """

    # Important: factories must be top-level functions (picklable)
    env = env_factory()
    policy = policy_factory(env).to(device)
    policy.eval()

    while True:
        msg = params_queue.get()  # block until we get a command
        if msg == "STOP":
            break

        state_dict = msg
        policy.load_state_dict(state_dict)

        batch = _collect_rollout(env, policy, cfg.horizon, device)
        rollout_queue.put((worker_id, batch))

    env.close()


def _collect_rollout(
    env: Env,
    policy: ActorCritic,
    horizon: int,
    device: str,
) -> RolloutBatch:
    """Collect a single rollout from the environment.
    
    Args:
        env: Environment to collect from.
        policy: Policy to use for action selection.
        horizon: Number of steps to collect.
        device: Device policy is on.
        
    Returns:
        RolloutBatch containing all collected data.
    """
    obs_buf: Dict[str, list] = defaultdict(list)
    act_buf, logp_buf, rew_buf, done_buf = [], [], [], []

    obs = env.reset()
    for _ in range(horizon):
        with torch.no_grad():
            action, logp_t = policy.act(obs, deterministic=False)

        action = action.squeeze(0).cpu().numpy()
        logp = logp_t.squeeze(0).cpu().numpy()

        step_res = env.step(action)

        for key, val in obs.items():
            obs_buf[key].append(val)
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(step_res.reward)
        done_buf.append(step_res.done)

        obs = step_res.obs if not step_res.done else env.reset()

    last_obs = obs

    return RolloutBatch(
        obs={k: np.array(v, dtype=np.float32) for k, v in obs_buf.items()},
        actions=np.array(act_buf, dtype=np.float32),
        rewards=np.array(rew_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=np.bool_),
        log_probs=np.array(logp_buf, dtype=np.float32),
        last_obs={k: np.array(v, dtype=np.float32) for k, v in last_obs.items()},
    )