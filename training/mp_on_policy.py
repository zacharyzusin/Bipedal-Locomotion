# training/mp_on_policy.py
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Dict, Any, List

import multiprocessing as mp
import numpy as np
import torch

from core.base_env import Env
from policies.actor_critic import ActorCritic
from algorithms.ppo import PPO, PPOConfig, compute_gae
from sampling.rollout_worker import (
    EnvFactory,
    PolicyFactory,
    WorkerConfig,
    RolloutBatch,
    worker_loop,
)


@dataclass
class MPTrainConfig:
    total_steps: int = 1_000_000
    horizon: int = 1024
    num_workers: int = 8
    log_interval: int = 10
    device: str = "cpu"
    checkpoint_path: str = "checkpoints/walker_ppo_mp.pt"


AlgoFactory = Callable[[ActorCritic], PPO]


class MultiProcessOnPolicyTrainer:
    def __init__(
        self,
        env_factory: EnvFactory,
        policy_factory: PolicyFactory,
        algo_factory: AlgoFactory | None = None,
        train_cfg: MPTrainConfig | None = None,
        ppo_cfg: PPOConfig | None = None,
    ):
        self.env_factory = env_factory
        self.policy_factory = policy_factory

        self.cfg = train_cfg or MPTrainConfig()
        self.device = self.cfg.device

        # Create a reference env to get spec (no need for rendering)
        ref_env = env_factory()
        self.policy = policy_factory(ref_env).to(self.device)
        ppo_cfg = ppo_cfg or PPOConfig()
        self.algo = algo_factory(self.policy) if algo_factory else PPO(
            self.policy, ppo_cfg, device=self.device
        )
        ref_env.close()

        self.num_workers = self.cfg.num_workers
        self.horizon = self.cfg.horizon

        # Multiprocessing primitives
        self.params_queues: List[mp.Queue] = []
        self.rollout_queue: mp.Queue = mp.Queue()

        self.workers: List[mp.Process] = []
        self._start_workers()

    # ----------------------------------------------------------------------
    # Worker management
    # ----------------------------------------------------------------------
    def _start_workers(self) -> None:
        ctx = mp.get_context("spawn")  # safer with PyTorch
        worker_cfg = WorkerConfig(horizon=self.horizon)

        for wid in range(self.num_workers):
            params_q = ctx.Queue()
            self.params_queues.append(params_q)

            p = ctx.Process(
                target=worker_loop,
                args=(
                    wid,
                    self.env_factory,
                    self.policy_factory,
                    params_q,
                    self.rollout_queue,
                    worker_cfg,
                    self.device,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def _stop_workers(self) -> None:
        for q in self.params_queues:
            q.put("STOP")
        for p in self.workers:
            p.join()

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------
    def run(self) -> None:
        import time

        start_time = time.time()
        total_steps = 0
        iteration = 0
        highest_return = -np.inf

        try:
            while total_steps < self.cfg.total_steps:
                iteration += 1

                # Broadcast latest parameters
                state_dict = self.policy.state_dict()
                for q in self.params_queues:
                    q.put(state_dict)

                # Collect one batch per worker
                batches = []
                for _ in range(self.num_workers):
                    wid, batch = self.rollout_queue.get()
                    batches.append(batch)

                combined = self._combine_batches(batches)
                batch_steps = combined["actions"].shape[0]
                total_steps += batch_steps

                metrics = self.algo.update(combined)

                avg_return = combined["returns"].mean()
                if avg_return > highest_return:
                    highest_return = avg_return
                    if self.cfg.checkpoint_path:
                        import os
                        os.makedirs(os.path.dirname(self.cfg.checkpoint_path), exist_ok=True)
                        torch.save(self.policy.state_dict(), self.cfg.checkpoint_path)

                if iteration % self.cfg.log_interval == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = total_steps / max(1.0, elapsed)

                    print(
                        f"[Iter {iteration}] Steps={total_steps}  "
                        f"Steps/sec={steps_per_sec:.1f}  "
                        f"Return={avg_return:.2f}  "
                        f"pi_loss={metrics['policy_loss']:.3f}  "
                        f"v_loss={metrics['value_loss']:.3f}  "
                        f"entropy={metrics['entropy']:.3f}"
                    )

        finally:
            # Print final total timing and throughput
            total_time = time.time() - start_time
            steps_per_sec = total_steps / max(1.0, total_time)
            print("==================================================")
            print("Training finished.")
            print(f"Total steps:        {total_steps}")
            print(f"Total time:         {total_time:.2f} seconds")
            print(f"Average steps/sec:  {steps_per_sec:.1f}")
            print("==================================================")

            self._stop_workers()


    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def _combine_batches(self, batches: List[RolloutBatch]) -> Dict[str, Any]:
        """
        - Compute values & advantages for each worker's trajectory.
        - Concatenate into a single big batch for PPO.
        """
        all_obs: Dict[str, list] = defaultdict(list)
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []

        gamma = self.algo.cfg.gamma
        lam = self.algo.cfg.lam

        for b in batches:
            # Compute values on obs
            with torch.no_grad():
                values = self.policy.value(b.obs).cpu().numpy()
            
            # Bootstrap last value from last_obs
            with torch.no_grad():
                last_value = self.policy.value(b.last_obs).item()

            adv, ret = compute_gae(
                rewards=b.rewards,
                values=values,
                dones=b.dones,
                last_value=last_value,
                gamma=gamma,
                lam=lam,
            )

            for key, val in b.obs.items():
                all_obs[key].append(val)
            all_actions.append(b.actions)
            all_log_probs.append(b.log_probs)
            all_advantages.append(adv)
            all_returns.append(ret)

        obs_dict = {k: np.concatenate(v, axis=0) for k, v in all_obs.items()}
        act_arr = np.concatenate(all_actions, axis=0)
        logp_arr = np.concatenate(all_log_probs, axis=0)
        adv_arr = np.concatenate(all_advantages, axis=0)
        ret_arr = np.concatenate(all_returns, axis=0)

        return {
            "obs": obs_dict,
            "actions": act_arr,
            "log_probs": logp_arr,
            "advantages": adv_arr,
            "returns": ret_arr,
        }