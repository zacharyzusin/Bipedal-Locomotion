# scripts/biped/evaluate_policies.py
"""
Compare performance of different policies on the biped environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, List, Dict
import numpy as np
import torch

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from core.base_env import Env
from policies.actor_critic import ActorCritic
from policies.reference_policy import (
    ReferenceWalkerPolicy,
    GaitParams,
)
from control.ik_2r import Planar2RLegConfig
from control.pd import PDConfig

from tasks.biped.reward import reward
from tasks.biped.done import done


@dataclass
class EvalStats:
    """Statistics from policy evaluation."""
    name: str
    num_episodes: int
    
    # Per-episode metrics (arrays)
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    episode_distances: np.ndarray
    
    # Aggregate metrics
    mean_reward: float
    std_reward: float
    mean_length: float
    std_length: float
    mean_distance: float
    std_distance: float
    mean_velocity: float
    
    # Control metrics
    mean_torque: float
    max_torque: float
    torque_std: float
    
    # Reward components (if available)
    reward_components: Dict[str, float]
    
    def __str__(self) -> str:
        lines = [
            f"\n{'='*50}",
            f"Policy: {self.name}",
            f"{'='*50}",
            f"Episodes evaluated: {self.num_episodes}",
            f"",
            f"Episode Reward:   {self.mean_reward:8.2f} ± {self.std_reward:.2f}",
            f"Episode Length:   {self.mean_length:8.1f} ± {self.std_length:.1f} steps",
            f"Distance:         {self.mean_distance:8.3f} ± {self.std_distance:.3f} m",
            f"Mean Velocity:    {self.mean_velocity:8.4f} m/step",
            f"",
            f"Mean |Torque|:    {self.mean_torque:8.4f}",
            f"Max |Torque|:     {self.max_torque:8.4f}",
            f"Torque Std:       {self.torque_std:8.4f}",
        ]
        
        if self.reward_components:
            lines.append("")
            lines.append("Reward Components (mean per step):")
            for name, value in self.reward_components.items():
                lines.append(f"  {name:20s}: {value:8.4f}")
        
        return "\n".join(lines)


def evaluate_policy(
    env_factory: Callable[[], Env],
    policy_factory: Callable[[Any], Any],
    policy_name: str,
    num_episodes: int = 10,
    max_steps_per_episode: int = 5000,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    deterministic: bool = True,
) -> EvalStats:
    """
    Evaluate a policy over multiple episodes and collect statistics.
    """
    env = env_factory()
    policy = policy_factory(env)
    
    if hasattr(policy, "to"):
        policy = policy.to(device)
    if hasattr(policy, "eval"):
        policy.eval()
    
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if hasattr(policy, "load_state_dict"):
            policy.load_state_dict(state_dict)
    
    # Storage for episode-level metrics
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_distances: List[float] = []
    
    # Storage for step-level metrics (aggregated)
    all_torques: List[float] = []
    reward_components_acc: Dict[str, List[float]] = {}
    
    for ep in range(num_episodes):
        obs = env.reset(seed=ep)
        if hasattr(policy, "reset"):
            policy.reset()
        
        ep_reward = 0.0
        ep_length = 0
        start_x = env.data.qpos[0] if hasattr(env, "data") else 0.0
        
        for step in range(max_steps_per_episode):
            with torch.no_grad():
                action, _ = policy.act(obs, deterministic=deterministic)
            
            # Handle different action formats from different policies
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).cpu().numpy()
            elif isinstance(action, np.ndarray):
                action = action.squeeze()
            # else: assume it's already in the right format
            
            step_res = env.step(action)
            
            ep_reward += step_res.reward
            ep_length += 1
            
            # Track torques
            if hasattr(env, "data"):
                torques = np.abs(env.data.ctrl)
                all_torques.extend(torques.tolist())
            
            # Track reward components
            if "reward_components" in step_res.info:
                for name, val in step_res.info["reward_components"].items():
                    if name not in reward_components_acc:
                        reward_components_acc[name] = []
                    reward_components_acc[name].append(val)
            
            if step_res.done:
                break
            obs = step_res.obs
        
        # End position
        end_x = env.data.qpos[0] if hasattr(env, "data") else 0.0
        distance = end_x - start_x
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_distances.append(distance)
        
        print(f"  Episode {ep+1}/{num_episodes}: "
              f"reward={ep_reward:.2f}, length={ep_length}, distance={distance:.3f}m")
    
    env.close()
    
    # Convert to arrays
    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)
    episode_distances = np.array(episode_distances)
    all_torques = np.array(all_torques) if all_torques else np.array([0.0])
    
    # Compute reward component means
    reward_components_mean = {
        name: np.mean(vals) for name, vals in reward_components_acc.items()
    }
    
    return EvalStats(
        name=policy_name,
        num_episodes=num_episodes,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_distances=episode_distances,
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        mean_length=float(np.mean(episode_lengths)),
        std_length=float(np.std(episode_lengths)),
        mean_distance=float(np.mean(episode_distances)),
        std_distance=float(np.std(episode_distances)),
        mean_velocity=float(np.mean(episode_distances / episode_lengths)),
        mean_torque=float(np.mean(all_torques)),
        max_torque=float(np.max(all_torques)),
        torque_std=float(np.std(all_torques)),
        reward_components=reward_components_mean,
    )


def print_comparison_table(stats_list: List[EvalStats]) -> None:
    """Print a side-by-side comparison table."""
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    # Header
    header = f"{'Metric':<25}"
    for s in stats_list:
        header += f"{s.name:>20}"
    print(header)
    print("-"*70)
    
    # Rows
    metrics = [
        ("Episode Reward", lambda s: f"{s.mean_reward:.2f} ± {s.std_reward:.2f}"),
        ("Episode Length", lambda s: f"{s.mean_length:.0f} ± {s.std_length:.0f}"),
        ("Distance (m)", lambda s: f"{s.mean_distance:.3f} ± {s.std_distance:.3f}"),
        ("Mean Velocity (m/step)", lambda s: f"{s.mean_velocity:.5f}"),
        ("Mean |Torque|", lambda s: f"{s.mean_torque:.4f}"),
        ("Max |Torque|", lambda s: f"{s.max_torque:.4f}"),
    ]
    
    for name, getter in metrics:
        row = f"{name:<25}"
        for s in stats_list:
            row += f"{getter(s):>20}"
        print(row)
    
    print("="*70)


def generate_latex_table(stats_list: List[EvalStats]) -> str:
    """Generate a LaTeX table for the report."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Policy Performance Comparison}",
        r"\label{tab:policy-comparison}",
        r"\begin{tabular}{l" + "c" * len(stats_list) + "}",
        r"\toprule",
        r"Metric & " + " & ".join(s.name for s in stats_list) + r" \\",
        r"\midrule",
    ]
    
    metrics = [
        ("Episode Reward", lambda s: f"${s.mean_reward:.2f} \\pm {s.std_reward:.2f}$"),
        ("Episode Length", lambda s: f"${s.mean_length:.0f} \\pm {s.std_length:.0f}$"),
        ("Distance (m)", lambda s: f"${s.mean_distance:.3f} \\pm {s.std_distance:.3f}$"),
        ("Mean Velocity (m/step)", lambda s: f"${s.mean_velocity:.5f}$"),
        ("Mean $|\\tau|$", lambda s: f"${s.mean_torque:.4f}$"),
        ("Max $|\\tau|$", lambda s: f"${s.max_torque:.4f}$"),
    ]
    
    for name, getter in metrics:
        row = name + " & " + " & ".join(getter(s) for s in stats_list) + r" \\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Environment and Policy Factories
# -----------------------------------------------------------------------------

def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=5_000,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
        reset_noise_scale=0.0,
        render=False,  # No rendering for faster evaluation
        reward_fn=reward,
        done_fn=done,
        base_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    return MujocoEnv(cfg)


def make_actor_critic_policy(env: MujocoEnv) -> ActorCritic:
    return ActorCritic(env.spec, hidden_sizes=(64, 64))


def make_reference_policy(env: MujocoEnv) -> ReferenceWalkerPolicy:
    gait_params = GaitParams(
        step_length=0.02,
        step_height=0.01,
        cycle_duration=1.25,
    )
    
    leg_geom = Planar2RLegConfig(
        L1=0.05,
        L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0,
        ankle_offset=-np.pi/2,
    )
    
    pd_cfg = PDConfig(kp=5.0, kd=1.0, torque_limit=None)
    
    return ReferenceWalkerPolicy(
        env=env,
        gait_params=gait_params,
        left_leg_geom=leg_geom,
        right_leg_geom=leg_geom,
        pd_config=pd_cfg,
        desired_foot_angle=0.0,
    )


def main():
    num_episodes = 10
    device = "cpu"
    
    print("Evaluating policies...")
    
    # Evaluate learned policy
    print("\n[1/2] Evaluating Learned Policy (ActorCritic)...")
    learned_stats = evaluate_policy(
        env_factory=make_env,
        policy_factory=make_actor_critic_policy,
        policy_name="Learned (PPO)",
        num_episodes=num_episodes,
        checkpoint_path="checkpoints/biped_ppo_mp.pt",
        device=device,
    )
    
    # Evaluate reference policy
    print("\n[2/2] Evaluating Reference Policy...")
    reference_stats = evaluate_policy(
        env_factory=make_env,
        policy_factory=make_reference_policy,
        policy_name="Reference",
        num_episodes=num_episodes,
        checkpoint_path=None,  # No checkpoint needed
        device=device,
    )
    
    # Print individual stats
    print(learned_stats)
    print(reference_stats)
    
    # Print comparison table
    print_comparison_table([learned_stats, reference_stats])
    
    # Generate LaTeX table
    latex = generate_latex_table([learned_stats, reference_stats])
    print("\nLaTeX Table:")
    print(latex)
    
    # Save LaTeX to file
    with open("recordings/policy_comparison.tex", "w") as f:
        f.write(latex)
    print("\nLaTeX table saved to recordings/policy_comparison.tex")


if __name__ == "__main__":
    main()