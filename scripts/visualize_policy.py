import numpy as np
import matplotlib.pyplot as plt

recording_path = "recordings/biped_reference_recording_4096.npz"
data = np.load(recording_path)

print(f"Loaded {len(data['time'])} timesteps from {recording_path}")
print(f"Keys: {list(data.keys())}")
print(f"qpos shape: {data['qpos'].shape}")  # (timesteps, num_joints)
print(f"qvel shape: {data['qvel'].shape}")

cycle_duration = 1.25  # seconds per gait cycle

# Optional: Define joint names if known (adjust based on your model)
# For a typical biped: root (7 DoF: 3 pos + 4 quat) + leg joints
JOINT_NAMES = None  # Set to list of names if available, e.g., ["root_x", "root_y", ...]


def plot_qpos(data: np.ndarray, save_dir: str = ".", cycle_duration: float = None, joint_names: list = None):
    """Plot all joint positions over time."""
    time = data["time"]
    qpos = data["qpos"]
    num_joints = qpos.shape[1]
    
    # Create subplots - group into reasonable chunks
    joints_per_fig = 6
    num_figs = (num_joints + joints_per_fig - 1) // joints_per_fig
    
    for fig_idx in range(num_figs):
        start_j = fig_idx * joints_per_fig
        end_j = min(start_j + joints_per_fig, num_joints)
        n_plots = end_j - start_j
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        for i, j in enumerate(range(start_j, end_j)):
            ax = axes[i]
            ax.plot(time, qpos[:, j], linewidth=1.5)
            
            label = joint_names[j] if joint_names else f"qpos[{j}]"
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add min/max annotations
            min_val, max_val = qpos[:, j].min(), qpos[:, j].max()
            ax.axhline(y=min_val, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=max_val, color='gray', linestyle=':', alpha=0.5)
            ax.text(time[-1], max_val, f' {max_val:.3f}', va='center', fontsize=8)
            ax.text(time[-1], min_val, f' {min_val:.3f}', va='center', fontsize=8)
            
            # Cycle boundaries
            if cycle_duration is not None:
                for ct in np.arange(0, time[-1], cycle_duration):
                    ax.axvline(x=ct, color='purple', linestyle='--', alpha=0.4, linewidth=1)
        
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        fig.suptitle(f'Joint Positions (qpos) - Part {fig_idx + 1}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/qpos_part{fig_idx + 1}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_qvel(data: np.ndarray, save_dir: str = ".", cycle_duration: float = None, joint_names: list = None):
    """Plot all joint velocities over time."""
    time = data["time"]
    qvel = data["qvel"]
    num_joints = qvel.shape[1]
    
    joints_per_fig = 6
    num_figs = (num_joints + joints_per_fig - 1) // joints_per_fig
    
    for fig_idx in range(num_figs):
        start_j = fig_idx * joints_per_fig
        end_j = min(start_j + joints_per_fig, num_joints)
        n_plots = end_j - start_j
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        for i, j in enumerate(range(start_j, end_j)):
            ax = axes[i]
            ax.plot(time, qvel[:, j], linewidth=1.5, color='#ff7f0e')
            
            label = joint_names[j] if joint_names else f"qvel[{j}]"
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            if cycle_duration is not None:
                for ct in np.arange(0, time[-1], cycle_duration):
                    ax.axvline(x=ct, color='purple', linestyle='--', alpha=0.4, linewidth=1)
        
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        fig.suptitle(f'Joint Velocities (qvel) - Part {fig_idx + 1}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/qvel_part{fig_idx + 1}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_phase_portrait(data: np.ndarray, joint_idx: int, save_dir: str = ".", joint_name: str = None):
    """Plot phase portrait (position vs velocity) for a joint."""
    qpos = data["qpos"][:, joint_idx]
    qvel = data["qvel"][:, joint_idx]
    time = data["time"]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(qpos, qvel, c=time, cmap='viridis', s=2, alpha=0.7)
    plt.colorbar(label='Time (s)')
    plt.xlabel('Position', fontsize=10)
    plt.ylabel('Velocity', fontsize=10)
    label = joint_name if joint_name else f"Joint {joint_idx}"
    plt.title(f'Phase Portrait: {label}', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/phase_joint{joint_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()


# Generate plots
plot_qpos(data, cycle_duration=cycle_duration, joint_names=JOINT_NAMES)
plot_qvel(data, cycle_duration=cycle_duration, joint_names=JOINT_NAMES)

# Plot phase portraits for a few interesting joints (adjust indices as needed)
num_joints = data["qpos"].shape[1]
for j in range(min(3, num_joints)):  # First 3 joints as example
    plot_phase_portrait(data, j)