"""Termination condition for bipedal locomotion task.

This module defines when an episode should terminate, typically when
the robot falls or fails to maintain balance.
"""
import mujoco


def done(model: mujoco.MjModel, data: mujoco.MjData, t: int) -> bool:
    """Check if episode should terminate.
    
    Terminates when the robot's hip height falls below a threshold,
    indicating the robot has fallen.
    
    Args:
        model: MuJoCo model.
        data: MuJoCo data (current state).
        t: Current time step.
        
    Returns:
        True if episode should terminate, False otherwise.
    """
    # Get torso world position via named access
    torso_pos = data.body('hips').xpos  # shape (3,)
    height = torso_pos[2]  # z-component (vertical height)

    # Terminate if hips fall too low (robot has fallen)
    return height < 0.12