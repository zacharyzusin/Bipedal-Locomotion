import mujoco

def done(model: mujoco.MjModel, data: mujoco.MjData, t: int) -> bool:
    # Get torso world position via named access
    torso_pos = data.body('hips').xpos  # shape (3,)
    height = torso_pos[2]

    # Terminate if hips falls too low
    return height < 0.12