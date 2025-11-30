import numpy as np

def reward(model, data):
    """
    Simplified locomotion-first reward.
    Goal: get ANY forward motion before shaping gait.
    """

    # --- 1. Forward COM velocity (dominant term) ---
    com_vel = data.qvel[0]
    r_forward = 5.0 * com_vel     # huge emphasis on forward motion

    # --- 2. Upright torso (weaker) ---
    torso_id = model.body("hips").id
    quat = data.xquat[torso_id]
    w,x,y,z = quat
    pitch_roll_mag = np.sqrt(x*x + y*y)
    r_upright = np.exp(-4.0 * pitch_roll_mag)

    # --- 3. Angular velocity penalty (light) ---
    ang_vel = data.cvel[torso_id, 3:]
    r_torso_stable = -0.1 * np.linalg.norm(ang_vel)

    # --- 4. Control penalty (very weak for exploration) ---
    torque = data.ctrl
    r_energy = -0.0001 * np.sum(torque * torque)

    # --- 5. Height penalty (only if falling) ---
    height = data.xpos[torso_id, 2]
    r_height = -10.0 if height < 0.10 else 0.0

    reward = (
        r_forward +
        0.3 * r_upright +
        r_torso_stable +
        r_energy +
        r_height
    )

    info = dict(
        forward=r_forward,
        upright=r_upright,
        ang=r_torso_stable,
        energy=r_energy,
        height=r_height,
    )

    return reward, info
