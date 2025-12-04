import numpy as np

def reward(model, data, t, dt, action):
    """
    Reward for stable, symmetric biped walking.
    Works with your model's free-floating root joint ('<unnamed_0>')
    and torso body named 'biped'.
    """

    # -----------------------------------------
    # 1. Forward velocity reward (slow walking)
    # -----------------------------------------
    # Free root joint: data.qvel[0] = forward velocity
    forward_speed = data.qvel[0]
    target_speed = 0.8         # slower walking target
    speed_error = forward_speed - target_speed
    speed_reward = np.exp(-0.5 * (speed_error ** 2)) * 1.0

    # -----------------------------------------
    # 2. Upright reward using torso (body 'biped')
    # -----------------------------------------
    torso_id = model.body("biped").id
    torso_mat = data.xmat[torso_id].reshape(3, 3)

    # World z-axis alignment (how vertical the torso is)
    torso_up = torso_mat[2, 2]
    upright_reward = 0.5 * torso_up

    # -----------------------------------------
    # 3. Control effort penalty
    # -----------------------------------------
    ctrl_cost = 0.001 * np.sum(action ** 2)

    # -----------------------------------------
    # 4. Bounce penalty to discourage hopping
    # -----------------------------------------
    vertical_vel = data.qvel[2]    # upward/downward velocity
    bounce_cost = 0.05 * (vertical_vel ** 2)

    # -----------------------------------------
    # 5. Gait symmetry reward (alternate stepping)
    # -----------------------------------------
    lh_vel = data.qvel[model.joint("left_hip_joint").id]
    rh_vel = data.qvel[model.joint("right_hip_joint").id]

    gait_sym = (lh_vel - rh_vel) ** 2
    gait_sym_reward = 0.02 * gait_sym

    # -----------------------------------------
    # 6. Foot clearance reward
    # -----------------------------------------
    lf_z = data.xpos[model.body("left_foot").id][2]
    rf_z = data.xpos[model.body("right_foot").id][2]

    foot_height_nom = 0.03
    foot_alpha = 200.0

    lf_clear = np.exp(-foot_alpha * (lf_z - foot_height_nom) ** 2)
    rf_clear = np.exp(-foot_alpha * (rf_z - foot_height_nom) ** 2)
    clearance_reward = 0.01 * (lf_clear + rf_clear)

    # -----------------------------------------
    # 7. Alive bonus
    # -----------------------------------------
    alive = 0.2

    # -----------------------------------------
    # Total reward
    # -----------------------------------------
    total = (
        speed_reward
        + upright_reward
        + gait_sym_reward
        + clearance_reward
        + alive
        - ctrl_cost
        - bounce_cost
    )

    return float(total)
