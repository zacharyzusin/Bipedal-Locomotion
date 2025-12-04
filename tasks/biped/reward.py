import numpy as np

def reward(model, data, t, dt, action):
    """
    Stable symmetric walking with flat feet, 
    without collapsing into stand-still local optimum.
    """

    # ------------------------------------------------------------
    # 1. Forward velocity reward (dominant term)
    # ------------------------------------------------------------
    forward_speed = max(0.0, data.qvel[0])
    target_speed = 0.6

    speed_reward = forward_speed / target_speed
    speed_reward = np.clip(speed_reward, 0.0, 1.5)

    # Slow penalty: large negative gradient when not moving
    slow_penalty = -0.5 * np.exp(-5 * forward_speed)

    # ------------------------------------------------------------
    # 2. Strictly limited upright reward
    # ------------------------------------------------------------
    torso_id = model.body("biped").id
    R = data.xmat[torso_id].reshape(3, 3)
    upright_raw = R[2, 2]

    upright_reward = 0.2 * np.clip(upright_raw, -1.0, 1.0)

    # ------------------------------------------------------------
    # 3. Weak gait symmetry cue
    # ------------------------------------------------------------
    left_knee_q = data.qpos[8]
    right_knee_q = data.qpos[11]

    gait_sym = np.sin(left_knee_q - right_knee_q)
    gait_sym_reward = 0.05 * gait_sym

    # ------------------------------------------------------------
    # 4. Very small foot clearance reward
    # ------------------------------------------------------------
    left_foot = model.body("left_foot").id
    right_foot = model.body("right_foot").id

    left_z = data.xpos[left_foot][2]
    right_z = data.xpos[right_foot][2]

    clearance_reward = 0.02 * (np.tanh(10 * left_z) + np.tanh(10 * right_z))

    # ------------------------------------------------------------
    # 5. Small foot flatness reward
    # ------------------------------------------------------------
    def flatness(body_name):
        bid = model.body(body_name).id
        Rf = data.xmat[bid].reshape(3, 3)
        up = Rf[2, 2]
        z = data.xpos[bid][2]
        stance = np.exp(-600 * (z - 0.02) ** 2)
        return up * stance

    foot_orientation_reward = 0.05 * (
        flatness("left_foot") + flatness("right_foot")
    )

    # ------------------------------------------------------------
    # 6. Alive bonus
    # ------------------------------------------------------------
    alive = 0.5

    # ------------------------------------------------------------
    # 7. Control + bounce penalties
    # ------------------------------------------------------------
    ctrl_cost = 0.005 * np.sum(action ** 2)
    bounce_cost = 0.1 * (data.qvel[2] ** 2)

    # ------------------------------------------------------------
    # Total
    # ------------------------------------------------------------
    total = (
        speed_reward +
        slow_penalty +
        upright_reward +
        gait_sym_reward +
        clearance_reward +
        foot_orientation_reward +
        alive -
        ctrl_cost -
        bounce_cost
    )

    components = {
        "speed": float(speed_reward),
        "slow": float(slow_penalty),
        "upright": float(upright_reward),
        "gait_sym": float(gait_sym_reward),
        "clear": float(clearance_reward),
        "flat": float(foot_orientation_reward),
        "alive": float(alive),
        "ctrl": float(-ctrl_cost),
        "bounce": float(-bounce_cost),
    }

    return float(total), components
