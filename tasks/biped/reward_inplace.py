import numpy as np

# ----------------------------------------------------------
# GLOBAL GAIT PHASE
# ----------------------------------------------------------
phase = 0.0
PHASE_SPEED = 1.6   # steps/sec


def reward(model, data):
    """
    Improved walking reward:
    - Eliminates static leaning local optimum.
    - Forces alternating left/right stepping.
    - Links forward velocity reward to actual stepping.
    - Penalizes feet being too close for too long.
    - Encourages swing leg to pass ahead of stance leg.
    """

    global phase

    # -----------------------------
    # Phase update (MuJoCo 2.x)
    # -----------------------------
    dt = model.opt.timestep * 5.0   # your env uses frame_skip=5
    phase = (phase + dt * 2 * np.pi * PHASE_SPEED) % (2 * np.pi)

    # -----------------------------
    # Body and foot IDs
    # -----------------------------
    hips_id  = model.body("hips").id
    left_id  = model.body("left_foot").id
    right_id = model.body("right_foot").id

    # -----------------------------
    # State extraction
    # -----------------------------
    hip_pos = data.xpos[hips_id]
    hip_x, hip_z = hip_pos[0], hip_pos[2]

    left_pos  = data.xpos[left_id]
    right_pos = data.xpos[right_id]

    left_x, left_z   = left_pos[0], left_pos[2]
    right_x, right_z = right_pos[0], right_pos[2]

    # Spatial velocity: [wx wy wz vx vy vz]
    hip_cvel = data.cvel[hips_id]
    vx, vy, vz = hip_cvel[3:6]

    # Foot velocities for swing detection
    left_vx  = abs(data.cvel[left_id][3])
    right_vx = abs(data.cvel[right_id][3])

    # Orientation (pitch)
    zaxis = data.xmat[hips_id, 6:9]
    pitch = np.arctan2(zaxis[0], zaxis[2])
    pitch_rate = data.qvel[3]

    # -----------------------------------------------------
    # FALL CHECK — terminate reward early
    # -----------------------------------------------------
    neutral_height = 0.171

    if hip_z < neutral_height - 0.06:
        return -5.0
    if abs(pitch) > 1.0:
        return -5.0

    # -----------------------------------------------------
    # BASIC POSTURE & STABILITY
    # -----------------------------------------------------
    height_term  = np.exp(-40.0 * (hip_z - neutral_height)**2)
    upright_term = np.exp(-10.0 * pitch**2)
    stable_term  = np.exp(-4.0 * pitch_rate**2)

    # -----------------------------------------------------
    # COM BALANCE
    # -----------------------------------------------------
    com_x = data.subtree_com[hips_id][0]
    stance_mid = 0.5 * (left_x + right_x)
    com_error = com_x - stance_mid
    com_balance = np.exp(-80.0 * com_error**2)

    # -----------------------------------------------------
    # FOOT CONTACT (MuJoCo contact force)
    # -----------------------------------------------------
    left_contact  = data.cfrc_ext[left_id][2]  < -5.0
    right_contact = data.cfrc_ext[right_id][2] < -5.0

    # -----------------------------------------------------
    # STEP REWARD — encourages alternating stepping
    # -----------------------------------------------------
    swing_left  = left_vx  * np.exp(-3 * right_vx)
    swing_right = right_vx * np.exp(-3 * left_vx)
    step_reward = swing_left + swing_right

    # -----------------------------------------------------
    # FORCE THE ROBOT TO STEP!
    #
    # Penalize feet staying too close = static stance
    # -----------------------------------------------------
    foot_separation = abs(left_x - right_x)
    no_step_penalty = -0.5 if foot_separation < 0.02 else 0.0

    # -----------------------------------------------------
    # GAIT PHASE CONSISTENCY
    # -----------------------------------------------------
    left_phase  = (phase < np.pi)
    right_phase = (phase >= np.pi)

    if left_phase:
        # left leg should swing forward
        phase_penalty = -0.3 if left_x <= right_x + 0.005 else 0.0
        phase_reward = left_vx * np.exp(-3 * right_vx)
    else:
        # right leg should swing forward
        phase_penalty = -0.3 if right_x <= left_x + 0.005 else 0.0
        phase_reward = right_vx * np.exp(-3 * left_vx)

    phase_reward *= 0.7

    # -----------------------------------------------------
    # FOOT FORWARD PROGRESSION — swing must pass stance
    # -----------------------------------------------------
    if left_z > right_z:  # left is swing
        swing_x = left_x
        stance_x = right_x
    else:
        swing_x = right_x
        stance_x = left_x

    foot_forward_progress = max(0.0, swing_x - stance_x)
    foot_forward_reward = 3.0 * foot_forward_progress * abs(vx)

    # -----------------------------------------------------
    # STEP COMPLETION — COM must shift over stance
    # -----------------------------------------------------
    stance_error = com_x - stance_mid
    step_completion_reward = max(0.0, stance_error) * abs(vx)
    step_completion_reward *= 2.0

    # -----------------------------------------------------
    # FORWARD SPEED — ONLY IF STEPPING
    # -----------------------------------------------------
    movement_factor = max(0.0, vx - 0.03)
    if step_reward > 0.01:
        movement_reward = movement_factor
    else:
        movement_reward = 0.0  # cannot "lean" for reward

    target_speed = 0.30
    speed_term = np.exp(-6.0 * (vx - target_speed)**2)

    # -----------------------------------------------------
    # ENERGY
    # -----------------------------------------------------
    ctrl_cost = 0.0003 * np.sum(data.ctrl**2)

    # -----------------------------------------------------
    # TOTAL REWARD
    # -----------------------------------------------------
    total = (
        3.0 * movement_reward +
        3.0 * speed_term +
        1.2 * step_completion_reward +
        1.5 * foot_forward_reward +
        movement_reward * (
            1.0 * height_term +
            1.0 * upright_term +
            0.7 * stable_term +
            1.0 * com_balance +
            0.8 * step_reward
        )
        + phase_reward
        + no_step_penalty
        + phase_penalty
        - 1.2 * (vy * vy)
        - 0.4 * abs(vz)
        - ctrl_cost
        + 0.02
    )

    return float(total)
