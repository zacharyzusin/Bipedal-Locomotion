# scripts/biped/stream_biped_ik.py

import numpy as np
import uvicorn
from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from core.biped_ik_env import BipedIKEnv, FootActionConfig
from core.biped_obs_wrapper import BipedSensorWrapper
from control.ik_2r import Planar2RLegConfig
from control.pd import PDConfig
from policies.actor_critic import ActorCritic
from policies.reference_policy import WalkerJointMap, LegJointIndices

from tasks.biped.reward_inplace import reward
from tasks.biped.done import done

from streaming.mjpeg_server import create_app


# ------------------------------------------------------------
# Create the same IK env as training, but with render=True
# ------------------------------------------------------------
def make_env_render():
    # Low-level Mujoco env with render=True
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=300,
        frame_skip=5,
        ctrl_scale=1.0,
        reset_noise_scale=0.01,
        reward_fn=reward,
        done_fn=done,
        render=True,             # <--- FORCE OFFSCREEN RENDERING
        width=640,
        height=480,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    base = MujocoEnv(cfg)
    low_env = BipedSensorWrapper(base)

    # Joint indexing must match XML exactly
    joint_map = WalkerJointMap(
        left=LegJointIndices(hip=7, knee=8, ankle=9),
        right=LegJointIndices(hip=10, knee=11, ankle=12),
    )

    # IK geometry
    leg_l = Planar2RLegConfig(L1=0.05, L2=0.058,
                              knee_sign=1.0,
                              hip_offset=np.pi/2,
                              knee_offset=0.0,
                              ankle_offset=-np.pi/2)

    leg_r = Planar2RLegConfig(L1=0.05, L2=0.058,
                              knee_sign=1.0,
                              hip_offset=np.pi/2,
                              knee_offset=0.0,
                              ankle_offset=-np.pi/2)

    pd_cfg = PDConfig(
        kp=600.0,
        kd=40.0,
        torque_limit=150.0,
    )
    foot_cfg = FootActionConfig(
        max_dx=0.05,
        max_dz=0.04,
    )


    """pd_cfg = PDConfig(kp=50.0, kd=1.0, torque_limit=None)
    foot_cfg = FootActionConfig(max_dx=0.02, max_dz=0.015)"""

    env = BipedIKEnv(low_env, joint_map, leg_l, leg_r, pd_cfg, foot_cfg)
    return env


# ------------------------------------------------------------
# Policy
# ------------------------------------------------------------
def make_policy(env):
    return ActorCritic(env.spec, hidden_sizes=(128, 128))


# ------------------------------------------------------------
# Main entry: load policy + start web server
# ------------------------------------------------------------
def main():
    checkpoint_path = "checkpoints/biped_ik.pt"
    device = "cpu"

    app = create_app(
        env_factory=make_env_render,   # <--- USE OUR RENDER ENV
        policy_factory=make_policy,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
