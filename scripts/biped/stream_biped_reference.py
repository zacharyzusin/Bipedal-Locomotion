# scripts/stream_walker_reference.py
from __future__ import annotations

import uvicorn
import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig
from streaming.mjpeg_server import create_app

from policies.reference_policy import (
    ReferenceWalkerPolicy,
    GaitParams,
    WalkerJointMap,
    LegJointIndices,
)
from control.ik_2r import Planar2RLegConfig
from control.pd import PDConfig

from tasks.biped.reward import reward as reward
from tasks.biped.done import done as done

def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/biped/biped.xml",
        episode_length=5_000,
        frame_skip=5,
        ctrl_scale=0.1,
        reset_noise_scale=0.01,
        render=True,
        reward_fn=reward,
        done_fn=done,
        width=640,
        height=480,
        hip_site="base",
        left_foot_site="left_foot_ik",
        right_foot_site="right_foot_ik",
    )
    return MujocoEnv(cfg)


def make_policy(env: MujocoEnv):
    # ---- Gait parameters ----
    gait_params = GaitParams(
        step_length=0.02,      # tune to be realistic for your tiny biped
        step_height=0.01,      # front foot height
        cycle_duration=1.25,    # 1 second per full step R->L
    )

    leg_geom_left = Planar2RLegConfig(
        L1=0.05,   # thigh length [m]
        L2=0.058,   # shank length [m]
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0,
        ankle_offset=-np.pi/2,

    )
    leg_geom_right = Planar2RLegConfig(
        L1=0.05,
        L2=0.058,
        knee_sign=1.0,
        hip_offset=np.pi/2,
        knee_offset=0,
        ankle_offset=-np.pi/2,
    )

    pd_cfg = PDConfig(
        kp=50.0,
        kd=1.0,
        torque_limit=None,
    )

    joint_map = WalkerJointMap(
        left=LegJointIndices(
            hip=7,
            knee=8,
            ankle=9,
        ),
        right=LegJointIndices(
            hip=10,
            knee=11,
            ankle=12,
        ),
    )

    return ReferenceWalkerPolicy(
        env=env,
        gait_params=gait_params,
        joint_map=joint_map,
        left_leg_geom=leg_geom_left,
        right_leg_geom=leg_geom_right,
        pd_config=pd_cfg,
        desired_foot_angle=0.0,  # parallel to ground
    )


def main():
    device = "cpu"

    app = create_app(
        env_factory=make_env,
        policy_factory=make_policy,
        checkpoint_path=None,
        device=device,
    )

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()