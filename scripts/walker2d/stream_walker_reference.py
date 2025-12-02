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

from tasks.walker2d.reward import reward
from tasks.walker2d.done import done

def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path="assets/walker2d/walker2d.xml",
        episode_length=5_000,
        frame_skip=5,
        pd_cfg=PDConfig(kp=5.0, kd=1.0, torque_limit=1.0),
        reset_noise_scale=0.01,
        render=True,
        reward_fn=reward,
        done_fn=done,
        width=640,
        height=480,
    )
    return MujocoEnv(cfg)


def make_policy(env: MujocoEnv):
    # ---- Gait parameters ----
    gait_params = GaitParams(
        stride_length=0.3,
        stride_height=0.2,
        cycle_duration=2.0,
        stance_fraction=0.6,
    )

    leg_geom_left = Planar2RLegConfig(
        L1=0.45,   # thigh length [m]
        L2=0.5,   # shank length [m]
        knee_sign=1.0,
        ankle_offset=0.0,
    )
    leg_geom_right = Planar2RLegConfig(
        L1=0.45,
        L2=0.5,
        knee_sign=1.0,
        ankle_offset=0.0,
    )

    pd_cfg = PDConfig(
        kp=5.0,
        kd=1.0,
        torque_limit=0.1,
    )

    joint_map = WalkerJointMap(
        left=LegJointIndices(
            hip=6,
            knee=7,
            ankle=8,
        ),
        right=LegJointIndices(
            hip=3,
            knee=4,
            ankle=5,
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