"""Hand-crafted reference policy for bipedal locomotion.

This module implements a baseline controller using a hand-designed gait
pattern, inverse kinematics, and PD control. The policy generates periodic
foot trajectories and uses IK to convert them to joint angles, which are
then tracked using PD control.

This serves as a baseline for comparison with learned RL policies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from control.ik_2r import Planar2RLegConfig, Planar2RLegIK
from control.pd import PDConfig, PDController


@dataclass
class GaitParams:
    """Parameters for periodic gait pattern.
    
    Attributes:
        step_length: Maximum horizontal foot displacement from neutral [m].
        step_height: Maximum vertical foot lift during swing [m].
        cycle_duration: Duration of one full gait cycle (L->R->L) [s].
    """
    step_length: float
    step_height: float
    cycle_duration: float


@dataclass
class FootDelta:
    """Foot target relative to neutral position.
    
    Attributes:
        delta: 2D displacement (dx, dz) relative to foot base position.
        foot_angle: Desired foot pitch angle [rad].
    """
    delta: np.ndarray
    foot_angle: float

def gait_targets(t: float, params: GaitParams) -> Dict[str, FootDelta]:
    """Generate foot targets for symmetric 2-phase gait.
    
    Implements a periodic gait where legs alternate between swing and stance:
    - Phase 0.0-0.5: LEFT leg swings, RIGHT leg in stance
    - Phase 0.5-1.0: RIGHT leg swings, LEFT leg in stance
    
    The swing leg follows a sinusoidal trajectory for smooth motion.
    
    Args:
        t: Current time [s].
        params: Gait parameters.
        
    Returns:
        Dictionary with "left" and "right" FootDelta objects.
        Delta is relative to each foot's neutral/base position.
    """
    # Time → phase in [0, 1)
    phase = (t / params.cycle_duration) % 1.0

    # ----------------------------
    # Phase allocation
    # ----------------------------
    if phase < 0.5:
        # LEFT swings: use phase 0.0-0.5 mapped to 0-1 for the swing
        swing_phase = phase * 2.0  # 0 to 1
        omega = 2.0 * np.pi * swing_phase
        dx = params.step_length * np.sin(omega)
        dz = params.step_height * max(0.0, np.sin(omega))
        
        left_dx = +dx
        left_dz = +dz
        right_dx = -dx
        right_dz = 0.0     # stance leg stays on ground
    else:
        # RIGHT swings: use phase 0.5-1.0 mapped to 0-1 for the swing
        swing_phase = (phase - 0.5) * 2.0  # 0 to 1
        omega = 2.0 * np.pi * swing_phase
        dx = params.step_length * np.sin(omega)
        dz = params.step_height * max(0.0, np.sin(omega))
        
        left_dx = -dx
        left_dz = 0.0
        right_dx = +dx
        right_dz = +dz

    left_delta  = np.array([left_dx,  left_dz ], dtype=np.float32)
    right_delta = np.array([right_dx, right_dz], dtype=np.float32)

    return {
        "left":  FootDelta(delta=left_delta,  foot_angle=0.0),
        "right": FootDelta(delta=right_delta, foot_angle=0.0),
    }

# ------------------------------------------------------------
# Reference walker policy (hand-crafted gait + IK + PD)
# ------------------------------------------------------------

@dataclass
class LegJointIndices:
    """Joint indices for a single leg.
    
    These indices map to positions in the full joint/action vectors.
    Must be adapted to match the specific robot model's joint ordering.
    
    Attributes:
        hip: Index of hip joint.
        knee: Index of knee joint.
        ankle: Index of ankle joint.
    """
    hip: int
    knee: int
    ankle: int


@dataclass
class WalkerJointMap:
    """Mapping of joint indices for both legs.
    
    Attributes:
        left: Joint indices for left leg.
        right: Joint indices for right leg.
    """
    left: LegJointIndices
    right: LegJointIndices

class ReferenceWalkerPolicy(nn.Module):
    """Hand-crafted baseline controller for bipedal walking.
    
    This policy implements a deterministic control pipeline:
    1. Generate periodic gait pattern (foot targets)
    2. Solve inverse kinematics to get desired joint angles
    3. Use PD control to track joint angles (output torques)
    
    The interface matches ActorCritic.act() for compatibility, but
    returns zero log probabilities since this is deterministic.
    
    Attributes:
        env: Environment instance (for accessing dt, foot positions, etc.).
        gait_params: Gait pattern parameters.
        ik_left: Inverse kinematics solver for left leg.
        ik_right: Inverse kinematics solver for right leg.
        pd: PD controller for joint tracking.
        _time: Internal time counter for gait phase.
    """
    def __init__(
        self,
        env,  # MujocoEnv instance so we can get dt, act_dim, etc.
        gait_params: GaitParams,
        left_leg_geom: Planar2RLegConfig,
        right_leg_geom: Planar2RLegConfig,
        pd_config: PDConfig,
        desired_foot_angle: float = 0.0,
    ):
        super().__init__()

        self.env = env
        self.gait_params = gait_params
        self.desired_foot_angle = desired_foot_angle

        # Simulation timestep (approx): dt = timestep * frame_skip
        dt = env.model.opt.timestep * env.cfg.frame_skip
        self.dt = float(dt)
        self._time = 0.0

        # IK for each leg
        self.ik_left = Planar2RLegIK(left_leg_geom)
        self.ik_right = Planar2RLegIK(right_leg_geom)

        # PD controller: one PD over all controlled joints
        self.pd = PDController(pd_config)

        # Action dimension (number of actuators)
        self.act_dim = env.spec.act.shape[0]

        # get neutral foot positions from env
        if not env.foot_base_pos:
            print("[ReferenceWalkerPolicy] WARNING: env.foot_base_pos is empty; "
                  "using zeros for foot bases.")
            self.left_foot_base = np.zeros(2, dtype=np.float32)
            self.right_foot_base = np.zeros(2, dtype=np.float32)
        else:
            self.left_foot_base = env.foot_base_pos["left"].copy()
            self.right_foot_base = env.foot_base_pos["right"].copy()

    def compute_q_ref(self, t: float) -> np.ndarray:
        """
        Compute reference joint positions q_ref at time t
        (ignores the current state, purely kinematic).
        """
        # 1) Gait deltas
        targets = gait_targets(t, self.gait_params)
        left_tgt = targets["left"]
        right_tgt = targets["right"]

        left_ankle_target = self.left_foot_base + left_tgt.delta
        right_ankle_target = self.right_foot_base + right_tgt.delta

        # 2) IK per leg
        hip_L_des, knee_L_des, ankle_L_des = self.ik_left.solve(
            target_x=float(left_ankle_target[0]),
            target_z=float(left_ankle_target[1]),
            compute_ankle=True,
            desired_foot_angle=float(left_tgt.foot_angle),
        )
        hip_R_des, knee_R_des, ankle_R_des = self.ik_right.solve(
            target_x=float(right_ankle_target[0]),
            target_z=float(right_ankle_target[1]),
            compute_ankle=True,
            desired_foot_angle=float(right_tgt.foot_angle),
        )

        # TODO: adapt to varying joint order
        return [hip_L_des, knee_L_des, ankle_L_des, hip_R_des, knee_R_des, ankle_R_des]

    def reset(self):
        """Reset internal episode state (e.g. gait phase)."""
        self._time = 0.0

    def act(self, obs: dict, deterministic: bool = True):
        joint_pos = obs["joint_pos"]
        joint_vel = obs["joint_vel"]

        # Time update
        t = self._time
        self._time += self.dt

        ref_joint_pos = self.compute_q_ref(t)

        action = self.pd.compute(
            q=joint_pos,
            qd=joint_vel,
            q_des=ref_joint_pos,
            qd_des=None,
        )  # shape (n_ctrl,)

        dummy_logp = torch.zeros(1, dtype=torch.float32)  # not used

        return action, dummy_logp