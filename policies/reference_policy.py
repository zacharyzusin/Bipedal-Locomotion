# policies/reference_policy.py
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
    step_length: float       # max |dx| around neutral [m]
    step_height: float       # max upward dz [m]
    cycle_duration: float    # seconds per full L->R->L cycle
    ramp_up: float = 1.0     # seconds to fade in the gait


@dataclass
class FootDelta:
    delta: np.ndarray   # (dx, dz)
    foot_angle: float

def gait_targets(t: float, params: GaitParams) -> Dict[str, FootDelta]:
    """
    Symmetric 2-phase gait:
      - Phase 0.0–0.5:  LEFT swings, RIGHT stance.
      - Phase 0.5–1.0:  RIGHT swings, LEFT stance.
      - dz is always >= 0 (we only LIFT).
      - Starts with LEFT swinging.

    delta is relative to the foot's neutral/base position.
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

    # ----------------------------
    # Soft start of gait
    # ----------------------------
    ramp = min(t / params.ramp_up, 1.0)

    left_delta  = np.array([left_dx,  left_dz ], dtype=np.float32) * ramp
    right_delta = np.array([right_dx, right_dz], dtype=np.float32) * ramp

    return {
        "left":  FootDelta(delta=left_delta,  foot_angle=0.0),
        "right": FootDelta(delta=right_delta, foot_angle=0.0),
    }

# ------------------------------------------------------------
# Reference walker policy (hand-crafted gait + IK + PD)
# ------------------------------------------------------------

@dataclass
class LegJointIndices:
    """
    Indices of the leg joints in the full q/qdot/ctrl/action vectors.
    These are robot-specific and MUST be adapted to your biped.xml.
    """
    hip: int
    knee: int
    ankle: int


@dataclass
class WalkerJointMap:
    left: LegJointIndices
    right: LegJointIndices


class ReferenceWalkerPolicy(nn.Module):
    """
    Hand-crafted baseline controller:

      gait (foot targets) -> 2R leg IK -> desired joint angles -> PD -> torques.

    The interface matches ActorCritic.act: it returns (action_tensor, logp_tensor),
    but logp is just zeros since this is not stochastic.
    """
    def __init__(
        self,
        env,  # MujocoEnv instance so we can get dt, act_dim, etc.
        gait_params: GaitParams,
        joint_map: WalkerJointMap,
        left_leg_geom: Planar2RLegConfig,
        right_leg_geom: Planar2RLegConfig,
        pd_config: PDConfig,
        desired_foot_angle: float = 0.0,
    ):
        super().__init__()

        self.env = env
        self.gait_params = gait_params
        self.joint_map = joint_map
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

        # ------------------------------------------------------------------
        # Joint index bookkeeping
        # ------------------------------------------------------------------
        # Joint indices in q (qpos) space that we want to control
        jmL = self.joint_map.left
        jmR = self.joint_map.right

        # Vector of q indices for all controlled joints (left + right leg)
        self._q_indices = np.array(
            [
                jmL.hip, jmL.knee, jmL.ankle,
                jmR.hip, jmR.knee, jmR.ankle,
            ],
            dtype=int,
        )
        
        # For MuJoCo models with a floating base, nq != nv.
        # Typically: nq = 7 + n_hinge, nv = 6 + n_hinge → offset = nq - nv = 1.
        # For non-base joints: qd_index = q_index - offset
        model = self.env.model
        nq = model.nq
        nv = model.nv
        offset = nq - nv

        self._qd_indices = self._q_indices - offset

        # ------------------------------------------------------------------
        # Build mapping from q_idx -> actuator index
        # ------------------------------------------------------------------
        # joint_id -> q_idx (from MuJoCo)
        jnt_to_q: dict[int, int] = {}
        for j in range(model.njnt):
            q_idx = int(model.jnt_qposadr[j])
            if q_idx >= 0:
                jnt_to_q[j] = q_idx

        # q_idx -> actuator index: uses actuator_trnid[a, 0] = joint_id
        self._act_for_q: dict[int, int] = {}
        for a in range(model.nu):
            j_id = int(model.actuator_trnid[a, 0])
            if j_id in jnt_to_q:
                q_idx = jnt_to_q[j_id]
                self._act_for_q[q_idx] = a

        # get neutral foot positions from env
        if not env.foot_base_pos:
            print("[ReferenceWalkerPolicy] WARNING: env.foot_base_pos is empty; "
                  "using zeros for foot bases.")
            self.left_foot_base = np.zeros(2, dtype=np.float32)
            self.right_foot_base = np.zeros(2, dtype=np.float32)
        else:
            self.left_foot_base = env.foot_base_pos["left"].copy()
            self.right_foot_base = env.foot_base_pos["right"].copy()

    def reset(self):
        """Reset internal episode state (e.g. gait phase)."""
        self._time = 0.0

    def forward(self, *args, **kwargs):
        # We don't use forward; we use act() like ActorCritic.
        raise NotImplementedError

    def act(self, obs: torch.Tensor, deterministic: bool = True):
        obs_np = obs.detach().cpu().numpy().squeeze(0)
        nq = self.env.model.nq
        nv = self.env.model.nv

        q = obs_np[:nq]
        qd = obs_np[nq:nq + nv]

        t = self._time
        self._time += self.dt

        # 1) Gait deltas (dx, dz) around neutral base pose
        targets = gait_targets(t, self.gait_params)
        left_tgt = targets["left"]
        right_tgt = targets["right"]

        # Combine base + delta: full ankle targets in hip frame
        left_ankle_target = self.left_foot_base + left_tgt.delta
        right_ankle_target = self.right_foot_base + right_tgt.delta

        # 2) IK per leg with per-foot angle
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

        # 3) Desired joint positions in full q vector
        q_des = q.copy()
        jmL = self.joint_map.left
        jmR = self.joint_map.right

        q_des[jmL.hip] = hip_L_des
        q_des[jmL.knee] = knee_L_des
        q_des[jmL.ankle] = ankle_L_des

        q_des[jmR.hip] = hip_R_des
        q_des[jmR.knee] = knee_R_des
        q_des[jmR.ankle] = ankle_R_des

        # 4) Extract controlled joints only
        q_ctrl = q[self._q_indices]           # shape (n_ctrl,)
        qd_ctrl = qd[self._qd_indices]        # shape (n_ctrl,)
        q_des_ctrl = q_des[self._q_indices]   # shape (n_ctrl,)

         # 5) PD control in joint space
        tau_ctrl = self.pd.compute(
            q=q_ctrl,
            qd=qd_ctrl,
            q_des=q_des_ctrl,
            qd_des=None,
        )  # shape (n_ctrl,)

        # 6) Scatter torques into full action vector
        action = np.zeros(self.act_dim, dtype=np.float32)

        for q_idx, tau in zip(self._q_indices, tau_ctrl):
            act_idx = self._act_for_q.get(int(q_idx), None)
            if act_idx is not None and 0 <= act_idx < self.act_dim:
                action[act_idx] = tau

        action_t = torch.as_tensor(
            action, dtype=obs.dtype, device=obs.device
        ).unsqueeze(0)
        logp_dummy = torch.zeros_like(action_t[..., 0:1])

        return action_t, logp_dummy