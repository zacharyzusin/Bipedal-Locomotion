# core/biped_ik_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import mujoco

from core.base_env import Env, StepResult
from core.specs import EnvSpec, SpaceSpec

from control.ik_2r import Planar2RLegConfig, Planar2RLegIK
from control.pd import PDConfig, PDController
from policies.reference_policy import LegJointIndices, WalkerJointMap


@dataclass
class FootActionConfig:
    """
    High-level action parameterization for the IK+PD env.

    Actions are interpreted as:
      [dx_L, dz_L, dx_R, dz_R] in meters, relative to neutral base positions.
    """
    max_dx: float = 0.06   # +/- 6cm forward/back
    max_dz: float = 0.04  # +/- 4cm up/down around neutral


class BipedIKEnv(Env):
    """
    High-level biped env:

    - Observation: inherited from base_env (e.g., BipedSensorWrapper).
    - Action: 4D foot deltas [dx_L, dz_L, dx_R, dz_R] in hip frame.

    Internally:
    - Uses IK+PD to convert foot deltas into joint torques for hip/knee/ankle.
    """

    def __init__(
        self,
        base_env: Env,                    # e.g. BipedSensorWrapper(MujocoEnv(...))
        joint_map: WalkerJointMap,
        left_leg_geom: Planar2RLegConfig,
        right_leg_geom: Planar2RLegConfig,
        pd_cfg: PDConfig,
        foot_cfg: FootActionConfig = FootActionConfig(),
        desired_foot_angle: float = 0.0,
    ):
        self.base_env = base_env
        self.model: mujoco.MjModel = base_env.model
        self.data: mujoco.MjData = base_env.data

        self.joint_map = joint_map
        self.desired_foot_angle = desired_foot_angle
        self.foot_cfg = foot_cfg

        # IK for each leg
        self.ik_left = Planar2RLegIK(left_leg_geom)
        self.ik_right = Planar2RLegIK(right_leg_geom)

        # PD controller for all 6 controlled joints
        self.pd = PDController(pd_cfg)

        # ----------------------------------------------------
        # Joint & actuator mapping (same logic as ReferenceWalkerPolicy)
        # ----------------------------------------------------
        jmL = self.joint_map.left
        jmR = self.joint_map.right

        # q indices for the controlled joints
        self._q_indices = np.array(
            [jmL.hip, jmL.knee, jmL.ankle,
             jmR.hip, jmR.knee, jmR.ankle],
            dtype=int,
        )

        # For floating base: nq != nv, typically nq = 7 + n_hinge, nv = 6 + n_hinge
        model = self.model
        nq = model.nq
        nv = model.nv
        offset = nq - nv  # qd index = q index - offset for hinge joints

        self._qd_indices = self._q_indices - offset

        # Map q_idx -> actuator index
        jnt_to_q: Dict[int, int] = {}
        for j in range(model.njnt):
            q_idx = int(model.jnt_qposadr[j])
            if q_idx >= 0:
                jnt_to_q[j] = q_idx

        self._act_for_q: Dict[int, int] = {}
        for a in range(model.nu):
            j_id = int(model.actuator_trnid[a, 0])
            if j_id in jnt_to_q:
                q_idx = jnt_to_q[j_id]
                self._act_for_q[q_idx] = a

        # Get neutral foot positions from base_env
        if not getattr(self.base_env, "foot_base_pos", None):
            print(
                "[BipedIKEnv] WARNING: base_env.foot_base_pos is empty; "
                "using zeros for foot bases."
            )
            self.left_foot_base = np.zeros(2, dtype=np.float32)
            self.right_foot_base = np.zeros(2, dtype=np.float32)
        else:
            self.left_foot_base = self.base_env.foot_base_pos["left"].copy()
            self.right_foot_base = self.base_env.foot_base_pos["right"].copy()

        # Spec: obs same as base, actions are 4D foot deltas
        obs_spec = self.base_env.spec.obs
        self.spec = EnvSpec(
            obs=obs_spec,
            act=SpaceSpec(shape=(4,), dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Env interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = self.base_env.reset(seed=seed)
        return obs

    def step(self, action: np.ndarray) -> StepResult:
        # High-level action -> numpy
        a = np.asarray(action, dtype=np.float32).reshape(4,)

        # Clip to allowed range
        max_dx = self.foot_cfg.max_dx
        max_dz = self.foot_cfg.max_dz

        dx_L = np.clip(a[0], -max_dx, max_dx)
        dz_L = np.clip(a[1], -max_dz, max_dz)
        dx_R = np.clip(a[2], -max_dx, max_dx)
        dz_R = np.clip(a[3], -max_dz, max_dz)

        # Current joint state
        q = np.asarray(self.data.qpos, dtype=np.float32).copy()
        qd = np.asarray(self.data.qvel, dtype=np.float32).copy()

        # Build ankle targets in hip frame: base + deltas
        left_tgt = self.left_foot_base + np.array([dx_L, dz_L], dtype=np.float32)
        right_tgt = self.right_foot_base + np.array([dx_R, dz_R], dtype=np.float32)

        # IK per leg
        hip_L_des, knee_L_des, ankle_L_des = self.ik_left.solve(
            target_x=float(left_tgt[0]),
            target_z=float(left_tgt[1]),
            compute_ankle=True,
            desired_foot_angle=self.desired_foot_angle,
        )
        hip_R_des, knee_R_des, ankle_R_des = self.ik_right.solve(
            target_x=float(right_tgt[0]),
            target_z=float(right_tgt[1]),
            compute_ankle=True,
            desired_foot_angle=self.desired_foot_angle,
        )

        # Desired q for the controlled joints
        jmL = self.joint_map.left
        jmR = self.joint_map.right

        q_des = q.copy()
        q_des[jmL.hip] = hip_L_des
        q_des[jmL.knee] = knee_L_des
        q_des[jmL.ankle] = ankle_L_des

        q_des[jmR.hip] = hip_R_des
        q_des[jmR.knee] = knee_R_des
        q_des[jmR.ankle] = ankle_R_des

        # Extract controlled joints
        q_ctrl = q[self._q_indices]
        qd_ctrl = qd[self._qd_indices]
        q_des_ctrl = q_des[self._q_indices]

        # PD torques
        tau_ctrl = self.pd.compute(
            q=q_ctrl,
            qd=qd_ctrl,
            q_des=q_des_ctrl,
            qd_des=None,
        )

        TAU_SCALE = 40.0   # try anywhere 10–50
        tau_ctrl *= TAU_SCALE


        # Scatter into full actuator vector
        act_dim = self.model.nu
        torques = np.zeros(act_dim, dtype=np.float32)
        for q_idx, tau in zip(self._q_indices, tau_ctrl):
            act_idx = self._act_for_q.get(int(q_idx), None)
            if act_idx is not None and 0 <= act_idx < act_dim:
                torques[act_idx] = tau

        # Step underlying env with torques
        step_res = self.base_env.step(torques)

        # Just forward obs/reward/done/frame
        return StepResult(
            obs=step_res.obs,
            reward=step_res.reward,
            done=step_res.done,
            info=step_res.info,
            frame=step_res.frame,
        )

    def close(self) -> None:
        self.base_env.close()

    # Delegate attributes like .model, .data, .set_camera, etc.
    def __getattr__(self, name: str):
        return getattr(self.base_env, name)
