"""Proportional-Derivative (PD) controller for joint-level control.

This module implements a PD controller that converts desired joint positions
to torques. The controller can be configured with per-joint or global gains
and optional torque limits.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PDConfig:
    """Configuration for PD controller.
    
    The controller computes: τ = kp * (q_des - q) + kd * (qd_des - qd)
    
    Attributes:
        kp: Proportional gain. Can be scalar (applied to all joints) or
            array of shape (n_joints,) for per-joint gains.
        kd: Derivative gain. Same format as kp.
        torque_limit: Optional maximum torque magnitude. If None, no limit.
            Can be scalar or per-joint array.
    """
    kp: float | np.ndarray
    kd: float | np.ndarray
    torque_limit: Optional[float | np.ndarray] = None


class PDController:
    """Proportional-Derivative controller for joint control.
    
    Computes control torques based on position and velocity errors.
    Supports per-joint or global gains and optional torque saturation.
    """
    def __init__(self, cfg: PDConfig):
        """Initialize PD controller.
        
        Args:
            cfg: PD controller configuration.
        """
        self.kp = np.array(cfg.kp, dtype=np.float32) if not np.isscalar(cfg.kp) else cfg.kp
        self.kd = np.array(cfg.kd, dtype=np.float32) if not np.isscalar(cfg.kd) else cfg.kd
        if cfg.torque_limit is None:
            self.torque_limit = None
        else:
            self.torque_limit = (
                np.array(cfg.torque_limit, dtype=np.float32)
                if not np.isscalar(cfg.torque_limit)
                else cfg.torque_limit
            )

    def compute(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_des: np.ndarray,
        qd_des: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute control torques.
        
        Computes: τ = kp * (q_des - q) + kd * (qd_des - qd)
        with optional torque limiting.
        
        Args:
            q: Current joint positions, shape (n_joints,).
            qd: Current joint velocities, shape (n_joints,).
            q_des: Desired joint positions, shape (n_joints,).
            qd_des: Optional desired joint velocities, shape (n_joints,).
                Defaults to zero if None.
                
        Returns:
            Control torques, shape (n_joints,).
            
        Raises:
            AssertionError: If input shapes don't match.
        """
        q = np.asarray(q, dtype=np.float32)
        qd = np.asarray(qd, dtype=np.float32)
        q_des = np.asarray(q_des, dtype=np.float32)

        assert q.shape == qd.shape == q_des.shape, (
            f"PDController expects q, qd, q_des same shape, got "
            f"{q.shape}, {qd.shape}, {q_des.shape}"
        )

        if qd_des is None:
            qd_des = np.zeros_like(q)
        else:
            qd_des = np.asarray(qd_des, dtype=np.float32)
            assert qd_des.shape == q.shape

        # Broadcast scalar gains if needed
        if np.isscalar(self.kp):
            kp = np.full_like(q, float(self.kp))
        else:
            kp = np.asarray(self.kp, dtype=np.float32)
            if kp.shape != q.shape:
                kp = np.broadcast_to(kp, q.shape)

        if np.isscalar(self.kd):
            kd = np.full_like(q, float(self.kd))
        else:
            kd = np.asarray(self.kd, dtype=np.float32)
            if kd.shape != q.shape:
                kd = np.broadcast_to(kd, q.shape)

        pos_err = q_des - q
        vel_err = qd_des - qd

        tau = kp * pos_err + kd * vel_err

        if self.torque_limit is not None:
            if np.isscalar(self.torque_limit):
                limit = np.full_like(tau, float(self.torque_limit))
            else:
                limit = np.asarray(self.torque_limit, dtype=np.float32)
                if limit.shape != tau.shape:
                    limit = np.broadcast_to(limit, tau.shape)
            tau = np.clip(tau, -limit, limit)

        return tau