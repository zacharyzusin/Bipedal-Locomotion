# policies/reference_policy.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GaitParams:
    stride_length: float      # [m] front-back distance covered by each foot per step
    stride_height: float      # [m] max height of the foot during swing
    cycle_duration: float     # [s] time for a full gait cycle
    stance_fraction: float = 0.6  # fraction of cycle in stance, >0.5 for double support


def _foot_trajectory_phase(phase: float, params: GaitParams) -> np.ndarray:
    """
    Compute (x, z) foot target in body frame for a single leg, given phase in [0, 1).

    Convention:
      - stance: phase ∈ [0, stance_fraction)
      - swing: phase ∈ [stance_fraction, 1)
      - x: forward (+) / backward (-) relative to body
      - z: height above ground (0 = on ground)
    """
    phase = phase % 1.0
    L = params.stride_length
    H = params.stride_height
    sf = params.stance_fraction

    if phase < sf:
        # ---- STANCE PHASE ----
        # normalized stance phase in [0,1)
        s = phase / sf
        # x: +L/2 -> -L/2
        x = L * (0.5 - s)    # = +L/2 at s=0, -L/2 at s=1
        z = 0.0
    else:
        # ---- SWING PHASE ----
        # normalized swing phase in [0,1)
        s = (phase - sf) / (1.0 - sf)
        # x: -L/2 -> +L/2
        x = L * (s - 0.5)    # = -L/2 at s=0, +L/2 at s=1
        # z: smooth bump, 0 at ends, H in the middle
        z = H * np.sin(np.pi * s)

    return np.array([x, z], dtype=np.float32)


def foot_trajectory(
    t: float,
    params: GaitParams,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """
    Time-based wrapper: given current time t, returns (x,z) for one foot.

    phase_offset lets you shift legs relative to each other, e.g.:
      left:  phase_offset = 0.0
      right: phase_offset = 0.5
    """
    phase = ((t / params.cycle_duration) + phase_offset) % 1.0
    return _foot_trajectory_phase(phase, params)


def gait_targets(
    t: float,
    params: GaitParams,
) -> dict[str, np.ndarray]:
    """
    Convenience function: returns targets for left and right feet at time t.

    Returns:
      {
        "left":  np.array([x_left,  z_left],  dtype=float32),
        "right": np.array([x_right, z_right], dtype=float32),
      }
    """
    left = foot_trajectory(t, params, phase_offset=0.0)
    right = foot_trajectory(t, params, phase_offset=0.5)  # half-cycle offset

    return {
        "left": left,
        "right": right,
    }
