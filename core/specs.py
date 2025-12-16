"""Environment specification definitions.

This module defines data structures for describing observation and action
spaces, enabling type checking and automatic policy/algorithm configuration.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class SpaceSpec:
    """Specification for an observation or action space.
    
    Attributes:
        shape: Tuple defining the shape of the space (e.g., (64,) for a
            64-dimensional vector, (84, 84, 3) for an image).
        dtype: NumPy dtype of the space elements (e.g., np.float32).
    """
    shape: Tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class EnvSpec:
    """Complete environment specification.
    
    Contains both observation and action space specifications, allowing
    algorithms and policies to automatically configure themselves based on
    the environment's requirements.
    
    Attributes:
        obs: Observation space specification.
        act: Action space specification.
    """
    obs: SpaceSpec
    act: SpaceSpec