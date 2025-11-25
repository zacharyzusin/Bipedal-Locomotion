# core/specs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class SpaceSpec:
    shape: Tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class EnvSpec:
    obs: SpaceSpec
    act: SpaceSpec