# core/base_env.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from .specs import EnvSpec


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]
    # Optional RGB frame; typically None during training, used in eval/streaming.
    frame: Optional[np.ndarray] = None


class Env(ABC):
    spec: EnvSpec

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> "Env":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()