"""Base environment interface for reinforcement learning.

This module defines the abstract base class for RL environments, providing
a standardized interface for reset, step, and cleanup operations. All
concrete environment implementations should inherit from the Env class.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from .specs import EnvSpec


@dataclass
class StepResult:
    """Result of an environment step.
    
    Attributes:
        obs: Observation array after taking the action.
        reward: Scalar reward signal for the transition.
        done: Boolean indicating episode termination.
        info: Dictionary of additional information (e.g., reward components).
        frame: Optional RGB frame for visualization (H, W, 3), typically None
            during training to save computation.
    """
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]
    frame: Optional[np.ndarray] = None


class Env(ABC):
    """Abstract base class for reinforcement learning environments.
    
    This class defines the standard RL environment interface. All environments
    must implement reset(), step(), and close() methods. The class also supports
    context manager protocol for automatic resource cleanup.
    
    Attributes:
        spec: Environment specification containing observation and action spaces.
    """
    spec: EnvSpec

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment to an initial state.
        
        Args:
            seed: Optional random seed for reproducible initial states.
            
        Returns:
            Initial observation array.
        """
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Execute one environment step.
        
        Args:
            action: Action array to apply to the environment.
            
        Returns:
            StepResult containing observation, reward, done flag, info dict,
            and optional frame.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources.
        
        Should release any allocated resources (e.g., renderers, file handles).
        """
        ...

    def __enter__(self) -> "Env":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit, ensures cleanup."""
        self.close()