"""
Base Reward Function Interface

Provides abstract base class for reward functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseRewardFunction(ABC):
    """
    Abstract base class for reward functions.
    
    All custom reward functions should inherit from this class
    and implement the compute() method.
    """
    
    def __init__(self, reward_scale: float = 1.0):
        """
        Args:
            reward_scale: Scaling factor for the final reward
        """
        self.reward_scale = reward_scale
    
    @abstractmethod
    def compute(
        self, 
        env: Any, 
        action: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute the reward for the current state.
        
        Args:
            env: The LIBERO environment instance
            action: The action taken (optional, may not be used)
            
        Returns:
            float: The computed reward value
        """
        raise NotImplementedError
    
    def reset(self) -> None:
        """
        Reset any internal state of the reward function.
        Called when the environment resets.
        """
        pass
    
    def get_info(self) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.
        Useful for debugging and analysis.
        
        Returns:
            Dict containing individual reward components
        """
        return {}


class SparseRewardFunction(BaseRewardFunction):
    """
    Sparse reward function - only gives reward on task success.
    This is the default LIBERO behavior.
    """
    
    def compute(self, env: Any, action: Optional[np.ndarray] = None) -> float:
        reward = 0.0
        if env._check_success():
            reward = 1.0
        return reward * self.reward_scale



