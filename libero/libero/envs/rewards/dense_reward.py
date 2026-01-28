"""
Dense Reward Implementation for LIBERO

This module provides dense reward computation that enables
reward_shaping=True to work in LIBERO environments.

Dense reward components:
1. Reaching reward: Distance from gripper to object of interest
2. Grasping reward: Bonus when object is grasped
3. Task-specific progress reward: Based on predicate satisfaction
"""

import numpy as np
from typing import Any, Optional, Dict, Union, Tuple

from .base_reward import BaseRewardFunction
from .reward_utils import (
    compute_reaching_reward,
    compute_grasping_reward,
    compute_action_penalty,
    compute_gripper_distance_to_objects,
)


class DenseRewardFunction(BaseRewardFunction):
    """
    Dense reward function for LIBERO environments.
    
    Combines multiple reward components:
    - Reaching reward (distance-based)
    - Grasping reward (contact-based)
    - Success reward (sparse, on completion)
    """
    
    def __init__(
        self,
        reward_scale: float = 1.0,
        reaching_scale: float = 0.5,
        grasping_reward: float = 0.25,
        success_reward: float = 2.0,
        action_penalty_scale: float = 0.0,
        normalize_reward: bool = True,
    ):
        """
        Args:
            reward_scale: Overall scaling factor
            reaching_scale: Scale for reaching reward component
            grasping_reward: Fixed reward for grasping
            success_reward: Reward for task completion
            action_penalty_scale: Penalty for large actions (0 to disable)
            normalize_reward: If True, normalize final reward
        """
        super().__init__(reward_scale)
        self.reaching_scale = reaching_scale
        self.grasping_reward = grasping_reward
        self.success_reward = success_reward
        self.action_penalty_scale = action_penalty_scale
        self.normalize_reward = normalize_reward
        
        # For normalization
        self._max_reward = success_reward + reaching_scale + grasping_reward
        
        # Store last computed components for debugging
        self._last_components: Dict[str, float] = {}
    
    def compute(
        self, 
        env: Any, 
        action: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute dense reward for the current state.
        
        Args:
            env: LIBERO environment instance
            action: The action taken
            
        Returns:
            float: Dense reward value
        """
        reward = 0.0
        components = {}
        
        # 1. Check for task success (sparse component)
        if env._check_success():
            components['success'] = self.success_reward
            reward += self.success_reward
        else:
            components['success'] = 0.0
            
            # 2. Reaching reward (only if not yet successful)
            reaching = compute_reaching_reward(
                env,
                target_object_name=None,  # Uses first obj of interest
                scale=self.reaching_scale,
                tanh_scale=10.0,
            )
            components['reaching'] = reaching
            reward += reaching
            
            # 3. Grasping reward
            grasping = compute_grasping_reward(
                env,
                target_object_name=None,
                reward_value=self.grasping_reward,
            )
            components['grasping'] = grasping
            reward += grasping
        
        # 4. Action penalty (optional)
        if self.action_penalty_scale > 0 and action is not None:
            penalty = compute_action_penalty(action, self.action_penalty_scale)
            components['action_penalty'] = penalty
            reward += penalty
        else:
            components['action_penalty'] = 0.0
        
        # Normalize if requested
        if self.normalize_reward and self._max_reward > 0:
            reward = reward / self._max_reward
        
        # Apply overall scale
        reward *= self.reward_scale
        
        # Store for debugging
        self._last_components = components
        self._last_components['total'] = reward
        
        return reward
    
    def get_info(self) -> Dict[str, float]:
        """Get the last computed reward components."""
        return self._last_components.copy()


# Singleton instance for default use
_default_dense_reward = DenseRewardFunction()


def compute_dense_reward(
    env: Any,
    action: Optional[np.ndarray] = None,
    reward_scale: float = 1.0,
    return_components: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Convenience function to compute dense reward.
    
    This is the main entry point for computing dense rewards
    in LIBERO environments.
    
    Args:
        env: LIBERO environment instance
        action: The action taken
        reward_scale: Scaling factor for the reward
        return_components: If True, also return reward components dict
        
    Returns:
        float: Dense reward value
        OR
        tuple: (reward, components_dict) if return_components=True
        
    Example:
        >>> reward = compute_dense_reward(env, action)
        >>> # or with components
        >>> reward, components = compute_dense_reward(env, action, return_components=True)
        >>> print(components)  # {'reaching': 0.3, 'grasping': 0.0, 'success': 0.0, ...}
    """
    # Create reward function with specified scale
    reward_fn = DenseRewardFunction(reward_scale=reward_scale)
    
    # Compute reward
    reward = reward_fn.compute(env, action)
    
    if return_components:
        return reward, reward_fn.get_info()
    return reward


def get_reward_info(env: Any) -> Dict[str, Any]:
    """
    Get detailed information about reward-relevant state.
    Useful for debugging and visualization.
    
    Args:
        env: LIBERO environment instance
        
    Returns:
        dict: Detailed reward information including distances, grasp state, etc.
    """
    info = {}
    
    # Get distances to objects
    info['distances'] = compute_gripper_distance_to_objects(env)
    
    # Get objects of interest
    if hasattr(env, 'obj_of_interest'):
        info['objects_of_interest'] = list(env.obj_of_interest)
    
    # Check success
    try:
        info['success'] = env._check_success()
    except:
        info['success'] = False
    
    # Compute current dense reward
    try:
        reward, components = compute_dense_reward(env, return_components=True)
        info['reward'] = reward
        info['reward_components'] = components
    except:
        info['reward'] = 0.0
        info['reward_components'] = {}
    
    return info

