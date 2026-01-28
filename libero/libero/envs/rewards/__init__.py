"""
LIBERO Dense Reward Module

This module provides dense reward functions for LIBERO environments.
It enables reward_shaping=True to work properly.

Usage:
    from libero.libero.envs.rewards import compute_dense_reward
"""

from .dense_reward import compute_dense_reward
from .base_reward import BaseRewardFunction
from .reward_utils import (
    compute_reaching_reward,
    compute_grasping_reward,
    compute_goal_progress_reward,
)

__all__ = [
    "compute_dense_reward",
    "BaseRewardFunction",
    "compute_reaching_reward",
    "compute_grasping_reward",
    "compute_goal_progress_reward",
]



