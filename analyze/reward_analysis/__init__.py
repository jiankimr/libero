"""
Reward Analysis Module for LIBERO

This module provides tools to analyze and visualize rewards
from clean vs noisy (poisoned) trajectories.

Used to assess the stealthiness of data poisoning attacks
by comparing reward distributions.
"""

from .reward_visualizer import (
    visualize_reward_comparison,
    plot_reward_distribution,
    plot_reward_curves,
)

__all__ = [
    "visualize_reward_comparison",
    "plot_reward_distribution", 
    "plot_reward_curves",
]



