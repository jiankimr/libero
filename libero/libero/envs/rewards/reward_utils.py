"""
Reward Utility Functions

Provides helper functions for computing various reward components.
"""

import numpy as np
from typing import Any, Optional, List


def compute_reaching_reward(
    env: Any,
    target_object_name: Optional[str] = None,
    scale: float = 1.0,
    tanh_scale: float = 10.0,
) -> float:
    """
    Compute reaching reward based on distance from gripper to target object.
    
    Reward = scale * (1 - tanh(tanh_scale * distance))
    
    Args:
        env: LIBERO environment instance
        target_object_name: Name of target object. If None, uses first object of interest.
        scale: Scaling factor for the reward
        tanh_scale: Scale factor for tanh function (higher = sharper falloff)
        
    Returns:
        float: Reaching reward in range [0, scale]
    """
    try:
        # Get gripper position
        gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        
        # Get target object position
        if target_object_name is None:
            # Use first object of interest if not specified
            if hasattr(env, 'obj_of_interest') and len(env.obj_of_interest) > 0:
                target_object_name = env.obj_of_interest[0]
            else:
                return 0.0
        
        # Try to get object position
        if target_object_name in env.obj_body_id:
            object_pos = env.sim.data.body_xpos[env.obj_body_id[target_object_name]]
        else:
            return 0.0
        
        # Compute distance and reward
        dist = np.linalg.norm(gripper_site_pos - object_pos)
        reaching_reward = scale * (1.0 - np.tanh(tanh_scale * dist))
        
        return reaching_reward
        
    except (AttributeError, KeyError, IndexError) as e:
        # Graceful degradation if something goes wrong
        return 0.0


def compute_grasping_reward(
    env: Any,
    target_object_name: Optional[str] = None,
    reward_value: float = 0.25,
) -> float:
    """
    Compute grasping reward - non-zero if gripper is grasping the target object.
    
    Args:
        env: LIBERO environment instance
        target_object_name: Name of target object. If None, uses first object of interest.
        reward_value: Reward value when grasping
        
    Returns:
        float: Grasping reward (0 or reward_value)
    """
    try:
        if target_object_name is None:
            if hasattr(env, 'obj_of_interest') and len(env.obj_of_interest) > 0:
                target_object_name = env.obj_of_interest[0]
            else:
                return 0.0
        
        # Check if object exists
        if target_object_name not in env.objects_dict:
            return 0.0
        
        target_object = env.objects_dict[target_object_name]
        
        # Check grasp using robosuite's check_grasp method
        if hasattr(env, '_check_grasp'):
            is_grasping = env._check_grasp(
                gripper=env.robots[0].gripper,
                object_geoms=target_object
            )
            return reward_value if is_grasping else 0.0
        
        return 0.0
        
    except (AttributeError, KeyError, IndexError) as e:
        return 0.0


def compute_goal_progress_reward(
    env: Any,
    goal_objects: Optional[List[str]] = None,
    goal_positions: Optional[List[np.ndarray]] = None,
    scale: float = 0.5,
) -> float:
    """
    Compute goal progress reward based on how close objects are to their goal positions.
    
    This is useful for tasks where objects need to be moved to specific locations.
    
    Args:
        env: LIBERO environment instance
        goal_objects: List of object names to track
        goal_positions: Corresponding goal positions
        scale: Scaling factor
        
    Returns:
        float: Goal progress reward
    """
    try:
        if goal_objects is None or goal_positions is None:
            return 0.0
        
        total_reward = 0.0
        for obj_name, goal_pos in zip(goal_objects, goal_positions):
            if obj_name in env.obj_body_id:
                current_pos = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
                dist = np.linalg.norm(current_pos - goal_pos)
                total_reward += 1.0 - np.tanh(5.0 * dist)
        
        if len(goal_objects) > 0:
            total_reward /= len(goal_objects)
        
        return scale * total_reward
        
    except (AttributeError, KeyError, IndexError) as e:
        return 0.0


def compute_action_penalty(
    action: np.ndarray,
    penalty_scale: float = 0.01,
) -> float:
    """
    Compute action penalty to encourage smooth actions.
    
    Args:
        action: The action taken
        penalty_scale: Scale of the penalty
        
    Returns:
        float: Negative penalty value
    """
    if action is None:
        return 0.0
    
    action_magnitude = np.linalg.norm(action)
    return -penalty_scale * action_magnitude


def compute_gripper_distance_to_objects(env: Any) -> dict:
    """
    Compute distances from gripper to all objects of interest.
    Useful for debugging and analysis.
    
    Args:
        env: LIBERO environment instance
        
    Returns:
        dict: Object name -> distance mapping
    """
    distances = {}
    try:
        gripper_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        
        if hasattr(env, 'obj_of_interest'):
            for obj_name in env.obj_of_interest:
                if obj_name in env.obj_body_id:
                    obj_pos = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
                    distances[obj_name] = float(np.linalg.norm(gripper_pos - obj_pos))
    except:
        pass
    
    return distances

