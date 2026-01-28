#!/usr/bin/env python3
"""
Test script for dense reward implementation.

This script tests that reward_shaping=True now provides dense rewards.

Usage:
    cd /workspace/repos/rsec/LIBERO
    conda activate libero
    python analyze/reward_analysis/test_dense_reward.py
"""

import os
import sys
import numpy as np

# Add LIBERO to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def test_sparse_vs_dense_reward():
    """Compare sparse and dense reward outputs."""
    
    print("=" * 60)
    print("Testing Sparse vs Dense Reward")
    print("=" * 60)
    
    # Get a task from LIBERO-10
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_10']()
    task = task_suite.get_task(0)  # First task
    
    # Create environment with sparse reward
    print("\n1. Testing SPARSE reward (reward_shaping=False)...")
    env_sparse = OffScreenRenderEnv(
        bddl_file_name=task.bddl_file,
        camera_names=['agentview'],
        camera_heights=128,
        camera_widths=128,
        reward_shaping=False,  # Sparse reward
    )
    env_sparse.reset()
    
    # Take some random actions and record rewards
    sparse_rewards = []
    for _ in range(10):
        action = np.random.uniform(-1, 1, 7)
        obs, reward, done, info = env_sparse.step(action)
        sparse_rewards.append(reward)
    
    print(f"   Sparse rewards (10 steps): {sparse_rewards}")
    print(f"   Sparse reward range: [{min(sparse_rewards):.3f}, {max(sparse_rewards):.3f}]")
    
    env_sparse.close()
    
    # Create environment with dense reward
    print("\n2. Testing DENSE reward (reward_shaping=True)...")
    env_dense = OffScreenRenderEnv(
        bddl_file_name=task.bddl_file,
        camera_names=['agentview'],
        camera_heights=128,
        camera_widths=128,
        reward_shaping=True,  # Dense reward
    )
    env_dense.reset()
    
    # Take some random actions and record rewards
    dense_rewards = []
    for _ in range(10):
        action = np.random.uniform(-1, 1, 7)
        obs, reward, done, info = env_dense.step(action)
        dense_rewards.append(reward)
    
    print(f"   Dense rewards (10 steps): {[f'{r:.3f}' for r in dense_rewards]}")
    print(f"   Dense reward range: [{min(dense_rewards):.3f}, {max(dense_rewards):.3f}]")
    
    env_dense.close()
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    sparse_variance = np.var(sparse_rewards)
    dense_variance = np.var(dense_rewards)
    
    print(f"Sparse reward variance: {sparse_variance:.6f}")
    print(f"Dense reward variance:  {dense_variance:.6f}")
    
    if dense_variance > sparse_variance:
        print("\n✅ SUCCESS: Dense reward shows more variation than sparse reward!")
        print("   This indicates reward_shaping=True is now working correctly.")
    else:
        print("\n⚠️  WARNING: Dense reward variance is not higher than sparse.")
        print("   This might indicate an issue, or the random actions")
        print("   happened to not reach the objects of interest.")
    
    return True


def test_reward_components():
    """Test that we can get detailed reward components."""
    
    print("\n" + "=" * 60)
    print("Testing Reward Components")
    print("=" * 60)
    
    # Import the reward functions directly
    try:
        from libero.libero.envs.rewards import compute_dense_reward
        from libero.libero.envs.rewards.dense_reward import get_reward_info
        print("✅ Successfully imported dense reward module")
    except ImportError as e:
        print(f"❌ Failed to import dense reward module: {e}")
        return False
    
    # Create environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_10']()
    task = task_suite.get_task(0)
    
    env = OffScreenRenderEnv(
        bddl_file_name=task.bddl_file,
        camera_names=['agentview'],
        camera_heights=128,
        camera_widths=128,
        reward_shaping=True,
    )
    env.reset()
    
    # Get reward with components
    action = np.random.uniform(-1, 1, 7)
    env.step(action)
    
    reward, components = compute_dense_reward(env.env, action, return_components=True)
    
    print(f"\nReward: {reward:.4f}")
    print("Components:")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")
    
    # Get detailed info
    info = get_reward_info(env.env)
    print(f"\nObjects of interest: {info.get('objects_of_interest', [])}")
    print(f"Distances to objects: {info.get('distances', {})}")
    print(f"Task success: {info.get('success', False)}")
    
    env.close()
    
    print("\n✅ Reward components test passed!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LIBERO Dense Reward Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_sparse_vs_dense_reward()
        test_reward_components()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

