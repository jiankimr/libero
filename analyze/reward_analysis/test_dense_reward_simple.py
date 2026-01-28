#!/usr/bin/env python3
"""
Simple Dense Reward Test (No GPU Inference Required)

This script tests the dense reward implementation using random actions.
It's designed to be lightweight and not interfere with ongoing training.

Usage:
    cd /workspace/repos/rsec/LIBERO
    conda activate libero
    python analyze/reward_analysis/test_dense_reward_simple.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Limit CPU threads to avoid interfering with GPU training
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# Add LIBERO to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_dense_vs_sparse_reward():
    """
    Compare dense vs sparse reward using random actions.
    No model inference required - uses random actions.
    """
    import os
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    
    print("=" * 60)
    print("Dense vs Sparse Reward Test (Random Actions)")
    print("=" * 60)
    
    # Get first task from LIBERO-10
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_10']()
    task = task_suite.get_task(0)
    
    # Get full BDDL path
    libero_bddl_path = get_libero_path('bddl_files')
    bddl_file = os.path.join(libero_bddl_path, 'libero_10', os.path.basename(task.bddl_file))
    
    print(f"\nTask: {task.language}")
    print(f"BDDL: {bddl_file}")
    
    num_episodes = 5
    steps_per_episode = 50
    
    results = {
        'sparse': {'rewards': [], 'all_rewards': []},
        'dense': {'rewards': [], 'all_rewards': []},
    }
    
    for reward_mode in ['sparse', 'dense']:
        print(f"\n--- Testing {reward_mode.upper()} reward ---")
        
        use_shaping = (reward_mode == 'dense')
        
        # Create environment with minimal rendering
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_names=['agentview'],
            camera_heights=64,  # Small resolution to save memory
            camera_widths=64,
            reward_shaping=use_shaping,
            render_gpu_device_id=-1,  # Try to use CPU if possible
        )
        
        for ep in range(num_episodes):
            env.reset()
            episode_rewards = []
            
            for step in range(steps_per_episode):
                # Random action
                action = np.random.uniform(-0.5, 0.5, 7)
                action[-1] = np.random.choice([-1, 1])  # Gripper open/close
                
                obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                
                if done:
                    break
            
            total_reward = sum(episode_rewards)
            results[reward_mode]['rewards'].append(total_reward)
            results[reward_mode]['all_rewards'].extend(episode_rewards)
            
            print(f"  Episode {ep+1}: steps={len(episode_rewards)}, "
                  f"total_reward={total_reward:.4f}, "
                  f"reward_range=[{min(episode_rewards):.4f}, {max(episode_rewards):.4f}]")
        
        env.close()
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    for mode in ['sparse', 'dense']:
        rewards = results[mode]['all_rewards']
        print(f"\n{mode.upper()} Reward Statistics:")
        print(f"  Mean: {np.mean(rewards):.6f}")
        print(f"  Std:  {np.std(rewards):.6f}")
        print(f"  Min:  {np.min(rewards):.6f}")
        print(f"  Max:  {np.max(rewards):.6f}")
        print(f"  Unique values: {len(set([round(r, 4) for r in rewards]))}")
    
    # Check if dense reward is actually dense
    sparse_unique = len(set([round(r, 4) for r in results['sparse']['all_rewards']]))
    dense_unique = len(set([round(r, 4) for r in results['dense']['all_rewards']]))
    
    print("\n" + "=" * 60)
    if dense_unique > sparse_unique * 2:
        print("✅ SUCCESS: Dense reward shows more variation!")
        print(f"   Sparse unique values: {sparse_unique}")
        print(f"   Dense unique values:  {dense_unique}")
    else:
        print("⚠️  Dense reward may not be working correctly")
        print(f"   Sparse unique values: {sparse_unique}")
        print(f"   Dense unique values:  {dense_unique}")
    
    return results


def plot_results(results: dict, output_dir: str = "./reward_test_output"):
    """Plot and save reward comparison."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors
    colors = {'sparse': '#FFB6C1', 'dense': '#90EE90'}
    
    # Left: Histogram
    ax1 = axes[0]
    for mode in ['sparse', 'dense']:
        rewards = results[mode]['all_rewards']
        ax1.hist(rewards, bins=30, alpha=0.6, label=f"{mode} (μ={np.mean(rewards):.4f})",
                 color=colors[mode])
    ax1.set_xlabel('Reward', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Reward Distribution (Random Actions)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Box plot
    ax2 = axes[1]
    data = [results['sparse']['all_rewards'], results['dense']['all_rewards']]
    bp = ax2.boxplot(data, labels=['Sparse', 'Dense'], patch_artist=True)
    for i, (patch, mode) in enumerate(zip(bp['boxes'], ['sparse', 'dense'])):
        patch.set_facecolor(colors[mode])
        patch.set_alpha(0.6)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Comparison', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Dense vs Sparse Reward (Random Actions)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / "dense_vs_sparse_reward_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")
    
    plt.close()
    
    return output_file


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LIBERO Dense Reward Simple Test")
    print("(Using random actions - no GPU inference required)")
    print("=" * 60 + "\n")
    
    try:
        results = test_dense_vs_sparse_reward()
        plot_results(results, output_dir="/workspace/results/reward_test")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

