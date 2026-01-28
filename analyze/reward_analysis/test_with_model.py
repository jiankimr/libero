#!/usr/bin/env python3
"""
Test Dense vs Sparse Reward with Trained Model

Runs evaluation and collects BOTH sparse and dense rewards simultaneously
from the same trajectories for fair comparison.

Usage:
    python test_with_model.py --host localhost --port 5555 --num_episodes 10 --num_tasks 1
"""

import argparse
import collections
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import json

# Add LIBERO to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from service import ExternalRobotInferenceClient

# Import dense reward function
try:
    from libero.libero.envs.rewards import compute_dense_reward
    DENSE_REWARD_AVAILABLE = True
except ImportError:
    DENSE_REWARD_AVAILABLE = False
    print("Warning: Dense reward module not available")


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # Dummy action for initial wait


def collect_episode_rewards(
    env,
    client,
    task_description: str,
    max_steps: int = 520,  # libero_10's longest demo has 505 steps
    num_steps_wait: int = 10,  # Wait for objects to stabilize
) -> dict:
    """
    Collect BOTH sparse and dense rewards for a single episode.
    Same trajectory, both reward types logged simultaneously.
    """
    import robosuite.utils.transform_utils as T
    
    sparse_rewards = []
    dense_rewards = []
    action_plan = collections.deque()
    
    # Reset environment
    obs = env.reset()
    
    # Wait for objects to stabilize (like eval.py does)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    
    for step in range(max_steps):
        # Get action from model
        if len(action_plan) == 0:
            # Get observation images (rotate 180 degrees to match train preprocessing)
            raw_img = obs.get("agentview_image", obs.get("agentview_rgb"))
            raw_wrist_img = obs.get("robot0_eye_in_hand_image", obs.get("eye_in_hand_rgb"))
            
            # IMPORTANT: rotate 180 degrees like eval.py does
            img = np.ascontiguousarray(raw_img[::-1, ::-1]) if raw_img is not None else None
            wrist_img = np.ascontiguousarray(raw_wrist_img[::-1, ::-1]) if raw_wrist_img is not None else None
            
            if img is None:
                print(f"    Warning: No image at step {step}")
                break
            
            # Get EEF state
            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = obs.get("robot0_eef_quat", np.array([1, 0, 0, 0]))
            gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
            
            # Convert quaternion to axis angle
            try:
                axis_angle = T.quat2axisangle(eef_quat)
            except:
                axis_angle = np.zeros(3)
            
            # Build element dict matching model's expected format
            element = {
                "video.image": np.expand_dims(img, axis=0),
                "video.wrist_image": np.expand_dims(wrist_img if wrist_img is not None else img, axis=0),
                "state.x": eef_pos[np.newaxis, 0:1],
                "state.y": eef_pos[np.newaxis, 1:2],
                "state.z": eef_pos[np.newaxis, 2:3],
                "state.axis_angle1": axis_angle[np.newaxis, 0:1],
                "state.axis_angle2": axis_angle[np.newaxis, 1:2],
                "state.axis_angle3": axis_angle[np.newaxis, 2:3],
                "state.gripper_left_finger": gripper_qpos[np.newaxis, 0:1],
                "state.gripper_right_finger": gripper_qpos[np.newaxis, 1:2],
                "annotation.human.action.task_description": [str(task_description)],
            }
            
            # Query model for actions
            try:
                action_chunk = client.get_action(element)
                
                # Action format: action.x, action.y, action.z, action.axis_angle1/2/3, action.gripper
                ax = action_chunk.get("action.x", np.zeros((16, 1)))
                ay = action_chunk.get("action.y", np.zeros((16, 1)))
                az = action_chunk.get("action.z", np.zeros((16, 1)))
                aa1 = action_chunk.get("action.axis_angle1", np.zeros((16, 1)))
                aa2 = action_chunk.get("action.axis_angle2", np.zeros((16, 1)))
                aa3 = action_chunk.get("action.axis_angle3", np.zeros((16, 1)))
                grip = action_chunk.get("action.gripper", np.zeros((16, 1)))
                
                # Stack into 7D actions
                # Model outputs gripper in [0, 1] (0=open, 1=close)
                # LIBERO expects [-1, 1] (+1=open, -1=close)
                # Conversion: gripper_libero = 1.0 - 2.0 * gripper_model
                num_steps = min(16, len(ax))
                for t in range(num_steps):
                    gripper_libero = 1.0 - 2.0 * grip[t, 0]
                    action = np.array([
                        ax[t, 0], ay[t, 0], az[t, 0],
                        aa1[t, 0], aa2[t, 0], aa3[t, 0],
                        gripper_libero,
                    ])
                    action_plan.append(action)
                    
            except Exception as e:
                print(f"    Error getting action: {e}")
                break
        
        # Get next action
        if len(action_plan) == 0:
            break
            
        action = action_plan.popleft()
        
        # Step environment - this returns SPARSE reward by default
        obs, sparse_reward, done, info = env.step(action)
        sparse_rewards.append(sparse_reward)
        
        # Compute DENSE reward for the same state
        if DENSE_REWARD_AVAILABLE:
            dense_reward = compute_dense_reward(env.env, action)
        else:
            dense_reward = sparse_reward  # Fallback
        dense_rewards.append(dense_reward)
        
        if done:
            break
    
    success = env.env._check_success()
    
    return {
        'sparse_rewards': sparse_rewards,
        'dense_rewards': dense_rewards,
        'success': success,
        'steps': len(sparse_rewards),
    }


def run_test(
    host: str,
    port: int,
    num_episodes: int,
    num_tasks: int,
    output_dir: str,
):
    """Run dense vs sparse reward test with model inference."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Dense vs Sparse Reward Test (Same Trajectory)")
    print("=" * 60)
    print(f"Host: {host}:{port}")
    print(f"Episodes per task: {num_episodes}")
    print(f"Tasks: {num_tasks}")
    print(f"Dense reward available: {DENSE_REWARD_AVAILABLE}")
    
    # Initialize client
    print("\nConnecting to inference server...")
    client = ExternalRobotInferenceClient(host=host, port=port)
    
    # Get tasks
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_10']()
    libero_bddl_path = get_libero_path('bddl_files')
    
    all_results = {
        'sparse': {'all_rewards': [], 'episodes': []},
        'dense': {'all_rewards': [], 'episodes': []},
        'successes': 0,
        'total': 0,
    }
    
    for task_id in range(min(num_tasks, 10)):
        task = task_suite.get_task(task_id)
        task_description = task.language
        bddl_file = os.path.join(libero_bddl_path, 'libero_10', os.path.basename(task.bddl_file))
        
        print(f"\n{'='*60}")
        print(f"Task {task_id + 1}/{num_tasks}: {task_description}")
        print('='*60)
        
        # Create environment (sparse reward - we compute dense separately)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_names=['agentview', 'robot0_eye_in_hand'],
            camera_heights=256,
            camera_widths=256,
            reward_shaping=False,  # Use sparse, compute dense manually
        )
        
        for ep in range(num_episodes):
            print(f"  Episode {ep + 1}/{num_episodes}...", end=" ")
            
            result = collect_episode_rewards(env, client, task_description)
            
            # Store results
            all_results['sparse']['episodes'].append(result['sparse_rewards'])
            all_results['sparse']['all_rewards'].extend(result['sparse_rewards'])
            all_results['dense']['episodes'].append(result['dense_rewards'])
            all_results['dense']['all_rewards'].extend(result['dense_rewards'])
            all_results['total'] += 1
            if result['success']:
                all_results['successes'] += 1
            
            sparse_sum = sum(result['sparse_rewards'])
            dense_sum = sum(result['dense_rewards'])
            print(f"steps={result['steps']}, sparse={sparse_sum:.2f}, dense={dense_sum:.2f}, success={result['success']}")
        
        env.close()
    
    # Compute statistics
    success_rate = all_results['successes'] / all_results['total']
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.1%} ({all_results['successes']}/{all_results['total']})")
    
    for mode in ['sparse', 'dense']:
        rewards = all_results[mode]['all_rewards']
        print(f"\n{mode.upper()} Reward:")
        print(f"  Mean: {np.mean(rewards):.6f}")
        print(f"  Std:  {np.std(rewards):.6f}")
        print(f"  Min:  {np.min(rewards):.6f}")
        print(f"  Max:  {np.max(rewards):.6f}")
        print(f"  Unique values: {len(set([round(r, 4) for r in rewards]))}")
    
    # Plot results
    plot_results(all_results, output_path, success_rate)
    
    # Save raw data
    results_data = {
        'success_rate': success_rate,
        'sparse': {
            'episodes': [[float(r) for r in ep] for ep in all_results['sparse']['episodes']],
            'mean': float(np.mean(all_results['sparse']['all_rewards'])),
            'std': float(np.std(all_results['sparse']['all_rewards'])),
        },
        'dense': {
            'episodes': [[float(r) for r in ep] for ep in all_results['dense']['episodes']],
            'mean': float(np.mean(all_results['dense']['all_rewards'])),
            'std': float(np.std(all_results['dense']['all_rewards'])),
        },
    }
    with open(output_path / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return all_results


def plot_results(results: Dict, output_path: Path, success_rate: float):
    """Plot reward comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'sparse': '#FFB6C1', 'dense': '#90EE90'}
    
    # 1. Histogram
    ax1 = axes[0]
    for mode in ['sparse', 'dense']:
        rewards = results[mode]['all_rewards']
        ax1.hist(rewards, bins=30, alpha=0.6, 
                label=f"{mode} (μ={np.mean(rewards):.4f}, σ={np.std(rewards):.4f})",
                color=colors[mode])
    ax1.set_xlabel('Reward', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Reward Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[1]
    data = [results['sparse']['all_rewards'], results['dense']['all_rewards']]
    bp = ax2.boxplot(data, labels=['Sparse', 'Dense'], patch_artist=True)
    for i, (patch, mode) in enumerate(zip(bp['boxes'], ['sparse', 'dense'])):
        patch.set_facecolor(colors[mode])
        patch.set_alpha(0.6)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Comparison (Box Plot)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward curves over time (mean ± std)
    ax3 = axes[2]
    for mode in ['sparse', 'dense']:
        episodes = results[mode]['episodes']
        if not episodes:
            continue
        
        # Pad to same length
        max_len = max(len(ep) for ep in episodes)
        padded = np.full((len(episodes), max_len), np.nan)
        for i, ep in enumerate(episodes):
            padded[i, :len(ep)] = ep
        
        mean_rewards = np.nanmean(padded, axis=0)
        std_rewards = np.nanstd(padded, axis=0)
        steps = np.arange(len(mean_rewards))
        
        ax3.plot(steps, mean_rewards, label=mode, color=colors[mode], linewidth=2)
        ax3.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                        alpha=0.3, color=colors[mode])
    
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Reward', fontsize=12)
    ax3.set_title('Reward Over Time (mean ± std)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sparse vs Dense Reward (Same Trajectory) - Success Rate: {success_rate:.1%}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / "reward_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test dense vs sparse reward with model inference")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./reward_test_output")
    
    args = parser.parse_args()
    
    run_test(
        host=args.host,
        port=args.port,
        num_episodes=args.num_episodes,
        num_tasks=args.num_tasks,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
