#!/usr/bin/env python3
"""
Run Reward Comparison: Clean vs Noisy (Poisoned) Models

This script compares reward distributions between models trained on
clean data vs models trained on noisy (poisoned) data.

Goal: Assess the stealthiness of data poisoning attacks by checking
whether normal and poisoned trajectories are indistinguishable under
reward-based evaluation.

Usage:
    cd /workspace/repos/rsec/LIBERO
    conda activate libero
    python analyze/reward_analysis/run_reward_comparison.py \
        --clean_model_host localhost --clean_model_port 5555 \
        --noisy_model_host localhost --noisy_model_port 5556 \
        --num_episodes 10 --output_dir ./reward_comparison_output
"""

import argparse
import os
import sys
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import collections
import json

# Add LIBERO to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from service import ExternalRobotInferenceClient
from reward_visualizer import visualize_reward_comparison


def collect_episode_rewards(
    env,
    client: ExternalRobotInferenceClient,
    task_description: str,
    max_steps: int = 300,
    use_dense_reward: bool = True,
) -> tuple:
    """
    Collect rewards for a single episode.
    
    Args:
        env: LIBERO environment
        client: Inference client for the model
        task_description: Task description string
        max_steps: Maximum steps per episode
        use_dense_reward: Whether to use dense or sparse reward
        
    Returns:
        tuple: (list of rewards, success flag)
    """
    rewards = []
    
    # Reset environment
    env.reset()
    obs = env.env.reset()
    
    action_plan = collections.deque()
    
    for step in range(max_steps):
        # Get action from model
        if len(action_plan) == 0:
            # Get observation images
            image = obs.get("agentview_image", obs.get("agentview_rgb"))
            wrist_image = obs.get("robot0_eye_in_hand_image", obs.get("eye_in_hand_rgb"))
            
            if image is None:
                break
            
            # Query model for actions
            actions = client(
                images={"cam_high": image, "cam_right_wrist": wrist_image},
                instruction=task_description,
            )
            action_plan.extend(actions)
        
        # Get next action
        if len(action_plan) == 0:
            break
            
        action = action_plan.popleft()
        
        # Step environment
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        
        if done:
            break
    
    success = env.env._check_success()
    return rewards, success


def run_comparison(
    clean_host: str,
    clean_port: int,
    noisy_host: str,
    noisy_port: int,
    num_episodes: int = 10,
    num_tasks: int = 5,
    use_dense_reward: bool = True,
    output_dir: str = "./reward_comparison_output",
) -> Dict:
    """
    Run full comparison between clean and noisy models.
    
    Args:
        clean_host: Host for clean model server
        clean_port: Port for clean model server
        noisy_host: Host for noisy model server
        noisy_port: Port for noisy model server
        num_episodes: Episodes per task
        num_tasks: Number of tasks to evaluate
        use_dense_reward: Use dense reward
        output_dir: Output directory
        
    Returns:
        Dict with comparison results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize clients
    clean_client = ExternalRobotInferenceClient(host=clean_host, port=clean_port)
    noisy_client = ExternalRobotInferenceClient(host=noisy_host, port=noisy_port)
    
    # Get tasks
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_10']()
    
    all_results = {
        'clean': {'episodes': [], 'successes': 0, 'total': 0},
        'noisy': {'episodes': [], 'successes': 0, 'total': 0},
    }
    
    for task_id in range(min(num_tasks, 10)):
        task = task_suite.get_task(task_id)
        task_description = task.language
        
        print(f"\n{'='*60}")
        print(f"Task {task_id + 1}/{num_tasks}: {task_description}")
        print('='*60)
        
        # Create environment with reward_shaping
        env = OffScreenRenderEnv(
            bddl_file_name=task.bddl_file,
            camera_names=['agentview', 'robot0_eye_in_hand'],
            camera_heights=256,
            camera_widths=256,
            reward_shaping=use_dense_reward,
        )
        
        for ep in range(num_episodes):
            print(f"\n  Episode {ep + 1}/{num_episodes}")
            
            # Collect from clean model
            print("    Collecting clean model rewards...", end=" ")
            clean_rewards, clean_success = collect_episode_rewards(
                env, clean_client, task_description, use_dense_reward=use_dense_reward
            )
            all_results['clean']['episodes'].append(clean_rewards)
            all_results['clean']['total'] += 1
            if clean_success:
                all_results['clean']['successes'] += 1
            print(f"Done (success={clean_success}, steps={len(clean_rewards)})")
            
            # Collect from noisy model
            print("    Collecting noisy model rewards...", end=" ")
            noisy_rewards, noisy_success = collect_episode_rewards(
                env, noisy_client, task_description, use_dense_reward=use_dense_reward
            )
            all_results['noisy']['episodes'].append(noisy_rewards)
            all_results['noisy']['total'] += 1
            if noisy_success:
                all_results['noisy']['successes'] += 1
            print(f"Done (success={noisy_success}, steps={len(noisy_rewards)})")
        
        env.close()
    
    # Compute success rates
    for condition in ['clean', 'noisy']:
        total = all_results[condition]['total']
        successes = all_results[condition]['successes']
        all_results[condition]['success_rate'] = successes / total if total > 0 else 0
    
    # Prepare data for visualization
    rewards_dict = {
        'Clean Model': all_results['clean']['episodes'],
        'Noisy Model': all_results['noisy']['episodes'],
    }
    
    # Generate visualizations
    print("\n\nGenerating visualizations...")
    analysis_results = visualize_reward_comparison(
        rewards_dict,
        output_dir=output_dir,
        title_prefix="Clean vs Noisy Model: ",
    )
    
    # Add success rates to results
    analysis_results['success_rates'] = {
        'clean': all_results['clean']['success_rate'],
        'noisy': all_results['noisy']['success_rate'],
    }
    
    # Save raw data
    with open(output_path / "raw_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'clean': {
                'episodes': [[float(r) for r in ep] for ep in all_results['clean']['episodes']],
                'success_rate': all_results['clean']['success_rate'],
            },
            'noisy': {
                'episodes': [[float(r) for r in ep] for ep in all_results['noisy']['episodes']],
                'success_rate': all_results['noisy']['success_rate'],
            },
        }
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Clean Model Success Rate: {all_results['clean']['success_rate']:.2%}")
    print(f"Noisy Model Success Rate: {all_results['noisy']['success_rate']:.2%}")
    
    if 'statistical_tests' in analysis_results:
        for comparison, test in analysis_results['statistical_tests'].items():
            print(f"\n{comparison}:")
            print(f"  P-value: {test['p_value']:.6f}")
            if test['significant']:
                print("  → Reward distributions are DISTINGUISHABLE (attack detectable)")
            else:
                print("  → Reward distributions are INDISTINGUISHABLE (stealthy attack)")
    
    return analysis_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare reward distributions between clean and noisy models"
    )
    parser.add_argument("--clean_model_host", type=str, default="localhost")
    parser.add_argument("--clean_model_port", type=int, default=5555)
    parser.add_argument("--noisy_model_host", type=str, default="localhost")
    parser.add_argument("--noisy_model_port", type=int, default=5556)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--sparse_reward", action="store_true",
                        help="Use sparse reward instead of dense")
    parser.add_argument("--output_dir", type=str, default="./reward_comparison_output")
    
    args = parser.parse_args()
    
    results = run_comparison(
        clean_host=args.clean_model_host,
        clean_port=args.clean_model_port,
        noisy_host=args.noisy_model_host,
        noisy_port=args.noisy_model_port,
        num_episodes=args.num_episodes,
        num_tasks=args.num_tasks,
        use_dense_reward=not args.sparse_reward,
        output_dir=args.output_dir,
    )
    
    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()



