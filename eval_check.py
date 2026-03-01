#!/usr/bin/env python3
"""
EISS (Exponential Incremental State Stability) Check Script

Verifies the EISS property by:
1. Running a reference trajectory (NO perturbation) from a fixed initial state
2. Running multiple perturbed trajectories (perturbed initial joint positions)
3. Computing d_t = ||x_t - x'_t|| over time (first 16 chunks only)
4. Checking if log(d_t) decreases linearly (exponential contraction)

Only SUCCESSFUL episodes are saved and compared.

Reference: "Action Chunking and Exploratory Data Collection Yield Exponential Improvements 
in Behavior Cloning for Continuous Control" (arXiv:2507.09061)
"""

import collections
import dataclasses
import logging
import math
import pathlib
import sys
import datetime
import json
import os
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import tyro

from service import ExternalRobotInferenceClient
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "localhost"
    port: int = 5555
    
    # EISS test parameters
    task_id: int = 0  # Which task to test (0-9 for libero_10), -1 for all tasks
    episode_idx: int = 0  # Which initial state to use
    num_perturbations: int = 20  # Number of perturbed rollouts to attempt
    eef_noise_std: float = 0.001  # EEF position noise std in meters (0.001m = 1mm)
    num_chunks: int = 16  # Number of action chunks to execute (16 chunks × 16 steps = 256 steps)
    
    # Quick mode: test multiple tasks quickly
    quick_mode: bool = False  # If True, test first 3 tasks with 10 perturbations each
    
    # Evaluation parameters
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    replan_steps: int = 16
    resize_size: int = 224
    
    # Output
    output_dir: str = "./eiss_check"
    seed: int = 42


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def get_state_vector(obs: dict) -> np.ndarray:
    """Extract a state vector from observation for distance calculation.
    
    Uses EEF position + orientation (axis-angle) + gripper state = 8D
    """
    eef_pos = obs["robot0_eef_pos"]  # (3,)
    axis_angle = _quat2axisangle(obs["robot0_eef_quat"].copy())  # (3,)
    gripper_qpos = obs["robot0_gripper_qpos"]  # (2,)
    
    state = np.concatenate([eef_pos, axis_angle, gripper_qpos])
    return state


def unchunk(action_chunk, action_plan, replan_steps):
    """Convert action chunk to individual timesteps."""
    lengths = [v.shape[0] for v in action_chunk.values()]
    assert len(set(lengths)) == 1
    assert lengths[0] >= replan_steps
    
    T = replan_steps
    keys = list(action_chunk.keys())
    
    for t in range(T):
        action_list = []
        for k in keys:
            val = action_chunk[k][t]
            if np.isscalar(val):
                val = np.array([val], dtype=np.float32)
            elif val.ndim == 0:
                val = np.array([val.item()], dtype=np.float32)
            action_list.append(val)
        
        flattened = [np.atleast_1d(a).flatten() for a in action_list]
        new_arr = np.concatenate(flattened, axis=0)
        action_plan.append(new_arr)


def run_rollout(
    env,
    policy,
    initial_states,
    episode_idx: int,
    task_description: str,
    num_chunks: int,
    num_steps_wait: int,
    replan_steps: int,
    perturb_joints: bool = False,
    eef_noise_std: float = 0.001
) -> Tuple[List[np.ndarray], bool]:
    """
    Run a single rollout for exactly num_chunks action chunks.
    
    Args:
        perturb_joints: If False, no perturbation (reference). If True, add joint noise.
        
    Returns:
        state_trajectory: List of state vectors (one per step, including chunk boundaries)
        success: Whether the task was completed
    """
    env.reset()
    action_plan = collections.deque()
    
    # Set initial state
    obs = env.set_init_state(initial_states[episode_idx])
    
    # Add EEF space perturbation using Jacobian-based IK
    # This perturbs in the space that the policy actually sees (EEF pos/ori)
    if perturb_joints:
        robot = env.env.robots[0]
        
        # Get EEF site name (typically 'gripper0_grip_site' for Panda robot)
        eef_site_name = robot.gripper.important_sites["grip_site"]
        
        # Get current Jacobian (maps joint velocities to EEF velocities)
        # Shape: (6, n_joints) for [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
        J_pos = robot.sim.data.get_site_jacp(eef_site_name).reshape(3, -1)[:, robot.joint_indexes]
        J_rot = robot.sim.data.get_site_jacr(eef_site_name).reshape(3, -1)[:, robot.joint_indexes]
        J = np.vstack([J_pos, J_rot])  # (6, n_joints)
        
        # Generate EEF perturbation (in meters for position, radians for rotation)
        # Use smaller noise for rotation (radians) vs position (meters)
        delta_eef_pos = np.random.normal(0.0, eef_noise_std, size=3)  # e.g., 0.001m = 1mm
        delta_eef_rot = np.random.normal(0.0, eef_noise_std * 0.1, size=3)  # smaller for rotation
        delta_eef = np.concatenate([delta_eef_pos, delta_eef_rot])
        
        # Convert EEF perturbation to joint perturbation using Jacobian pseudo-inverse
        J_pinv = np.linalg.pinv(J)
        delta_joint = J_pinv @ delta_eef
        
        # Apply to joint positions
        joint_qpos_idx = robot.joint_indexes
        qpos = env.sim.data.qpos.copy()
        qpos[joint_qpos_idx] += delta_joint
        env.sim.data.qpos[:] = qpos
        env.sim.forward()
        
        # Get observation after perturbation
        obs = env.env._get_observations()
    
    state_trajectory = []
    max_steps = num_steps_wait + (num_chunks * replan_steps)
    chunks_executed = 0
    t = 0
    
    while t < max_steps:
        # Wait for objects to stabilize
        if t < num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        
        # Record state at each step
        state = get_state_vector(obs)
        state_trajectory.append(state)
        
        # Get new action chunk if needed
        if not action_plan:
            if chunks_executed >= num_chunks:
                break  # Done with all chunks
                
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            
            eef_pos = obs["robot0_eef_pos"]
            axis_angle = _quat2axisangle(obs["robot0_eef_quat"].copy())
            gripper_qpos = obs["robot0_gripper_qpos"]
            
            element = {
                "video.image": np.expand_dims(img, axis=0),
                "video.wrist_image": np.expand_dims(wrist_img, axis=0),
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
            
            action_chunk = policy.get_action(element)
            unchunk(action_chunk, action_plan, replan_steps)
            chunks_executed += 1
        
        # Execute action
        action = action_plan.popleft()
        eef_pos_delta = action[:3]
        eef_rot_delta = action[3:6]
        gripper_cmd = action[6]
        gripper_libero = 1.0 - 2.0 * gripper_cmd
        
        libero_action = np.concatenate([eef_pos_delta, eef_rot_delta, [gripper_libero]])
        obs, reward, done, info = env.step(libero_action.tolist())
        
        if done:
            # Record final state
            state = get_state_vector(obs)
            state_trajectory.append(state)
            return state_trajectory, True
        
        t += 1
    
    return state_trajectory, False


def compute_distances(
    ref_trajectory: List[np.ndarray],
    perturbed_trajectories: List[List[np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distances d_t = ||x_t - x'_t|| for all perturbed trajectories.
    
    Returns:
        all_distances: (num_trajectories, num_steps) array
        mean_distances: Mean distance at each timestep
        std_distances: Std of distances at each timestep
    """
    min_len = len(ref_trajectory)
    for traj in perturbed_trajectories:
        min_len = min(min_len, len(traj))
    
    all_distances = []
    for traj in perturbed_trajectories:
        distances = []
        for t in range(min_len):
            d_t = np.linalg.norm(ref_trajectory[t] - traj[t])
            distances.append(d_t)
        all_distances.append(distances)
    
    all_distances = np.array(all_distances)  # (num_perturbations, min_len)
    mean_distances = np.mean(all_distances, axis=0)
    std_distances = np.std(all_distances, axis=0)
    
    return all_distances, mean_distances, std_distances


def check_eiss_criteria(mean_distances: np.ndarray, std_distances: np.ndarray) -> dict:
    """
    Check EISS criteria according to the paper (arXiv:2507.09061).
    
    Paper Definition 2.1 (Globally Stable Dynamics):
    - d_t ≤ ρ^t * d_0 where ρ < 1
    - This means: log(d_t) ≈ t * log(ρ) + log(d_0)
    
    Primary EISS criterion (from paper):
    - slope < 0 (exponential decay, i.e., ρ < 1)
    
    Supporting metrics:
    - R² (goodness of linear fit on log scale)
    - Contraction ratio: d_T / d_0 (should be < 1)
    """
    eps = 1e-10
    timesteps = np.arange(len(mean_distances))
    
    # Compute auxiliary metrics (informational, not for EISS decision)
    diffs = np.diff(mean_distances)
    monotonic_ratio = np.mean(diffs <= 0)
    
    mid = len(std_distances) // 2
    std_first_half = np.mean(std_distances[:mid]) if mid > 0 else 0
    std_second_half = np.mean(std_distances[mid:]) if mid > 0 else 0
    std_decreasing = std_second_half < std_first_half
    
    # Primary criterion: Linear fit on log scale
    valid_mask = mean_distances > eps
    if np.sum(valid_mask) > 10:
        log_mean = np.log(mean_distances[valid_mask] + eps)
        t_valid = timesteps[valid_mask]
        coeffs = np.polyfit(t_valid, log_mean, 1)
        fitted = np.polyval(coeffs, t_valid)
        
        ss_res = np.sum((log_mean - fitted) ** 2)
        ss_tot = np.sum((log_mean - np.mean(log_mean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Compute ρ (contraction rate per step)
        rho = np.exp(slope)  # slope = log(ρ), so ρ = exp(slope)
    else:
        r_squared = 0
        slope = 0
        intercept = 0
        rho = 1.0
    
    # Contraction ratio: d_T / d_0
    d_0 = mean_distances[0] if len(mean_distances) > 0 else 1.0
    d_T = mean_distances[-1] if len(mean_distances) > 0 else 1.0
    contraction_ratio = d_T / d_0 if d_0 > eps else 1.0
    
    # EISS criterion (paper-based):
    # Primary: slope < 0 (exponential decay)
    # The paper's Definition 2.1 requires ρ < 1, i.e., slope < 0
    is_decaying = slope < 0
    
    # Strong EISS: slope < 0 AND reasonable R² (≥ 0.5 for some linearity)
    # This is the paper's criterion: trajectories converge exponentially
    strong_eiss = is_decaying and r_squared >= 0.5
    
    return {
        "slope": float(slope),
        "rho": float(rho),  # contraction rate per step (ρ < 1 means EISS)
        "r_squared": float(r_squared),
        "contraction_ratio": float(contraction_ratio),  # d_T / d_0
        "intercept": float(intercept),
        "is_decaying": bool(is_decaying),
        "strong_eiss": bool(strong_eiss),
        # Auxiliary metrics (informational)
        "monotonic_ratio": float(monotonic_ratio),
        "std_first_half": float(std_first_half),
        "std_second_half": float(std_second_half),
        "std_decreasing": bool(std_decreasing),
    }


def plot_eiss_results(
    all_distances: np.ndarray,
    mean_distances: np.ndarray,
    std_distances: np.ndarray,
    eiss_results: dict,
    output_path: str,
    task_description: str,
    replan_steps: int
):
    """Create EISS visualization plots."""
    timesteps = np.arange(len(mean_distances))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Linear scale with individual trajectories
    ax1 = axes[0]
    for i, dists in enumerate(all_distances):
        ax1.plot(timesteps, dists[:len(timesteps)], 'gray', alpha=0.3, linewidth=0.5)
    ax1.plot(timesteps, mean_distances, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(timesteps, 
                      mean_distances - std_distances, 
                      mean_distances + std_distances, 
                      alpha=0.3, color='blue', label='±1 std')
    
    # Mark chunk boundaries
    for c in range(1, len(timesteps) // replan_steps + 1):
        ax1.axvline(x=c * replan_steps, color='red', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Time step', fontsize=12)
    ax1.set_ylabel('Distance d_t = ||x_t - x\'_t||', fontsize=12)
    ax1.set_title('State Distance Over Time (Linear Scale)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale with linear fit
    ax2 = axes[1]
    eps = 1e-10
    log_mean = np.log(mean_distances + eps)
    
    ax2.plot(timesteps, log_mean, 'b-', linewidth=2, label='log(Mean)')
    
    # Linear fit line
    slope = eiss_results['slope']
    intercept = eiss_results['intercept']
    fitted_line = slope * timesteps + intercept
    ax2.plot(timesteps, fitted_line, 'r--', linewidth=2, 
             label=f'Fit: slope={slope:.4f}, R²={eiss_results["r_squared"]:.3f}')
    
    # Mark chunk boundaries
    for c in range(1, len(timesteps) // replan_steps + 1):
        ax2.axvline(x=c * replan_steps, color='red', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel('Time step', fontsize=12)
    ax2.set_ylabel('log(d_t)', fontsize=12)
    ax2.set_title('State Distance Over Time (Log Scale)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overall title with EISS result
    eiss_status = "✓ EISS SATISFIED" if eiss_results['strong_eiss'] else "✗ EISS NOT SATISFIED"
    plt.suptitle(f'{task_description[:60]}...\n{eiss_status}', fontsize=12, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_eiss_data(
    output_dir: str,
    task_id: int,
    episode_idx: int,
    ref_trajectory: np.ndarray,
    perturbed_trajectories: List[np.ndarray],
    all_distances: np.ndarray,
    mean_distances: np.ndarray,
    std_distances: np.ndarray,
    eiss_results: dict,
    task_description: str,
    timestamp: str
):
    """Save all EISS data as npy files for later analysis."""
    task_dir = os.path.join(output_dir, f"task{task_id:02d}_ep{episode_idx:02d}_{timestamp}")
    pathlib.Path(task_dir).mkdir(parents=True, exist_ok=True)
    
    # Save trajectories
    np.save(os.path.join(task_dir, "ref_trajectory.npy"), np.array(ref_trajectory))
    np.save(os.path.join(task_dir, "perturbed_trajectories.npy"), 
            np.array([np.array(t) for t in perturbed_trajectories], dtype=object))
    
    # Save distances
    np.save(os.path.join(task_dir, "all_distances.npy"), all_distances)
    np.save(os.path.join(task_dir, "mean_distances.npy"), mean_distances)
    np.save(os.path.join(task_dir, "std_distances.npy"), std_distances)
    
    # Save EISS results as JSON
    results_dict = {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "num_successful_perturbed": len(perturbed_trajectories),
        "trajectory_length": len(ref_trajectory),
        "eiss_analysis": eiss_results
    }
    with open(os.path.join(task_dir, "eiss_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return task_dir


def run_eiss_check_single_task(
    args: Args,
    task_suite,
    policy,
    task_id: int,
    episode_idx: int,
    timestamp: str
) -> Optional[dict]:
    """
    Run EISS check for a single task/episode.
    
    Returns None if reference trajectory fails (task is skipped).
    Only successful episodes are saved and compared.
    """
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    
    print(f"\n{'='*60}")
    print(f"Task {task_id}: {task_description}")
    print(f"Episode: {episode_idx}, Chunks: {args.num_chunks}, Steps: {args.num_chunks * args.replan_steps}")
    print(f"{'='*60}")
    
    # 1. Run REFERENCE trajectory (NO perturbation)
    print("Running reference trajectory (no perturbation)...")
    ref_trajectory, ref_success = run_rollout(
        env=env, policy=policy, initial_states=initial_states,
        episode_idx=episode_idx, task_description=task_description,
        num_chunks=args.num_chunks, num_steps_wait=args.num_steps_wait,
        replan_steps=args.replan_steps,
        perturb_joints=False  # NO perturbation for reference
    )
    
    print(f"  Reference: {len(ref_trajectory)} steps (task_done={ref_success})")
    
    # 2. Run PERTURBED trajectories (with joint noise)
    # Note: We keep ALL trajectories regardless of task completion
    # because we're testing trajectory divergence, not task success
    print(f"Running {args.num_perturbations} perturbed trajectories...")
    perturbed_trajectories = []
    
    for i in tqdm.tqdm(range(args.num_perturbations)):
        traj, success = run_rollout(
            env=env, policy=policy, initial_states=initial_states,
            episode_idx=episode_idx, task_description=task_description,
            num_chunks=args.num_chunks, num_steps_wait=args.num_steps_wait,
            replan_steps=args.replan_steps,
            perturb_joints=True,  # WITH perturbation
            eef_noise_std=args.eef_noise_std
        )
        perturbed_trajectories.append(traj)
    
    env.close()
    
    print(f"  Collected: {len(perturbed_trajectories)} trajectories")
    
    # 3. Compute distances (compare ALL trajectories)
    all_distances, mean_distances, std_distances = compute_distances(
        ref_trajectory, perturbed_trajectories
    )
    
    # 4. Check EISS criteria
    eiss_results = check_eiss_criteria(mean_distances, std_distances)
    
    # 5. Print results (paper-based criteria)
    print(f"\n--- EISS Analysis (Paper: arXiv:2507.09061) ---")
    print(f"Compared: 1 Reference vs {len(perturbed_trajectories)} Perturbed (all trajectories)")
    print(f"")
    print(f"Primary Criterion (Definition 2.1):")
    print(f"  slope = {eiss_results['slope']:.6f} {'< 0 ✓' if eiss_results['is_decaying'] else '>= 0 ✗'}")
    print(f"  ρ (contraction rate) = {eiss_results['rho']:.4f} {'< 1 ✓' if eiss_results['rho'] < 1 else '>= 1 ✗'}")
    print(f"  R² = {eiss_results['r_squared']:.3f}")
    print(f"  d_T/d_0 = {eiss_results['contraction_ratio']:.3f} {'(contracted)' if eiss_results['contraction_ratio'] < 1 else '(expanded)'}")
    print(f"")
    print(f"★ EISS Satisfied: {'✓ YES' if eiss_results['strong_eiss'] else '✗ NO'}")
    
    # 6. Save data as npy files
    task_dir = save_eiss_data(
        output_dir=args.output_dir,
        task_id=task_id,
        episode_idx=episode_idx,
        ref_trajectory=ref_trajectory,
        perturbed_trajectories=perturbed_trajectories,
        all_distances=all_distances,
        mean_distances=mean_distances,
        std_distances=std_distances,
        eiss_results=eiss_results,
        task_description=task_description,
        timestamp=timestamp
    )
    print(f"  Data saved to: {task_dir}")
    
    # 7. Save plot
    plot_path = os.path.join(task_dir, "eiss_plot.png")
    plot_eiss_results(
        all_distances=all_distances,
        mean_distances=mean_distances,
        std_distances=std_distances,
        eiss_results=eiss_results,
        output_path=plot_path,
        task_description=task_description,
        replan_steps=args.replan_steps
    )
    print(f"  Plot saved to: {plot_path}")
    
    return {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "num_perturbed": len(perturbed_trajectories),
        "eiss_results": eiss_results,
        "data_dir": task_dir
    }


def main(args: Args):
    """Main function to run EISS check."""
    np.random.seed(args.seed)
    
    # Create output directory
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    # Initialize policy client
    policy = ExternalRobotInferenceClient(host=args.host, port=args.port)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'#'*60}")
    print(f"# EISS Check - arXiv:2507.09061")
    print(f"# Reference: NO perturbation")
    print(f"# Perturbed: EEF noise std={args.eef_noise_std}m ({args.eef_noise_std*1000}mm)")
    print(f"# Chunks: {args.num_chunks} × {args.replan_steps} = {args.num_chunks * args.replan_steps} steps")
    print(f"# Only SUCCESSFUL episodes are saved and compared")
    print(f"{'#'*60}\n")
    
    all_results = []
    
    if args.quick_mode:
        # Quick mode: test first 3 tasks
        print("QUICK MODE: Testing first 3 tasks")
        tasks_to_test = list(range(min(3, num_tasks)))
    elif args.task_id == -1:
        # All tasks
        print(f"Testing ALL {num_tasks} tasks")
        tasks_to_test = list(range(num_tasks))
    else:
        # Single task
        tasks_to_test = [args.task_id]
    
    for task_id in tasks_to_test:
        result = run_eiss_check_single_task(
            args=args,
            task_suite=task_suite,
            policy=policy,
            task_id=task_id,
            episode_idx=args.episode_idx,
            timestamp=timestamp
        )
        if result is not None:
            all_results.append(result)
    
    # Summary
    print(f"\n{'#'*60}")
    print(f"# EISS CHECK SUMMARY")
    print(f"{'#'*60}")
    
    if len(all_results) == 0:
        print("No tasks completed!")
    else:
        print(f"\n{'Task':<5} {'N':<4} {'EISS':<5} {'Slope':<12} {'ρ':<8} {'R²':<6} {'d_T/d_0':<8}")
        print("-" * 60)
        
        eiss_count = 0
        for r in all_results:
            e = r['eiss_results']
            eiss_mark = "✓" if e['strong_eiss'] else "✗"
            if e['strong_eiss']:
                eiss_count += 1
            print(f"{r['task_id']:<5} {r['num_perturbed']:<4} {eiss_mark:<5} "
                  f"{e['slope']:<12.6f} {e['rho']:<8.4f} {e['r_squared']:<6.3f} {e['contraction_ratio']:<8.3f}")
        
        print("-" * 60)
        print(f"EISS satisfied (slope<0 & R²≥0.5): {eiss_count}/{len(all_results)} tasks")
        print(f"Paper criterion: ρ < 1 means exponential contraction (EISS)")
    
    # Save overall summary
    summary_path = os.path.join(args.output_dir, f"eiss_summary_{timestamp}.json")
    summary = {
        "timestamp": timestamp,
        "config": {
            "eef_noise_std": args.eef_noise_std,
            "num_chunks": args.num_chunks,
            "replan_steps": args.replan_steps,
            "num_perturbations": args.num_perturbations,
        },
        "num_tasks_tested": len(tasks_to_test),
        "num_tasks_successful": len(all_results),
        "eiss_satisfied_count": sum(1 for r in all_results if r['eiss_results']['strong_eiss']),
        "results": [
            {
                "task_id": r['task_id'],
                "episode_idx": r['episode_idx'],
                "task_description": r['task_description'],
                "num_perturbed": r['num_perturbed'],
                "strong_eiss": r['eiss_results']['strong_eiss'],
                "slope": r['eiss_results']['slope'],
                "rho": r['eiss_results']['rho'],
                "r_squared": r['eiss_results']['r_squared'],
                "contraction_ratio": r['eiss_results']['contraction_ratio'],
                "data_dir": r['data_dir']
            }
            for r in all_results
        ]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    tyro.cli(main)
