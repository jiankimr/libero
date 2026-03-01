#!/usr/bin/env python3
"""
EISS Results Visualization for PPT

Generates three types of plots:
1. Perturbation method explanation (diagram)
2. Distance over time (d_t vs t) with linear and log scale
3. EEF position trajectories (3D and 2D projections)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

# Set style for PPT
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})


def load_eiss_data(data_dir):
    """Load all EISS data from a task directory."""
    data = {}
    
    # Load numpy files
    data['ref_trajectory'] = np.load(os.path.join(data_dir, 'ref_trajectory.npy'))
    data['perturbed_trajectories'] = np.load(
        os.path.join(data_dir, 'perturbed_trajectories.npy'), 
        allow_pickle=True
    )
    data['all_distances'] = np.load(os.path.join(data_dir, 'all_distances.npy'))
    data['mean_distances'] = np.load(os.path.join(data_dir, 'mean_distances.npy'))
    data['std_distances'] = np.load(os.path.join(data_dir, 'std_distances.npy'))
    
    # Load JSON results
    with open(os.path.join(data_dir, 'eiss_results.json'), 'r') as f:
        data['results'] = json.load(f)
    
    return data


def plot_perturbation_method(output_path, eef_noise_std=0.001):
    """
    Plot 1: Perturbation method explanation diagram
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Conceptual diagram
    ax1 = axes[0]
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(-1, 5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('EEF Space Perturbation Method', fontsize=14, fontweight='bold')
    
    # Draw robot arm (simplified)
    ax1.plot([0, 1.5, 3], [0, 2, 3], 'b-', linewidth=8, solid_capstyle='round', label='Robot Arm')
    ax1.scatter([3], [3], s=200, c='red', marker='o', zorder=5, label='EEF (Reference)')
    
    # Draw perturbation sphere
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.5
    ax1.plot(3 + r*np.cos(theta), 3 + r*np.sin(theta), 'g--', linewidth=2, alpha=0.7)
    ax1.fill(3 + r*np.cos(theta), 3 + r*np.sin(theta), alpha=0.1, color='green')
    
    # Draw perturbed positions
    np.random.seed(42)
    for i in range(5):
        dx, dy = np.random.randn(2) * 0.3
        ax1.scatter([3 + dx], [3 + dy], s=100, c='orange', marker='x', zorder=5)
    ax1.scatter([], [], s=100, c='orange', marker='x', label='Perturbed EEF')
    
    # Annotations
    ax1.annotate('σ = 1mm', xy=(3.5, 3.5), fontsize=11, color='green')
    ax1.annotate('Jacobian IK\n→ Joint angles', xy=(1, 3.5), fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax1.legend(loc='lower right')
    
    # Right: Mathematical formulation
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('Mathematical Formulation', fontsize=14, fontweight='bold')
    
    text = """
    EEF Perturbation Method:
    
    1. Generate EEF noise:
       Δx_eef ~ N(0, σ²I)     where σ = 0.001m (1mm)
    
    2. Convert to joint space via Jacobian:
       J = [J_pos; J_rot]     (6 × n_joints)
       Δq = J⁺ · Δx_eef       (pseudo-inverse)
    
    3. Apply to initial state:
       q_perturbed = q_init + Δq
    
    4. Compare trajectories:
       d_t = ‖x_t^ref - x_t^perturbed‖
    
    EISS Criterion (Paper Definition 2.1):
       d_t ≤ ρᵗ · d_0,  where ρ < 1
       ⟹ slope of log(d_t) < 0
    """
    ax2.text(0.1, 0.9, text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_distance_over_time(data, output_path, task_name="Task"):
    """
    Plot 2: Distance over time (linear and log scale)
    """
    mean_d = data['mean_distances']
    std_d = data['std_distances']
    all_d = data['all_distances']
    results = data['results']['eiss_analysis']
    
    timesteps = np.arange(len(mean_d))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Linear scale
    ax1 = axes[0]
    
    # Plot individual trajectories (faint)
    for i, d in enumerate(all_d):
        ax1.plot(timesteps, d[:len(timesteps)], 'gray', alpha=0.2, linewidth=0.5)
    
    # Plot mean with std band
    ax1.plot(timesteps, mean_d, 'b-', linewidth=2, label='Mean distance')
    ax1.fill_between(timesteps, mean_d - std_d, mean_d + std_d, 
                      alpha=0.3, color='blue', label='±1 std')
    
    # Mark chunk boundaries
    replan_steps = 16
    for c in range(1, len(timesteps) // replan_steps + 1):
        ax1.axvline(x=c * replan_steps, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('Time step t')
    ax1.set_ylabel('Distance d_t = ||x_t - x\'_t||')
    ax1.set_title(f'State Distance Over Time (Linear Scale)\n{task_name}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right: Log scale with linear fit
    ax2 = axes[1]
    
    eps = 1e-10
    log_mean = np.log(mean_d + eps)
    
    ax2.plot(timesteps, log_mean, 'b-', linewidth=2, label='log(Mean distance)')
    
    # Linear fit
    slope = results['slope']
    intercept = results['intercept']
    fitted = slope * timesteps + intercept
    ax2.plot(timesteps, fitted, 'r--', linewidth=2, 
             label=f'Linear fit: slope={slope:.4f}')
    
    # Annotations
    rho = results['rho']
    r2 = results['r_squared']
    eiss = results['strong_eiss']
    
    info_text = f"ρ = {rho:.4f}\nR² = {r2:.3f}\nEISS: {'✓' if eiss else '✗'}"
    ax2.text(0.95, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax2.set_xlabel('Time step t')
    ax2.set_ylabel('log(d_t)')
    ax2.set_title(f'State Distance Over Time (Log Scale)\nslope {"< 0 ✓" if slope < 0 else "> 0 ✗"} EISS')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_eef_trajectories(data, output_path, task_name="Task"):
    """
    Plot 3: EEF position trajectories (3D and 2D projections)
    """
    ref = data['ref_trajectory']  # (T, 8) - [eef_pos(3), eef_ori(3), gripper(2)]
    perturbed = data['perturbed_trajectories']  # array of (T, 8)
    
    # Extract EEF positions (first 3 dimensions)
    ref_pos = ref[:, :3]  # (T, 3)
    
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Reference trajectory
    ax1.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], 
             'b-', linewidth=2, label='Reference')
    ax1.scatter(ref_pos[0, 0], ref_pos[0, 1], ref_pos[0, 2], 
                c='green', s=100, marker='o', label='Start')
    ax1.scatter(ref_pos[-1, 0], ref_pos[-1, 1], ref_pos[-1, 2], 
                c='red', s=100, marker='x', label='End')
    
    # Perturbed trajectories (sample a few)
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, min(5, len(perturbed))))
    for i, traj in enumerate(perturbed[:5]):
        traj = np.array(traj)
        if len(traj) > 0:
            pos = traj[:, :3]
            ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                     color=colors[i], alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D EEF Trajectories')
    ax1.legend(loc='upper left')
    
    # Plot 2: XY projection
    ax2 = fig.add_subplot(132)
    ax2.plot(ref_pos[:, 0], ref_pos[:, 1], 'b-', linewidth=2, label='Reference')
    ax2.scatter(ref_pos[0, 0], ref_pos[0, 1], c='green', s=100, marker='o')
    ax2.scatter(ref_pos[-1, 0], ref_pos[-1, 1], c='red', s=100, marker='x')
    
    for i, traj in enumerate(perturbed[:5]):
        traj = np.array(traj)
        if len(traj) > 0:
            pos = traj[:, :3]
            ax2.plot(pos[:, 0], pos[:, 1], color=colors[i], alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection (Top View)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend()
    
    # Plot 3: XZ projection
    ax3 = fig.add_subplot(133)
    ax3.plot(ref_pos[:, 0], ref_pos[:, 2], 'b-', linewidth=2, label='Reference')
    ax3.scatter(ref_pos[0, 0], ref_pos[0, 2], c='green', s=100, marker='o')
    ax3.scatter(ref_pos[-1, 0], ref_pos[-1, 2], c='red', s=100, marker='x')
    
    for i, traj in enumerate(perturbed[:5]):
        traj = np.array(traj)
        if len(traj) > 0:
            pos = traj[:, :3]
            ax3.plot(pos[:, 0], pos[:, 2], color=colors[i], alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Projection (Side View)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle(f'EEF Position Trajectories: {task_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_initial_perturbation(data, output_path, task_name="Task"):
    """
    Plot 4: Initial perturbation visualization (d_0 distribution)
    """
    ref = data['ref_trajectory']
    perturbed = data['perturbed_trajectories']
    
    # Get initial states
    ref_init = ref[0, :3]  # EEF position at t=0
    
    perturbed_inits = []
    for traj in perturbed:
        traj = np.array(traj)
        if len(traj) > 0:
            perturbed_inits.append(traj[0, :3])
    perturbed_inits = np.array(perturbed_inits)
    
    # Compute initial distances
    d_0_values = np.linalg.norm(perturbed_inits - ref_init, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Histogram of d_0
    ax1 = axes[0]
    ax1.hist(d_0_values * 1000, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(d_0_values) * 1000, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(d_0_values)*1000:.2f} mm')
    ax1.set_xlabel('Initial Distance d_0 (mm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Initial Perturbation Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Scatter of initial positions
    ax2 = axes[1]
    ax2.scatter(ref_init[0], ref_init[1], c='blue', s=200, marker='o', 
                label='Reference', zorder=5)
    ax2.scatter(perturbed_inits[:, 0], perturbed_inits[:, 1], c='orange', 
                s=50, marker='x', alpha=0.7, label='Perturbed')
    
    # Draw circle showing perturbation range
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.mean(d_0_values)
    ax2.plot(ref_init[0] + r*np.cos(theta), ref_init[1] + r*np.sin(theta), 
             'g--', linewidth=2, alpha=0.5, label=f'Mean radius: {r*1000:.2f}mm')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Initial EEF Positions (XY)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle(f'Initial Perturbation Analysis: {task_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot EISS results for PPT')
    parser.add_argument('--data-dir', type=str, 
                        default='/home/taywonmin/rsec/repos/rsec/LIBERO/eiss_check/task00_ep00_20260130_130830',
                        help='Path to EISS data directory')
    parser.add_argument('--output-dir', type=str, 
                        default='/home/taywonmin/rsec/repos/rsec/LIBERO/plots/eiss',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get task name from directory
    task_name = os.path.basename(args.data_dir)
    
    print(f"\n=== EISS Visualization ===")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    
    # Generate Plot 1: Perturbation method
    plot_perturbation_method(
        os.path.join(args.output_dir, 'eiss_01_perturbation_method.png')
    )
    
    # Load data
    data = load_eiss_data(args.data_dir)
    
    # Generate Plot 2: Distance over time
    plot_distance_over_time(
        data, 
        os.path.join(args.output_dir, f'eiss_02_distance_over_time_{task_name}.png'),
        task_name
    )
    
    # Generate Plot 3: EEF trajectories
    plot_eef_trajectories(
        data,
        os.path.join(args.output_dir, f'eiss_03_eef_trajectories_{task_name}.png'),
        task_name
    )
    
    # Generate Plot 4: Initial perturbation
    plot_initial_perturbation(
        data,
        os.path.join(args.output_dir, f'eiss_04_initial_perturbation_{task_name}.png'),
        task_name
    )
    
    print(f"\n✓ All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()


