#!/usr/bin/env python3
"""
Torque Trajectory Visualization per Joints
- Shows torque time series for each joint (0-6)
- Compares baseline vs noisy conditions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

# Set style
try:
    plt.style.use('seaborn-whitegrid')
except:
    pass

plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9


def find_torque_files(analysis_dir, task_pattern=None):
    """Find torque_current npy files in analysis directory."""
    pattern = f"{analysis_dir}/torque_current_*.npy"
    files = glob(pattern)
    if task_pattern:
        files = [f for f in files if task_pattern in f]
    return sorted(files)


def load_torque_data(filepath):
    """Load torque data from npy file."""
    data = np.load(filepath)  # Shape: (timesteps, 7) for 7 joints
    return data


def plot_torque_trajectory_comparison(baseline_dir, noisy_dir, task_name, output_path, 
                                       baseline_label="Baseline", noisy_label="Noisy"):
    """Create side-by-side comparison of torque trajectories."""
    
    # Find matching files
    baseline_files = find_torque_files(baseline_dir, task_name)
    noisy_files = find_torque_files(noisy_dir, task_name)
    
    if not baseline_files or not noisy_files:
        print(f"Warning: No matching files for task '{task_name}'")
        return None
    
    # Load first matching file from each
    baseline_data = load_torque_data(baseline_files[0])
    noisy_data = load_torque_data(noisy_files[0])
    
    # Create figure with 2 rows x 7 columns
    fig, axes = plt.subplots(2, 7, figsize=(18, 6))
    fig.suptitle(f'Torque Trajectory per Joints: {task_name[:50]}...', 
                 fontsize=14, fontweight='bold', color='#3366cc', y=1.02)
    
    joint_names = [f'Joint {i}' for i in range(7)]
    
    for col in range(7):
        # Baseline row
        ax_baseline = axes[0, col]
        if col < baseline_data.shape[1]:
            timesteps = np.arange(len(baseline_data))
            ax_baseline.plot(timesteps, baseline_data[:, col], color='#e8a83c', linewidth=0.8)
        ax_baseline.set_title(joint_names[col])
        ax_baseline.set_ylim(-60, 60)
        ax_baseline.set_ylabel('Torque [N·m]' if col == 0 else '')
        if col == 0:
            ax_baseline.text(-0.35, 0.5, baseline_label, transform=ax_baseline.transAxes, 
                           fontsize=12, fontweight='bold', va='center', ha='right')
        
        # Noisy row
        ax_noisy = axes[1, col]
        if col < noisy_data.shape[1]:
            timesteps = np.arange(len(noisy_data))
            ax_noisy.plot(timesteps, noisy_data[:, col], color='#e8a83c', linewidth=0.8)
        ax_noisy.set_ylim(-60, 60)
        ax_noisy.set_xlabel('Time step')
        ax_noisy.set_ylabel('Torque [N·m]' if col == 0 else '')
        if col == 0:
            ax_noisy.text(-0.35, 0.5, noisy_label, transform=ax_noisy.transAxes,
                         fontsize=12, fontweight='bold', va='center', ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_torque_multi_noise_comparison(analysis_dirs, noise_labels, task_name, output_path):
    """Create multi-row comparison for different noise levels."""
    
    n_conditions = len(analysis_dirs)
    
    fig, axes = plt.subplots(n_conditions, 7, figsize=(18, 3 * n_conditions))
    fig.suptitle(f'Torque Trajectory per Joints\n{task_name[:60]}...', 
                 fontsize=14, fontweight='bold', color='#3366cc', y=1.02)
    
    joint_names = [f'Joint {i}' for i in range(7)]
    
    for row, (analysis_dir, label) in enumerate(zip(analysis_dirs, noise_labels)):
        files = find_torque_files(analysis_dir, task_name)
        if not files:
            continue
            
        data = load_torque_data(files[0])
        
        for col in range(7):
            if n_conditions == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            if col < data.shape[1]:
                timesteps = np.arange(len(data))
                ax.plot(timesteps, data[:, col], color='#e8a83c', linewidth=0.8)
            
            ax.set_ylim(-60, 60)
            
            if row == 0:
                ax.set_title(joint_names[col])
            if row == n_conditions - 1:
                ax.set_xlabel('Time step')
            if col == 0:
                ax.set_ylabel('Torque [N·m]')
                ax.text(-0.4, 0.5, label, transform=ax.transAxes,
                       fontsize=11, fontweight='bold', va='center', ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_all_noise_levels_torque(output_dir):
    """Plot torque trajectories for all noise levels."""
    
    base_path = Path('/workspace/repos/rsec/LIBERO/analysis')
    
    # Define analysis directories for each noise level
    # NOTE: 0.0 excluded - rollout_noise03 model already learned noisy behavior
    noise_configs = {
        # '0.0 (baseline)': excluded - meaningless for comparison
        '0.1': 'analysis_libero_10_20251206_052507_noise_01000_dim_action.x_rollout_noise_01_actionx',
        '0.2': 'analysis_libero_10_20251206_052507_noise_02000_dim_action.x_rollout_noise_02_actionx',
        '0.3': 'analysis_libero_10_20251202_171746_noise_03000_dim_action.x_clean_noise',
        '0.4': 'analysis_libero_10_20251206_052507_noise_04000_dim_action.x_rollout_noise_04_actionx',
        '0.5': 'analysis_libero_10_20251206_052507_noise_05000_dim_action.x_rollout_noise_05_actionx',
        '0.6': 'analysis_libero_10_20251206_052507_noise_06000_dim_action.x_rollout_noise_06_actionx',
    }
    
    # Find a common task across all directories
    # Get task list from first available config (0.1)
    first_config = list(noise_configs.values())[0]
    baseline_dir = base_path / first_config
    baseline_files = find_torque_files(str(baseline_dir))
    
    if not baseline_files:
        print("No baseline torque files found!")
        return
    
    # Extract task names from filenames
    task_names = set()
    for f in baseline_files:
        # Extract task name from filename like: torque_current_task_name_0_success.npy
        fname = os.path.basename(f)
        parts = fname.replace('torque_current_', '').replace('.npy', '')
        # Remove the trial number and status suffix
        task_parts = parts.rsplit('_', 2)[0]  # Remove _N_success
        task_names.add(task_parts)
    
    print(f"Found {len(task_names)} unique tasks")
    
    # Select a few representative tasks
    sample_tasks = list(task_names)[:3]
    
    for task in sample_tasks:
        analysis_dirs = []
        noise_labels = []
        
        for label, dirname in noise_configs.items():
            dir_path = base_path / dirname
            if dir_path.exists():
                files = find_torque_files(str(dir_path), task)
                if files:
                    analysis_dirs.append(str(dir_path))
                    noise_labels.append(f'Noise {label}')
        
        if len(analysis_dirs) >= 2:
            safe_task = task.replace('/', '_')[:30]
            output_path = os.path.join(output_dir, f'torque_trajectory_{safe_task}.png')
            plot_torque_multi_noise_comparison(analysis_dirs, noise_labels, task, output_path)


def plot_comparison_clean_vs_noisy_with_noise(output_dir):
    """Compare Clean vs Noisy models, both with and without action noise."""
    
    base_path = Path('/workspace/repos/rsec/LIBERO/analysis')
    
    configs = [
        ('Clean (no noise)', 'analysis_libero_10_20251202_170035_noise_00000_clean'),
        ('Noisy (no noise)', 'analysis_libero_10_20251202_170044_noise_00000_noisy'),
        ('Clean (0.3, x)', 'analysis_libero_10_20251202_171746_noise_03000_dim_action.x_clean_noise'),
        ('Noisy (0.3, x)', 'analysis_libero_10_20251202_171749_noise_03000_dim_action.x_noisy_noise'),
    ]
    
    # Find common task
    first_dir = base_path / configs[0][1]
    files = find_torque_files(str(first_dir))
    
    if not files:
        print("No files found in first directory!")
        return
    
    # Get task from first file
    fname = os.path.basename(files[0])
    task = fname.replace('torque_current_', '').rsplit('_', 2)[0]
    
    analysis_dirs = []
    noise_labels = []
    
    for label, dirname in configs:
        dir_path = base_path / dirname
        if dir_path.exists():
            task_files = find_torque_files(str(dir_path), task)
            if task_files:
                analysis_dirs.append(str(dir_path))
                noise_labels.append(label)
    
    if len(analysis_dirs) >= 2:
        output_path = os.path.join(output_dir, 'torque_trajectory_clean_vs_noisy.png')
        plot_torque_multi_noise_comparison(analysis_dirs, noise_labels, task, output_path)


def main():
    output_dir = '/workspace/repos/rsec/LIBERO/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Torque Trajectory Visualization")
    print("=" * 60)
    
    # Plot comparison: Clean vs Noisy with/without noise
    print("\n1. Plotting Clean vs Noisy comparison...")
    plot_comparison_clean_vs_noisy_with_noise(output_dir)
    
    # Plot all noise levels
    print("\n2. Plotting all noise levels...")
    plot_all_noise_levels_torque(output_dir)
    
    print("\n" + "=" * 60)
    print(f"All torque plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

