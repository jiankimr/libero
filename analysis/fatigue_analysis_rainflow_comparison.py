#!/usr/bin/env python3
"""
Fatigue Analysis: Rainflow Counting Comparison
===============================================

This script compares two rainflow counting implementations:
- rainflow: 3-point algorithm (ASTM-like)
- fatpack: 4-point algorithm (Amzallag-like)

The 4-point algorithm (fatpack) tends to preserve more large-range cycles
because it compares inner vs outer ranges and keeps the outer envelope,
while the 3-point algorithm may close smaller cycles earlier.

Usage:
    python fatigue_analysis_rainflow_comparison.py [data_dir] [label]
    
Examples:
    python fatigue_analysis_rainflow_comparison.py  # defaults to noisy
    python fatigue_analysis_rainflow_comparison.py analysis_libero_10_20251202_170035_noise_00000_clean clean
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rainflow
import fatpack

# Configuration - can be overridden by command line arguments
DEFAULT_DATA_DIR = "analysis_libero_10_20251202_170044_noise_00000_noisy"
DEFAULT_LABEL = "noisy"
TORQUE_FILE = "torque_current_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_0_success.npy"
FOCUS_JOINT = 5  # Joint index to analyze (0-indexed)
NUM_JOINTS = 7

# Fatigue damage parameters (S-N curve)
# For robot joints/mechanical components, typical values:
# N = C / (S_a^m)  where N = cycles to failure
SN_EXPONENT = 3.0  # m: Wöhler exponent (3-5 for steel, 5-12 for aluminum)
SN_CONSTANT = 1e8  # C: Material constant (adjusted for torque in N·m)


def load_torque_data(filepath):
    """
    Load torque data from .npy file.
    
    Parameters
    ----------
    filepath : str
        Path to the .npy file
        
    Returns
    -------
    data : np.ndarray
        Array of shape (T, J) where T is time steps and J is number of joints
    """
    data = np.load(filepath)
    print(f"Loaded torque data: shape = {data.shape}")
    print(f"  Time steps: {data.shape[0]}")
    print(f"  Joints: {data.shape[1]}")
    return data


def plot_raw_torque(torque_signal, joint_idx, filename):
    """
    Plot raw torque signal over time.
    
    Parameters
    ----------
    torque_signal : np.ndarray
        1D array of torque values
    joint_idx : int
        Joint index for labeling
    filename : str
        Filename for the title
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(torque_signal, linewidth=0.8, color='steelblue')
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Torque (N·m)', fontsize=12)
    ax.set_title(f'Raw Torque Signal - {filename} - Joint {joint_idx}', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'raw_torque_joint_{joint_idx}.png', dpi=150)
    plt.show()
    print(f"✓ Saved: raw_torque_joint_{joint_idx}.png")


def plot_raw_torque_labeled(torque_signal, joint_idx, filename, label):
    """Plot raw torque signal with label in filename."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(torque_signal, linewidth=0.8, color='steelblue')
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Torque (N·m)', fontsize=12)
    ax.set_title(f'Raw Torque Signal [{label.upper()}] - Joint {joint_idx}', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = f'raw_torque_joint_{joint_idx}_{label}.png'
    plt.savefig(outfile, dpi=150)
    plt.show()
    print(f"✓ Saved: {outfile}")


def extract_turning_points(signal):
    """
    Extract turning points (local extrema) from a signal.
    This is required for rainflow counting algorithms.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values
        
    Returns
    -------
    turning_points : np.ndarray
        Array of turning point values
    turning_indices : np.ndarray
        Array of indices where turning points occur
    """
    # Find local maxima and minima
    turning_indices = [0]  # Always start with the first point
    
    for i in range(1, len(signal) - 1):
        # Local maximum
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            turning_indices.append(i)
        # Local minimum
        elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            turning_indices.append(i)
    
    turning_indices.append(len(signal) - 1)  # Always end with the last point
    turning_indices = np.array(turning_indices)
    turning_points = signal[turning_indices]
    
    print(f"  Extracted {len(turning_points)} turning points from {len(signal)} data points")
    return turning_points, turning_indices


def compute_cycles_rainflow(signal):
    """
    Compute rainflow cycles using the 'rainflow' package (3-point ASTM-like algorithm).
    
    Parameters
    ----------
    signal : np.ndarray
        1D torque signal
        
    Returns
    -------
    df_cycles : pd.DataFrame
        DataFrame with columns: range, mean, amplitude, count
    """
    print("\n=== Rainflow Package (3-point ASTM-like) ===")
    
    # Extract turning points for rainflow algorithm
    turning_points, turning_indices = extract_turning_points(signal)
    
    # Run rainflow counting using extract_cycles for detailed info
    cycles = list(rainflow.extract_cycles(turning_points))
    
    # Process cycles
    cycle_data = []
    for cycle_range, cycle_mean, cycle_count, i_start, i_end in cycles:
        amplitude = cycle_range / 2.0
        cycle_data.append({
            'range': cycle_range,
            'mean': cycle_mean,
            'amplitude': amplitude,
            'count': cycle_count,
            'i_start': turning_indices[i_start] if i_start < len(turning_indices) else None,
            'i_end': turning_indices[i_end] if i_end < len(turning_indices) else None,
        })
    
    df_cycles = pd.DataFrame(cycle_data)
    
    print(f"  Total cycles detected: {len(df_cycles)}")
    print(f"  Total cycle count (including 0.5 for half): {df_cycles['count'].sum():.1f}")
    if len(df_cycles) > 0:
        print(f"  Max range: {df_cycles['range'].max():.4f} N·m")
        print(f"  Mean range: {df_cycles['range'].mean():.4f} N·m")
    
    return df_cycles


def compute_cycles_fatpack(signal):
    """
    Compute rainflow cycles using the 'fatpack' package (4-point Amzallag-like algorithm).
    
    Parameters
    ----------
    signal : np.ndarray
        1D torque signal
        
    Returns
    -------
    df_cycles : pd.DataFrame
        DataFrame with columns: range, mean, amplitude, count, i_start, i_end
    """
    print("\n=== Fatpack Package (4-point Amzallag-like) ===")
    
    # Extract turning points
    turning_points, turning_indices = extract_turning_points(signal)
    
    # Run rainflow counting
    # fatpack.find_rainflow_cycles returns (Nx2) array where each row is [start, end] values
    cycles, residue = fatpack.find_rainflow_cycles(turning_points)
    
    # Process cycles and match them to indices
    cycle_data = []
    for cycle in cycles:
        start_val, end_val = cycle[0], cycle[1]
        cycle_range = abs(end_val - start_val)
        cycle_mean = (start_val + end_val) / 2.0
        amplitude = cycle_range / 2.0
        
        # Find indices of these values in turning_points
        # Note: there may be multiple occurrences, we take the first match
        start_idx = np.where(turning_points == start_val)[0]
        end_idx = np.where(turning_points == end_val)[0]
        
        i_start = turning_indices[start_idx[0]] if len(start_idx) > 0 else None
        i_end = turning_indices[end_idx[0]] if len(end_idx) > 0 else None
        
        # fatpack returns full cycles (count = 1.0)
        cycle_data.append({
            'range': cycle_range,
            'mean': cycle_mean,
            'amplitude': amplitude,
            'count': 1.0,
            'i_start': i_start,
            'i_end': i_end,
        })
    
    df_cycles = pd.DataFrame(cycle_data)
    
    print(f"  Total cycles detected: {len(df_cycles)}")
    print(f"  Total cycle count: {df_cycles['count'].sum():.1f}")
    print(f"  Residue length: {len(residue)}")
    if len(df_cycles) > 0:
        print(f"  Max range: {df_cycles['range'].max():.4f} N·m")
        print(f"  Mean range: {df_cycles['range'].mean():.4f} N·m")
    
    return df_cycles


def plot_histogram_comparison(df_rainflow, df_fatpack):
    """
    Plot overlaid histograms of cycle ranges for both methods.
    
    Parameters
    ----------
    df_rainflow : pd.DataFrame
        Cycles from rainflow package
    df_fatpack : pd.DataFrame
        Cycles from fatpack package
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine bin edges
    all_ranges = np.concatenate([df_rainflow['range'].values, df_fatpack['range'].values])
    bins = np.linspace(all_ranges.min(), all_ranges.max(), 30)
    
    # Plot histograms
    ax.hist(df_rainflow['range'], bins=bins, alpha=0.6, label='three_point (ASTM-like)', 
            color='steelblue', edgecolor='black', linewidth=0.5)
    ax.hist(df_fatpack['range'], bins=bins, alpha=0.6, label='four_point (Amzallag-like)', 
            color='coral', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Cycle range (N·m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Comparison of Cycle Ranges: Rainflow vs Fatpack', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('histogram_comparison.png', dpi=150)
    plt.show()
    print("✓ Saved: histogram_comparison.png")


def plot_histogram_comparison_labeled(df_rainflow, df_fatpack, label):
    """Plot histogram comparison with label in filename."""
    fig, ax = plt.subplots(figsize=(10, 6))
    all_ranges = np.concatenate([df_rainflow['range'].values, df_fatpack['range'].values])
    bins = np.linspace(all_ranges.min(), all_ranges.max(), 30)
    ax.hist(df_rainflow['range'], bins=bins, alpha=0.6, label='three_point (ASTM-like)', 
            color='steelblue', edgecolor='black', linewidth=0.5)
    ax.hist(df_fatpack['range'], bins=bins, alpha=0.6, label='four_point (Amzallag-like)', 
            color='coral', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Cycle range (N·m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Comparison of Cycle Ranges [{label.upper()}]: Rainflow vs Fatpack', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = f'histogram_comparison_{label}.png'
    plt.savefig(outfile, dpi=150)
    plt.show()
    print(f"✓ Saved: {outfile}")


def plot_mean_amplitude_scatter(df_rainflow, df_fatpack):
    """
    Plot mean vs amplitude scatter plots for both methods.
    
    Parameters
    ----------
    df_rainflow : pd.DataFrame
        Cycles from rainflow package
    df_fatpack : pd.DataFrame
        Cycles from fatpack package
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rainflow scatter
    axes[0].scatter(df_rainflow['mean'], df_rainflow['amplitude'], 
                   alpha=0.6, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('Mean (S_m) [N·m]', fontsize=11)
    axes[0].set_ylabel('Amplitude (S_a) [N·m]', fontsize=11)
    axes[0].set_title('Rainflow (3-point ASTM-like)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Fatpack scatter
    axes[1].scatter(df_fatpack['mean'], df_fatpack['amplitude'], 
                   alpha=0.6, s=30, color='coral', edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('Mean (S_m) [N·m]', fontsize=11)
    axes[1].set_ylabel('Amplitude (S_a) [N·m]', fontsize=11)
    axes[1].set_title('Fatpack (4-point Amzallag-like)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mean_amplitude_scatter.png', dpi=150)
    plt.show()
    print("✓ Saved: mean_amplitude_scatter.png")


def plot_mean_amplitude_scatter_labeled(df_rainflow, df_fatpack, label):
    """Plot mean vs amplitude scatter with label in filename."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(df_rainflow['mean'], df_rainflow['amplitude'], 
                   alpha=0.6, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('Mean (S_m) [N·m]', fontsize=11)
    axes[0].set_ylabel('Amplitude (S_a) [N·m]', fontsize=11)
    axes[0].set_title(f'Rainflow (3-point) [{label.upper()}]', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(df_fatpack['mean'], df_fatpack['amplitude'], 
                   alpha=0.6, s=30, color='coral', edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('Mean (S_m) [N·m]', fontsize=11)
    axes[1].set_ylabel('Amplitude (S_a) [N·m]', fontsize=11)
    axes[1].set_title(f'Fatpack (4-point) [{label.upper()}]', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = f'mean_amplitude_scatter_{label}.png'
    plt.savefig(outfile, dpi=150)
    plt.show()
    print(f"✓ Saved: {outfile}")


def plot_cycles_overlay(signal, df_cycles, method_name, joint_idx):
    """
    Plot the original signal with cycle overlays.
    
    Parameters
    ----------
    signal : np.ndarray
        1D torque signal
    df_cycles : pd.DataFrame
        Cycles dataframe (must have i_start, i_end, range, mean, amplitude)
    method_name : str
        Name of the method (for labeling)
    joint_idx : int
        Joint index (for labeling)
    """
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Plot original signal
    ax.plot(signal, linewidth=1.0, color='black', label='Torque signal', zorder=1)
    
    # Filter cycles that have valid indices
    cycles_with_indices = df_cycles[df_cycles['i_start'].notna() & df_cycles['i_end'].notna()].copy()
    
    if len(cycles_with_indices) == 0:
        print(f"  Warning: No cycles with valid indices for {method_name}")
        ax.set_xlabel('Time step', fontsize=12)
        ax.set_ylabel('Torque (N·m)', fontsize=12)
        ax.set_title(f'{method_name} - Joint {joint_idx} - No cycles with indices', fontsize=13)
        plt.tight_layout()
        return
    
    # Find the largest-range cycle
    max_idx = cycles_with_indices['range'].idxmax()
    max_cycle = cycles_with_indices.loc[max_idx]
    
    # Plot all cycles as line segments (color by count: full vs half cycle)
    for idx, cycle in cycles_with_indices.iterrows():
        i_start = int(cycle['i_start'])
        i_end = int(cycle['i_end'])
        
        if idx == max_idx:
            continue  # Will plot separately
        
        # Color code: full cycles (1.0) = gray, half cycles (0.5) = lightblue
        if cycle['count'] == 1.0:
            color = 'gray'
            alpha = 0.3
        else:  # half cycle (0.5)
            color = 'lightblue'
            alpha = 0.5
        
        ax.plot([i_start, i_end], [signal[i_start], signal[i_end]], 
               color=color, alpha=alpha, linewidth=0.8, zorder=2)
    
    # Highlight the largest-range cycle
    i_start_max = int(max_cycle['i_start'])
    i_end_max = int(max_cycle['i_end'])
    s_max = max(signal[i_start_max], signal[i_end_max])
    s_min = min(signal[i_start_max], signal[i_end_max])
    s_mean = max_cycle['mean']
    s_amp = max_cycle['amplitude']
    
    # Red line for the cycle
    ax.plot([i_start_max, i_end_max], [signal[i_start_max], signal[i_end_max]], 
           color='red', linewidth=2.5, label=f'Largest cycle (range={max_cycle["range"]:.3f})', zorder=4)
    
    # Mean line (horizontal dashed line)
    ax.axhline(s_mean, color='blue', linestyle='--', linewidth=1.5, 
              label=f'Mean (S_m={s_mean:.3f})', zorder=3, 
              xmin=(i_start_max/len(signal))*0.95, xmax=(i_end_max/len(signal))*1.05)
    
    # Amplitude arrow
    mid_point = (i_start_max + i_end_max) / 2
    ax.annotate('', xy=(mid_point, s_mean + s_amp), xytext=(mid_point, s_mean),
               arrowprops=dict(arrowstyle='<->', color='green', lw=2), zorder=5)
    ax.text(mid_point + len(signal)*0.01, s_mean + s_amp/2, f'S_a={s_amp:.3f}', 
           color='green', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Torque (N·m)', fontsize=12)
    ax.set_title(f'{method_name} - Joint {joint_idx} - Cycles Overlay', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Count full and half cycles
    full_cycles = cycles_with_indices[cycles_with_indices['count'] == 1.0]
    half_cycles = cycles_with_indices[cycles_with_indices['count'] == 0.5]
    
    # Add text annotation
    textstr = (f'Gray segments: Full cycles (count=1.0) = {len(full_cycles)}\n'
               f'Light blue: Half cycles (count=0.5) = {len(half_cycles)}\n'
               f'Red segment: Largest-range cycle\n'
               f'Total cycle count: {cycles_with_indices["count"].sum():.1f}')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = f'cycles_overlay_{method_name.lower().replace(" ", "_")}_joint_{joint_idx}.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"✓ Saved: {filename}")


def plot_cycles_overlay_labeled(signal, df_cycles, method_name, joint_idx, label):
    """Plot cycles overlay with label in filename."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(signal, linewidth=1.0, color='black', label='Torque signal', zorder=1)
    cycles_with_indices = df_cycles[df_cycles['i_start'].notna() & df_cycles['i_end'].notna()].copy()
    
    if len(cycles_with_indices) == 0:
        print(f"  Warning: No cycles with valid indices for {method_name}")
        return
    
    max_idx = cycles_with_indices['range'].idxmax()
    max_cycle = cycles_with_indices.loc[max_idx]
    
    for idx, cycle in cycles_with_indices.iterrows():
        i_start = int(cycle['i_start'])
        i_end = int(cycle['i_end'])
        if idx == max_idx:
            continue
        if cycle['count'] == 1.0:
            color = 'gray'
            alpha = 0.3
        else:
            color = 'lightblue'
            alpha = 0.5
        ax.plot([i_start, i_end], [signal[i_start], signal[i_end]], 
               color=color, alpha=alpha, linewidth=0.8, zorder=2)
    
    i_start_max = int(max_cycle['i_start'])
    i_end_max = int(max_cycle['i_end'])
    s_mean = max_cycle['mean']
    s_amp = max_cycle['amplitude']
    
    ax.plot([i_start_max, i_end_max], [signal[i_start_max], signal[i_end_max]], 
           color='red', linewidth=2.5, label=f'Largest cycle (range={max_cycle["range"]:.3f})', zorder=4)
    ax.axhline(s_mean, color='blue', linestyle='--', linewidth=1.5, 
              label=f'Mean (S_m={s_mean:.3f})', zorder=3)
    mid_point = (i_start_max + i_end_max) / 2
    ax.annotate('', xy=(mid_point, s_mean + s_amp), xytext=(mid_point, s_mean),
               arrowprops=dict(arrowstyle='<->', color='green', lw=2), zorder=5)
    ax.text(mid_point + len(signal)*0.01, s_mean + s_amp/2, f'S_a={s_amp:.3f}', 
           color='green', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Torque (N·m)', fontsize=12)
    ax.set_title(f'{method_name} [{label.upper()}] - Joint {joint_idx} - Cycles Overlay', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    full_cycles = cycles_with_indices[cycles_with_indices['count'] == 1.0]
    half_cycles = cycles_with_indices[cycles_with_indices['count'] == 0.5]
    textstr = (f'Gray: Full cycles = {len(full_cycles)}\n'
               f'Light blue: Half cycles = {len(half_cycles)}\n'
               f'Total count: {cycles_with_indices["count"].sum():.1f}')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = f'cycles_overlay_{method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_joint_{joint_idx}_{label}.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"✓ Saved: {filename}")


def print_summary(df_rainflow, df_fatpack):
    """
    Print numerical summary comparing both methods.
    
    Parameters
    ----------
    df_rainflow : pd.DataFrame
        Cycles from rainflow package
    df_fatpack : pd.DataFrame
        Cycles from fatpack package
    """
    print("\n" + "="*70)
    print("NUMERIC SUMMARY")
    print("="*70)
    
    print("\n--- Rainflow Package (3-point ASTM-like) ---")
    print(f"  Total number of closed cycles: {df_rainflow['count'].sum():.1f}")
    if len(df_rainflow) > 0:
        max_idx = df_rainflow['range'].idxmax()
        max_cycle = df_rainflow.loc[max_idx]
        print(f"  Maximum range: {max_cycle['range']:.4f} N·m")
        print(f"    → Amplitude: {max_cycle['amplitude']:.4f} N·m")
        print(f"    → Mean: {max_cycle['mean']:.4f} N·m")
        if 'damage' in df_rainflow.columns:
            print(f"  Total accumulated damage: {df_rainflow['damage'].sum():.6f}")
    
    print("\n--- Fatpack Package (4-point Amzallag-like) ---")
    print(f"  Total number of closed cycles: {df_fatpack['count'].sum():.1f}")
    if len(df_fatpack) > 0:
        max_idx = df_fatpack['range'].idxmax()
        max_cycle = df_fatpack.loc[max_idx]
        print(f"  Maximum range: {max_cycle['range']:.4f} N·m")
        print(f"    → Amplitude: {max_cycle['amplitude']:.4f} N·m")
        print(f"    → Mean: {max_cycle['mean']:.4f} N·m")
        if 'damage' in df_fatpack.columns:
            print(f"  Total accumulated damage: {df_fatpack['damage'].sum():.6f}")
    
    print("\n--- Comparison Notes ---")
    print("  The 4-point algorithm (fatpack) typically preserves more large-range")
    print("  cycles compared to the 3-point algorithm (rainflow) because:")
    print("  • It compares inner vs outer ranges and keeps the outer envelope")
    print("  • It delays closing cycles until the full 4-point pattern is identified")
    print("  • The 3-point algorithm may close smaller cycles earlier")
    print("="*70 + "\n")


def calculate_fatigue_damage(df_cycles, m=SN_EXPONENT, C=SN_CONSTANT):
    """
    Calculate fatigue damage using Palmgren-Miner rule and S-N curve.
    
    S-N Curve: N = C / (S_a^m)
    where N = cycles to failure at stress amplitude S_a
    
    Palmgren-Miner: D = Σ(n_i / N_i)
    where n_i = actual cycles, N_i = cycles to failure
    
    Parameters
    ----------
    df_cycles : pd.DataFrame
        Cycles dataframe with 'amplitude' and 'count' columns
    m : float
        Wöhler exponent (material parameter)
    C : float
        Material constant
        
    Returns
    -------
    df_cycles : pd.DataFrame
        Input dataframe with added 'N_f' and 'damage' columns
    """
    df = df_cycles.copy()
    
    # Cycles to failure for each amplitude: N_f = C / (S_a^m)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    df['N_f'] = C / (np.maximum(df['amplitude'], epsilon) ** m)
    
    # Damage contribution: D_i = n_i / N_f_i
    df['damage'] = df['count'] / df['N_f']
    
    return df


def plot_damage_comparison(df_rainflow, df_fatpack):
    """
    Create comprehensive damage comparison figure for both methods.
    Shows amplitude, mean, cycle counts, and damage contributions.
    
    Parameters
    ----------
    df_rainflow : pd.DataFrame
        Cycles from rainflow with damage calculations
    df_fatpack : pd.DataFrame
        Cycles from fatpack with damage calculations
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    # Sort by damage for better visualization
    df_rf_sorted = df_rainflow.sort_values('damage', ascending=False).reset_index(drop=True)
    df_fp_sorted = df_fatpack.sort_values('damage', ascending=False).reset_index(drop=True)
    
    total_damage_rf = df_rainflow['damage'].sum()
    total_damage_fp = df_fatpack['damage'].sum()
    
    # ============== RAINFLOW (Left Column) ==============
    
    # Top: Amplitude vs Damage scatter
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(df_rainflow['amplitude'], df_rainflow['damage'], 
                          c=df_rainflow['mean'], cmap='RdYlBu_r', 
                          s=df_rainflow['count']*100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Amplitude S_a (N·m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Damage Contribution D_i', fontsize=11, fontweight='bold')
    ax1.set_title('Rainflow (3-point): Amplitude vs Damage', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Mean S_m (N·m)', fontsize=9)
    
    # Add top 3 damaging cycles annotation
    top3_rf = df_rf_sorted.head(3)
    for i, (idx, row) in enumerate(top3_rf.iterrows()):
        ax1.annotate(f'#{i+1}', xy=(row['amplitude'], row['damage']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='red')
    
    # Middle: Cumulative damage bar chart (top 20)
    ax3 = fig.add_subplot(gs[1, 0])
    top20_rf = df_rf_sorted.head(20)
    colors_rf = ['red' if i < 3 else 'steelblue' for i in range(len(top20_rf))]
    bars1 = ax3.barh(range(len(top20_rf)), top20_rf['damage'], color=colors_rf, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Damage Contribution D_i', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cycle Rank (by damage)', fontsize=11, fontweight='bold')
    ax3.set_title('Rainflow: Top 20 Damaging Cycles', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Bottom: Summary text with formulas
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    
    summary_text_rf = f"""
RAINFLOW (3-point ASTM-like) - DAMAGE SUMMARY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
S-N Curve (Wöhler):  N = C / (Sₐᵐ)
  • m (exponent) = {SN_EXPONENT}
  • C (constant) = {SN_CONSTANT:.2e}
  
Palmgren-Miner:  D = Σ(nᵢ / Nᵢ)
  • nᵢ = actual cycle count
  • Nᵢ = cycles to failure at amplitude Sₐᵢ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Cycles Detected: {len(df_rainflow)}
Total Cycle Count: {df_rainflow['count'].sum():.1f}
  (includes {len(df_rainflow[df_rainflow['count']==0.5])} half-cycles)

TOTAL ACCUMULATED DAMAGE:  D = {total_damage_rf:.6f}
  (D ≥ 1.0 indicates failure predicted)

Top 3 Contributing Cycles:
"""
    for i, (idx, row) in enumerate(top3_rf.iterrows()):
        summary_text_rf += f"\n  #{i+1}: Sₐ={row['amplitude']:.3f} N·m, Sₘ={row['mean']:.3f} N·m"
        summary_text_rf += f"\n      n={row['count']:.1f}, Nf={row['N_f']:.1e}, D={row['damage']:.6f}"
        summary_text_rf += f" ({row['damage']/total_damage_rf*100:.1f}%)"
    
    ax5.text(0.05, 0.95, summary_text_rf, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ============== FATPACK (Right Column) ==============
    
    # Top: Amplitude vs Damage scatter
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(df_fatpack['amplitude'], df_fatpack['damage'], 
                          c=df_fatpack['mean'], cmap='RdYlBu_r', 
                          s=df_fatpack['count']*100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Amplitude S_a (N·m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Damage Contribution D_i', fontsize=11, fontweight='bold')
    ax2.set_title('Fatpack (4-point): Amplitude vs Damage', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Mean S_m (N·m)', fontsize=9)
    
    # Add top 3 damaging cycles annotation
    top3_fp = df_fp_sorted.head(3)
    for i, (idx, row) in enumerate(top3_fp.iterrows()):
        ax2.annotate(f'#{i+1}', xy=(row['amplitude'], row['damage']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='red')
    
    # Middle: Cumulative damage bar chart (top 20)
    ax4 = fig.add_subplot(gs[1, 1])
    top20_fp = df_fp_sorted.head(20)
    colors_fp = ['red' if i < 3 else 'coral' for i in range(len(top20_fp))]
    bars2 = ax4.barh(range(len(top20_fp)), top20_fp['damage'], color=colors_fp, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Damage Contribution D_i', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cycle Rank (by damage)', fontsize=11, fontweight='bold')
    ax4.set_title('Fatpack: Top 20 Damaging Cycles', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Bottom: Summary text with formulas
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    summary_text_fp = f"""
FATPACK (4-point Amzallag-like) - DAMAGE SUMMARY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
S-N Curve (Wöhler):  N = C / (Sₐᵐ)
  • m (exponent) = {SN_EXPONENT}
  • C (constant) = {SN_CONSTANT:.2e}
  
Palmgren-Miner:  D = Σ(nᵢ / Nᵢ)
  • nᵢ = actual cycle count
  • Nᵢ = cycles to failure at amplitude Sₐᵢ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Cycles Detected: {len(df_fatpack)}
Total Cycle Count: {df_fatpack['count'].sum():.1f}
  (all full cycles, residue={3})

TOTAL ACCUMULATED DAMAGE:  D = {total_damage_fp:.6f}
  (D ≥ 1.0 indicates failure predicted)

Top 3 Contributing Cycles:
"""
    for i, (idx, row) in enumerate(top3_fp.iterrows()):
        summary_text_fp += f"\n  #{i+1}: Sₐ={row['amplitude']:.3f} N·m, Sₘ={row['mean']:.3f} N·m"
        summary_text_fp += f"\n      n={row['count']:.1f}, Nf={row['N_f']:.1e}, D={row['damage']:.6f}"
        summary_text_fp += f" ({row['damage']/total_damage_fp*100:.1f}%)"
    
    ax6.text(0.05, 0.95, summary_text_fp, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    # Overall title
    fig.suptitle('Fatigue Damage Comparison: Rainflow (3-point) vs Fatpack (4-point)', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('damage_comparison_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: damage_comparison_detailed.png")


def plot_damage_comparison_labeled(df_rainflow, df_fatpack, label):
    """Create damage comparison figure with label in filename (simplified version)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    df_rf_sorted = df_rainflow.sort_values('damage', ascending=False).reset_index(drop=True)
    df_fp_sorted = df_fatpack.sort_values('damage', ascending=False).reset_index(drop=True)
    total_damage_rf = df_rainflow['damage'].sum()
    total_damage_fp = df_fatpack['damage'].sum()
    
    # Top Left: Rainflow Amplitude vs Damage
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(df_rainflow['amplitude'], df_rainflow['damage'], 
                          c=df_rainflow['mean'], cmap='RdYlBu_r', 
                          s=df_rainflow['count']*100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Amplitude S_a (N·m)', fontsize=11)
    ax1.set_ylabel('Damage D_i', fontsize=11)
    ax1.set_title(f'Rainflow (3-point) [{label.upper()}]', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Mean (N·m)')
    
    # Top Right: Fatpack Amplitude vs Damage
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df_fatpack['amplitude'], df_fatpack['damage'], 
                          c=df_fatpack['mean'], cmap='RdYlBu_r', 
                          s=df_fatpack['count']*100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Amplitude S_a (N·m)', fontsize=11)
    ax2.set_ylabel('Damage D_i', fontsize=11)
    ax2.set_title(f'Fatpack (4-point) [{label.upper()}]', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Mean (N·m)')
    
    # Bottom Left: Rainflow Top 20
    ax3 = axes[1, 0]
    top20_rf = df_rf_sorted.head(20)
    colors_rf = ['red' if i < 3 else 'steelblue' for i in range(len(top20_rf))]
    ax3.barh(range(len(top20_rf)), top20_rf['damage'], color=colors_rf, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Damage D_i', fontsize=11)
    ax3.set_ylabel('Rank', fontsize=11)
    ax3.set_title(f'Rainflow: Top 20 (Total D={total_damage_rf:.6f})', fontsize=11)
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Bottom Right: Fatpack Top 20
    ax4 = axes[1, 1]
    top20_fp = df_fp_sorted.head(20)
    colors_fp = ['red' if i < 3 else 'coral' for i in range(len(top20_fp))]
    ax4.barh(range(len(top20_fp)), top20_fp['damage'], color=colors_fp, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Damage D_i', fontsize=11)
    ax4.set_ylabel('Rank', fontsize=11)
    ax4.set_title(f'Fatpack: Top 20 (Total D={total_damage_fp:.6f})', fontsize=11)
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(f'Fatigue Damage Comparison [{label.upper()}] - m={SN_EXPONENT}, C={SN_CONSTANT:.0e}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = f'damage_comparison_{label}.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {outfile}")


def print_summary(df_rainflow, df_fatpack):
    """
    Print numerical summary comparing both methods.
    
    Parameters
    ----------
    df_rainflow : pd.DataFrame
        Cycles from rainflow package
    df_fatpack : pd.DataFrame
        Cycles from fatpack package
    """
    print("\n" + "="*70)
    print("NUMERIC SUMMARY")
    print("="*70)
    
    print("\n--- Rainflow Package (3-point ASTM-like) ---")
    print(f"  Total number of closed cycles: {df_rainflow['count'].sum():.1f}")
    if len(df_rainflow) > 0:
        max_idx = df_rainflow['range'].idxmax()
        max_cycle = df_rainflow.loc[max_idx]
        print(f"  Maximum range: {max_cycle['range']:.4f} N·m")
        print(f"    → Amplitude: {max_cycle['amplitude']:.4f} N·m")
        print(f"    → Mean: {max_cycle['mean']:.4f} N·m")
        if 'damage' in df_rainflow.columns:
            print(f"  Total accumulated damage: {df_rainflow['damage'].sum():.6f}")
    
    print("\n--- Fatpack Package (4-point Amzallag-like) ---")
    print(f"  Total number of closed cycles: {df_fatpack['count'].sum():.1f}")
    if len(df_fatpack) > 0:
        max_idx = df_fatpack['range'].idxmax()
        max_cycle = df_fatpack.loc[max_idx]
        print(f"  Maximum range: {max_cycle['range']:.4f} N·m")
        print(f"    → Amplitude: {max_cycle['amplitude']:.4f} N·m")
        print(f"    → Mean: {max_cycle['mean']:.4f} N·m")
        if 'damage' in df_fatpack.columns:
            print(f"  Total accumulated damage: {df_fatpack['damage'].sum():.6f}")
    
    print("\n--- Comparison Notes ---")
    print("  The 4-point algorithm (fatpack) typically preserves more large-range")
    print("  cycles compared to the 3-point algorithm (rainflow) because:")
    print("  • It compares inner vs outer ranges and keeps the outer envelope")
    print("  • It delays closing cycles until the full 4-point pattern is identified")
    print("  • The 3-point algorithm may close smaller cycles earlier")
    
    if 'damage' in df_rainflow.columns and 'damage' in df_fatpack.columns:
        damage_diff = abs(df_rainflow['damage'].sum() - df_fatpack['damage'].sum())
        damage_pct = damage_diff / df_rainflow['damage'].sum() * 100
        print(f"\n  Damage difference: {damage_diff:.6f} ({damage_pct:.2f}%)")
    
    print("="*70 + "\n")


def main(data_dir=None, label=None):
    """Main execution function.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the torque data
    label : str
        Label for output files (e.g., 'clean', 'noisy')
    """
    # Use defaults if not provided
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if label is None:
        label = DEFAULT_LABEL
    
    data_file = f"{data_dir}/{TORQUE_FILE}"
    
    print("="*70)
    print(f"FATIGUE ANALYSIS: RAINFLOW COUNTING COMPARISON [{label.upper()}]")
    print("="*70)
    print(f"Data: {data_file}")
    print(f"Label: {label}")
    
    # Step 1: Load data
    print("\n[Step 1] Loading torque data...")
    torque_data = load_torque_data(data_file)
    
    # Extract joint 5 signal
    torque_joint5 = torque_data[:, FOCUS_JOINT]
    print(f"\nFocusing on Joint {FOCUS_JOINT}")
    print(f"  Signal length: {len(torque_joint5)} time steps")
    print(f"  Signal range: [{torque_joint5.min():.4f}, {torque_joint5.max():.4f}] N·m")
    
    # Step 2: Plot raw torque
    print("\n[Step 2] Plotting raw torque signal...")
    plot_raw_torque_labeled(torque_joint5, FOCUS_JOINT, TORQUE_FILE, label)
    
    # Step 3: Rainflow counting with 'rainflow' package
    print("\n[Step 3] Rainflow counting with 'rainflow' package...")
    df_rainflow = compute_cycles_rainflow(torque_joint5)
    
    # Step 4: Rainflow counting with 'fatpack' package
    print("\n[Step 4] Rainflow counting with 'fatpack' package...")
    df_fatpack = compute_cycles_fatpack(torque_joint5)
    
    # Step 5: Visual comparisons
    print("\n[Step 5] Creating visual comparisons...")
    if len(df_rainflow) > 0 and len(df_fatpack) > 0:
        plot_histogram_comparison_labeled(df_rainflow, df_fatpack, label)
        plot_mean_amplitude_scatter_labeled(df_rainflow, df_fatpack, label)
    else:
        print("  Warning: One or both methods returned no cycles. Skipping comparisons.")
    
    # Step 6: Overlay cycles on signal
    print("\n[Step 6] Creating cycle overlay plots...")
    if 'i_start' in df_rainflow.columns and 'i_end' in df_rainflow.columns:
        plot_cycles_overlay_labeled(torque_joint5, df_rainflow, "Rainflow (3-point)", FOCUS_JOINT, label)
    else:
        print("  Warning: Rainflow cycles missing index information. Skipping overlay.")
    
    if 'i_start' in df_fatpack.columns and 'i_end' in df_fatpack.columns:
        plot_cycles_overlay_labeled(torque_joint5, df_fatpack, "Fatpack (4-point)", FOCUS_JOINT, label)
    else:
        print("  Warning: Fatpack cycles missing index information. Skipping overlay.")
    
    # Step 7: Calculate fatigue damage
    print("\n[Step 7] Calculating fatigue damage...")
    print(f"  S-N Curve parameters: m={SN_EXPONENT}, C={SN_CONSTANT:.2e}")
    df_rainflow = calculate_fatigue_damage(df_rainflow)
    df_fatpack = calculate_fatigue_damage(df_fatpack)
    print(f"  Rainflow total damage: {df_rainflow['damage'].sum():.6f}")
    print(f"  Fatpack total damage: {df_fatpack['damage'].sum():.6f}")
    
    # Step 8: Create comprehensive damage comparison figure
    print("\n[Step 8] Creating comprehensive damage comparison figure...")
    plot_damage_comparison_labeled(df_rainflow, df_fatpack, label)
    
    # Step 9: Print summary
    print("\n[Step 9] Printing numeric summary...")
    print_summary(df_rainflow, df_fatpack)
    
    print("\n✓ Analysis complete!")
    print(f"  Generated plots saved with '{label}' prefix")
    
    return df_rainflow, df_fatpack


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 3:
        data_dir = sys.argv[1]
        label = sys.argv[2]
    elif len(sys.argv) >= 2:
        data_dir = sys.argv[1]
        label = "custom"
    else:
        data_dir = DEFAULT_DATA_DIR
        label = DEFAULT_LABEL
    
    main(data_dir, label)

