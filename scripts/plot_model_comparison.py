#!/usr/bin/env python3
"""
Model Comparison Visualization
Compare different model configurations (clean_noise, noisy_02, noisy_mixed)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-whitegrid')
except:
    pass

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
COLORS = ['#e8a83c', '#6baed6', '#74c476']


def parse_comparison_file(filepath):
    """Parse a comparison analysis file and extract key metrics."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = {
        'rainflow_life_ratio': {},
        'fatpack_life_ratio': {},
        'rainflow_avg': None,
        'fatpack_avg': None,
        'energy': {},
        'success_rate': {}
    }
    
    # Parse average Rainflow life ratio
    match = re.search(r'Rainflow.*?평균 수명비:\s+([\d.]+)x', content, re.DOTALL)
    if match:
        data['rainflow_avg'] = float(match.group(1))
    
    # Parse average Fatpack life ratio
    match = re.search(r'Fatpack.*?평균 수명비:\s+([\d.]+)x', content, re.DOTALL)
    if match:
        data['fatpack_avg'] = float(match.group(1))
    
    # Parse Rainflow Joint-wise life ratio
    rainflow_pattern = r'Joint_(\d)\s+\|\s+[\d.e+\-]+\s+\|\s+[\d.e+\-]+\s+\|\s+([\d.]+)x'
    rainflow_section = re.search(r'RAINFLOW 패키지 기반.*?FATPACK 패키지', content, re.DOTALL)
    if rainflow_section:
        matches = re.findall(rainflow_pattern, rainflow_section.group())
        for joint, ratio in matches:
            data['rainflow_life_ratio'][f'Joint_{joint}'] = float(ratio)
    
    # Parse Fatpack Joint-wise life ratio
    fatpack_section = re.search(r'FATPACK 패키지 기반.*?패키지 간 비교', content, re.DOTALL)
    if fatpack_section:
        matches = re.findall(rainflow_pattern, fatpack_section.group())
        for joint, ratio in matches:
            data['fatpack_life_ratio'][f'Joint_{joint}'] = float(ratio)
    
    # Parse Energy metrics
    energy_patterns = {
        'draw_sum': r'energy_draw\.sum\s+\|\s+Before:\s+([\d.]+)±\s*[\d.]+\s+→\s+After:\s+([\d.]+)',
        'regen_sum': r'energy_regen\.sum\s+\|\s+Before:\s+([\d.]+)±\s*[\d.]+\s+→\s+After:\s+([\d.]+)',
        'net_sum': r'energy_net\.sum\s+\|\s+Before:\s+([\d.]+)±\s*[\d.]+\s+→\s+After:\s+([\d.]+)',
    }
    
    for key, pattern in energy_patterns.items():
        match = re.search(pattern, content)
        if match:
            before, after = float(match.group(1)), float(match.group(2))
            change_pct = ((after - before) / before) * 100 if before != 0 else 0
            data['energy'][key] = {
                'before': before,
                'after': after,
                'change_pct': change_pct
            }
    
    # Parse success rate
    success_pattern = r'After \(With Noise\):\s+([\d.]+)%'
    match = re.search(success_pattern, content)
    if match:
        data['success_rate']['after'] = float(match.group(1))
    
    before_success_pattern = r'Before \(No Noise\):\s+([\d.]+)%'
    match = re.search(before_success_pattern, content)
    if match:
        data['success_rate']['before'] = float(match.group(1))
    
    return data


def plot_three_models_comparison(output_dir):
    """Compare 3 different model configurations."""
    
    # Define the 3 files with different configurations
    files = {
        'Clean + 0.3 Noise': '/workspace/repos/rsec/LIBERO/results/comparison_analysis_libero_10_20251202_171746_noise_03000_dim_action.x_clean_noise.txt',
        'Noisy_02 (baseline)': '/workspace/repos/rsec/LIBERO/results/comparison_analysis_libero_10_20251205_043000_noise_00000_noisy_02.txt',
        'Noisy_Mixed (baseline)': '/workspace/repos/rsec/LIBERO/results/comparison_analysis_libero_10_20251205_085034_noise_00000_noisy_mixed.txt',
    }
    
    # Parse all files
    all_data = {}
    for label, filepath in files.items():
        if os.path.exists(filepath):
            all_data[label] = parse_comparison_file(filepath)
            print(f"Parsed: {label}")
        else:
            print(f"File not found: {filepath}")
    
    if len(all_data) < 2:
        print("Not enough data to compare!")
        return
    
    labels = list(all_data.keys())
    x = np.arange(len(labels))
    width = 0.25
    
    # ========== 1. Life Ratio Comparison (Average) ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Rainflow
    rainflow_values = [all_data[l].get('rainflow_avg', 0) or 0 for l in labels]
    bars1 = axes[0].bar(x, rainflow_values, width=0.5, color=COLORS[0], edgecolor='#b8831c', alpha=0.9)
    axes[0].set_ylabel('Average Life Ratio (x)')
    axes[0].set_title('Rainflow - Average Life Ratio', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars1, rainflow_values):
        axes[0].annotate(f'{val:.2f}x', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Fatpack
    fatpack_values = [all_data[l].get('fatpack_avg', 0) or 0 for l in labels]
    bars2 = axes[1].bar(x, fatpack_values, width=0.5, color=COLORS[1], edgecolor='#4292c6', alpha=0.9)
    axes[1].set_ylabel('Average Life Ratio (x)')
    axes[1].set_title('Fatpack - Average Life Ratio', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars2, fatpack_values):
        axes[1].annotate(f'{val:.2f}x', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Model Comparison: Average Life Ratio\n(Higher = More Fatigue Damage)', 
                fontsize=16, fontweight='bold', color='#3366cc', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison_life_ratio_avg.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # ========== 2. Energy Change Comparison ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    
    draw_changes = [all_data[l]['energy'].get('draw_sum', {}).get('change_pct', 0) for l in labels]
    regen_changes = [all_data[l]['energy'].get('regen_sum', {}).get('change_pct', 0) for l in labels]
    net_changes = [all_data[l]['energy'].get('net_sum', {}).get('change_pct', 0) for l in labels]
    
    bars1 = ax.bar(x - width, draw_changes, width, label='draw_sum Δ%', color='#e8a83c', edgecolor='#b8831c')
    bars2 = ax.bar(x, regen_changes, width, label='regen_sum Δ%', color='#6baed6', edgecolor='#4292c6')
    bars3 = ax.bar(x + width, net_changes, width, label='net_sum Δ%', color='#74c476', edgecolor='#41ab5d')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -12),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Model Configuration', fontsize=14)
    ax.set_ylabel('Change (%)', fontsize=14)
    ax.set_title('Model Comparison: Energy Change (%)\n(Before vs After for each configuration)', 
                fontsize=16, fontweight='bold', color='#3366cc')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison_energy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # ========== 3. Success Rate Comparison ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    before_success = [all_data[l]['success_rate'].get('before', 0) for l in labels]
    after_success = [all_data[l]['success_rate'].get('after', 0) for l in labels]
    
    bars1 = ax.bar(x - width/2, before_success, width, label='Before (No Noise)', color='#74c476', edgecolor='#41ab5d')
    bars2 = ax.bar(x + width/2, after_success, width, label='After (With Config)', color='#e8a83c', edgecolor='#b8831c')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model Configuration', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_title('Model Comparison: Task Success Rate', fontsize=16, fontweight='bold', color='#3366cc')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison_success_rate.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # ========== 4. Joint-wise Life Ratio Comparison ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    joints = [f'Joint_{i}' for i in range(7)]
    x_joints = np.arange(len(joints))
    width_j = 0.25
    
    for ax_idx, (method, method_key) in enumerate([('Rainflow', 'rainflow_life_ratio'), 
                                                    ('Fatpack', 'fatpack_life_ratio')]):
        ax = axes[ax_idx]
        
        for i, label in enumerate(labels):
            values = [all_data[label].get(method_key, {}).get(j, 0) for j in joints]
            offset = (i - 1) * width_j
            ax.bar(x_joints + offset, values, width_j, label=label, color=COLORS[i], alpha=0.85)
        
        ax.set_xlabel('Joint', fontsize=12)
        ax.set_ylabel('Life Ratio (x)', fontsize=12)
        ax.set_title(f'{method} - Joint-wise Life Ratio', fontsize=14, fontweight='bold')
        ax.set_xticks(x_joints)
        ax.set_xticklabels([j.replace('Joint_', 'J') for j in joints])
        ax.legend(loc='upper right', fontsize=9)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_yscale('log')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Model Comparison: Joint-wise Life Ratio (log scale)', 
                fontsize=16, fontweight='bold', color='#3366cc', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison_joint_life_ratio.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = '/workspace/repos/rsec/LIBERO/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Model Comparison Visualization")
    print("=" * 60)
    print("\nComparing 3 different configurations:")
    print("  1. Clean + 0.3 Noise (action.x)")
    print("  2. Noisy_02 (baseline evaluation)")
    print("  3. Noisy_Mixed (baseline evaluation)")
    print("=" * 60)
    
    plot_three_models_comparison(output_dir)
    
    print("\n" + "=" * 60)
    print(f"All comparison plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()




