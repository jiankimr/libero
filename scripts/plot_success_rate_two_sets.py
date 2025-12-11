#!/usr/bin/env python3
"""
Task Success Rate Visualization for Two Sets
Set 1: Noise Scale (0.1~0.6)
Set 2: Different Model Configurations
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Set style
try:
    plt.style.use('seaborn-whitegrid')
except:
    pass

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14


def plot_set1_noise_scale_success_rate(output_dir):
    """Set 1: Success Rate vs Noise Scale (0.1~0.6)"""
    
    # Data from comparison files (After With Noise)
    noise_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    success_rates = [87.4, 87.4, 87.0, 87.2, 85.0, 81.6]
    baseline = 87.0  # Before (No Noise)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar chart
    colors = ['#2ca02c' if sr >= baseline else '#d62728' for sr in success_rates]
    bars = ax.bar(noise_scales, success_rates, width=0.07, color=colors, 
                  edgecolor='#1a6b1a', alpha=0.85)
    
    # Line connecting points
    ax.plot(noise_scales, success_rates, 'o-', color='#1f77b4', linewidth=2.5, 
            markersize=10, markerfacecolor='white', markeredgewidth=2)
    
    # Baseline line
    ax.axhline(y=baseline, color='#ff7f0e', linestyle='--', linewidth=2, 
               label=f'Baseline (No Noise): {baseline}%')
    
    # Add value labels
    for i, (x, y) in enumerate(zip(noise_scales, success_rates)):
        diff = y - baseline
        color = '#2ca02c' if diff >= 0 else '#d62728'
        sign = '+' if diff >= 0 else ''
        ax.annotate(f'{y:.1f}%\n({sign}{diff:.1f}%)', 
                   (x, y), 
                   textcoords="offset points", 
                   xytext=(0, 12), 
                   ha='center', 
                   fontsize=11, 
                   fontweight='bold',
                   color=color)
    
    ax.set_xlabel('Noise Scale (action.x)', fontsize=14)
    ax.set_ylabel('Task Success Rate (%)', fontsize=14)
    ax.set_title('Set 1: Task Success Rate vs Noise Scale\n(Rollout Model + Clean Noise @ 0.3)', 
                fontsize=16, fontweight='bold', color='#3366cc')
    ax.set_xticks(noise_scales)
    ax.set_xticklabels([f'{n}' for n in noise_scales])
    ax.set_ylim(75, 95)
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'set1_success_rate_vs_noise.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def plot_set2_model_comparison_success_rate(output_dir):
    """Set 2: Success Rate for Different Model Configurations"""
    
    # Data from comparison files
    models = ['Clean + 0.3 Noise', 'Noisy_02\n(baseline)', 'Noisy_Mixed\n(baseline)']
    before_rates = [87.0, 87.0, 87.0]  # Before (No Noise) - same baseline
    after_rates = [87.0, 88.0, 92.0]   # After (With Config)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Before bars
    bars1 = ax.bar(x - width/2, before_rates, width, label='Before (No Noise)', 
                   color='#74c476', edgecolor='#41ab5d', alpha=0.85)
    
    # After bars
    bars2 = ax.bar(x + width/2, after_rates, width, label='After (With Config)', 
                   color='#6baed6', edgecolor='#4292c6', alpha=0.85)
    
    # Add value labels and difference
    for i, (before, after) in enumerate(zip(before_rates, after_rates)):
        # Before label
        ax.annotate(f'{before:.1f}%', 
                   (x[i] - width/2, before), 
                   textcoords="offset points", 
                   xytext=(0, 5), 
                   ha='center', 
                   fontsize=11, 
                   fontweight='bold')
        
        # After label with difference
        diff = after - before
        sign = '+' if diff >= 0 else ''
        color = '#2ca02c' if diff >= 0 else '#d62728'
        ax.annotate(f'{after:.1f}%\n({sign}{diff:.1f}%)', 
                   (x[i] + width/2, after), 
                   textcoords="offset points", 
                   xytext=(0, 5), 
                   ha='center', 
                   fontsize=11, 
                   fontweight='bold',
                   color=color if diff != 0 else 'black')
    
    ax.set_xlabel('Model Configuration', fontsize=14)
    ax.set_ylabel('Task Success Rate (%)', fontsize=14)
    ax.set_title('Set 2: Task Success Rate by Model Configuration\n(Before vs After Comparison)', 
                fontsize=16, fontweight='bold', color='#3366cc')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(80, 100)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'set2_success_rate_model_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def main():
    output_dir = '/workspace/repos/rsec/LIBERO/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Task Success Rate Visualization - Two Sets")
    print("=" * 60)
    
    print("\n📊 Set 1: Noise Scale (0.1~0.6)")
    plot_set1_noise_scale_success_rate(output_dir)
    
    print("\n📊 Set 2: Model Configuration Comparison")
    plot_set2_model_comparison_success_rate(output_dir)
    
    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()




