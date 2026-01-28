#!/usr/bin/env python3
"""
Compare reward distributions between different models (100% noisy vs 25% noisy)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(result_dir):
    """Load results from a model's output directory"""
    result_path = Path(result_dir) / "results.json"
    with open(result_path, 'r') as f:
        return json.load(f)

def compute_episode_stats(episodes_data):
    """Compute per-episode statistics"""
    episode_means = []
    episode_sums = []
    for ep in episodes_data:
        episode_means.append(np.mean(ep))
        episode_sums.append(np.sum(ep))
    return episode_means, episode_sums

def create_comparison_plot(results_100, results_25, output_path):
    """Create comparison plots between 100% and 25% noisy models"""
    
    # Pastel colors
    color_100_sparse = '#FF9999'   # light red
    color_100_dense = '#99FF99'    # light green  
    color_25_sparse = '#9999FF'    # light blue
    color_25_dense = '#FFCC99'     # light orange
    
    fig = plt.figure(figsize=(16, 12))
    
    # Title with success rates
    sr_100 = results_100['success_rate'] * 100
    sr_25 = results_25['success_rate'] * 100
    fig.suptitle(f'Reward Comparison: 100% Noisy (SR={sr_100:.0f}%) vs 25% Noisy (SR={sr_25:.0f}%)', 
                 fontsize=16, fontweight='bold')
    
    # ============== 1. Dense Reward Distribution Comparison ==============
    ax1 = fig.add_subplot(2, 3, 1)
    
    dense_100 = np.concatenate(results_100['dense']['episodes'])
    dense_25 = np.concatenate(results_25['dense']['episodes'])
    
    ax1.hist(dense_100, bins=50, alpha=0.6, label=f'100% Noisy (μ={np.mean(dense_100):.3f})', 
             color=color_100_dense, edgecolor='darkgreen')
    ax1.hist(dense_25, bins=50, alpha=0.6, label=f'25% Noisy (μ={np.mean(dense_25):.3f})', 
             color=color_25_dense, edgecolor='darkorange')
    ax1.set_xlabel('Dense Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Dense Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ============== 2. Per-Episode Dense Reward Sum ==============
    ax2 = fig.add_subplot(2, 3, 2)
    
    _, sums_100 = compute_episode_stats(results_100['dense']['episodes'])
    _, sums_25 = compute_episode_stats(results_25['dense']['episodes'])
    
    x = np.arange(len(sums_100))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, sums_100, width, label='100% Noisy', color=color_100_dense, edgecolor='darkgreen')
    bars2 = ax2.bar(x + width/2, sums_25, width, label='25% Noisy', color=color_25_dense, edgecolor='darkorange')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Dense Reward')
    ax2.set_title('Per-Episode Dense Reward (Sum)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Ep{i+1}' for i in x])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ============== 3. Box Plot Comparison ==============
    ax3 = fig.add_subplot(2, 3, 3)
    
    data = [dense_100, dense_25]
    bp = ax3.boxplot(data, labels=['100% Noisy', '25% Noisy'], patch_artist=True)
    bp['boxes'][0].set_facecolor(color_100_dense)
    bp['boxes'][1].set_facecolor(color_25_dense)
    
    ax3.set_ylabel('Dense Reward')
    ax3.set_title('Dense Reward Box Plot')
    ax3.grid(True, alpha=0.3)
    
    # ============== 4. Dense Reward Over Time (Mean ± Std) ==============
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Pad episodes to same length
    max_len_100 = max(len(ep) for ep in results_100['dense']['episodes'])
    max_len_25 = max(len(ep) for ep in results_25['dense']['episodes'])
    max_len = max(max_len_100, max_len_25)
    
    # Pad with NaN for proper mean/std calculation
    padded_100 = np.full((len(results_100['dense']['episodes']), max_len), np.nan)
    padded_25 = np.full((len(results_25['dense']['episodes']), max_len), np.nan)
    
    for i, ep in enumerate(results_100['dense']['episodes']):
        padded_100[i, :len(ep)] = ep
    for i, ep in enumerate(results_25['dense']['episodes']):
        padded_25[i, :len(ep)] = ep
    
    mean_100 = np.nanmean(padded_100, axis=0)
    std_100 = np.nanstd(padded_100, axis=0)
    mean_25 = np.nanmean(padded_25, axis=0)
    std_25 = np.nanstd(padded_25, axis=0)
    
    steps = np.arange(max_len)
    
    ax4.plot(steps, mean_100, color='green', label='100% Noisy', linewidth=2)
    ax4.fill_between(steps, mean_100 - std_100, mean_100 + std_100, color=color_100_dense, alpha=0.3)
    
    ax4.plot(steps, mean_25, color='orange', label='25% Noisy', linewidth=2)
    ax4.fill_between(steps, mean_25 - std_25, mean_25 + std_25, color=color_25_dense, alpha=0.3)
    
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Dense Reward')
    ax4.set_title('Dense Reward Over Time (mean ± std)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ============== 5. Success Rate Comparison ==============
    ax5 = fig.add_subplot(2, 3, 5)
    
    models = ['100% Noisy', '25% Noisy']
    success_rates = [sr_100, sr_25]
    colors = [color_100_dense, color_25_dense]
    
    bars = ax5.bar(models, success_rates, color=colors, edgecolor=['darkgreen', 'darkorange'], linewidth=2)
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title('Success Rate Comparison')
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, sr in zip(bars, success_rates):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{sr:.0f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # ============== 6. Summary Statistics Table ==============
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', '100% Noisy', '25% Noisy', 'Difference'],
        ['Success Rate', f'{sr_100:.0f}%', f'{sr_25:.0f}%', f'{sr_25 - sr_100:+.0f}%'],
        ['Dense Mean', f'{np.mean(dense_100):.4f}', f'{np.mean(dense_25):.4f}', 
         f'{np.mean(dense_25) - np.mean(dense_100):+.4f}'],
        ['Dense Std', f'{np.std(dense_100):.4f}', f'{np.std(dense_25):.4f}',
         f'{np.std(dense_25) - np.std(dense_100):+.4f}'],
        ['Dense Max', f'{np.max(dense_100):.4f}', f'{np.max(dense_25):.4f}',
         f'{np.max(dense_25) - np.max(dense_100):+.4f}'],
        ['Avg Episode Sum', f'{np.mean(sums_100):.2f}', f'{np.mean(sums_25):.2f}',
         f'{np.mean(sums_25) - np.mean(sums_100):+.2f}'],
    ]
    
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#CCCCCC')
        table[(0, i)].set_text_props(fontweight='bold')
    
    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plot saved to: {output_path}")
    return output_path

def main():
    # Paths
    results_100_dir = "/workspace/outputs/reward_test_noisy"      # 100% noisy (gr00t_rollout_noise03)
    results_25_dir = "/workspace/outputs/reward_test_25rollout"   # 25% noisy rollout
    output_path = "/workspace/outputs/reward_comparison_100_vs_25.png"
    
    print("Loading results...")
    results_100 = load_results(results_100_dir)
    results_25 = load_results(results_25_dir)
    
    print(f"100% Noisy: {results_100['success_rate']*100:.0f}% success rate")
    print(f"25% Noisy: {results_25['success_rate']*100:.0f}% success rate")
    
    print("\nCreating comparison plot...")
    create_comparison_plot(results_100, results_25, output_path)

if __name__ == "__main__":
    main()


