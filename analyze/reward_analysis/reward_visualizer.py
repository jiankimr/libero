"""
Reward Visualization for Clean vs Noisy Trajectory Comparison

This module provides visualization tools to compare reward distributions
between clean and noisy (poisoned) model trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from scipy import stats


def plot_reward_curves(
    rewards_dict: Dict[str, List[List[float]]],
    title: str = "Reward Over Time",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot reward curves over time for multiple conditions.
    
    Args:
        rewards_dict: Dict mapping condition name to list of episode rewards
                     e.g., {"clean": [[r1, r2, ...], [r1, r2, ...]], "noisy": [[...], [...]]}
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pastel colors
    colors = {
        'clean': '#90EE90',      # Light green
        'noisy': '#FFB6C1',      # Light pink
        '25%': '#87CEEB',        # Sky blue
        '50%': '#DDA0DD',        # Plum
        '100%': '#F0E68C',       # Khaki
    }
    
    for condition_name, episodes in rewards_dict.items():
        if not episodes:
            continue
            
        # Convert to numpy array (pad shorter episodes)
        max_len = max(len(ep) for ep in episodes)
        padded = np.full((len(episodes), max_len), np.nan)
        for i, ep in enumerate(episodes):
            padded[i, :len(ep)] = ep
        
        # Compute mean and std
        mean_rewards = np.nanmean(padded, axis=0)
        std_rewards = np.nanstd(padded, axis=0)
        
        steps = np.arange(len(mean_rewards))
        
        # Get color
        color = colors.get(condition_name.lower(), plt.cm.tab10(len(colors) % 10))
        
        # Plot mean with confidence interval
        ax.plot(steps, mean_rewards, label=condition_name, color=color, linewidth=2)
        ax.fill_between(
            steps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.3,
            color=color,
        )
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved reward curves to {output_path}")
    
    return fig


def plot_reward_distribution(
    rewards_dict: Dict[str, List[float]],
    title: str = "Reward Distribution Comparison",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot reward distributions for multiple conditions.
    
    Args:
        rewards_dict: Dict mapping condition name to list of rewards
                     e.g., {"clean": [0.5, 0.6, ...], "noisy": [0.4, 0.5, ...]}
        title: Plot title
        output_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Pastel colors
    colors = ['#90EE90', '#FFB6C1', '#87CEEB', '#DDA0DD', '#F0E68C']
    
    # Left: Histogram
    ax1 = axes[0]
    for i, (condition_name, rewards) in enumerate(rewards_dict.items()):
        ax1.hist(
            rewards, 
            bins=30, 
            alpha=0.6, 
            label=f"{condition_name} (μ={np.mean(rewards):.3f})",
            color=colors[i % len(colors)],
        )
    ax1.set_xlabel('Reward', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Histogram', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Box plot
    ax2 = axes[1]
    data = list(rewards_dict.values())
    labels = list(rewards_dict.keys())
    
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Box Plot', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved reward distribution to {output_path}")
    
    return fig


def compute_statistical_test(
    rewards_a: List[float],
    rewards_b: List[float],
    test_type: str = "ks",
) -> Dict[str, float]:
    """
    Perform statistical test to compare two reward distributions.
    
    Args:
        rewards_a: First reward distribution
        rewards_b: Second reward distribution
        test_type: Type of test - "ks" (Kolmogorov-Smirnov), "t" (t-test), "mw" (Mann-Whitney)
        
    Returns:
        Dict with test statistic and p-value
    """
    results = {}
    
    if test_type == "ks":
        stat, pvalue = stats.ks_2samp(rewards_a, rewards_b)
        results['test'] = 'Kolmogorov-Smirnov'
    elif test_type == "t":
        stat, pvalue = stats.ttest_ind(rewards_a, rewards_b)
        results['test'] = 't-test'
    elif test_type == "mw":
        stat, pvalue = stats.mannwhitneyu(rewards_a, rewards_b, alternative='two-sided')
        results['test'] = 'Mann-Whitney U'
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    results['statistic'] = stat
    results['p_value'] = pvalue
    results['significant'] = pvalue < 0.05
    
    return results


def visualize_reward_comparison(
    rewards_dict: Dict[str, List[List[float]]],
    output_dir: str = "./reward_analysis_output",
    title_prefix: str = "",
) -> Dict[str, any]:
    """
    Generate complete reward comparison visualization.
    
    Args:
        rewards_dict: Dict mapping condition name to list of episode rewards
        output_dir: Directory to save output files
        title_prefix: Prefix for plot titles
        
    Returns:
        Dict with analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Plot reward curves over time
    plot_reward_curves(
        rewards_dict,
        title=f"{title_prefix}Reward Over Time",
        output_path=str(output_path / "reward_curves.png"),
    )
    
    # 2. Aggregate rewards for distribution comparison
    aggregated = {}
    for condition, episodes in rewards_dict.items():
        all_rewards = [r for ep in episodes for r in ep]
        aggregated[condition] = all_rewards
    
    # 3. Plot distributions
    plot_reward_distribution(
        aggregated,
        title=f"{title_prefix}Reward Distribution",
        output_path=str(output_path / "reward_distribution.png"),
    )
    
    # 4. Statistical tests
    conditions = list(aggregated.keys())
    if len(conditions) >= 2:
        results['statistical_tests'] = {}
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                cond_a, cond_b = conditions[i], conditions[j]
                test_result = compute_statistical_test(
                    aggregated[cond_a],
                    aggregated[cond_b],
                    test_type="ks",
                )
                results['statistical_tests'][f"{cond_a}_vs_{cond_b}"] = test_result
    
    # 5. Summary statistics
    results['summary'] = {}
    for condition, rewards in aggregated.items():
        results['summary'][condition] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'median': np.median(rewards),
            'count': len(rewards),
        }
    
    # 6. Save results to text file
    with open(output_path / "analysis_results.txt", 'w') as f:
        f.write(f"{title_prefix}Reward Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Summary Statistics:\n")
        f.write("-" * 40 + "\n")
        for condition, stats_dict in results['summary'].items():
            f.write(f"\n{condition}:\n")
            for key, value in stats_dict.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        
        if 'statistical_tests' in results:
            f.write("\n\nStatistical Tests:\n")
            f.write("-" * 40 + "\n")
            for comparison, test_result in results['statistical_tests'].items():
                f.write(f"\n{comparison}:\n")
                f.write(f"  Test: {test_result['test']}\n")
                f.write(f"  Statistic: {test_result['statistic']:.4f}\n")
                f.write(f"  P-value: {test_result['p_value']:.6f}\n")
                f.write(f"  Significant (p<0.05): {test_result['significant']}\n")
                
                if test_result['significant']:
                    f.write("  → Distributions are DISTINGUISHABLE\n")
                else:
                    f.write("  → Distributions are INDISTINGUISHABLE\n")
    
    print(f"\nAnalysis results saved to {output_path}")
    print(f"Files: reward_curves.png, reward_distribution.png, analysis_results.txt")
    
    return results



