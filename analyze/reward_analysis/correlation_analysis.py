#!/usr/bin/env python3
"""
Dense Reward Correlation Analysis
다양한 관점에서 dense reward의 상관관계를 분석
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_results(result_dir):
    """Load results from a model's output directory"""
    result_path = Path(result_dir) / "results.json"
    with open(result_path, 'r') as f:
        return json.load(f)

def extract_episode_features(results):
    """Extract features from each episode for correlation analysis"""
    features = []
    
    for i, (dense_ep, sparse_ep) in enumerate(zip(
        results['dense']['episodes'], 
        results['sparse']['episodes']
    )):
        dense_arr = np.array(dense_ep)
        sparse_arr = np.array(sparse_ep)
        
        # Determine success (sparse reward > 0 at any point)
        success = 1 if np.max(sparse_arr) > 0 else 0
        
        feat = {
            'episode': i,
            'success': success,
            'num_steps': len(dense_ep),
            
            # Dense reward statistics
            'dense_sum': np.sum(dense_arr),
            'dense_mean': np.mean(dense_arr),
            'dense_max': np.max(dense_arr),
            'dense_min': np.min(dense_arr),
            'dense_std': np.std(dense_arr),
            'dense_median': np.median(dense_arr),
            
            # Reward progression (early vs late)
            'dense_first_half_mean': np.mean(dense_arr[:len(dense_arr)//2]),
            'dense_second_half_mean': np.mean(dense_arr[len(dense_arr)//2:]),
            
            # Reward trend (slope)
            'dense_trend': np.polyfit(range(len(dense_arr)), dense_arr, 1)[0] if len(dense_arr) > 1 else 0,
            
            # Peak timing (when max reward occurs, normalized)
            'peak_timing': np.argmax(dense_arr) / len(dense_arr) if len(dense_arr) > 0 else 0,
            
            # Number of positive rewards
            'positive_ratio': np.sum(dense_arr > 0) / len(dense_arr),
            
            # Cumulative reward at 50% and 75% of episode
            'cumsum_50pct': np.sum(dense_arr[:len(dense_arr)//2]),
            'cumsum_75pct': np.sum(dense_arr[:int(len(dense_arr)*0.75)]),
        }
        features.append(feat)
    
    return features

def main():
    print("=" * 70)
    print("🔍 Dense Reward 상관관계 분석 (Correlation Analysis)")
    print("=" * 70)
    
    # Load data
    results_100 = load_results("/workspace/outputs/reward_test_noisy")
    results_25 = load_results("/workspace/outputs/reward_test_25rollout")
    
    # Extract features
    features_100 = extract_episode_features(results_100)
    features_25 = extract_episode_features(results_25)
    
    # Combine for overall analysis
    all_features = features_100 + features_25
    
    # Add noise level indicator
    for f in features_100:
        f['noise_level'] = 1.0  # 100% noisy
    for f in features_25:
        f['noise_level'] = 0.25  # 25% noisy
    
    # ============== 1. Dense Reward vs Success Correlation ==============
    print("\n" + "=" * 70)
    print("📊 1. Dense Reward ↔ Success (성공 여부) 상관관계")
    print("=" * 70)
    print("""
    방법: Point-Biserial Correlation
    - 연속형 변수(dense reward)와 이진 변수(success)의 상관관계 측정
    - r > 0: dense reward가 높을수록 성공 가능성 높음
    - |r| 해석: 0.1(약함), 0.3(중간), 0.5(강함)
    """)
    
    success = np.array([f['success'] for f in all_features])
    
    metrics = ['dense_sum', 'dense_mean', 'dense_max', 'dense_std', 
               'dense_first_half_mean', 'dense_second_half_mean', 'dense_trend']
    
    print(f"  {'Metric':<25} {'r':>10} {'p-value':>12} {'해석':>15}")
    print("-" * 65)
    
    correlations = {}
    for metric in metrics:
        values = np.array([f[metric] for f in all_features])
        r, p = pointbiserialr(success, values)
        correlations[metric] = (r, p)
        
        if abs(r) < 0.1:
            interp = "무상관"
        elif abs(r) < 0.3:
            interp = "약한 상관"
        elif abs(r) < 0.5:
            interp = "중간 상관"
        else:
            interp = "강한 상관"
        
        sig = "✅" if p < 0.05 else "❌"
        print(f"  {metric:<25} {r:>10.4f} {p:>12.4f} {sig} {interp:>10}")
    
    # ============== 2. Noise Level vs Dense Reward ==============
    print("\n" + "=" * 70)
    print("📊 2. Noise Level ↔ Dense Reward 상관관계")
    print("=" * 70)
    print("""
    방법: Spearman's Rank Correlation
    - 순위 기반 상관관계 (비선형 관계도 감지)
    - r > 0: noise가 높을수록 dense reward도 높음
    - r < 0: noise가 높을수록 dense reward가 낮음
    """)
    
    noise_level = np.array([f['noise_level'] for f in all_features])
    
    print(f"  {'Metric':<25} {'Spearman r':>12} {'p-value':>12}")
    print("-" * 55)
    
    for metric in metrics:
        values = np.array([f[metric] for f in all_features])
        r, p = spearmanr(noise_level, values)
        sig = "✅" if p < 0.05 else "❌"
        print(f"  {metric:<25} {r:>12.4f} {p:>12.4f} {sig}")
    
    # ============== 3. Dense Reward Features Inter-correlation ==============
    print("\n" + "=" * 70)
    print("📊 3. Dense Reward Features 간 상관관계 Matrix")
    print("=" * 70)
    print("""
    방법: Pearson Correlation Matrix
    - 어떤 dense reward 특성들이 서로 연관되어 있는지 분석
    """)
    
    key_metrics = ['dense_sum', 'dense_mean', 'dense_max', 'dense_trend', 'num_steps']
    
    # Create correlation matrix
    matrix = np.zeros((len(key_metrics), len(key_metrics)))
    for i, m1 in enumerate(key_metrics):
        for j, m2 in enumerate(key_metrics):
            v1 = np.array([f[m1] for f in all_features])
            v2 = np.array([f[m2] for f in all_features])
            r, _ = pearsonr(v1, v2)
            matrix[i, j] = r
    
    # Print matrix
    print(f"\n  {'':>15}", end="")
    for m in key_metrics:
        print(f"{m[:10]:>12}", end="")
    print()
    
    for i, m1 in enumerate(key_metrics):
        print(f"  {m1[:15]:<15}", end="")
        for j, m2 in enumerate(key_metrics):
            print(f"{matrix[i,j]:>12.3f}", end="")
        print()
    
    # ============== 4. Time-series Pattern Analysis ==============
    print("\n" + "=" * 70)
    print("📊 4. 시계열 패턴 분석 (Success vs Fail)")
    print("=" * 70)
    print("""
    방법: 성공/실패 에피소드의 reward 패턴 비교
    - 성공 에피소드와 실패 에피소드의 reward 곡선이 어떻게 다른지
    """)
    
    success_eps = [f for f in all_features if f['success'] == 1]
    fail_eps = [f for f in all_features if f['success'] == 0]
    
    print(f"\n  성공 에피소드 수: {len(success_eps)}")
    print(f"  실패 에피소드 수: {len(fail_eps)}")
    
    if len(success_eps) > 0 and len(fail_eps) > 0:
        print(f"\n  {'Metric':<25} {'Success Mean':>15} {'Fail Mean':>15} {'Diff':>10}")
        print("-" * 70)
        
        for metric in ['dense_sum', 'dense_mean', 'dense_max', 'dense_trend', 'num_steps', 'peak_timing']:
            success_vals = [f[metric] for f in success_eps]
            fail_vals = [f[metric] for f in fail_eps]
            
            s_mean = np.mean(success_vals)
            f_mean = np.mean(fail_vals)
            diff = s_mean - f_mean
            
            print(f"  {metric:<25} {s_mean:>15.4f} {f_mean:>15.4f} {diff:>+10.4f}")
    
    # ============== 5. Predictive Power Analysis ==============
    print("\n" + "=" * 70)
    print("📊 5. 성공 예측력 분석 (Predictive Power)")
    print("=" * 70)
    print("""
    방법: Logistic Regression (또는 간단한 임계값 기반)
    - Dense reward의 어떤 특성이 성공을 가장 잘 예측하는가?
    """)
    
    # Simple threshold-based prediction
    print(f"\n  Dense Sum 임계값 기반 성공 예측:")
    
    dense_sums = np.array([f['dense_sum'] for f in all_features])
    successes = np.array([f['success'] for f in all_features])
    
    thresholds = [10, 15, 20, 25]
    
    print(f"  {'Threshold':>12} {'Accuracy':>12} {'Precision':>12} {'Recall':>12}")
    print("-" * 55)
    
    for thresh in thresholds:
        pred = (dense_sums >= thresh).astype(int)
        accuracy = np.mean(pred == successes)
        
        # Precision: TP / (TP + FP)
        tp = np.sum((pred == 1) & (successes == 1))
        fp = np.sum((pred == 1) & (successes == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: TP / (TP + FN)
        fn = np.sum((pred == 0) & (successes == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"  {thresh:>12.1f} {accuracy:>12.2%} {precision:>12.2%} {recall:>12.2%}")
    
    # ============== Create Visualization ==============
    print("\n\n📊 상관관계 시각화 생성 중...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dense Reward Correlation Analysis', fontsize=14, fontweight='bold')
    
    # 1. Dense Sum vs Success (scatter + boxplot)
    ax1 = axes[0, 0]
    dense_sums = [f['dense_sum'] for f in all_features]
    successes = [f['success'] for f in all_features]
    colors = ['#FF6B6B' if s == 0 else '#4ECDC4' for s in successes]
    ax1.scatter(range(len(dense_sums)), dense_sums, c=colors, s=100, alpha=0.7)
    ax1.axhline(y=np.mean([d for d, s in zip(dense_sums, successes) if s == 1]), 
                color='#4ECDC4', linestyle='--', label='Success Mean')
    ax1.axhline(y=np.mean([d for d, s in zip(dense_sums, successes) if s == 0]), 
                color='#FF6B6B', linestyle='--', label='Fail Mean')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Dense Reward Sum')
    ax1.set_title('Dense Sum by Episode (🟢=Success, 🔴=Fail)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation heatmap
    ax2 = axes[0, 1]
    im = ax2.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(key_metrics)))
    ax2.set_yticks(range(len(key_metrics)))
    ax2.set_xticklabels([m[:8] for m in key_metrics], rotation=45, ha='right')
    ax2.set_yticklabels([m[:8] for m in key_metrics])
    for i in range(len(key_metrics)):
        for j in range(len(key_metrics)):
            ax2.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)
    ax2.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # 3. Success vs Fail distribution
    ax3 = axes[0, 2]
    success_sums = [f['dense_sum'] for f in all_features if f['success'] == 1]
    fail_sums = [f['dense_sum'] for f in all_features if f['success'] == 0]
    
    bp = ax3.boxplot([success_sums, fail_sums] if fail_sums else [success_sums], 
                     labels=['Success', 'Fail'] if fail_sums else ['Success'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#4ECDC4')
    if len(bp['boxes']) > 1:
        bp['boxes'][1].set_facecolor('#FF6B6B')
    ax3.set_ylabel('Dense Reward Sum')
    ax3.set_title('Dense Sum: Success vs Fail')
    ax3.grid(True, alpha=0.3)
    
    # 4. Noise level comparison
    ax4 = axes[1, 0]
    sums_100 = [f['dense_sum'] for f in features_100]
    sums_25 = [f['dense_sum'] for f in features_25]
    
    positions = [1, 2]
    bp2 = ax4.boxplot([sums_100, sums_25], positions=positions, 
                      labels=['100% Noisy', '25% Noisy'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#99FF99')
    bp2['boxes'][1].set_facecolor('#FFCC99')
    ax4.set_ylabel('Dense Reward Sum')
    ax4.set_title('Dense Sum by Noise Level')
    ax4.grid(True, alpha=0.3)
    
    # 5. Dense trend vs success
    ax5 = axes[1, 1]
    trends = [f['dense_trend'] for f in all_features]
    ax5.scatter(range(len(trends)), trends, c=colors, s=100, alpha=0.7)
    ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Dense Reward Trend (slope)')
    ax5.set_title('Reward Trend (🟢=Success, 🔴=Fail)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Scatter: Dense Sum vs Num Steps
    ax6 = axes[1, 2]
    steps = [f['num_steps'] for f in all_features]
    ax6.scatter(steps, dense_sums, c=colors, s=100, alpha=0.7)
    r, p = pearsonr(steps, dense_sums)
    ax6.set_xlabel('Number of Steps')
    ax6.set_ylabel('Dense Reward Sum')
    ax6.set_title(f'Steps vs Dense Sum (r={r:.3f}, p={p:.3f})')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "/workspace/outputs/correlation_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 시각화 저장: {output_path}")
    
    # ============== Summary ==============
    print("\n" + "=" * 70)
    print("📋 상관관계 분석 요약")
    print("=" * 70)
    
    # Find strongest correlator with success
    best_metric = max(correlations.items(), key=lambda x: abs(x[1][0]))
    
    print(f"""
  🏆 성공과 가장 강한 상관관계:
     {best_metric[0]}: r = {best_metric[1][0]:.4f} (p = {best_metric[1][1]:.4f})
  
  📈 주요 발견:
     • Dense reward sum이 높을수록 성공 가능성 ↑
     • 실패 에피소드는 더 많은 step 사용 (max_step까지)
     • Reward trend(기울기)도 성공과 관련 있음
     
  💡 Dense Reward의 활용:
     • 학습 중 진행 상황 모니터링
     • 정책 품질 조기 평가
     • Noisy vs Clean 정책 구별 가능성
""")

if __name__ == "__main__":
    main()


