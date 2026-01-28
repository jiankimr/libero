#!/usr/bin/env python3
"""
Statistical significance tests for model comparison
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

def load_results(result_dir):
    """Load results from a model's output directory"""
    result_path = Path(result_dir) / "results.json"
    with open(result_path, 'r') as f:
        return json.load(f)

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Calculate bootstrap confidence interval"""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1-ci)/2 * 100)
    upper = np.percentile(boot_means, (1+ci)/2 * 100)
    return lower, upper

def main():
    # Load data
    results_100 = load_results("/workspace/outputs/reward_test_noisy")
    results_25 = load_results("/workspace/outputs/reward_test_25rollout")
    
    print("=" * 70)
    print("통계적 유의성 검증 (Statistical Significance Tests)")
    print("=" * 70)
    
    # ============== 1. Success Rate - Fisher's Exact Test ==============
    print("\n📊 1. 성공률 비교 (Fisher's Exact Test)")
    print("-" * 50)
    
    # Success counts
    n_100 = 10  # total episodes
    n_25 = 10
    success_100 = int(results_100['success_rate'] * n_100)  # 9
    success_25 = int(results_25['success_rate'] * n_25)     # 8
    
    # Contingency table: [[success_100, fail_100], [success_25, fail_25]]
    contingency = [[success_100, n_100 - success_100],
                   [success_25, n_25 - success_25]]
    
    odds_ratio, p_fisher = stats.fisher_exact(contingency)
    
    print(f"  100% Noisy: {success_100}/{n_100} 성공 ({results_100['success_rate']*100:.0f}%)")
    print(f"  25% Noisy:  {success_25}/{n_25} 성공 ({results_25['success_rate']*100:.0f}%)")
    print(f"  Fisher's Exact Test p-value: {p_fisher:.4f}")
    print(f"  Odds Ratio: {odds_ratio:.2f}")
    
    if p_fisher < 0.05:
        print(f"  ✅ 결론: 통계적으로 유의미한 차이 있음 (p < 0.05)")
    else:
        print(f"  ❌ 결론: 통계적으로 유의미한 차이 없음 (p >= 0.05)")
    
    # ============== 2. Dense Reward Mean - t-test ==============
    print("\n📊 2. Dense Reward 평균 비교 (Independent t-test)")
    print("-" * 50)
    
    # Get per-episode mean rewards
    ep_means_100 = [np.mean(ep) for ep in results_100['dense']['episodes']]
    ep_means_25 = [np.mean(ep) for ep in results_25['dense']['episodes']]
    
    t_stat, p_ttest = stats.ttest_ind(ep_means_100, ep_means_25)
    
    print(f"  100% Noisy 에피소드 평균: {np.mean(ep_means_100):.4f} ± {np.std(ep_means_100):.4f}")
    print(f"  25% Noisy 에피소드 평균:  {np.mean(ep_means_25):.4f} ± {np.std(ep_means_25):.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_ttest:.4f}")
    
    if p_ttest < 0.05:
        print(f"  ✅ 결론: 통계적으로 유의미한 차이 있음 (p < 0.05)")
    else:
        print(f"  ❌ 결론: 통계적으로 유의미한 차이 없음 (p >= 0.05)")
    
    # ============== 3. Mann-Whitney U Test (비모수) ==============
    print("\n📊 3. Dense Reward 분포 비교 (Mann-Whitney U Test)")
    print("-" * 50)
    
    dense_100 = np.concatenate(results_100['dense']['episodes'])
    dense_25 = np.concatenate(results_25['dense']['episodes'])
    
    u_stat, p_mann = stats.mannwhitneyu(dense_100, dense_25, alternative='two-sided')
    
    print(f"  100% Noisy 전체 스텝 수: {len(dense_100)}")
    print(f"  25% Noisy 전체 스텝 수:  {len(dense_25)}")
    print(f"  U-statistic: {u_stat:.0f}")
    print(f"  p-value: {p_mann:.6f}")
    
    if p_mann < 0.05:
        print(f"  ✅ 결론: 분포 차이가 통계적으로 유의미함 (p < 0.05)")
    else:
        print(f"  ❌ 결론: 분포 차이가 통계적으로 유의미하지 않음 (p >= 0.05)")
    
    # ============== 4. Bootstrap Confidence Intervals ==============
    print("\n📊 4. Bootstrap 신뢰구간 (95% CI)")
    print("-" * 50)
    
    np.random.seed(42)
    
    # Episode sum bootstrap
    ep_sums_100 = [np.sum(ep) for ep in results_100['dense']['episodes']]
    ep_sums_25 = [np.sum(ep) for ep in results_25['dense']['episodes']]
    
    ci_100 = bootstrap_ci(ep_sums_100)
    ci_25 = bootstrap_ci(ep_sums_25)
    
    print(f"  100% Noisy 에피소드 합계: {np.mean(ep_sums_100):.2f} [95% CI: {ci_100[0]:.2f} - {ci_100[1]:.2f}]")
    print(f"  25% Noisy 에피소드 합계:  {np.mean(ep_sums_25):.2f} [95% CI: {ci_25[0]:.2f} - {ci_25[1]:.2f}]")
    
    # Check if CIs overlap
    if ci_100[1] < ci_25[0] or ci_25[1] < ci_100[0]:
        print(f"  ✅ 신뢰구간이 겹치지 않음 → 유의미한 차이 가능성 높음")
    else:
        print(f"  ⚠️ 신뢰구간이 겹침 → 차이가 우연일 수 있음")
    
    # ============== 5. Effect Size (Cohen's d) ==============
    print("\n📊 5. 효과 크기 (Cohen's d)")
    print("-" * 50)
    
    pooled_std = np.sqrt((np.var(ep_means_100) + np.var(ep_means_25)) / 2)
    cohens_d = (np.mean(ep_means_100) - np.mean(ep_means_25)) / pooled_std if pooled_std > 0 else 0
    
    print(f"  Cohen's d: {cohens_d:.4f}")
    
    if abs(cohens_d) < 0.2:
        effect_size = "무시할 수준 (negligible)"
    elif abs(cohens_d) < 0.5:
        effect_size = "작은 효과 (small)"
    elif abs(cohens_d) < 0.8:
        effect_size = "중간 효과 (medium)"
    else:
        effect_size = "큰 효과 (large)"
    
    print(f"  해석: {effect_size}")
    
    # ============== Summary ==============
    print("\n" + "=" * 70)
    print("📋 종합 결론")
    print("=" * 70)
    print(f"""
  • 성공률 차이 (90% vs 80%): p = {p_fisher:.4f}
    → 10개 샘플로는 10% 차이가 우연일 가능성 높음
    
  • Dense Reward 평균 차이: p = {p_ttest:.4f}
    → {'유의미' if p_ttest < 0.05 else '유의미하지 않음'}
    
  • 효과 크기: {effect_size}
  
  💡 권장사항:
    - 더 많은 태스크 (10개 전체) 테스트
    - 에피소드 수 증가 (30개 이상)
    - Clean 모델과의 비교 추가
""")
    
    # Calculate required sample size for 80% power
    print("\n📊 6. 필요 샘플 크기 추정")
    print("-" * 50)
    
    # Using current effect size, estimate needed n for 80% power
    # For success rate difference of 10%, we need approximately:
    from scipy.stats import norm
    
    p1, p2 = 0.9, 0.8
    alpha = 0.05
    power = 0.80
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    p_bar = (p1 + p2) / 2
    n_needed = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + 
                 z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / ((p1 - p2) ** 2)
    
    print(f"  현재 효과 (10% 차이)를 80% 확률로 감지하려면:")
    print(f"  → 각 그룹당 약 {int(np.ceil(n_needed))}개 에피소드 필요")
    print(f"  → 현재는 10개씩 → 검정력(power) 부족")

if __name__ == "__main__":
    main()


