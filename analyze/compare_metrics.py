"""
Compare Metrics between two conditions (before/after noise)
-----------------------------------------------------------

두 조건(noise 없음/있음)의 물리량을 전반적으로 비교하는 스크립트.

1. NPY 파일 기반: torque_current_*.npy를 통한 피로 손상(damage) 비교 → life_ratio 계산
2. CSV 파일 기반: analysis_summary*.csv를 통한 주요 물리량 비교

사용 예시:
    python analyze/compare_metrics.py \
    --before_dir ./analysis/analysis_libero_10_20251107_172053_noise_00000/ \
    --after_dir ./analysis/analysis_libero_10_20251107_145142_noise_05000_dim_action.eef_pos_delta[2]/
    (output txt 자동생성)

python analyze/compare_metrics.py \
    --before_dir ./analysis/analysis_libero_10_20251202_170035_noise_00000_clean/ \
    --after_dir ./analysis/analysis_libero_10_20251202_171746_noise_03000_dim_action.x_clean_noise/


(--output_npy_csv 옵션)
--output_npy_csv ./results/comparison_npy.csv
"""

import argparse
import numpy as np
import pandas as pd
import pathlib
import csv
import logging
from typing import Optional, Dict, Tuple, List
from collections import defaultdict

try:
    import rainflow
    HAS_RAINFLOW = True
except ImportError:
    HAS_RAINFLOW = False
    logging.warning("rainflow package not installed. Life ratio calculation will be skipped.")

try:
    import fatpack
    HAS_FATPACK = True
except ImportError:
    HAS_FATPACK = False
    logging.warning("fatpack package not installed. Fatpack-based life ratio calculation will be skipped.")

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)


# 주요 비교 메트릭 정의
COMPARISON_METRICS = {
    "🔋 Energy (전기 에너지)": [
        "energy_draw.mean",
        "energy_draw.sum",
        "energy_net.mean",
        "energy_net.sum",
        "energy_regen.mean",
        "energy_regen.sum",
    ],
    "⚙️  Torque (모터 토크)": [
        "torque_average.mean",
        "torque_average.max",
        "torque_average.std",
        "torque_current.mean",
        "torque_current.max",
        "torque_current.std",
    ],
    "🏃 Kinematics (운동학)": [
        "acceleration.mean",
        "acceleration.max",
        "acceleration.std",
        "velocity.mean",
        "velocity.max",
        "velocity.std",
    ],
    "⚡ Jerk (부드러움/평탄성)": [
        "kinematic_jerk_delta.mean",
        "kinematic_jerk_delta.max",
        "kinematic_jerk_delta.std",
        "actuator_jerk_delta.mean",
        "actuator_jerk_delta.max",
        "actuator_jerk_delta.std",
    ],
}


def find_csv_file(directory: pathlib.Path) -> Optional[pathlib.Path]:
    """
    analysis_summary*.csv 파일 찾기
    
    Args:
        directory: 검색할 디렉토리 (e.g., LIBERO/analysis/analysis_libero_10_...)
        
    Returns:
        찾은 CSV 파일 경로, 없으면 None
    """
    import re
    
    dir_path = pathlib.Path(directory)
    parent_dir = dir_path.parent  # LIBERO/analysis/
    
    # 디렉토리 이름 추출
    # e.g., "analysis_libero_10_20251107_145142_noise_05000_dim_action.eef_pos_delta[2]_noisy"
    dir_name = dir_path.name
    
    # 1️⃣ 부모 디렉토리(LIBERO/analysis/)에서 모든 CSV 찾기
    csv_files = sorted(parent_dir.glob("analysis_summary*.csv"))
    
    logger.debug(f"   🔍 Looking for CSV in {parent_dir}")
    logger.debug(f"   📁 Directory name: {dir_name}")
    logger.debug(f"   📋 Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
    
    if not csv_files:
        logger.debug(f"   ⚠️  No CSV files found in {parent_dir}")
        return None
    
    def extract_core_pattern(name: str) -> str:
        """
        마지막 suffix를 제거하여 핵심 패턴 추출
        
        형식: {task_suite}_{datetime}_{noise_str}{dim_str}_{suffix}
        예시: "libero_10_20251202_170044_noise_00000_dim_action.x_noisy"
              → "libero_10_20251202_170044_noise_00000_dim_action.x"
        
        또는 dim이 없는 경우:
        "libero_10_20251202_170044_noise_00000_clean"
        → "libero_10_20251202_170044_noise_00000"
        
        규칙: 마지막 '_'로 시작하는 단어가 suffix로 간주됨 (단, noise_나 dim_은 제외)
        """
        # 이름을 '_'로 분할
        parts = name.split('_')
        
        # 뒤에서부터 검사하여 suffix 찾기
        # noise_, dim_으로 시작하지 않는 마지막 부분이 suffix
        core_parts = []
        found_suffix = False
        
        for i in range(len(parts)):
            part = parts[i]
            # 앞부분은 무조건 포함
            if i < len(parts) - 1:
                core_parts.append(part)
            else:
                # 마지막 부분: noise나 dim으로 시작하면 포함, 아니면 suffix로 간주
                if part.startswith('noise') or part.startswith('dim'):
                    core_parts.append(part)
                # 마지막이 숫자로만 이루어져 있으면 포함 (예: noise_00000)
                elif part.isdigit() or all(c.isdigit() or c == '.' for c in part.replace('[', '').replace(']', '')):
                    core_parts.append(part)
                # 그 외는 suffix로 간주하여 제외
                else:
                    found_suffix = True
                    logger.debug(f"      → Detected suffix: '{part}'")
        
        return '_'.join(core_parts)
    
    # 2️⃣ 디렉토리 이름과 CSV 파일명 매칭 (suffix 무시)
    # 디렉토리 이름에서 "analysis_" 제거
    dir_name_suffix = dir_name.replace("analysis_", "")
    dir_name_core = extract_core_pattern(dir_name_suffix)
    
    logger.debug(f"   📝 Dir core (suffix removed): '{dir_name_core}'")
    
    # 정확한 매칭 시도 (suffix 제거 후)
    for csv_file in csv_files:
        csv_name = csv_file.stem  # "analysis_summary_libero_10_20251107_145142..."
        csv_name_suffix = csv_name.replace("analysis_summary_", "")
        csv_name_core = extract_core_pattern(csv_name_suffix)
        
        logger.debug(f"   Comparing cores: '{dir_name_core}' vs '{csv_name_core}'")
        
        if dir_name_core == csv_name_core:
            logger.info(f"   ✅ CSV found (exact match without suffix): {csv_file.name}")
            return csv_file
    
    # 3️⃣ 부분 매칭 (핵심 부분이 포함되어 있으면 매칭)
    for csv_file in csv_files:
        csv_name = csv_file.stem
        csv_name_suffix = csv_name.replace("analysis_summary_", "")
        csv_name_core = extract_core_pattern(csv_name_suffix)
        
        # 더 짧은 것이 긴 것에 포함되면 매칭
        if len(dir_name_core) <= len(csv_name_core):
            if dir_name_core in csv_name_core:
                logger.info(f"   ✅ CSV found (partial match): {csv_file.name}")
                return csv_file
        else:
            if csv_name_core in dir_name_core:
                logger.info(f"   ✅ CSV found (partial match): {csv_file.name}")
                return csv_file
    
    logger.warning(f"   ⚠️  No matching CSV found for directory: {dir_name}")
    return None


def compare_csv_metrics(before_csv: pathlib.Path, after_csv: pathlib.Path) -> Tuple[str, Dict]:
    """
    CSV 파일 기반 물리량 비교
    
    Returns:
        Tuple[출력 텍스트, 비교 결과 딕셔너리]
    """
    logger.info(f"\n📊 CSV 파일 비교 시작...")
    
    try:
        df_before = pd.read_csv(before_csv)
        df_after = pd.read_csv(after_csv)
        
        logger.info(f"   Before (No Noise): {df_before.shape[0]} episodes, {df_before.shape[1]} metrics")
        logger.info(f"   After (With Noise): {df_after.shape[0]} episodes, {df_after.shape[1]} metrics")
    except Exception as e:
        logger.error(f"CSV 파일 로드 실패: {e}")
        return "", {}
    
    # 성공한 에피소드만 필터링 (물리량 비교용)
    if "success" in df_before.columns and "success" in df_after.columns:
        df_before_success = df_before[df_before["success"] == 1].copy()
        df_after_success = df_after[df_after["success"] == 1].copy()
        
        logger.info(f"   Before (Success only): {df_before_success.shape[0]} episodes")
        logger.info(f"   After (Success only): {df_after_success.shape[0]} episodes")
    else:
        # success 컬럼이 없으면 전체 사용
        df_before_success = df_before.copy()
        df_after_success = df_after.copy()
    
    output_lines = []
    output_lines.append("\n" + "=" * 120)
    output_lines.append("📊 CSV 기반 물리량 비교 (BASELINE vs WITH NOISE)")
    output_lines.append("=" * 120)
    output_lines.append("※ 물리량 비교는 양쪽 모두 성공한 에피소드만 사용합니다.\n")
    
    comparison_results = {}
    all_pct_changes = []
    
    # 각 카테고리별 비교
    for category, metrics in COMPARISON_METRICS.items():
        output_lines.append(f"\n{category}")
        output_lines.append("-" * 120)
        
        category_results = {}
        
        for metric in metrics:
            if metric not in df_before_success.columns or metric not in df_after_success.columns:
                continue
            
            before_vals = df_before_success[metric].dropna().astype(float)
            after_vals = df_after_success[metric].dropna().astype(float)
            
            if len(before_vals) == 0 or len(after_vals) == 0:
                continue
            
            # 통계 계산
            before_mean = before_vals.mean()
            before_std = before_vals.std()
            before_min = before_vals.min()
            before_max = before_vals.max()
            
            after_mean = after_vals.mean()
            after_std = after_vals.std()
            after_min = after_vals.min()
            after_max = after_vals.max()
            
            # 차이 계산
            mean_diff = after_mean - before_mean
            mean_pct = (mean_diff / before_mean * 100) if before_mean != 0 else 0
            
            category_results[metric] = {
                "before_mean": before_mean,
                "before_std": before_std,
                "after_mean": after_mean,
                "after_std": after_std,
                "mean_diff": mean_diff,
                "mean_pct": mean_pct,
            }
            
            all_pct_changes.append(abs(mean_pct))
            
            # 출력 포매팅 (변화율에 따라 색상 힌트 추가)
            symbol = "↑" if mean_pct > 0 else "↓" if mean_pct < 0 else "→"
            output_lines.append(f"  {metric:45s} | Before: {before_mean:12.4f}±{before_std:8.4f} → After: {after_mean:12.4f}±{after_std:8.4f} | Δ {mean_pct:+6.1f}% {symbol}")
        
        comparison_results[category] = category_results
    
    # 성공률 비교
    output_lines.append(f"\n✅ 작업 성공률 비교")
    output_lines.append("-" * 120)
    
    if "success" in df_before.columns and "success" in df_after.columns:
        success_before = df_before["success"].sum() / len(df_before) * 100
        success_after = df_after["success"].sum() / len(df_after) * 100
        success_diff = success_after - success_before
        
        output_lines.append(f"  Before (No Noise):  {success_before:6.1f}% ({int(df_before['success'].sum())}/{len(df_before)} episodes)")
        output_lines.append(f"  After (With Noise): {success_after:6.1f}% ({int(df_after['success'].sum())}/{len(df_after)} episodes)")
        
        if success_diff > 0:
            output_lines.append(f"  Difference:         {success_diff:+6.1f}% (개선 ✅)")
        elif success_diff < 0:
            output_lines.append(f"  Difference:         {success_diff:+6.1f}% (악화 ⚠️)")
        else:
            output_lines.append(f"  Difference:         {success_diff:+6.1f}% (변화 없음 ➡️)")
        
        comparison_results["Success Rate"] = {
            "before": success_before,
            "after": success_after,
            "diff": success_diff,
        }
    
    # 1️⃣ 전체 요약 통계
    output_lines.append(f"\n📈 전체 물리량 변화 요약 (모든 메트릭 종합)")
    output_lines.append("-" * 120)
    
    if all_pct_changes:
        mean_change = np.mean(all_pct_changes)
        max_change = np.max(all_pct_changes)
        min_change = np.min(all_pct_changes)
        
        output_lines.append(f"  평균 변화율:  {mean_change:+.2f}% (모든 메트릭 평균)")
        output_lines.append(f"  최대 변화율:  {max_change:+.2f}% (가장 큰 증가)")
        output_lines.append(f"  최소 변화율:  {min_change:+.2f}% (가장 큰 감소)")
        
        # 전체 평가
        if mean_change > 5:
            output_lines.append(f"\n  ⚠️  전체 평가: 노이즈로 인해 시스템 성능 악화 (평균 +{mean_change:.1f}%)")
        elif mean_change < -5:
            output_lines.append(f"\n  ✅ 전체 평가: 노이즈로 인해 시스템 효율화 (평균 {mean_change:.1f}%) - 예외적")
        else:
            output_lines.append(f"\n  ➡️  전체 평가: 노이즈의 영향 미미 (평균 {mean_change:+.1f}%)")
    
    # Joint별 실제 토크 분석 (torque_current: 실제 사용되는 토크)
    output_lines.append(f"\n⚙️  Joint별 실제 토크 분석 (torque_current - 실제 모터 사용 토크):\n")
    
    joint_torque_stats = defaultdict(lambda: {"before_mean": [], "after_mean": [], "before_max": [], "after_max": []})
    
    for j in range(7):
        for stat_type in ["mean", "max"]:
            col_name = f"torque_current.joint_{j}_{stat_type}"
            if col_name in df_before_success.columns and col_name in df_after_success.columns:
                before_vals = df_before_success[col_name].dropna().astype(float)
                after_vals = df_after_success[col_name].dropna().astype(float)
                
                if len(before_vals) > 0 and len(after_vals) > 0:
                    if stat_type == "mean":
                        joint_torque_stats[j]["before_mean"].append(before_vals.mean())
                        joint_torque_stats[j]["after_mean"].append(after_vals.mean())
                    else:  # stat_type == "max"
                        joint_torque_stats[j]["before_max"].append(before_vals.max())
                        joint_torque_stats[j]["after_max"].append(after_vals.max())
    
    if joint_torque_stats:
        output_lines.append(f"{'Joint':8s} | {'Before Mean':15s} | {'After Mean':15s} | {'Before Max':15s} | {'After Max':15s} | {'Status':20s}")
        output_lines.append(f"{'-'*120}")
        
        for j in range(7):
            if j in joint_torque_stats:
                stats = joint_torque_stats[j]
                if stats["before_mean"] and stats["after_mean"]:
                    b_mean = stats["before_mean"][0]
                    a_mean = stats["after_mean"][0]
                    b_max = stats["before_max"][0] if stats["before_max"] else b_mean
                    a_max = stats["after_max"][0] if stats["after_max"] else a_mean
                    
                    pct_change_mean = ((a_mean - b_mean) / b_mean * 100) if b_mean != 0 else 0
                    pct_change_max = ((a_max - b_max) / b_max * 100) if b_max != 0 else 0
                    
                    status = "📈 증가" if pct_change_mean > 0 else "📉 감소" if pct_change_mean < 0 else "➡️ 동일"
                    
                    output_lines.append(
                        f"Joint_{j}  | {b_mean:13.4f}  | {a_mean:13.4f}  | {b_max:13.4f}  | {a_max:13.4f}  | {status} ({pct_change_mean:+.1f}%)"
                    )
    
    # Joint별 가속도 분석
    output_lines.append(f"\n🏃 Joint별 가속도 분석 (각 조인트의 평균 가속도 비교):\n")
    
    joint_accel_stats = defaultdict(lambda: {"before_mean": [], "after_mean": [], "before_max": [], "after_max": []})
    
    for j in range(7):
        for stat_type in ["mean", "max"]:
            col_name = f"acceleration.joint_{j}_{stat_type}"
            if col_name in df_before_success.columns and col_name in df_after_success.columns:
                before_vals = df_before_success[col_name].dropna().astype(float)
                after_vals = df_after_success[col_name].dropna().astype(float)
                
                if len(before_vals) > 0 and len(after_vals) > 0:
                    if stat_type == "mean":
                        joint_accel_stats[j]["before_mean"].append(before_vals.mean())
                        joint_accel_stats[j]["after_mean"].append(after_vals.mean())
                    else:
                        joint_accel_stats[j]["before_max"].append(before_vals.mean())
                        joint_accel_stats[j]["after_max"].append(after_vals.mean())
    
    if joint_accel_stats:
        output_lines.append(f"{'Joint':8s} | {'Before Mean':15s} | {'After Mean':15s} | {'Before Max':15s} | {'After Max':15s} | {'Status':20s}")
        output_lines.append(f"{'-'*120}")
        
        for j in range(7):
            if j in joint_accel_stats:
                stats = joint_accel_stats[j]
                if stats["before_mean"] and stats["after_mean"]:
                    b_mean = stats["before_mean"][0]
                    a_mean = stats["after_mean"][0]
                    b_max = stats["before_max"][0] if stats["before_max"] else b_mean
                    a_max = stats["after_max"][0] if stats["after_max"] else a_mean
                    
                    pct_change_mean = ((a_mean - b_mean) / b_mean * 100) if b_mean != 0 else 0
                    pct_change_max = ((a_max - b_max) / b_max * 100) if b_max != 0 else 0
                    
                    status = "📈 증가" if pct_change_mean > 0 else "📉 감소" if pct_change_mean < 0 else "➡️ 동일"
                    
                    output_lines.append(
                        f"Joint_{j}  | {b_mean:13.4f}  | {a_mean:13.4f}  | {b_max:13.4f}  | {a_max:13.4f}  | {status} ({pct_change_mean:+.1f}%)"
                    )
    
    # Joint별 Average Torque 분석
    output_lines.append(f"\n⚙️  Joint별 Average Torque 분석 (torque_average - 평균 토크):\n")
    
    joint_avg_torque_stats = defaultdict(lambda: {"before_mean": [], "after_mean": [], "before_max": [], "after_max": [], "before_std": [], "after_std": []})
    
    for j in range(7):
        for stat_type in ["mean", "max", "std"]:
            col_name = f"torque_average.joint_{j}_{stat_type}"
            if col_name in df_before_success.columns and col_name in df_after_success.columns:
                before_vals = df_before_success[col_name].dropna().astype(float)
                after_vals = df_after_success[col_name].dropna().astype(float)
                
                if len(before_vals) > 0 and len(after_vals) > 0:
                    if stat_type == "mean":
                        joint_avg_torque_stats[j]["before_mean"].append(before_vals.mean())
                        joint_avg_torque_stats[j]["after_mean"].append(after_vals.mean())
                    elif stat_type == "max":
                        joint_avg_torque_stats[j]["before_max"].append(before_vals.mean())
                        joint_avg_torque_stats[j]["after_max"].append(after_vals.mean())
                    else:
                        joint_avg_torque_stats[j]["before_std"].append(before_vals.mean())
                        joint_avg_torque_stats[j]["after_std"].append(after_vals.mean())
    
    if joint_avg_torque_stats:
        output_lines.append(f"{'Joint':8s} | {'Before Mean':15s} | {'After Mean':15s} | {'Before Std':15s} | {'After Std':15s} | {'Status':20s}")
        output_lines.append(f"{'-'*120}")
        
        for j in range(7):
            if j in joint_avg_torque_stats:
                stats = joint_avg_torque_stats[j]
                if stats["before_mean"] and stats["after_mean"]:
                    b_mean = stats["before_mean"][0]
                    a_mean = stats["after_mean"][0]
                    b_std = stats["before_std"][0] if stats["before_std"] else 0
                    a_std = stats["after_std"][0] if stats["after_std"] else 0
                    
                    pct_change_mean = ((a_mean - b_mean) / abs(b_mean) * 100) if b_mean != 0 else 0
                    
                    status = "📈 증가" if pct_change_mean > 0 else "📉 감소" if pct_change_mean < 0 else "➡️ 동일"
                    
                    output_lines.append(
                        f"Joint_{j}  | {b_mean:13.4f}  | {a_mean:13.4f}  | {b_std:13.4f}  | {a_std:13.4f}  | {status} ({pct_change_mean:+.1f}%)"
                    )
    
    # Joint별 Kinematic Jerk 분석
    output_lines.append(f"\n⚡ Joint별 Kinematic Jerk 분석 (kinematic_jerk_delta - 운동학적 점프):\n")
    
    joint_kin_jerk_stats = defaultdict(lambda: {"before_mean": [], "after_mean": [], "before_max": [], "after_max": []})
    
    for j in range(7):
        for stat_type in ["mean", "max"]:
            col_name = f"kinematic_jerk_delta.joint_{j}_{stat_type}"
            if col_name in df_before_success.columns and col_name in df_after_success.columns:
                before_vals = df_before_success[col_name].dropna().astype(float)
                after_vals = df_after_success[col_name].dropna().astype(float)
                
                if len(before_vals) > 0 and len(after_vals) > 0:
                    if stat_type == "mean":
                        joint_kin_jerk_stats[j]["before_mean"].append(before_vals.mean())
                        joint_kin_jerk_stats[j]["after_mean"].append(after_vals.mean())
                    else:
                        joint_kin_jerk_stats[j]["before_max"].append(before_vals.mean())
                        joint_kin_jerk_stats[j]["after_max"].append(after_vals.mean())
    
    if joint_kin_jerk_stats:
        output_lines.append(f"{'Joint':8s} | {'Before Mean':15s} | {'After Mean':15s} | {'Before Max':15s} | {'After Max':15s} | {'Status':20s}")
        output_lines.append(f"{'-'*120}")
        
        for j in range(7):
            if j in joint_kin_jerk_stats:
                stats = joint_kin_jerk_stats[j]
                if stats["before_mean"] and stats["after_mean"]:
                    b_mean = stats["before_mean"][0]
                    a_mean = stats["after_mean"][0]
                    b_max = stats["before_max"][0] if stats["before_max"] else b_mean
                    a_max = stats["after_max"][0] if stats["after_max"] else a_mean
                    
                    pct_change_mean = ((a_mean - b_mean) / abs(b_mean) * 100) if b_mean != 0 else 0
                    
                    status = "📈 증가" if pct_change_mean > 0 else "📉 감소" if pct_change_mean < 0 else "➡️ 동일"
                    
                    output_lines.append(
                        f"Joint_{j}  | {b_mean:13.4f}  | {a_mean:13.4f}  | {b_max:13.4f}  | {a_max:13.4f}  | {status} ({pct_change_mean:+.1f}%)"
                    )
    
    # Joint별 Actuator Jerk 분석
    output_lines.append(f"\n🔧 Joint별 Actuator Jerk 분석 (actuator_jerk_delta - 구동기 점프):\n")
    
    joint_act_jerk_stats = defaultdict(lambda: {"before_mean": [], "after_mean": [], "before_max": [], "after_max": []})
    
    for j in range(7):
        for stat_type in ["mean", "max"]:
            col_name = f"actuator_jerk_delta.joint_{j}_{stat_type}"
            if col_name in df_before_success.columns and col_name in df_after_success.columns:
                before_vals = df_before_success[col_name].dropna().astype(float)
                after_vals = df_after_success[col_name].dropna().astype(float)
                
                if len(before_vals) > 0 and len(after_vals) > 0:
                    if stat_type == "mean":
                        joint_act_jerk_stats[j]["before_mean"].append(before_vals.mean())
                        joint_act_jerk_stats[j]["after_mean"].append(after_vals.mean())
                    else:
                        joint_act_jerk_stats[j]["before_max"].append(before_vals.mean())
                        joint_act_jerk_stats[j]["after_max"].append(after_vals.mean())
    
    if joint_act_jerk_stats:
        output_lines.append(f"{'Joint':8s} | {'Before Mean':15s} | {'After Mean':15s} | {'Before Max':15s} | {'After Max':15s} | {'Status':20s}")
        output_lines.append(f"{'-'*120}")
        
        for j in range(7):
            if j in joint_act_jerk_stats:
                stats = joint_act_jerk_stats[j]
                if stats["before_mean"] and stats["after_mean"]:
                    b_mean = stats["before_mean"][0]
                    a_mean = stats["after_mean"][0]
                    b_max = stats["before_max"][0] if stats["before_max"] else b_mean
                    a_max = stats["after_max"][0] if stats["after_max"] else a_mean
                    
                    pct_change_mean = ((a_mean - b_mean) / abs(b_mean) * 100) if b_mean != 0 else 0
                    
                    status = "📈 증가" if pct_change_mean > 0 else "📉 감소" if pct_change_mean < 0 else "➡️ 동일"
                    
                    output_lines.append(
                        f"Joint_{j}  | {b_mean:13.4f}  | {a_mean:13.4f}  | {b_max:13.4f}  | {a_max:13.4f}  | {status} ({pct_change_mean:+.1f}%)"
                    )
    
    # 동적으로 모든 Joint별 메트릭 수집 및 로깅
    output_lines.append(f"\n{'='*120}")
    output_lines.append("📋 추가 Joint별 세부 분석 (모든 메트릭)")
    output_lines.append(f"{'='*120}\n")
    
    # CSV 컬럼 중 joint_별 항목들 추출
    # 유효한 stat_type 정의 (before/after 같은 무의미한 값 제외)
    valid_stat_types = {"mean", "max", "min", "std", "sum"}
    
    # joint_metrics[joint_name][metric_name][stat_type] = {"before": val, "after": val}
    joint_metrics = defaultdict(lambda: defaultdict(lambda: {}))
    
    for col in df_before_success.columns:
        # joint_X_stat 패턴 찾기
        if ".joint_" in col:
            parts = col.split(".joint_")
            if len(parts) == 2:
                metric_name = parts[0]
                joint_stat = parts[1]  # e.g., "0_mean", "0_max", "0_std", "0_min"
                
                try:
                    joint_num = int(joint_stat.split("_")[0])
                    stat_type = "_".join(joint_stat.split("_")[1:])
                    
                    # 유효한 stat_type만 처리 (before/after 필터링)
                    if stat_type not in valid_stat_types:
                        continue
                    
                    if col in df_before_success.columns and col in df_after_success.columns:
                        before_vals = df_before_success[col].dropna().astype(float)
                        after_vals = df_after_success[col].dropna().astype(float)
                        
                        if len(before_vals) > 0 and len(after_vals) > 0:
                            joint_metrics[f"joint_{joint_num}"][metric_name][stat_type] = {
                                "before": before_vals.iloc[0] if len(before_vals) > 0 else 0,
                                "after": after_vals.iloc[0] if len(after_vals) > 0 else 0
                            }
                except (ValueError, IndexError):
                    pass
    
    # 수집된 joint 메트릭 로깅
    for joint_name in sorted(joint_metrics.keys(), key=lambda x: int(x.split("_")[1])):
        output_lines.append(f"\n{'='*120}")
        output_lines.append(f"🔍 {joint_name.upper()} 상세 분석")
        output_lines.append(f"{'='*120}")
        
        metrics = joint_metrics[joint_name]
        
        for metric_name in sorted(metrics.keys()):
            stats = metrics[metric_name]
            if stats:
                output_lines.append(f"\n  📊 {metric_name}:")
                output_lines.append(f"  {'-'*100}")
                output_lines.append(f"  {'Stat':10s} | {'Before':15s} | {'After':15s} | {'Change':15s} | {'%Change':12s} | {'Status':20s}")
                output_lines.append(f"  {'-'*100}")
                
                for stat_type in sorted(stats.keys()):
                    before_val = stats[stat_type].get("before", 0)
                    after_val = stats[stat_type].get("after", 0)
                    change = after_val - before_val
                    
                    if before_val != 0:
                        pct_change = (change / abs(before_val)) * 100
                    else:
                        pct_change = 0 if after_val == 0 else 100 if after_val > 0 else -100
                    
                    status = "📈 증가" if pct_change > 0.5 else "📉 감소" if pct_change < -0.5 else "➡️ 동일"
                    
                    output_lines.append(
                        f"  {stat_type:10s} | {before_val:15.4f} | {after_val:15.4f} | {change:15.4f} | {pct_change:+11.1f}% | {status:20s}"
                    )
    
    output_text = "\n".join(output_lines)
    return output_text, comparison_results


def calculate_damage(
    torque_array: np.ndarray,
    m: float = 3.0,
    use_goodman: bool = False,
    sigma_ult: float = 500.0,
    sigma_y: float = 400.0,
) -> Dict[str, float]:
    """
    Rainflow-based fatigue damage index.
    
    This returns a dimensionless damage index per joint and in total.
    It uses a Basquin-like form sum(amplitude**m * cycle_count) on
    rainflow-extracted cycles.
    
    ⚠️ IMPORTANT: Relative Comparison Only
    ---------------------------------------
    The input is torque in Nm, used only as a proxy for internal stress.
    The absolute magnitude is NOT a calibrated lifetime prediction for
    any real robot component; it is intended only for relative comparison
    between conditions (e.g., baseline vs noise injection).
    
    This is conceptually inspired by Basquin + Miner's rule, but it is
    NOT calibrated to any specific material S-N curve. The sampling rate
    is 20 Hz, and torque is used as a stress proxy without explicit
    torque-to-stress mapping.
    
    Args:
        torque_array: shape (T, n_joints) - 시계열 토크 데이터 (Nm)
        m: Basquin exponent (일반적으로 3~5, 기본값 3)
        use_goodman: Goodman 보정 적용 여부 (기본값: False)
                     For relative comparison, it is safer to keep this False.
        sigma_ult: 인장 강도 (기본값: 500 MPa) - only used if use_goodman=True
        sigma_y: 항복 강도 (기본값: 400 MPa) - only used if use_goodman=True
    
    Returns:
        각 조인트별 손상 지수 및 총 손상 지수:
        {
            "total_damage": D_total,
            "joint_0": D_0,
            "joint_1": D_1,
            ...
        }
        
    Note:
        If use_goodman=True, a Goodman-type correction is applied using
        sigma_ult and sigma_y, but this path is still approximate because
        the torque-to-stress mapping is not explicitly modeled.
    """
    if not HAS_RAINFLOW:
        logger.error("rainflow 패키지가 설치되지 않았습니다. 손상 계산을 건너뜁니다.")
        return None
    
    if torque_array.size == 0:
        logger.warning("빈 토크 배열입니다. 손상 계산을 건너뜁니다.")
        return None
    
    x = np.asarray(torque_array, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    
    damages = []
    for j in range(x.shape[1]):
        torque_data = x[:, j]
        
        # 평균 제거 (0 중심)
        mean_val = np.mean(torque_data)
        torque_centered = torque_data - mean_val
        
        try:
            # Rainflow로 사이클 추출
            cycles = rainflow.extract_cycles(torque_centered)
            cycles_list = list(cycles)
            num_cycles = len(cycles_list)
            
            # Miner의 규칙: D = sum(C * (sigma_a / sigma_f)^m * count)
            # 여기서 sigma_a: 응력 진폭, sigma_f: 피로 강도
            Dj = 0.0
            total_count = 0
            max_amp = 0.0
            total_half_cycles = 0
            
            for range_val, cycle_mean_stress, count, i_start, i_end in cycles_list:
                amp = range_val / 2.0  # 응력 진폭
                max_amp = max(max_amp, amp)
                
                # Goodman 보정 (평균 응력의 영향을 고려)
                # Goodman 선도: σ_f = σ_a × σ_ult / (σ_ult - σ_m)
                # σ_a: 응력 진폭, σ_m: 평균 응력
                if use_goodman:
                    # 전체 신호의 평균 응력 사용 (mean_val = np.mean(torque_data))
                    sigma_m = abs(mean_val)  # 평균 응력
                    
                    # σ_ult - σ_m > 0 인지 확인 (분모가 0이 되지 않도록)
                    denominator = sigma_ult - sigma_m
                    if denominator > 1e-6:
                        # Goodman 계수: σ_f / σ_a = σ_ult / (σ_ult - σ_m)
                        goodman_factor = sigma_ult / denominator
                        effective_amp = amp * goodman_factor
                    else:
                        # σ_m >= σ_ult인 경우 (매우 높은 평균 응력) - 보수적으로 처리
                        logger.warning(f"Joint {j}: 평균 응력({sigma_m:.4f})이 인장 강도({sigma_ult:.4f})를 초과합니다. Goodman 보정 스킵.")
                        effective_amp = amp * 100  # 극도로 보수적인 페널티
                else:
                    effective_amp = amp
                
                # 사이클 열림 여부에 따른 상수 C 설정
                # Rainflow에서 나온 count를 기반으로:
                # count = 1.0이면 완전한 사이클 (C=1)
                # count = 0.5이면 반 사이클 (C=0.5)
                # 보수적으로 모든 사이클을 완전 사이클(C=1)로 처리
                C = 1.0  # 보수적 평가
                
                # 손상 계산: D = C * (amplitude^m) * count
                Dj += C * (effective_amp ** m) * count
                total_count += count
                total_half_cycles += 1
            
            damages.append(Dj)
            logger.info(f"Joint {j}: 사이클={num_cycles}, 총카운트={total_count:.1f}, 최대진폭={max_amp:.4f}, Goodman={use_goodman}, 손상={Dj:.6e}")
        
        except Exception as e:
            logger.warning(f"Joint {j}의 손상 계산 실패: {e}")
            damages.append(0.0)
    
    D_total = sum(damages)
    result = {"total_damage": D_total}
    for j, D in enumerate(damages):
        result[f"joint_{j}"] = D
    
    logger.info(f"손상 계산 완료 (rainflow): total={D_total:.6e}")
    return result


def calculate_damage_fatpack(
    torque_array: np.ndarray,
    m: float = 3.0,
) -> Dict[str, float]:
    """
    Fatpack 기반 피로 손상 지수 계산 (rainflow의 대안 패키지)
    
    fatpack 패키지를 사용하여 Rainflow cycle counting 수행.
    rainflow 패키지와 동일한 Basquin 기반 손상 계산 로직 적용.
    
    Args:
        torque_array: shape (T, n_joints) - 시계열 토크 데이터 (Nm)
        m: Basquin exponent (일반적으로 3~5, 기본값 3)
    
    Returns:
        각 조인트별 손상 지수 및 총 손상 지수:
        {
            "total_damage": D_total,
            "joint_0": D_0,
            "joint_1": D_1,
            ...
        }
    """
    if not HAS_FATPACK:
        logger.error("fatpack 패키지가 설치되지 않았습니다. 손상 계산을 건너뜁니다.")
        return None
    
    if torque_array.size == 0:
        logger.warning("빈 토크 배열입니다. 손상 계산을 건너뜁니다.")
        return None
    
    x = np.asarray(torque_array, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    
    damages = []
    for j in range(x.shape[1]):
        torque_data = x[:, j]
        
        try:
            # 1. Find reversals (극값점 찾기)
            reversals, reversal_indices = fatpack.find_reversals(torque_data)
            
            if len(reversals) < 2:
                logger.warning(f"Joint {j}: reversals가 2개 미만입니다. 손상=0으로 설정.")
                damages.append(0.0)
                continue
            
            # 2. Find rainflow cycles
            cycles, residue = fatpack.find_rainflow_cycles(reversals)
            
            if len(cycles) == 0:
                logger.warning(f"Joint {j}: 추출된 사이클이 없습니다. 손상=0으로 설정.")
                damages.append(0.0)
                continue
            
            # 3. Calculate ranges (amplitude의 2배)
            ranges = np.abs(cycles[:, 1] - cycles[:, 0])
            amplitudes = ranges / 2.0
            
            # 4. Basquin 기반 손상 계산: D = sum(amplitude^m)
            # 각 사이클의 count는 1로 가정 (fatpack은 각 사이클을 별도로 반환)
            Dj = np.sum(amplitudes ** m)
            
            damages.append(Dj)
            logger.info(f"Joint {j} (fatpack): 사이클={len(cycles)}, 최대진폭={np.max(amplitudes):.4f}, 손상={Dj:.6e}")
        
        except Exception as e:
            logger.warning(f"Joint {j}의 fatpack 손상 계산 실패: {e}")
            damages.append(0.0)
    
    D_total = sum(damages)
    result = {"total_damage": D_total}
    for j, D in enumerate(damages):
        result[f"joint_{j}"] = D
    
    logger.info(f"손상 계산 완료 (fatpack): total={D_total:.6e}")
    return result


def find_matching_files(before_dir: pathlib.Path, after_dir: pathlib.Path, success_only: bool = True) -> list:
    """
    before와 after 디렉토리에서 매칭되는 torque_current_*.npy 파일 찾기
    
    Args:
        before_dir: Before 분석 디렉토리
        after_dir: After 분석 디렉토리
        success_only: True이면 성공한 에피소드만 매칭 (기본값: True)
    
    Returns:
        List of tuples: [(before_file, after_file, filename), ...]
    """
    before_files = sorted(before_dir.glob("torque_current_*.npy"))
    after_files = sorted(after_dir.glob("torque_current_*.npy"))
    
    # 성공한 에피소드만 필터링
    if success_only:
        before_files = [f for f in before_files if f.name.endswith("_success.npy")]
        after_files = [f for f in after_files if f.name.endswith("_success.npy")]
        logger.info(f"   Before 디렉토리({before_dir})에서 성공한 torque_current_*_success.npy 파일 {len(before_files)}개 발견")
        logger.info(f"   After 디렉토리({after_dir})에서 성공한 torque_current_*_success.npy 파일 {len(after_files)}개 발견")
    else:
        logger.info(f"   Before 디렉토리({before_dir})에서 torque_current_*.npy 파일 {len(before_files)}개 발견")
        logger.info(f"   After 디렉토리({after_dir})에서 torque_current_*.npy 파일 {len(after_files)}개 발견")
    
    if len(before_files) == 0:
        logger.warning(f"   ⚠️  Before 디렉토리에 torque_current_*.npy 파일이 없습니다: {before_dir}")
        logger.warning(f"   디렉토리 내용 (처음 5개):")
        for i, f in enumerate(sorted(before_dir.glob("*"))[:5]):
            logger.warning(f"     • {f.name}")
    
    if len(after_files) == 0:
        logger.warning(f"   ⚠️  After 디렉토리에 torque_current_*.npy 파일이 없습니다: {after_dir}")
        logger.warning(f"   디렉토리 내용 (처음 5개):")
        for i, f in enumerate(sorted(after_dir.glob("*"))[:5]):
            logger.warning(f"     • {f.name}")
    
    matches = []
    for bf in before_files:
        name = bf.name
        af = after_dir / name
        if af.exists():
            matches.append((bf, af, name))
        else:
            logger.warning(f"⚠️  매칭되는 파일 없음: {name}")
    
    return matches


def compare_conditions(
    before_file: pathlib.Path, 
    after_file: pathlib.Path, 
    m: float = 3.0,
    min_total_damage: float = 1e-8,
    min_joint_damage: float = 1e-10,
) -> Optional[Dict[str, float]]:
    """
    두 토크 파일 비교 및 상대 피로 손상 지수 비율 계산 (rainflow + fatpack 두 패키지 모두 사용)
    
    ⚠️ IMPORTANT: life_ratio is a Relative Fatigue Damage Index Ratio
    -----------------------------------------------------------------
    This is NOT an absolute lifetime ratio. It compares the relative
    fatigue damage index between two conditions (e.g., baseline vs noise).
    
    Args:
        before_file: Before torque_current_*.npy 파일 경로
        after_file: After torque_current_*.npy 파일 경로
        m: Basquin 지수
        min_total_damage: 신뢰할 수 있는 life_ratio 계산을 위한 최소 총 손상 임계값
        min_joint_damage: 신뢰할 수 있는 joint별 ratio 계산을 위한 최소 손상 임계값
    
    Returns:
        Before/After 손상 지수 및 비율 (rainflow + fatpack 모두 포함):
        {
            # rainflow 패키지 결과
            "rainflow_damage_before_total": D_b,
            "rainflow_damage_after_total": D_a,
            "rainflow_life_ratio": D_a / D_b,
            "rainflow_life_ratio_reliable": True/False,
            "rainflow_joint_0_damage_before": D_bj0,
            ...
            # fatpack 패키지 결과
            "fatpack_damage_before_total": D_b,
            "fatpack_damage_after_total": D_a,
            "fatpack_life_ratio": D_a / D_b,
            "fatpack_life_ratio_reliable": True/False,
            "fatpack_joint_0_damage_before": D_bj0,
            ...
        }
    """
    try:
        before_torque = np.load(before_file)
        after_torque = np.load(after_file)
        
        result = {}
        
        # ===== 1. Rainflow 패키지 기반 손상 계산 =====
        if HAS_RAINFLOW:
            damage_before_rf = calculate_damage(before_torque, m=m)
            damage_after_rf = calculate_damage(after_torque, m=m)
            
            if damage_before_rf is not None and damage_after_rf is not None:
                base_total_rf = damage_before_rf["total_damage"]
                after_total_rf = damage_after_rf["total_damage"]
                
                result["rainflow_damage_before_total"] = base_total_rf
                result["rainflow_damage_after_total"] = after_total_rf
                
                if base_total_rf <= 0.0:
                    result["rainflow_life_ratio"] = np.inf
                    result["rainflow_life_ratio_reliable"] = False
                else:
                    result["rainflow_life_ratio"] = after_total_rf / base_total_rf
                    result["rainflow_life_ratio_reliable"] = (base_total_rf >= min_total_damage)
                
                # 조인트별 손상 및 비율 (rainflow)
                n_joints = len(damage_before_rf) - 1
                for j in range(n_joints):
                    key = f"joint_{j}"
                    if key in damage_before_rf and key in damage_after_rf:
                        D_b = damage_before_rf[key]
                        D_a = damage_after_rf[key]
                        
                        result[f"rainflow_joint_{j}_damage_before"] = D_b
                        result[f"rainflow_joint_{j}_damage_after"] = D_a
                        
                        if D_b > min_joint_damage:
                            result[f"rainflow_joint_{j}_damage_ratio"] = D_a / D_b
                        else:
                            result[f"rainflow_joint_{j}_damage_ratio"] = np.inf
        
        # ===== 2. Fatpack 패키지 기반 손상 계산 =====
        if HAS_FATPACK:
            damage_before_fp = calculate_damage_fatpack(before_torque, m=m)
            damage_after_fp = calculate_damage_fatpack(after_torque, m=m)
            
            if damage_before_fp is not None and damage_after_fp is not None:
                base_total_fp = damage_before_fp["total_damage"]
                after_total_fp = damage_after_fp["total_damage"]
                
                result["fatpack_damage_before_total"] = base_total_fp
                result["fatpack_damage_after_total"] = after_total_fp
                
                if base_total_fp <= 0.0:
                    result["fatpack_life_ratio"] = np.inf
                    result["fatpack_life_ratio_reliable"] = False
                else:
                    result["fatpack_life_ratio"] = after_total_fp / base_total_fp
                    result["fatpack_life_ratio_reliable"] = (base_total_fp >= min_total_damage)
                
                # 조인트별 손상 및 비율 (fatpack)
                n_joints = len(damage_before_fp) - 1
                for j in range(n_joints):
                    key = f"joint_{j}"
                    if key in damage_before_fp and key in damage_after_fp:
                        D_b = damage_before_fp[key]
                        D_a = damage_after_fp[key]
                        
                        result[f"fatpack_joint_{j}_damage_before"] = D_b
                        result[f"fatpack_joint_{j}_damage_after"] = D_a
                        
                        if D_b > min_joint_damage:
                            result[f"fatpack_joint_{j}_damage_ratio"] = D_a / D_b
                        else:
                            result[f"fatpack_joint_{j}_damage_ratio"] = np.inf
        
        # 결과가 비어있으면 None 반환
        if not result:
            logger.warning(f"{before_file.name}: 어떤 패키지로도 손상 계산이 되지 않았습니다.")
            return None
        
        return result
    
    except Exception as e:
        logger.error(f"{before_file.name} 비교 실패: {e}")
        return None


def save_comparison_to_csv(results: list, output_file: pathlib.Path) -> None:
    """
    비교 결과를 CSV 파일에 저장
    
    CSV Column 구성:
    - file: 비교 파일명
    - damage_before_total: Before 총 손상
    - damage_after_total: After 총 손상
    - life_ratio: 수명비 (After/Before)
    - joint_0_damage_before ~ joint_6_damage_before: 각 조인트 Before 손상
    - joint_0_damage_after ~ joint_6_damage_after: 각 조인트 After 손상
    - joint_0_damage_ratio ~ joint_6_damage_ratio: 각 조인트 수명비
    
    Args:
        results: 비교 결과 딕셔너리 리스트
        output_file: 출력 CSV 파일 경로
    """
    if not results:
        logger.warning("저장할 결과가 없습니다.")
        return
    
    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 모든 가능한 키 수집
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # 키 정렬: 주요 메트릭 먼저, 그 다음 조인트별
    sorted_keys = []
    priority_order = [
        "file",
        "damage_before_total",
        "damage_after_total", 
        "life_ratio"
    ]
    
    for key in priority_order:
        if key in all_keys:
            sorted_keys.append(key)
            all_keys.discard(key)
    
    # 나머지 키 추가
    sorted_keys.extend(sorted(all_keys))
    
    try:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_keys)
            writer.writeheader()
            
            for result in results:
                row = {key: result.get(key, "") for key in sorted_keys}
                writer.writerow(row)
        
        logger.info(f"✅ NPY 비교 결과 저장 완료: {output_file}")
    
    except Exception as e:
        logger.error(f"CSV 저장 실패: {e}")


def compare_npy_metrics(
    before_dir: pathlib.Path, 
    after_dir: pathlib.Path, 
    m: float = 3.0,
    filter_unreliable: bool = False,
    min_total_damage: float = 1e-8,
    min_joint_damage: float = 1e-10,
) -> Tuple[str, list]:
    """
    NPY 파일 기반 피로 손상 지수 및 상대 비율 비교
    
    ⚠️ IMPORTANT: life_ratio is a Relative Fatigue Damage Index Ratio
    -----------------------------------------------------------------
    This compares relative fatigue damage indices, not absolute lifetimes.
    
    Args:
        before_dir: Before 분석 디렉토리
        after_dir: After 분석 디렉토리
        m: Basquin 지수
        filter_unreliable: True이면 요약 통계에서 신뢰할 수 없는 비율 제외
        min_total_damage: 신뢰할 수 있는 life_ratio 계산을 위한 최소 총 손상 임계값
        min_joint_damage: 신뢰할 수 있는 joint별 ratio 계산을 위한 최소 손상 임계값
    
    Returns:
        Tuple[출력 텍스트, 비교 결과 리스트]
    """
    logger.info(f"\n🔍 NPY 파일 비교 시작...")
    logger.info(f"   Before dir: {before_dir}")
    logger.info(f"   After dir:  {after_dir}")
    logger.info(f"   Basquin exponent (m): {m}")
    logger.info(f"   Filter unreliable: {filter_unreliable}")
    if filter_unreliable:
        logger.info(f"   Min total damage threshold: {min_total_damage:.2e}")
        logger.info(f"   Min joint damage threshold: {min_joint_damage:.2e}")
    
    # 매칭되는 파일 찾기 (성공한 에피소드만)
    matches = find_matching_files(before_dir, after_dir, success_only=True)
    
    logger.info(f"   📁 매칭된 torque_current_*.npy 파일: {len(matches)}개")
    
    if len(matches) == 0:
        logger.warning(f"   ⚠️  매칭된 NPY 파일이 없습니다!")
        logger.warning(f"   Before 디렉토리의 torque_current_*.npy 파일 개수: {len(list(before_dir.glob('torque_current_*.npy')))}")
        logger.warning(f"   After 디렉토리의 torque_current_*.npy 파일 개수: {len(list(after_dir.glob('torque_current_*.npy')))}")
    
    output_lines = []
    output_lines.append("\n" + "=" * 120)
    output_lines.append("⚙️  NPY 기반 피로 손상 지수 & 상대 비율 비교 (RAINFLOW + FATPACK 패키지)")
    output_lines.append("=" * 120)
    output_lines.append("※ life_ratio는 상대적 피로 손상 지수 비율이며, 절대 수명 예측이 아닙니다.")
    output_lines.append("※ 양쪽 모두 성공한 에피소드만 비교합니다.")
    output_lines.append(f"※ 사용 패키지: rainflow={'✅' if HAS_RAINFLOW else '❌'}, fatpack={'✅' if HAS_FATPACK else '❌'}\n")
    
    # 각 파일 쌍 비교
    results = []
    rainflow_ratios = []
    fatpack_ratios = []
    excluded_count_rf = 0
    excluded_count_fp = 0
    
    for before_file, after_file, filename in matches:
        logger.info(f"📈 처리 중: {filename}")
        
        comparison = compare_conditions(
            before_file, after_file, m=m,
            min_total_damage=min_total_damage,
            min_joint_damage=min_joint_damage,
        )
        if comparison is not None:
            comparison["file"] = filename
            results.append(comparison)
            
            # Rainflow 패키지 결과 수집
            ratio_rf = comparison.get("rainflow_life_ratio", np.inf)
            reliable_rf = comparison.get("rainflow_life_ratio_reliable", False)
            
            if filter_unreliable:
                if reliable_rf and not np.isinf(ratio_rf):
                    rainflow_ratios.append(ratio_rf)
                else:
                    excluded_count_rf += 1
            else:
                if not np.isinf(ratio_rf):
                    rainflow_ratios.append(ratio_rf)
            
            # Fatpack 패키지 결과 수집
            ratio_fp = comparison.get("fatpack_life_ratio", np.inf)
            reliable_fp = comparison.get("fatpack_life_ratio_reliable", False)
            
            if filter_unreliable:
                if reliable_fp and not np.isinf(ratio_fp):
                    fatpack_ratios.append(ratio_fp)
                else:
                    excluded_count_fp += 1
            else:
                if not np.isinf(ratio_fp):
                    fatpack_ratios.append(ratio_fp)
    
    # ===== 1️⃣ RAINFLOW 패키지 요약 =====
    output_lines.append(f"\n{'='*120}")
    output_lines.append(f"📦 RAINFLOW 패키지 기반 분석 결과")
    output_lines.append(f"{'='*120}")
    
    if filter_unreliable and excluded_count_rf > 0:
        output_lines.append(f"  ※ 베이스라인 손상이 극히 작은 파일 {excluded_count_rf}개가 요약 통계에서 제외되었습니다.\n")
    
    if rainflow_ratios and not all(np.isinf(r) for r in rainflow_ratios):
        finite_ratios_rf = [r for r in rainflow_ratios if not np.isinf(r)]
        if finite_ratios_rf:
            mean_ratio_rf = np.mean(finite_ratios_rf)
            min_ratio_rf = np.min(finite_ratios_rf)
            max_ratio_rf = np.max(finite_ratios_rf)
            
            output_lines.append(f"\n📊 Rainflow 전체 요약 통계:\n")
            output_lines.append(f"  평균 수명비:       {mean_ratio_rf:.4f}x")
            output_lines.append(f"  최소 수명비:       {min_ratio_rf:.4f}x (가장 양호)")
            output_lines.append(f"  최대 수명비:       {max_ratio_rf:.4f}x (가장 악화)")
            output_lines.append(f"  비교된 파일:       {len(finite_ratios_rf)}개\n")
            
            if mean_ratio_rf > 1.05:
                pct_reduction = (mean_ratio_rf - 1) * 100
                output_lines.append(f"  ⚠️  Rainflow 결론: 노이즈로 인해 평균 손상이 약 {pct_reduction:.1f}% 증가 → 수명 단축")
            elif mean_ratio_rf < 0.95:
                pct_improve = (1 - mean_ratio_rf) * 100
                output_lines.append(f"  ✅ Rainflow 결론: 노이즈로 인해 평균 손상이 약 {pct_improve:.1f}% 감소 → 수명 개선")
            else:
                output_lines.append(f"  ➡️  Rainflow 결론: 노이즈의 영향 미미 → 수명 변화 거의 없음")
    else:
        output_lines.append(f"\n  ⚠️  Rainflow 결과 없음 (패키지 미설치 또는 계산 불가)")
    
    # Rainflow Joint별 분석
    joint_stats_rf = defaultdict(lambda: {"damage_before": [], "damage_after": [], "ratio": []})
    
    for result in results:
        for j in range(7):
            key_before = f"rainflow_joint_{j}_damage_before"
            key_after = f"rainflow_joint_{j}_damage_after"
            key_ratio = f"rainflow_joint_{j}_damage_ratio"
            
            if key_before in result and key_after in result:
                d_b = result[key_before]
                d_a = result[key_after]
                ratio = result.get(key_ratio, np.inf)
                
                joint_stats_rf[j]["damage_before"].append(d_b)
                joint_stats_rf[j]["damage_after"].append(d_a)
                if not np.isinf(ratio):
                    joint_stats_rf[j]["ratio"].append(ratio)
    
    if any(joint_stats_rf[j]["damage_before"] for j in range(7)):
        output_lines.append(f"\n⚙️  Rainflow - Joint별 분석:\n")
        output_lines.append(f"{'Joint':8s} | {'Before Damage':20s} | {'After Damage':20s} | {'수명비':12s} | {'상태':30s}")
        output_lines.append(f"{'-'*120}")
        
        joint_assessments_rf = []
        for j in range(7):
            if j in joint_stats_rf and joint_stats_rf[j]["damage_before"]:
                avg_before = np.mean(joint_stats_rf[j]["damage_before"])
                avg_after = np.mean(joint_stats_rf[j]["damage_after"])
                avg_ratio = np.mean(joint_stats_rf[j]["ratio"]) if joint_stats_rf[j]["ratio"] else 1.0
                
                if avg_ratio > 1.1:
                    status, symbol = "📈 심각 (수명 크게 단축)", "🔴"
                elif avg_ratio > 1.0:
                    status, symbol = "📈 주의 (수명 단축)", "🟠"
                elif avg_ratio < 0.9:
                    status, symbol = "📉 개선 (수명 증가)", "🟢"
                elif avg_ratio < 1.0:
                    status, symbol = "📉 약간 개선", "🟡"
                else:
                    status, symbol = "➡️  변화 없음", "⚪"
                
                output_lines.append(
                    f"Joint_{j}  | {avg_before:18.6e} | {avg_after:18.6e} | {avg_ratio:10.4f}x | {symbol} {status}"
                )
                joint_assessments_rf.append((j, avg_ratio, status))
        
        if joint_assessments_rf:
            worst_joint_rf = max(joint_assessments_rf, key=lambda x: x[1])
            output_lines.append(f"\n  ⚠️  Rainflow 가장 취약한 조인트: Joint_{worst_joint_rf[0]} (수명비: {worst_joint_rf[1]:.4f}x)")
    
    # ===== 2️⃣ FATPACK 패키지 요약 =====
    output_lines.append(f"\n{'='*120}")
    output_lines.append(f"📦 FATPACK 패키지 기반 분석 결과")
    output_lines.append(f"{'='*120}")
    
    if filter_unreliable and excluded_count_fp > 0:
        output_lines.append(f"  ※ 베이스라인 손상이 극히 작은 파일 {excluded_count_fp}개가 요약 통계에서 제외되었습니다.\n")
    
    if fatpack_ratios and not all(np.isinf(r) for r in fatpack_ratios):
        finite_ratios_fp = [r for r in fatpack_ratios if not np.isinf(r)]
        if finite_ratios_fp:
            mean_ratio_fp = np.mean(finite_ratios_fp)
            min_ratio_fp = np.min(finite_ratios_fp)
            max_ratio_fp = np.max(finite_ratios_fp)
            
            output_lines.append(f"\n📊 Fatpack 전체 요약 통계:\n")
            output_lines.append(f"  평균 수명비:       {mean_ratio_fp:.4f}x")
            output_lines.append(f"  최소 수명비:       {min_ratio_fp:.4f}x (가장 양호)")
            output_lines.append(f"  최대 수명비:       {max_ratio_fp:.4f}x (가장 악화)")
            output_lines.append(f"  비교된 파일:       {len(finite_ratios_fp)}개\n")
            
            if mean_ratio_fp > 1.05:
                pct_reduction = (mean_ratio_fp - 1) * 100
                output_lines.append(f"  ⚠️  Fatpack 결론: 노이즈로 인해 평균 손상이 약 {pct_reduction:.1f}% 증가 → 수명 단축")
            elif mean_ratio_fp < 0.95:
                pct_improve = (1 - mean_ratio_fp) * 100
                output_lines.append(f"  ✅ Fatpack 결론: 노이즈로 인해 평균 손상이 약 {pct_improve:.1f}% 감소 → 수명 개선")
            else:
                output_lines.append(f"  ➡️  Fatpack 결론: 노이즈의 영향 미미 → 수명 변화 거의 없음")
    else:
        output_lines.append(f"\n  ⚠️  Fatpack 결과 없음 (패키지 미설치 또는 계산 불가)")
    
    # Fatpack Joint별 분석
    joint_stats_fp = defaultdict(lambda: {"damage_before": [], "damage_after": [], "ratio": []})
    
    for result in results:
        for j in range(7):
            key_before = f"fatpack_joint_{j}_damage_before"
            key_after = f"fatpack_joint_{j}_damage_after"
            key_ratio = f"fatpack_joint_{j}_damage_ratio"
            
            if key_before in result and key_after in result:
                d_b = result[key_before]
                d_a = result[key_after]
                ratio = result.get(key_ratio, np.inf)
                
                joint_stats_fp[j]["damage_before"].append(d_b)
                joint_stats_fp[j]["damage_after"].append(d_a)
                if not np.isinf(ratio):
                    joint_stats_fp[j]["ratio"].append(ratio)
    
    if any(joint_stats_fp[j]["damage_before"] for j in range(7)):
        output_lines.append(f"\n⚙️  Fatpack - Joint별 분석:\n")
        output_lines.append(f"{'Joint':8s} | {'Before Damage':20s} | {'After Damage':20s} | {'수명비':12s} | {'상태':30s}")
        output_lines.append(f"{'-'*120}")
        
        joint_assessments_fp = []
        for j in range(7):
            if j in joint_stats_fp and joint_stats_fp[j]["damage_before"]:
                avg_before = np.mean(joint_stats_fp[j]["damage_before"])
                avg_after = np.mean(joint_stats_fp[j]["damage_after"])
                avg_ratio = np.mean(joint_stats_fp[j]["ratio"]) if joint_stats_fp[j]["ratio"] else 1.0
                
                if avg_ratio > 1.1:
                    status, symbol = "📈 심각 (수명 크게 단축)", "🔴"
                elif avg_ratio > 1.0:
                    status, symbol = "📈 주의 (수명 단축)", "🟠"
                elif avg_ratio < 0.9:
                    status, symbol = "📉 개선 (수명 증가)", "🟢"
                elif avg_ratio < 1.0:
                    status, symbol = "📉 약간 개선", "🟡"
                else:
                    status, symbol = "➡️  변화 없음", "⚪"
                
                output_lines.append(
                    f"Joint_{j}  | {avg_before:18.6e} | {avg_after:18.6e} | {avg_ratio:10.4f}x | {symbol} {status}"
                )
                joint_assessments_fp.append((j, avg_ratio, status))
        
        if joint_assessments_fp:
            worst_joint_fp = max(joint_assessments_fp, key=lambda x: x[1])
            output_lines.append(f"\n  ⚠️  Fatpack 가장 취약한 조인트: Joint_{worst_joint_fp[0]} (수명비: {worst_joint_fp[1]:.4f}x)")
    
    # ===== 3️⃣ 패키지 간 비교 =====
    output_lines.append(f"\n{'='*120}")
    output_lines.append(f"📊 패키지 간 비교 (Rainflow vs Fatpack)")
    output_lines.append(f"{'='*120}")
    
    if rainflow_ratios and fatpack_ratios:
        finite_rf = [r for r in rainflow_ratios if not np.isinf(r)]
        finite_fp = [r for r in fatpack_ratios if not np.isinf(r)]
        
        if finite_rf and finite_fp:
            mean_rf = np.mean(finite_rf)
            mean_fp = np.mean(finite_fp)
            diff_pct = ((mean_fp - mean_rf) / mean_rf * 100) if mean_rf != 0 else 0
            
            output_lines.append(f"\n  {'패키지':15s} | {'평균 수명비':15s} | {'Min':12s} | {'Max':12s} | {'파일 수':10s}")
            output_lines.append(f"  {'-'*80}")
            output_lines.append(f"  {'Rainflow':15s} | {mean_rf:13.4f}x | {np.min(finite_rf):10.4f}x | {np.max(finite_rf):10.4f}x | {len(finite_rf):10d}")
            output_lines.append(f"  {'Fatpack':15s} | {mean_fp:13.4f}x | {np.min(finite_fp):10.4f}x | {np.max(finite_fp):10.4f}x | {len(finite_fp):10d}")
            output_lines.append(f"\n  📈 패키지 간 차이: Fatpack이 Rainflow 대비 {diff_pct:+.2f}%")
            
            if abs(diff_pct) < 5:
                output_lines.append(f"  ✅ 두 패키지의 결과가 유사합니다 (차이 < 5%).")
            else:
                output_lines.append(f"  ⚠️  두 패키지의 결과에 차이가 있습니다 (차이 >= 5%).")
    else:
        output_lines.append(f"\n  ⚠️  비교할 데이터가 부족합니다.")
    
    # ===== 4️⃣ 파일별 세부 수명비 분석 =====
    output_lines.append(f"\n{'─' * 120}")
    output_lines.append(f"\n📁 매칭된 파일 {len(results)}개 발견 - 파일별 세부 분석\n")
    output_lines.append(f"{'파일명':60s} | {'Rainflow 수명비':20s} | {'Fatpack 수명비':20s}")
    output_lines.append(f"{'-'*110}")
    
    for result in results:
        filename = result.get("file", "unknown")
        ratio_rf = result.get("rainflow_life_ratio", np.nan)
        ratio_fp = result.get("fatpack_life_ratio", np.nan)
        
        rf_str = f"{ratio_rf:.4f}x" if not np.isinf(ratio_rf) and not np.isnan(ratio_rf) else "N/A"
        fp_str = f"{ratio_fp:.4f}x" if not np.isinf(ratio_fp) and not np.isnan(ratio_fp) else "N/A"
        
        output_lines.append(f"  {filename:58s} | {rf_str:18s} | {fp_str:18s}")
    
    output_text = "\n".join(output_lines)
    return output_text, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="두 조건(noise 없음/있음)의 물리량을 전반적으로 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📋 CSV Column 설명 (NPY 비교 결과):
  - file: 비교한 NPY 파일명
  - damage_before_total: Before 조건의 총 피로 손상 (누적)
  - damage_after_total: After 조건의 총 피로 손상 (누적)
  - life_ratio: 상대 수명비 (= damage_after / damage_before)
              * > 1.0: 수명 단축 (노이즈의 부정적 영향)
              * < 1.0: 수명 증가
              * = 1.0: 변화 없음
  - joint_N_damage_before/after: 각 조인트별 개별 손상값
  - joint_N_damage_ratio: 각 조인트별 개별 수명비

예시:
  python compare_metrics.py \\
    --before_dir ./analysis/analysis_libero_10_20251107_172053_noise_00000/ \\
    --after_dir ./analysis/analysis_libero_10_20251107_145142_noise_05000_dim_action.eef_pos_delta[2]/ \\
    --output ./results/comparison_libero_10.txt
        """
    )
    
    parser.add_argument("--before_dir", type=str, required=True,
                        help="Before (baseline) 분석 디렉토리")
    parser.add_argument("--after_dir", type=str, required=True,
                        help="After (noise) 분석 디렉토리")
    parser.add_argument("--m", type=float, default=3.0,
                        help="Basquin 지수 (기본: 3.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 텍스트 파일 경로 (기본: after_dir 이름으로 자동 생성)")
    parser.add_argument("--output_npy_csv", type=str, default=None,
                        help="NPY 비교 결과를 저장할 CSV 파일 경로 (선택사항)")
    
    # 신뢰성 필터링 옵션 (기본: 비활성화)
    parser.add_argument("--filter_unreliable", action="store_true", default=False,
                        help="life_ratio 요약 통계에서 베이스라인 손상이 극히 작은 파일 제외 (기본: 비활성화)")
    parser.add_argument("--min_total_damage", type=float, default=1e-8,
                        help="신뢰할 수 있는 life_ratio 계산을 위한 최소 총 손상 임계값 (기본: 1e-8)")
    parser.add_argument("--min_joint_damage", type=float, default=1e-10,
                        help="신뢰할 수 있는 joint별 ratio 계산을 위한 최소 손상 임계값 (기본: 1e-10)")
    
    args = parser.parse_args()
    
    before_dir = pathlib.Path(args.before_dir)
    after_dir = pathlib.Path(args.after_dir)
    
    # Output 파일명 자동 생성
    if args.output is None:
        # after_dir 디렉토리 이름 추출 (e.g., "analysis_libero_10_20251107_145142_noise_05000_dim_action.eef_pos_delta[2]")
        after_dir_name = after_dir.name.rstrip("/")
        
        # 대괄호 처리 (파일시스템 호환성을 위해 언더스코어로 변환)
        safe_name = after_dir_name.replace("[", "_").replace("]", "_")
        
        output_file = pathlib.Path("./results") / f"comparison_{safe_name}.txt"
    else:
        output_file = pathlib.Path(args.output)
    
    # 디렉토리 존재 확인
    if not before_dir.exists():
        logger.error(f"Before 디렉토리를 찾을 수 없습니다: {before_dir}")
        exit(1)
    
    if not after_dir.exists():
        logger.error(f"After 디렉토리를 찾을 수 없습니다: {after_dir}")
        exit(1)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_output = []
    npy_results = []
    
    # 1️⃣ NPY 기반 비교 (기존 기능)
    logger.info(f"\n{'='*120}")
    logger.info("Step 1/2: NPY 기반 비교 시작...")
    npy_output, npy_results = compare_npy_metrics(
        before_dir, after_dir, 
        m=args.m,
        filter_unreliable=args.filter_unreliable,
        min_total_damage=args.min_total_damage,
        min_joint_damage=args.min_joint_damage,
    )
    all_output.append(npy_output)
    logger.info("✅ NPY 비교 완료")
    
    # 2️⃣ CSV 기반 비교 (신규 기능)
    logger.info(f"\n{'='*120}")
    logger.info("Step 2/2: CSV 기반 비교 시작...")
    
    before_csv = find_csv_file(before_dir)
    after_csv = find_csv_file(after_dir)
    
    logger.info(f"Before CSV: {before_csv}")
    logger.info(f"After CSV:  {after_csv}")
    
    if before_csv and after_csv:
        logger.info("✅ CSV 파일 발견, 비교 시작...")
        csv_output, csv_results = compare_csv_metrics(before_csv, after_csv)
        all_output.append(csv_output)
        logger.info("✅ CSV 비교 완료")
    else:
        missing = []
        if not before_csv:
            missing.append(f"Before CSV not found in {before_dir}")
        if not after_csv:
            missing.append(f"After CSV not found in {after_dir}")
        logger.error(f"❌ CSV 파일을 찾을 수 없습니다: {', '.join(missing)}")
        logger.error(f"   - {before_dir} 내용:")
        for item in before_dir.glob("*"):
            logger.error(f"     • {item.name}")
        logger.error(f"   - {after_dir} 내용:")
        for item in after_dir.glob("*"):
            logger.error(f"     • {item.name}")
    
    # 최종 결과 출력 및 저장
    logger.info(f"\n{'='*120}")
    logger.info("최종 결과 저장 중...")
    
    final_output = "\n".join(all_output)
    
    print(final_output)
    
    with open(output_file, "w") as f:
        f.write(final_output)
    
    file_size = output_file.stat().st_size
    logger.info(f"\n✅ 전체 비교 결과 저장 완료: {output_file}")
    logger.info(f"   파일 크기: {file_size:,} bytes")
    
    # NPY CSV 저장 (선택사항)
    if args.output_npy_csv and npy_results:
        npy_csv_file = pathlib.Path(args.output_npy_csv)
        save_comparison_to_csv(npy_results, npy_csv_file)
