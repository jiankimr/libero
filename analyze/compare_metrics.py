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
    dir_path = pathlib.Path(directory)
    parent_dir = dir_path.parent  # LIBERO/analysis/
    
    # 디렉토리 이름 추출
    # e.g., "analysis_libero_10_20251107_145142_noise_05000_dim_action.eef_pos_delta[2]"
    dir_name = dir_path.name
    
    # 1️⃣ 부모 디렉토리(LIBERO/analysis/)에서 모든 CSV 찾기
    csv_files = sorted(parent_dir.glob("analysis_summary*.csv"))
    
    logger.debug(f"   🔍 Looking for CSV in {parent_dir}")
    logger.debug(f"   📁 Directory name: {dir_name}")
    logger.debug(f"   📋 Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
    
    if not csv_files:
        logger.debug(f"   ⚠️  No CSV files found in {parent_dir}")
        return None
    
    # 2️⃣ 디렉토리 이름과 CSV 파일명 매칭
    # dir_name: "analysis_libero_10_20251107_145142_noise_05000_dim_action.eef_pos_delta[2]"
    # csv_name: "analysis_summary_libero_10_20251107_145142_noise_05000_dim_action.eef_pos_delta[2].csv"
    
    # 디렉토리 이름에서 "analysis_" 제거
    dir_name_suffix = dir_name.replace("analysis_", "")
    
    for csv_file in csv_files:
        csv_name = csv_file.stem  # "analysis_summary_libero_10_20251107_145142..."
        csv_name_suffix = csv_name.replace("analysis_summary_", "")
        
        logger.debug(f"   Comparing: '{dir_name_suffix}' vs '{csv_name_suffix}'")
        
        if dir_name_suffix == csv_name_suffix:
            logger.info(f"   ✅ CSV found: {csv_file.name}")
            return csv_file
    
    # 3️⃣ 부분 매칭 (예: 일부만 일치해도 찾기)
    for csv_file in csv_files:
        if dir_name_suffix in csv_file.name or dir_name in csv_file.name:
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
    
    output_lines = []
    output_lines.append("\n" + "=" * 120)
    output_lines.append("📊 CSV 기반 물리량 비교 (BASELINE vs WITH NOISE)")
    output_lines.append("=" * 120)
    
    comparison_results = {}
    all_pct_changes = []
    
    # 각 카테고리별 비교
    for category, metrics in COMPARISON_METRICS.items():
        output_lines.append(f"\n{category}")
        output_lines.append("-" * 120)
        
        category_results = {}
        
        for metric in metrics:
            if metric not in df_before.columns or metric not in df_after.columns:
                continue
            
            before_vals = df_before[metric].dropna().astype(float)
            after_vals = df_after[metric].dropna().astype(float)
            
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
            if col_name in df_before.columns and col_name in df_after.columns:
                before_vals = df_before[col_name].dropna().astype(float)
                after_vals = df_after[col_name].dropna().astype(float)
                
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
            if col_name in df_before.columns and col_name in df_after.columns:
                before_vals = df_before[col_name].dropna().astype(float)
                after_vals = df_after[col_name].dropna().astype(float)
                
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
            if col_name in df_before.columns and col_name in df_after.columns:
                before_vals = df_before[col_name].dropna().astype(float)
                after_vals = df_after[col_name].dropna().astype(float)
                
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
            if col_name in df_before.columns and col_name in df_after.columns:
                before_vals = df_before[col_name].dropna().astype(float)
                after_vals = df_after[col_name].dropna().astype(float)
                
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
            if col_name in df_before.columns and col_name in df_after.columns:
                before_vals = df_before[col_name].dropna().astype(float)
                after_vals = df_after[col_name].dropna().astype(float)
                
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
    
    for col in df_before.columns:
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
                    
                    if col in df_before.columns and col in df_after.columns:
                        before_vals = df_before[col].dropna().astype(float)
                        after_vals = df_after[col].dropna().astype(float)
                        
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


def calculate_damage(torque_array: np.ndarray, m: float = 3.0, use_goodman: bool = True, 
                     sigma_ult: float = 500.0, sigma_y: float = 400.0) -> Dict[str, float]:
    """
    Rainflow 알고리즘을 통한 피로 손상 계산 (Goodman 보정 적용)
    
    Args:
        torque_array: shape (T, n_joints) - 시계열 토크 데이터
        m: Basquin 지수 (일반적으로 3~5, 기본값 3)
        use_goodman: Goodman 보정 적용 여부 (기본값: True)
        sigma_ult: 인장 강도 (기본값: 500 MPa)
        sigma_y: 항복 강도 (기본값: 400 MPa)
    
    Returns:
        각 조인트별 손상 및 총 손상:
        {
            "total_damage": D_total,
            "joint_0": D_0,
            "joint_1": D_1,
            ...
        }
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
    
    logger.info(f"손상 계산 완료: total={D_total:.6e}")
    return result


def find_matching_files(before_dir: pathlib.Path, after_dir: pathlib.Path) -> list:
    """
    before와 after 디렉토리에서 매칭되는 torque_current_*.npy 파일 찾기
    
    Returns:
        List of tuples: [(before_file, after_file, filename), ...]
    """
    before_files = sorted(before_dir.glob("torque_current_*.npy"))
    after_files = sorted(after_dir.glob("torque_current_*.npy"))
    
    matches = []
    for bf in before_files:
        name = bf.name
        af = after_dir / name
        if af.exists():
            matches.append((bf, af, name))
        else:
            logger.warning(f"⚠️  매칭되는 파일 없음: {name}")
    
    return matches


def compare_conditions(before_file: pathlib.Path, after_file: pathlib.Path, 
                       m: float = 3.0) -> Optional[Dict[str, float]]:
    """
    두 토크 파일 비교 및 상대 수명비 계산
    
    Args:
        before_file: Before torque_current_*.npy 파일 경로
        after_file: After torque_current_*.npy 파일 경로
        m: Basquin 지수
    
    Returns:
        Before/After 손상 및 수명비:
        {
            "damage_before_total": D_b,
            "damage_after_total": D_a,
            "life_ratio": D_a / D_b,
            "joint_0_damage_before": D_bj0,
            ...
        }
    """
    try:
        before_torque = np.load(before_file)
        after_torque = np.load(after_file)
        
        # 손상 계산
        damage_before = calculate_damage(before_torque, m=m)
        damage_after = calculate_damage(after_torque, m=m)
        
        if damage_before is None or damage_after is None:
            return None
        
        # 결과 준비
        result = {}
        result["damage_before_total"] = damage_before["total_damage"]
        result["damage_after_total"] = damage_after["total_damage"]
        
        # 수명비 계산 (ratio > 1이면 수명 단축)
        if damage_before["total_damage"] > 0:
            result["life_ratio"] = damage_after["total_damage"] / damage_before["total_damage"]
        else:
            result["life_ratio"] = np.inf
        
        # 조인트별 손상 및 비율
        n_joints = len(damage_before) - 1  # total_damage 제외
        for j in range(n_joints):
            key_before = f"joint_{j}"
            key_after = f"joint_{j}"
            
            if key_before in damage_before and key_after in damage_after:
                D_b = damage_before[key_before]
                D_a = damage_after[key_after]
                
                result[f"joint_{j}_damage_before"] = D_b
                result[f"joint_{j}_damage_after"] = D_a
                
                if D_b > 0:
                    result[f"joint_{j}_damage_ratio"] = D_a / D_b
                else:
                    result[f"joint_{j}_damage_ratio"] = np.inf
        
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


def compare_npy_metrics(before_dir: pathlib.Path, after_dir: pathlib.Path, m: float = 3.0) -> Tuple[str, list]:
    """
    NPY 파일 기반 손상 및 수명비 비교
    
    Returns:
        Tuple[출력 텍스트, 비교 결과 리스트]
    """
    logger.info(f"\n🔍 NPY 파일 비교 시작...")
    logger.info(f"   Before dir: {before_dir}")
    logger.info(f"   After dir:  {after_dir}")
    logger.info(f"   Basquin exponent (m): {m}")
    
    # 매칭되는 파일 찾기
    matches = find_matching_files(before_dir, after_dir)
    
    output_lines = []
    output_lines.append("\n" + "=" * 120)
    output_lines.append("⚙️  NPY 기반 피로 손상 & 수명비 비교 (RAINFLOW 알고리즘)")
    output_lines.append("=" * 120)
    
    # 각 파일 쌍 비교
    results = []
    all_ratios = []
    
    for before_file, after_file, filename in matches:
        logger.info(f"📈 처리 중: {filename}")
        
        comparison = compare_conditions(before_file, after_file, m=m)
        if comparison is not None:
            comparison["file"] = filename
            results.append(comparison)
            
            ratio = comparison.get("life_ratio", np.inf)
            all_ratios.append(ratio)
    
    # 1️⃣ 전체 요약 먼저
    output_lines.append(f"\n📊 NPY 기반 전체 요약 통계 (모든 파일 종합):\n")
    
    if all_ratios and not all(np.isinf(r) for r in all_ratios):
        finite_ratios = [r for r in all_ratios if not np.isinf(r)]
        if finite_ratios:
            mean_ratio = np.mean(finite_ratios)
            min_ratio = np.min(finite_ratios)
            max_ratio = np.max(finite_ratios)
            
            output_lines.append(f"  평균 수명비:       {mean_ratio:.4f}x")
            output_lines.append(f"  최소 수명비:       {min_ratio:.4f}x (가장 양호)")
            output_lines.append(f"  최대 수명비:       {max_ratio:.4f}x (가장 악화)")
            output_lines.append(f"  비교된 파일:       {len(finite_ratios)}개\n")
            
            # 명확한 해석
            if mean_ratio > 1.05:
                pct_reduction = (mean_ratio - 1) * 100
                output_lines.append(f"  ⚠️  결론: 노이즈로 인해 평균 손상이 약 {pct_reduction:.1f}% 증가 → 수명 단축")
            elif mean_ratio < 0.95:
                pct_improve = (1 - mean_ratio) * 100
                output_lines.append(f"  ✅ 결론: 노이즈로 인해 평균 손상이 약 {pct_improve:.1f}% 감소 → 수명 개선 (예외적)")
            else:
                output_lines.append(f"  ➡️  결론: 노이즈의 영향 미미 → 수명 변화 거의 없음")
    
    # 2️⃣ 이어서 Joint별 분석
    output_lines.append(f"\n{'─' * 120}")
    output_lines.append(f"\n⚙️  Joint별 분석 (각 조인트별 평균 손상 및 수명비):\n")
    
    joint_stats = defaultdict(lambda: {"damage_before": [], "damage_after": [], "ratio": []})
    
    for result in results:
        for j in range(7):  # 7개 조인트
            key_before = f"joint_{j}_damage_before"
            key_after = f"joint_{j}_damage_after"
            key_ratio = f"joint_{j}_damage_ratio"
            
            if key_before in result and key_after in result:
                d_b = result[key_before]
                d_a = result[key_after]
                ratio = result.get(key_ratio, np.inf)
                
                joint_stats[j]["damage_before"].append(d_b)
                joint_stats[j]["damage_after"].append(d_a)
                if not np.isinf(ratio):
                    joint_stats[j]["ratio"].append(ratio)
    
    # Joint별 상세 테이블
    output_lines.append(f"{'Joint':8s} | {'Before Damage':20s} | {'After Damage':20s} | {'수명비':12s} | {'상태':30s}")
    output_lines.append(f"{'-'*120}")
    
    joint_assessments = []  # 각 조인트별 평가 저장
    
    for j in range(7):
        if j in joint_stats and joint_stats[j]["damage_before"]:
            avg_before = np.mean(joint_stats[j]["damage_before"])
            avg_after = np.mean(joint_stats[j]["damage_after"])
            avg_ratio = np.mean(joint_stats[j]["ratio"]) if joint_stats[j]["ratio"] else 1.0
            
            # 상태 판정
            if avg_ratio > 1.1:
                status = "📈 심각 (수명 크게 단축)"
                symbol = "🔴"
            elif avg_ratio > 1.0:
                status = "📈 주의 (수명 단축)"
                symbol = "🟠"
            elif avg_ratio < 0.9:
                status = "📉 개선 (수명 증가)"
                symbol = "🟢"
            elif avg_ratio < 1.0:
                status = "📉 약간 개선"
                symbol = "🟡"
            else:
                status = "➡️  변화 없음"
                symbol = "⚪"
            
            output_lines.append(
                f"Joint_{j}  | {avg_before:18.6e} | {avg_after:18.6e} | {avg_ratio:10.4f}x | {symbol} {status}"
            )
            
            joint_assessments.append((j, avg_ratio, status))
    
    # 가장 취약한 조인트 강조
    if joint_assessments:
        worst_joint = max(joint_assessments, key=lambda x: x[1])
        output_lines.append(f"\n⚠️  가장 취약한 조인트: Joint_{worst_joint[0]} (수명비: {worst_joint[1]:.4f}x)")
    
    # 3️⃣ 파일별 세부 수명비 분석
    output_lines.append(f"\n{'─' * 120}")
    output_lines.append(f"\n📁 매칭된 파일 {len(results)}개 발견\n")
    
    for result in results:
        filename = result.get("file", "unknown")
        ratio = result.get("life_ratio", 1.0)
        
        if ratio > 1.05:
            interpretation = "단축 ⬆️"
        elif ratio < 0.95:
            interpretation = "증가 ✅"
        else:
            interpretation = "유지 ➡️"
        
        output_lines.append(f"  {filename:80s} | 수명비: {ratio:.4f} ({interpretation})")
    
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
    npy_output, npy_results = compare_npy_metrics(before_dir, after_dir, m=args.m)
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
