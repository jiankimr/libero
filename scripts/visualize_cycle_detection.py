"""
원본 vs 4스텝 샘플링 토크에서 Rainflow/Fatpack 사이클 감지 시각화

두 패키지가 토크 시계열에서 어디를 사이클로, 어디를 진폭으로 보는지 상세 시각화
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import pathlib

try:
    import rainflow
    HAS_RAINFLOW = True
except ImportError:
    HAS_RAINFLOW = False

try:
    import fatpack
    HAS_FATPACK = True
except ImportError:
    HAS_FATPACK = False


# 경로 설정
ANALYSIS_DIR = pathlib.Path("/home/taywonmin/rsec/repos/rsec/LIBERO/analysis/analysis_libero_10_20251202_170035_noise_00000_clean")
OUTPUT_DIR = pathlib.Path("/home/taywonmin/rsec/repos/rsec/LIBERO/plots")


def get_rainflow_cycles_with_indices(signal):
    """Rainflow 사이클 추출 (인덱스 포함)"""
    signal_centered = signal - np.mean(signal)
    cycles = list(rainflow.extract_cycles(signal_centered))
    
    result = []
    for range_val, mean_val, count, i_start, i_end in cycles:
        amplitude = range_val / 2.0
        result.append({
            'range': range_val,
            'amplitude': amplitude,
            'mean': mean_val,
            'count': count,
            'i_start': i_start,
            'i_end': i_end,
        })
    return result


def get_fatpack_cycles_with_indices(signal):
    """Fatpack 사이클 추출 (인덱스 포함)"""
    reversals, reversal_indices = fatpack.find_reversals(signal)
    
    if len(reversals) < 2:
        return [], reversal_indices, reversals
    
    cycles, residue = fatpack.find_rainflow_cycles(reversals)
    
    result = []
    for cycle in cycles:
        start_val, end_val = cycle
        range_val = abs(end_val - start_val)
        amplitude = range_val / 2.0
        mean_val = (start_val + end_val) / 2.0
        
        # 인덱스 찾기 (근사값)
        i_start = np.argmin(np.abs(signal - start_val))
        i_end = np.argmin(np.abs(signal - end_val))
        
        result.append({
            'range': range_val,
            'amplitude': amplitude,
            'mean': mean_val,
            'start_val': start_val,
            'end_val': end_val,
            'i_start': i_start,
            'i_end': i_end,
        })
    
    return result, reversal_indices, reversals


def plot_cycle_detection_comparison(npy_file: pathlib.Path, joint_idx: int = 5, sample_step: int = 4):
    """원본 vs 샘플링 토크에서 사이클 감지 비교 시각화"""
    
    # 데이터 로드
    torque_data = np.load(npy_file)
    original_signal = torque_data[:, joint_idx]
    
    # 4스텝 샘플링
    sampled_signal = original_signal[::sample_step]
    
    # 시간 축
    dt = 1/20.0  # 20Hz
    t_original = np.arange(len(original_signal)) * dt
    t_sampled = np.arange(len(sampled_signal)) * dt * sample_step
    
    # 사이클 추출
    rf_cycles_orig = get_rainflow_cycles_with_indices(original_signal)
    rf_cycles_samp = get_rainflow_cycles_with_indices(sampled_signal)
    
    fp_cycles_orig, fp_rev_idx_orig, fp_rev_orig = get_fatpack_cycles_with_indices(original_signal)
    fp_cycles_samp, fp_rev_idx_samp, fp_rev_samp = get_fatpack_cycles_with_indices(sampled_signal)
    
    # 손상 계산
    m = 3.0
    rf_damage_orig = sum(c['amplitude']**m * c['count'] for c in rf_cycles_orig)
    rf_damage_samp = sum(c['amplitude']**m * c['count'] for c in rf_cycles_samp)
    fp_damage_orig = sum(c['amplitude']**m for c in fp_cycles_orig)
    fp_damage_samp = sum(c['amplitude']**m for c in fp_cycles_samp)
    
    # 그래프 생성 (2x2 레이아웃)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors_cycle = plt.cm.Set1(np.linspace(0, 1, 10))
    
    # ========== 1. 원본 + Rainflow ==========
    ax1 = axes[0, 0]
    ax1.plot(t_original, original_signal, 'b-', linewidth=1, alpha=0.7, label='Torque Signal')
    ax1.axhline(np.mean(original_signal), color='gray', linestyle='--', alpha=0.5, label='Mean')
    
    # Rainflow 사이클 표시 (상위 5개)
    sorted_cycles = sorted(rf_cycles_orig, key=lambda x: x['amplitude'], reverse=True)[:5]
    for i, cycle in enumerate(sorted_cycles):
        i_s, i_e = cycle['i_start'], cycle['i_end']
        if i_s < len(t_original) and i_e < len(t_original):
            # 사이클 구간 하이라이트
            ax1.axvspan(t_original[i_s], t_original[min(i_e, len(t_original)-1)], 
                       alpha=0.2, color=colors_cycle[i], label=f"Cycle {i+1}: amp={cycle['amplitude']:.2f}")
            # 진폭 화살표
            mid_idx = (i_s + i_e) // 2
            if mid_idx < len(t_original):
                ax1.annotate('', xy=(t_original[mid_idx], original_signal[mid_idx] + cycle['amplitude']),
                           xytext=(t_original[mid_idx], original_signal[mid_idx] - cycle['amplitude']),
                           arrowprops=dict(arrowstyle='<->', color=colors_cycle[i], lw=2))
    
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Torque (Nm)', fontsize=11)
    ax1.set_title(f'Original Signal + Rainflow\n{len(rf_cycles_orig)} cycles, Damage={rf_damage_orig:.2e}', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # ========== 2. 원본 + Fatpack ==========
    ax2 = axes[0, 1]
    ax2.plot(t_original, original_signal, 'b-', linewidth=1, alpha=0.7, label='Torque Signal')
    
    # Reversals (극값점) 표시
    if len(fp_rev_idx_orig) > 0:
        ax2.scatter(t_original[fp_rev_idx_orig], fp_rev_orig, 
                   c='red', s=50, zorder=5, marker='o', label=f'Reversals ({len(fp_rev_idx_orig)})')
    
    # Fatpack 사이클 표시 (상위 5개)
    sorted_fp_cycles = sorted(fp_cycles_orig, key=lambda x: x['amplitude'], reverse=True)[:5]
    for i, cycle in enumerate(sorted_fp_cycles):
        # 사이클 연결선
        i_s, i_e = cycle['i_start'], cycle['i_end']
        if i_s < len(t_original) and i_e < len(t_original):
            ax2.plot([t_original[i_s], t_original[i_e]], 
                    [original_signal[i_s], original_signal[i_e]],
                    color=colors_cycle[i], linewidth=2, linestyle='--',
                    label=f"Cycle {i+1}: amp={cycle['amplitude']:.2f}")
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Torque (Nm)', fontsize=11)
    ax2.set_title(f'Original Signal + Fatpack\n{len(fp_cycles_orig)} cycles, Damage={fp_damage_orig:.2e}', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # ========== 3. 4스텝 샘플링 + Rainflow ==========
    ax3 = axes[1, 0]
    ax3.plot(t_sampled, sampled_signal, 'g-', linewidth=1.5, alpha=0.8, marker='o', markersize=4, label='Sampled (4-step)')
    ax3.axhline(np.mean(sampled_signal), color='gray', linestyle='--', alpha=0.5, label='Mean')
    
    # Rainflow 사이클 표시 (상위 5개)
    sorted_cycles_samp = sorted(rf_cycles_samp, key=lambda x: x['amplitude'], reverse=True)[:5]
    for i, cycle in enumerate(sorted_cycles_samp):
        i_s, i_e = cycle['i_start'], cycle['i_end']
        if i_s < len(t_sampled) and i_e < len(t_sampled):
            ax3.axvspan(t_sampled[i_s], t_sampled[min(i_e, len(t_sampled)-1)], 
                       alpha=0.2, color=colors_cycle[i], label=f"Cycle {i+1}: amp={cycle['amplitude']:.2f}")
            mid_idx = (i_s + i_e) // 2
            if mid_idx < len(t_sampled):
                ax3.annotate('', xy=(t_sampled[mid_idx], sampled_signal[mid_idx] + cycle['amplitude']),
                           xytext=(t_sampled[mid_idx], sampled_signal[mid_idx] - cycle['amplitude']),
                           arrowprops=dict(arrowstyle='<->', color=colors_cycle[i], lw=2))
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Torque (Nm)', fontsize=11)
    ax3.set_title(f'4-step Sampled + Rainflow\n{len(rf_cycles_samp)} cycles, Damage={rf_damage_samp:.2e}', 
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # ========== 4. 4스텝 샘플링 + Fatpack ==========
    ax4 = axes[1, 1]
    ax4.plot(t_sampled, sampled_signal, 'g-', linewidth=1.5, alpha=0.8, marker='o', markersize=4, label='Sampled (4-step)')
    
    # Reversals 표시
    if len(fp_rev_idx_samp) > 0:
        t_sampled_rev = t_sampled[fp_rev_idx_samp] if len(fp_rev_idx_samp) <= len(t_sampled) else t_sampled[:len(fp_rev_idx_samp)]
        ax4.scatter(t_sampled[fp_rev_idx_samp[:len(t_sampled)]], fp_rev_samp[:len(t_sampled)], 
                   c='red', s=80, zorder=5, marker='o', label=f'Reversals ({len(fp_rev_idx_samp)})')
    
    # Fatpack 사이클 표시 (상위 5개)
    sorted_fp_cycles_samp = sorted(fp_cycles_samp, key=lambda x: x['amplitude'], reverse=True)[:5]
    for i, cycle in enumerate(sorted_fp_cycles_samp):
        i_s, i_e = cycle['i_start'], cycle['i_end']
        if i_s < len(t_sampled) and i_e < len(t_sampled):
            ax4.plot([t_sampled[i_s], t_sampled[i_e]], 
                    [sampled_signal[i_s], sampled_signal[i_e]],
                    color=colors_cycle[i], linewidth=2, linestyle='--',
                    label=f"Cycle {i+1}: amp={cycle['amplitude']:.2f}")
    
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Torque (Nm)', fontsize=11)
    ax4.set_title(f'4-step Sampled + Fatpack\n{len(fp_cycles_samp)} cycles, Damage={fp_damage_samp:.2e}', 
                 fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(alpha=0.3)
    
    # 전체 타이틀
    task_name = npy_file.stem.replace("torque_current_", "").replace("_success", "")[:40]
    fig.suptitle(f'Cycle Detection Comparison: Original vs 4-step Sampled (Joint {joint_idx})\n{task_name}', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 저장
    output_path = OUTPUT_DIR / f"cycle_detection_comparison_joint{joint_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 사이클 감지 비교 시각화 저장: {output_path}")
    
    # 요약 테이블 출력
    print(f"\n{'='*70}")
    print(f"📊 사이클 감지 비교 요약 (Joint {joint_idx})")
    print(f"{'='*70}")
    print(f"{'':20} | {'Original':20} | {'4-step Sampled':20}")
    print(f"{'-'*70}")
    print(f"{'Signal Length':20} | {len(original_signal):20} | {len(sampled_signal):20}")
    print(f"{'Rainflow Cycles':20} | {len(rf_cycles_orig):20} | {len(rf_cycles_samp):20}")
    print(f"{'Rainflow Damage':20} | {rf_damage_orig:20.2e} | {rf_damage_samp:20.2e}")
    print(f"{'Fatpack Cycles':20} | {len(fp_cycles_orig):20} | {len(fp_cycles_samp):20}")
    print(f"{'Fatpack Damage':20} | {fp_damage_orig:20.2e} | {fp_damage_samp:20.2e}")
    print(f"{'='*70}")
    
    # 샘플링 효과 분석
    print(f"\n📉 샘플링 효과:")
    if rf_damage_orig > 0:
        print(f"   Rainflow 손상 변화: {rf_damage_samp/rf_damage_orig:.2%} of original")
    if fp_damage_orig > 0:
        print(f"   Fatpack 손상 변화: {fp_damage_samp/fp_damage_orig:.2%} of original")


def create_detailed_visualization():
    """상세 시각화 생성"""
    
    # 샘플 파일 찾기
    npy_files = list(ANALYSIS_DIR.glob("torque_current_*_success.npy"))
    
    if not npy_files:
        print("⚠️ NPY 파일을 찾을 수 없습니다.")
        return
    
    # 첫 번째 파일로 시각화
    npy_file = npy_files[0]
    print(f"📄 분석 파일: {npy_file.name}")
    
    # Joint 5에 대해 시각화
    plot_cycle_detection_comparison(npy_file, joint_idx=5, sample_step=4)


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Cycle Detection Comparison: Original vs 4-step Sampled")
    print("=" * 60)
    
    if not HAS_RAINFLOW or not HAS_FATPACK:
        print("⚠️ 필요한 패키지가 설치되지 않았습니다.")
        exit(1)
    
    create_detailed_visualization()
    
    print("\n✅ 분석 완료!")





