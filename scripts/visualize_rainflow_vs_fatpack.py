"""
Rainflow vs Fatpack 분석 차이 시각화

두 패키지가 동일한 토크 시계열에서 어떻게 다르게 사이클을 추출하는지 시각화
"""
import numpy as np
import matplotlib.pyplot as plt
import pathlib

try:
    import rainflow
    HAS_RAINFLOW = True
except ImportError:
    HAS_RAINFLOW = False
    print("⚠️ rainflow 패키지가 설치되지 않았습니다.")

try:
    import fatpack
    HAS_FATPACK = True
except ImportError:
    HAS_FATPACK = False
    print("⚠️ fatpack 패키지가 설치되지 않았습니다.")


# 분석 디렉토리
ANALYSIS_DIR = pathlib.Path("/home/taywonmin/rsec/repos/rsec/LIBERO/analysis/analysis_libero_10_20251202_170035_noise_00000_clean")
OUTPUT_DIR = pathlib.Path("/home/taywonmin/rsec/repos/rsec/LIBERO/plots")


def find_sample_npy_files():
    """샘플 토크 NPY 파일 찾기"""
    npy_files = list(ANALYSIS_DIR.glob("torque_current_*_success.npy"))
    if npy_files:
        return npy_files[:2]  # 처음 2개만
    return []


def analyze_with_rainflow(torque_data: np.ndarray, joint_idx: int = 5):
    """Rainflow로 사이클 분석"""
    if not HAS_RAINFLOW:
        return None, None, None
    
    signal = torque_data[:, joint_idx]
    signal_centered = signal - np.mean(signal)
    
    cycles = list(rainflow.extract_cycles(signal_centered))
    
    ranges = []
    means = []
    counts = []
    
    for range_val, mean_val, count, i_start, i_end in cycles:
        ranges.append(range_val)
        means.append(mean_val)
        counts.append(count)
    
    return np.array(ranges), np.array(means), np.array(counts)


def analyze_with_fatpack(torque_data: np.ndarray, joint_idx: int = 5):
    """Fatpack으로 사이클 분석"""
    if not HAS_FATPACK:
        return None, None
    
    signal = torque_data[:, joint_idx]
    
    # Find reversals (극값점)
    reversals, reversal_indices = fatpack.find_reversals(signal)
    
    if len(reversals) < 2:
        return None, None
    
    # Find rainflow cycles
    cycles, residue = fatpack.find_rainflow_cycles(reversals)
    
    if len(cycles) == 0:
        return None, None
    
    # cycles: Nx2 array, each row is [start, end] of a cycle
    ranges = np.abs(cycles[:, 1] - cycles[:, 0])
    means = (cycles[:, 0] + cycles[:, 1]) / 2
    
    return ranges, means


def calculate_damage(ranges, m=3.0):
    """Basquin 기반 손상 계산"""
    amplitudes = ranges / 2.0
    return np.sum(amplitudes ** m)


def plot_comparison(npy_file: pathlib.Path, joint_idx: int = 5):
    """Rainflow vs Fatpack 비교 시각화"""
    
    torque_data = np.load(npy_file)
    signal = torque_data[:, joint_idx]
    time = np.arange(len(signal)) / 20.0  # 20Hz 샘플링
    
    # 분석 실행
    rf_ranges, rf_means, rf_counts = analyze_with_rainflow(torque_data, joint_idx)
    fp_ranges, fp_means = analyze_with_fatpack(torque_data, joint_idx)
    
    # 손상 계산
    m = 3.0
    rf_damage = calculate_damage(rf_ranges, m) if rf_ranges is not None else 0
    fp_damage = calculate_damage(fp_ranges, m) if fp_ranges is not None else 0
    
    # 그래프 생성 (3행 2열)
    fig = plt.figure(figsize=(14, 12))
    
    # 1. 원본 토크 신호
    ax1 = fig.add_subplot(3, 2, (1, 2))
    ax1.plot(time, signal, 'b-', linewidth=0.8, alpha=0.8)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Torque (Nm)', fontsize=11)
    ax1.set_title(f'Original Torque Signal - Joint {joint_idx}', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Fatpack reversals 표시
    if HAS_FATPACK:
        reversals, reversal_indices = fatpack.find_reversals(signal)
        ax1.scatter(time[reversal_indices], reversals, c='red', s=15, zorder=5, label='Reversals (Fatpack)')
        ax1.legend(loc='upper right', fontsize=9)
    
    # 2. Rainflow 사이클 범위 히스토그램
    ax2 = fig.add_subplot(3, 2, 3)
    if rf_ranges is not None and len(rf_ranges) > 0:
        ax2.hist(rf_ranges, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(rf_ranges), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rf_ranges):.3f}')
        ax2.set_xlabel('Cycle Range (Nm)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title(f'Rainflow: {len(rf_ranges)} cycles, Damage: {rf_damage:.2e}', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Rainflow data not available', ha='center', va='center', fontsize=12)
    
    # 3. Fatpack 사이클 범위 히스토그램
    ax3 = fig.add_subplot(3, 2, 4)
    if fp_ranges is not None and len(fp_ranges) > 0:
        ax3.hist(fp_ranges, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(fp_ranges), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(fp_ranges):.3f}')
        ax3.set_xlabel('Cycle Range (Nm)', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title(f'Fatpack: {len(fp_ranges)} cycles, Damage: {fp_damage:.2e}', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Fatpack data not available', ha='center', va='center', fontsize=12)
    
    # 4. Mean-Range scatter (Rainflow)
    ax4 = fig.add_subplot(3, 2, 5)
    if rf_ranges is not None and rf_means is not None and len(rf_ranges) > 0:
        scatter = ax4.scatter(rf_means, rf_ranges/2, c=rf_counts, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax4, label='Cycle Count')
        ax4.set_xlabel('Mean Stress (Nm)', fontsize=11)
        ax4.set_ylabel('Amplitude (Nm)', fontsize=11)
        ax4.set_title('Rainflow: Mean-Amplitude Distribution', fontsize=11, fontweight='bold')
        ax4.grid(alpha=0.3)
    
    # 5. Mean-Range scatter (Fatpack)
    ax5 = fig.add_subplot(3, 2, 6)
    if fp_ranges is not None and fp_means is not None and len(fp_ranges) > 0:
        ax5.scatter(fp_means, fp_ranges/2, c='#e74c3c', s=50, alpha=0.7)
        ax5.set_xlabel('Mean Stress (Nm)', fontsize=11)
        ax5.set_ylabel('Amplitude (Nm)', fontsize=11)
        ax5.set_title('Fatpack: Mean-Amplitude Distribution', fontsize=11, fontweight='bold')
        ax5.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # 파일명 정리
    task_name = npy_file.stem.replace("torque_current_", "").replace("_success", "")[:50]
    output_path = OUTPUT_DIR / f"rainflow_vs_fatpack_joint{joint_idx}_{task_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 비교 시각화 저장: {output_path}")
    
    # 요약 출력
    print(f"\n📊 분석 요약 (Joint {joint_idx}):")
    print(f"   신호 길이: {len(signal)} samples ({len(signal)/20:.1f} sec)")
    if rf_ranges is not None:
        print(f"   Rainflow: {len(rf_ranges)} cycles, Damage={rf_damage:.4e}")
    if fp_ranges is not None:
        print(f"   Fatpack:  {len(fp_ranges)} cycles, Damage={fp_damage:.4e}")
    if rf_damage > 0 and fp_damage > 0:
        ratio = fp_damage / rf_damage
        print(f"   Fatpack/Rainflow 손상비: {ratio:.2f}x")
    
    return rf_damage, fp_damage


def create_summary_comparison():
    """여러 파일에 대한 요약 비교 생성"""
    
    npy_files = find_sample_npy_files()
    
    if not npy_files:
        print("⚠️ 분석할 NPY 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 분석할 파일 {len(npy_files)}개 발견")
    
    all_rf_damages = []
    all_fp_damages = []
    
    for npy_file in npy_files:
        print(f"\n{'='*60}")
        print(f"📄 파일: {npy_file.name}")
        print(f"{'='*60}")
        
        for joint_idx in [3, 5]:  # Joint 3과 5에 대해 분석
            rf_damage, fp_damage = plot_comparison(npy_file, joint_idx=joint_idx)
            if rf_damage and fp_damage:
                all_rf_damages.append(rf_damage)
                all_fp_damages.append(fp_damage)
    
    # 전체 요약
    if all_rf_damages and all_fp_damages:
        print(f"\n{'='*60}")
        print("📊 전체 요약")
        print(f"{'='*60}")
        
        avg_rf = np.mean(all_rf_damages)
        avg_fp = np.mean(all_fp_damages)
        
        print(f"평균 Rainflow 손상: {avg_rf:.4e}")
        print(f"평균 Fatpack 손상:  {avg_fp:.4e}")
        print(f"Fatpack/Rainflow 비율: {avg_fp/avg_rf:.2f}x")
        
        print(f"\n💡 차이 원인:")
        print("   - Rainflow (3-point): 전체 사이클과 반사이클 구분")
        print("   - Fatpack (4-point): 극값점 기반 사이클 추출")
        print("   - Fatpack이 더 작은 사이클도 많이 감지하는 경향")


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Rainflow vs Fatpack Analysis Comparison")
    print("=" * 60)
    
    if not HAS_RAINFLOW or not HAS_FATPACK:
        print("⚠️ 필요한 패키지가 설치되지 않았습니다.")
        exit(1)
    
    create_summary_comparison()
    
    print("\n✅ 분석 완료!")

