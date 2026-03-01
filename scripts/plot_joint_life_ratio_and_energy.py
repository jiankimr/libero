"""
6개 비교 결과 파일에서 Joint별 수명비 그래프와 에너지 요약 파일 생성
- Rainflow와 Fatpack 두 가지 버전의 수명비 그래프 생성
- 16스텝, 4스텝 샘플링 모두 지원
"""
import re
import pathlib
import matplotlib.pyplot as plt
import numpy as np

# 결과 파일 경로
RESULT_DIR = pathlib.Path("/home/taywonmin/rsec/repos/rsec/LIBERO/results")
OUTPUT_DIR = pathlib.Path("/home/taywonmin/rsec/repos/rsec/LIBERO/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_files_for_step(step: int):
    """샘플링 스텝에 따른 파일 목록 반환"""
    return [
        (f"comparison_{step}step_noise_01000_rollout_noise_01_actionx.txt", "0.1", 0.1),
        (f"comparison_{step}step_noise_02000_rollout_noise_02_actionx.txt", "0.2", 0.2),
        (f"comparison_{step}step_noise_03000_rollout_noise_03_actionx.txt", "0.3", 0.3),
        (f"comparison_{step}step_noise_04000_rollout_noise_04_actionx.txt", "0.4", 0.4),
        (f"comparison_{step}step_noise_05000_rollout_noise_05_actionx.txt", "0.5", 0.5),
        (f"comparison_{step}step_noise_06000_rollout_noise_06_actionx.txt", "0.6", 0.6),
    ]


def extract_joint_life_ratios(filepath: pathlib.Path, method: str = "rainflow") -> dict:
    """
    파일에서 Joint별 수명비 추출
    
    Args:
        filepath: 결과 파일 경로
        method: "rainflow" 또는 "fatpack"
    
    Returns:
        {"Joint_0": 1.8818, "Joint_1": 1.5218, ...}
    """
    content = filepath.read_text()
    
    # 패턴: "Joint_0  |       1.441033e+03 |       1.854144e+03 |     1.8818x |"
    pattern = r"Joint_(\d)\s+\|\s+[\d.e+\-]+\s+\|\s+[\d.e+\-]+\s+\|\s+([\d.]+)x"
    matches = re.findall(pattern, content)
    
    # Rainflow는 첫 7개, Fatpack은 그 다음 7개
    if method.lower() == "rainflow":
        target_matches = matches[:7]
    else:  # fatpack
        target_matches = matches[7:14]
    
    result = {}
    for joint_num, life_ratio in target_matches:
        result[f"Joint_{joint_num}"] = float(life_ratio)
    
    return result


def extract_energy_metrics(filepath: pathlib.Path) -> dict:
    """파일에서 에너지 메트릭 추출 (sum 기준)"""
    content = filepath.read_text()
    result = {}
    
    patterns = [
        (r"energy_draw\.sum\s+\|\s+Before:\s+([\d.]+)±\s*[\d.]+\s+→\s+After:\s+([\d.]+)±\s*[\d.]+\s+\|\s+Δ\s+([+\-]?[\d.]+)%", "energy_draw_sum"),
        (r"energy_regen\.sum\s+\|\s+Before:\s+([\d.]+)±\s*[\d.]+\s+→\s+After:\s+([\d.]+)±\s*[\d.]+\s+\|\s+Δ\s+([+\-]?[\d.]+)%", "energy_regen_sum"),
        (r"energy_net\.sum\s+\|\s+Before:\s+([\d.]+)±\s*[\d.]+\s+→\s+After:\s+([\d.]+)±\s*[\d.]+\s+\|\s+Δ\s+([+\-]?[\d.]+)%", "energy_net_sum"),
    ]
    
    for pattern, key in patterns:
        match = re.search(pattern, content)
        if match:
            result[key] = {
                "before": float(match.group(1)),
                "after": float(match.group(2)),
                "change_pct": float(match.group(3)),
            }
    
    return result


def plot_joint_life_ratios_single(step: int, method: str = "rainflow"):
    """
    Joint별 수명비 꺾은선 그래프 생성 (log scale, 최대 joint 표시)
    """
    FILES = get_files_for_step(step)
    
    # 데이터 수집
    noise_vals = []
    joint_data = {f"Joint_{i}": [] for i in range(7)}
    
    for filename, label, noise_val in FILES:
        filepath = RESULT_DIR / filename
        if not filepath.exists():
            print(f"⚠️ 파일 없음: {filepath}")
            continue
        
        ratios = extract_joint_life_ratios(filepath, method=method)
        noise_vals.append(noise_val)
        
        for joint_name in joint_data.keys():
            joint_data[joint_name].append(ratios.get(joint_name, np.nan))
    
    if not noise_vals:
        print(f"⚠️ {step}스텝 데이터가 없습니다.")
        return
    
    # 노이즈 0 (baseline) 추가
    x_vals = [0] + noise_vals
    for joint_name in joint_data.keys():
        joint_data[joint_name] = [1.0] + joint_data[joint_name]
    
    # 색상 설정
    colors = ['#d4a017', '#5dade2', '#27ae60', '#e74c3c', '#f39c12', '#9b59b6', '#e91e63']
    
    # 꺾은선 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (joint_name, values) in enumerate(joint_data.items()):
        ax.plot(x_vals, values, marker='o', linewidth=2, markersize=6, 
                label=joint_name, color=colors[i])
    
    # 각 노이즈 레벨에서 최대 joint와 수명비 표시
    for idx, noise in enumerate(x_vals):
        if noise == 0:
            continue
        
        max_ratio = 0
        max_joint = ""
        for joint_name, values in joint_data.items():
            if values[idx] > max_ratio:
                max_ratio = values[idx]
                max_joint = joint_name
        
        ax.annotate(f'{max_joint}\n{max_ratio:.3f}x', 
                   xy=(noise, max_ratio), 
                   xytext=(noise, max_ratio * 1.3),
                   ha='center', va='bottom', fontsize=8,
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    
    ax.set_yscale('log')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('noise scale (raw)', fontsize=12)
    ax.set_ylabel('Life ratio (x, log scale)', fontsize=12)
    
    method_title = "Rainflow" if method.lower() == "rainflow" else "Fatpack"
    ax.set_title(f'{method_title} ({step}-step sampling) - Top: weakest joint & life ratio', fontsize=11, style='italic')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in x_vals])
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"joint_life_ratio_by_noise_{step}step_{method.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Joint 수명비 그래프 ({step}step, {method}) 저장: {output_path}")


def plot_energy_changes(step: int):
    """에너지 변화 막대 그래프 생성"""
    FILES = get_files_for_step(step)
    
    all_data = []
    for filename, label, noise_val in FILES:
        filepath = RESULT_DIR / filename
        if not filepath.exists():
            continue
        
        energy = extract_energy_metrics(filepath)
        all_data.append({
            "noise_val": noise_val,
            "draw": energy.get("energy_draw_sum", {}),
            "regen": energy.get("energy_regen_sum", {}),
        })
    
    if not all_data:
        print(f"⚠️ {step}스텝 에너지 데이터가 없습니다.")
        return
    
    noise_vals = [0] + [d["noise_val"] for d in all_data]
    draw_changes = [0] + [d["draw"].get("change_pct", 0) for d in all_data]
    regen_changes = [0] + [d["regen"].get("change_pct", 0) for d in all_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(noise_vals))
    width = 0.35
    
    ax.bar(x - width/2, draw_changes, width, label='draw_sum delta%', color='#d4a017', alpha=0.9)
    ax.bar(x + width/2, regen_changes, width, label='regen_sum delta%', color='#5dade2', alpha=0.9)
    
    ax.axhline(y=0, color='#d4a017', linestyle='--', linewidth=1, alpha=0.5)
    
    max_draw_idx = np.argmax(draw_changes)
    if draw_changes[max_draw_idx] > 0:
        ax.annotate(f'{draw_changes[max_draw_idx]:.1f}%', 
                   xy=(max_draw_idx - width/2, draw_changes[max_draw_idx]),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('noise scale (raw)', fontsize=12)
    ax.set_ylabel('Change (%)', fontsize=12)
    ax.set_title(f'Energy Change ({step}-step sampling)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in noise_vals])
    
    y_max = max(max(draw_changes), abs(min(regen_changes))) * 1.2
    ax.set_ylim(-30, y_max if y_max > 100 else 120)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"energy_change_by_noise_{step}step.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 에너지 변화 막대 그래프 ({step}step) 저장: {output_path}")


def generate_all_plots(step: int):
    """특정 스텝에 대한 모든 그래프 생성"""
    print(f"\n{'='*60}")
    print(f"📊 {step}-step sampling analysis")
    print(f"{'='*60}")
    
    plot_joint_life_ratios_single(step, method="rainflow")
    plot_joint_life_ratios_single(step, method="fatpack")
    plot_energy_changes(step)


if __name__ == "__main__":
    print("=" * 60)
    print("Joint Life Ratio & Energy Analysis")
    print("=" * 60)
    
    # 16스텝 분석
    generate_all_plots(16)
    
    # 4스텝 분석
    generate_all_plots(4)
    
    print("\n✅ All analysis completed!")
