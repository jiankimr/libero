"""
Rainflow vs Fatpack 알고리즘 원리 교육용 시각화

처음 보는 사람도 두 알고리즘의 작동 원리를 이해할 수 있도록 단계별 설명
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import pathlib

OUTPUT_DIR = pathlib.Path("/home/taywonmin/rsec/repos/rsec/LIBERO/plots")


def create_simple_signal():
    """설명용 간단한 토크 신호 생성"""
    # 이해하기 쉬운 패턴의 신호
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    signal = np.array([0, 4, 1, 3, -1, 5, 2, 4, 0, 3, 1])
    return t, signal


def create_explanation_figure():
    """Rainflow vs Fatpack 원리 설명 시각화"""
    
    t, signal = create_simple_signal()
    
    fig = plt.figure(figsize=(18, 14))
    
    # ========== 상단: 제목과 개요 ==========
    fig.suptitle('Rainflow vs Fatpack: How They Count Fatigue Cycles', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # ========== 1행: 원본 신호와 기본 개념 ==========
    
    # 1-1: 원본 신호
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(t, signal, 'b-o', linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Stress/Torque', fontsize=11)
    ax1.set_title('① Original Signal\n(Stress over time)', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-3, 7)
    
    # 극값점 표시
    peaks = [1, 3, 5, 7, 9]  # 인덱스
    valleys = [0, 2, 4, 6, 8, 10]
    for p in peaks:
        ax1.scatter(t[p], signal[p], c='red', s=150, zorder=5, marker='^', label='Peak' if p==1 else '')
    for v in valleys:
        ax1.scatter(t[v], signal[v], c='green', s=150, zorder=5, marker='v', label='Valley' if v==0 else '')
    ax1.legend(loc='upper right', fontsize=9)
    
    # 1-2: 사이클이란?
    ax2 = fig.add_subplot(3, 3, 2)
    # 단일 사이클 예시
    cycle_t = np.array([0, 1, 2])
    cycle_s = np.array([0, 4, 0])
    ax2.plot(cycle_t, cycle_s, 'b-o', linewidth=3, markersize=12, markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(cycle_t, 0, cycle_s, alpha=0.3, color='blue')
    
    # 진폭과 범위 표시
    ax2.annotate('', xy=(1, 4), xytext=(1, 0),
                arrowprops=dict(arrowstyle='<->', color='red', lw=3))
    ax2.text(1.3, 2, 'Range\n(4 units)', fontsize=11, color='red', fontweight='bold')
    ax2.text(1.3, 0.5, 'Amplitude\n= Range/2\n= 2 units', fontsize=10, color='darkred')
    
    ax2.axhline(2, color='orange', linestyle='--', linewidth=2, label='Mean')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Stress', fontsize=11)
    ax2.set_title('② What is a Cycle?\nOne complete load-unload', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-1, 6)
    ax2.legend(loc='upper right', fontsize=9)
    
    # 1-3: 피로 손상 공식
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.axis('off')
    
    # 수식 박스
    textbox = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax3.text(0.5, 0.75, 'Fatigue Damage Formula\n(Basquin\'s Law)', 
            fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='blue'))
    
    ax3.text(0.5, 0.5, r'$D = \sum_{i} (Amplitude_i)^m \times Count_i$', 
            fontsize=16, ha='center', va='center', 
            bbox=textbox)
    
    ax3.text(0.5, 0.25, 'where m = 3 (typical for metals)\n\n'
            'Larger amplitude → Much more damage!\n'
            '(amplitude doubled → damage × 8)', 
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    ax3.set_title('③ How Damage is Calculated', fontsize=12, fontweight='bold')
    
    # ========== 2행: Rainflow 설명 ==========
    
    # 2-1: Rainflow 원리
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.axis('off')
    
    ax4.text(0.5, 0.9, 'RAINFLOW Method', fontsize=14, fontweight='bold', 
            ha='center', color='blue')
    ax4.text(0.5, 0.75, '(3-Point Algorithm)', fontsize=11, ha='center', color='gray')
    
    explanation_rf = """
Step 1: Find peaks and valleys (reversal points)

Step 2: Look at 3 consecutive points (A, B, C)

Step 3: If |B-A| ≤ |C-B|, then A-B forms a cycle
        → Record the cycle (range, mean, count)
        → Remove point B, continue

Step 4: Repeat until no more cycles found

Key: Counts both FULL cycles (count=1.0)
     and HALF cycles (count=0.5)
"""
    ax4.text(0.05, 0.55, explanation_rf, fontsize=10, ha='left', va='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax4.set_title('④ Rainflow: The Idea', fontsize=12, fontweight='bold')
    
    # 2-2: Rainflow 시각적 예시
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(t[:6], signal[:6], 'b-o', linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
    
    # 3점 규칙 설명
    # 점 A(1,4), B(2,1), C(3,3) 표시
    ax5.annotate('A', (1, 4), xytext=(0.7, 4.5), fontsize=12, fontweight='bold', color='red')
    ax5.annotate('B', (2, 1), xytext=(1.7, 0.5), fontsize=12, fontweight='bold', color='red')
    ax5.annotate('C', (3, 3), xytext=(3.2, 3.2), fontsize=12, fontweight='bold', color='red')
    
    # |B-A| vs |C-B| 비교
    ax5.annotate('', xy=(1.5, 4), xytext=(1.5, 1),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax5.text(1.7, 2.5, '|B-A|=3', fontsize=10, color='green', fontweight='bold')
    
    ax5.annotate('', xy=(2.5, 1), xytext=(2.5, 3),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax5.text(2.7, 2, '|C-B|=2', fontsize=10, color='purple', fontweight='bold')
    
    # 결과
    ax5.text(0.5, -1.5, '|B-A|=3 > |C-B|=2\n→ No cycle yet, continue', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    ax5.set_xlabel('Time', fontsize=11)
    ax5.set_ylabel('Stress', fontsize=11)
    ax5.set_title('⑤ Rainflow: 3-Point Rule', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)
    ax5.set_ylim(-2.5, 6)
    
    # 2-3: Rainflow 사이클 추출 결과
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(t, signal, 'b-', linewidth=1, alpha=0.5)
    ax6.plot(t, signal, 'bo', markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    # 감지된 사이클들 (예시)
    cycles_rf = [
        {'start': 1, 'end': 2, 'color': 'red', 'range': 3},
        {'start': 3, 'end': 4, 'color': 'green', 'range': 4},
        {'start': 5, 'end': 6, 'color': 'orange', 'range': 3},
    ]
    
    for i, cyc in enumerate(cycles_rf):
        s, e = cyc['start'], cyc['end']
        ax6.fill_betweenx([signal[s], signal[e]], t[s], t[e], alpha=0.3, color=cyc['color'])
        ax6.plot([t[s], t[e]], [signal[s], signal[e]], color=cyc['color'], linewidth=3)
        ax6.text((t[s]+t[e])/2, max(signal[s], signal[e])+0.5, 
                f'Cycle {i+1}\nRange={cyc["range"]}', fontsize=9, ha='center', color=cyc['color'])
    
    ax6.set_xlabel('Time', fontsize=11)
    ax6.set_ylabel('Stress', fontsize=11)
    ax6.set_title('⑥ Rainflow: Detected Cycles', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)
    ax6.set_ylim(-2, 7)
    
    # ========== 3행: Fatpack 설명 ==========
    
    # 3-1: Fatpack 원리
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.axis('off')
    
    ax7.text(0.5, 0.9, 'FATPACK Method', fontsize=14, fontweight='bold', 
            ha='center', color='red')
    ax7.text(0.5, 0.75, '(4-Point Algorithm)', fontsize=11, ha='center', color='gray')
    
    explanation_fp = """
Step 1: Find reversals (direction changes)
        → Keep only peaks and valleys

Step 2: Look at 4 consecutive reversals (A, B, C, D)

Step 3: If inner range ≤ outer range:
        |C-B| ≤ |D-A| and |C-B| ≤ |B-A|
        → B-C forms a complete cycle
        → Remove B and C, continue

Step 4: Repeat until no more cycles found

Key: Only counts FULL cycles (count=1.0)
     More conservative than Rainflow
"""
    ax7.text(0.05, 0.55, explanation_fp, fontsize=10, ha='left', va='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.3))
    
    ax7.set_title('⑦ Fatpack: The Idea', fontsize=12, fontweight='bold')
    
    # 3-2: Fatpack 시각적 예시
    ax8 = fig.add_subplot(3, 3, 8)
    # 4점 예시 신호
    t4 = np.array([0, 1, 2, 3])
    s4 = np.array([0, 5, 2, 4])
    ax8.plot(t4, s4, 'r-o', linewidth=2, markersize=12, markerfacecolor='white', markeredgewidth=2)
    
    # 점 라벨
    labels = ['A', 'B', 'C', 'D']
    for i, label in enumerate(labels):
        ax8.annotate(label, (t4[i], s4[i]), xytext=(t4[i]-0.15, s4[i]+0.5), 
                    fontsize=14, fontweight='bold', color='red')
    
    # 내부 범위 (B-C)
    ax8.annotate('', xy=(1.5, 5), xytext=(1.5, 2),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax8.text(1.7, 3.5, 'Inner\n|C-B|=3', fontsize=10, color='blue', fontweight='bold')
    
    # 외부 범위 (A-D)
    ax8.annotate('', xy=(0.3, 0), xytext=(0.3, 5),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2, linestyle='--'))
    ax8.text(-0.3, 2.5, 'Outer\n|D-A|=4', fontsize=10, color='green', fontweight='bold', ha='right')
    
    # 결과
    ax8.text(1.5, -1, '|C-B|=3 ≤ |D-A|=4\n→ B-C is a cycle!', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    ax8.set_xlabel('Reversal Index', fontsize=11)
    ax8.set_ylabel('Stress', fontsize=11)
    ax8.set_title('⑧ Fatpack: 4-Point Rule', fontsize=12, fontweight='bold')
    ax8.grid(alpha=0.3)
    ax8.set_ylim(-2, 7)
    
    # 3-3: 비교 요약
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    # 비교 테이블
    comparison_text = """
┌─────────────────────────────────────────────────────────────┐
│            RAINFLOW  vs  FATPACK  Comparison                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Feature          │ Rainflow      │ Fatpack                 │
│  ─────────────────┼───────────────┼───────────────────────  │
│  Algorithm        │ 3-point       │ 4-point                 │
│  Half cycles      │ ✓ Yes         │ ✗ No                    │
│  Cycle count      │ More cycles   │ Fewer cycles            │
│  Damage estimate  │ Higher        │ Lower                   │
│  Conservatism     │ Conservative  │ Less conservative       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  💡 Key Insight:                                            │
│     Rainflow counts partial cycles (0.5 count)              │
│     Fatpack only counts complete cycles (1.0 count)         │
│                                                             │
│  → Same signal can give DIFFERENT damage values!            │
│  → Choose based on your application requirements            │
└─────────────────────────────────────────────────────────────┘
"""
    ax9.text(0.5, 0.5, comparison_text, fontsize=10, ha='center', va='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    ax9.set_title('⑨ Summary: Key Differences', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 저장
    output_path = OUTPUT_DIR / "rainflow_vs_fatpack_explanation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 알고리즘 설명 시각화 저장: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("📚 Rainflow vs Fatpack Algorithm Explanation")
    print("=" * 60)
    
    create_explanation_figure()
    
    print("\n✅ 완료!")





