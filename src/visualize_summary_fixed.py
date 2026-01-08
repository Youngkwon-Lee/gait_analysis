"""
RESULTS_SUMMARY.md 기반 성능 비교 시각화
(HPC 결과 다운로드 전 사전 시각화)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_PATH = Path(__file__).parent.parent
VISUALIZATIONS_PATH = BASE_PATH / 'visualizations'
VISUALIZATIONS_PATH.mkdir(exist_ok=True)

# 우리 결과 (RESULTS_SUMMARY.md 기준)
OUR_RESULTS = {
    'PD_Screening': {
        'auc': 0.963,
        'balanced_acc': 0.790,
        'sensitivity': 0.595,
        'specificity': 0.985
    },
    'OA_Screening': {
        'auc': 0.908,
        'balanced_acc': 0.786,
        'sensitivity': 0.668,
        'specificity': 0.904
    },
    'CVA_Detection': {
        'auc': 0.986,
        'balanced_acc': 0.936,
        'sensitivity': 0.958,
        'specificity': 0.914
    },
    'PD_vs_CVA': {
        'auc': 0.934,
        'balanced_acc': 0.880,
        'sensitivity': 0.942,
        'specificity': 0.819
    }
}

# 베이스라인 논문 결과
BASELINE_RESULTS = {
    'PD_Screening': {'auc': 0.821, 'balanced_acc': 0.639},
    'OA_Screening': {'auc': 0.990, 'balanced_acc': 0.942},
    'CVA_Detection': {'auc': 0.950, 'balanced_acc': 0.747},
    'PD_vs_CVA': {'auc': 0.657, 'balanced_acc': 0.607}
}

# Task 이름
TASK_NAMES = {
    'PD_Screening': 'PD Screening\n(HS vs PD)',
    'OA_Screening': 'OA Screening\n(HS vs HOA)',
    'CVA_Detection': 'CVA Detection\n(HS vs CVA)',
    'PD_vs_CVA': 'PD vs CVA'
}


def plot_auc_comparison():
    """ROC-AUC 비교 막대그래프"""

    tasks = list(TASK_NAMES.values())
    our_auc = [OUR_RESULTS[k]['auc'] for k in OUR_RESULTS.keys()]
    baseline_auc = [BASELINE_RESULTS[k]['auc'] for k in BASELINE_RESULTS.keys()]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width/2, our_auc, width, label='우리 결과',
                   color='#FF6B6B', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, baseline_auc, width, label='논문 (Baseline)',
                   color='#4ECDC4', edgecolor='black', linewidth=1.2)

    # 개선도 표시 (화살표)
    for i, (our, baseline) in enumerate(zip(our_auc, baseline_auc)):
        diff = our - baseline
        if diff > 0:
            # 상승 화살표
            ax.annotate('', xy=(i, our), xytext=(i, baseline),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(i, max(our, baseline) + 0.02, f'+{diff:.1%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color='green')
        else:
            # 하락 화살표
            ax.annotate('', xy=(i, our), xytext=(i, baseline),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text(i, min(our, baseline) - 0.02, f'{diff:.1%}',
                   ha='center', va='top', fontsize=10, fontweight='bold', color='red')

    # 막대 위에 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Task', fontsize=13, fontweight='bold')
    ax.set_ylabel('ROC-AUC', fontsize=13, fontweight='bold')
    ax.set_title('ROC-AUC 성능 비교 (우리 vs 베이스라인 논문)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0.5, 1.05])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='Excellent (0.9)')

    plt.tight_layout()

    save_path = VISUALIZATIONS_PATH / 'auc_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] AUC comparison saved: {save_path}")
    plt.close()


def plot_balanced_acc_comparison():
    """Balanced Accuracy 비교 막대그래프"""

    tasks = list(TASK_NAMES.values())
    our_acc = [OUR_RESULTS[k]['balanced_acc'] for k in OUR_RESULTS.keys()]
    baseline_acc = [BASELINE_RESULTS[k]['balanced_acc'] for k in BASELINE_RESULTS.keys()]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width/2, our_acc, width, label='우리 결과',
                   color='#95E1D3', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, baseline_acc, width, label='논문 (Baseline)',
                   color='#F38181', edgecolor='black', linewidth=1.2)

    # 개선도 표시
    for i, (our, baseline) in enumerate(zip(our_acc, baseline_acc)):
        diff = our - baseline
        if diff > 0:
            ax.annotate('', xy=(i, our), xytext=(i, baseline),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(i, max(our, baseline) + 0.02, f'+{diff:.1%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color='green')
        else:
            ax.annotate('', xy=(i, our), xytext=(i, baseline),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text(i, min(our, baseline) - 0.02, f'{diff:.1%}',
                   ha='center', va='top', fontsize=10, fontweight='bold', color='red')

    # 막대 위에 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Task', fontsize=13, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Balanced Accuracy 성능 비교 (우리 vs 베이스라인 논문)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0.5, 1.05])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()

    save_path = VISUALIZATIONS_PATH / 'balanced_acc_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Balanced Accuracy comparison saved: {save_path}")
    plt.close()


def plot_sensitivity_specificity():
    """Sensitivity & Specificity 비교 (우리 결과만)"""

    tasks = list(TASK_NAMES.values())
    sensitivity = [OUR_RESULTS[k]['sensitivity'] for k in OUR_RESULTS.keys()]
    specificity = [OUR_RESULTS[k]['specificity'] for k in OUR_RESULTS.keys()]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width/2, sensitivity, width, label='Sensitivity (민감도)',
                   color='#FFD93D', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, specificity, width, label='Specificity (특이도)',
                   color='#6BCB77', edgecolor='black', linewidth=1.2)

    # 막대 위에 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Task', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Sensitivity & Specificity 분석 (우리 결과)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim([0.5, 1.05])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='Good (0.9)')

    plt.tight_layout()

    save_path = VISUALIZATIONS_PATH / 'sensitivity_specificity.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Sensitivity/Specificity saved: {save_path}")
    plt.close()


def plot_improvement_heatmap():
    """개선도 히트맵"""

    tasks = list(TASK_NAMES.values())

    # 개선도 계산
    improvements = []
    for task_key in OUR_RESULTS.keys():
        auc_diff = OUR_RESULTS[task_key]['auc'] - BASELINE_RESULTS[task_key]['auc']
        acc_diff = OUR_RESULTS[task_key]['balanced_acc'] - BASELINE_RESULTS[task_key]['balanced_acc']
        improvements.append([auc_diff * 100, acc_diff * 100])

    improvements = np.array(improvements)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 컬러맵 (빨강=나쁨, 초록=좋음)
    im = ax.imshow(improvements.T, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=50)

    # 축 설정
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_yticklabels(['ROC-AUC', 'Balanced Accuracy'], fontsize=12, fontweight='bold')

    # 값 표시
    for i in range(len(tasks)):
        for j in range(2):
            value = improvements[i, j]
            color = 'white' if abs(value) > 20 else 'black'
            text = ax.text(i, j, f'{value:+.1f}%',
                          ha="center", va="center", color=color,
                          fontsize=11, fontweight='bold')

    ax.set_title('베이스라인 대비 개선도 (%)', fontsize=15, fontweight='bold', pad=20)

    # 컬러바
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()

    save_path = VISUALIZATIONS_PATH / 'improvement_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Improvement heatmap saved: {save_path}")
    plt.close()


def create_summary_table():
    """성능 요약 테이블 (마크다운)"""

    lines = []
    lines.append("# 성능 비교 요약표")
    lines.append("")
    lines.append("## 전체 비교")
    lines.append("")
    lines.append("| Task | 지표 | 우리 결과 | 논문 | 개선도 | 등급 |")
    lines.append("|------|------|-----------|------|--------|------|")

    for task_key, task_name in TASK_NAMES.items():
        our = OUR_RESULTS[task_key]
        baseline = BASELINE_RESULTS[task_key]

        # AUC
        auc_diff = our['auc'] - baseline['auc']
        auc_pct = (auc_diff / baseline['auc']) * 100
        auc_grade = "🔥🔥🔥" if auc_pct > 20 else "🔥🔥" if auc_pct > 10 else "✅" if auc_pct > 0 else "⚠️"

        lines.append(f"| {task_name.replace(chr(10), ' ')} | **AUC** | **{our['auc']:.3f}** | {baseline['auc']:.3f} | **{auc_diff:+.3f}** ({auc_pct:+.1f}%) | {auc_grade} |")

        # Balanced Accuracy
        acc_diff = our['balanced_acc'] - baseline['balanced_acc']
        acc_pct = (acc_diff / baseline['balanced_acc']) * 100
        acc_grade = "🔥🔥🔥" if acc_pct > 20 else "🔥🔥" if acc_pct > 10 else "✅" if acc_pct > 0 else "⚠️"

        lines.append(f"| {task_name.replace(chr(10), ' ')} | **Bal.Acc** | **{our['balanced_acc']:.3f}** | {baseline['balanced_acc']:.3f} | **{acc_diff:+.3f}** ({acc_pct:+.1f}%) | {acc_grade} |")

        # Sensitivity & Specificity (우리만)
        lines.append(f"| {task_name.replace(chr(10), ' ')} | Sensitivity | {our['sensitivity']:.3f} | - | - | - |")
        lines.append(f"| {task_name.replace(chr(10), ' ')} | Specificity | {our['specificity']:.3f} | - | - | - |")

    lines.append("")
    lines.append("## 주요 성과")
    lines.append("")
    lines.append("### 🏆 최대 개선")
    lines.append("- **PD vs CVA**: AUC +0.277 (+42.2%) - 가장 큰 개선")
    lines.append("- **CVA Detection**: Bal.Acc +0.189 (+25.3%)")
    lines.append("- **PD Screening**: AUC +0.142 (+17.3%)")
    lines.append("")
    lines.append("### ⚠️ 성능 저하")
    lines.append("- **OA Screening**: AUC -0.082 (-8.3%)")
    lines.append("  - 원인: 샘플 불균형 (HOA 74개 vs HS 360개)")
    lines.append("  - 개선 방향: Data Augmentation, Class Weighting 조정")
    lines.append("")
    lines.append("## 임상적 의의")
    lines.append("")
    lines.append("### CVA Detection (뇌졸중 검출)")
    lines.append("- **Sensitivity 95.8%**: 뇌졸중 환자 대부분 검출")
    lines.append("- **Specificity 91.4%**: 건강인 오진율 낮음")
    lines.append("- **임상 활용**: 조기 스크리닝 도구로 활용 가능")
    lines.append("")
    lines.append("### PD Screening (파킨슨병 스크리닝)")
    lines.append("- **Specificity 98.5%**: 건강인을 PD로 오진하는 경우 극히 드묾")
    lines.append("- **Sensitivity 59.5%**: 일부 PD 환자 미검출 - 개선 필요")
    lines.append("- **임상 활용**: False Positive 최소화, 2차 검사 의뢰 기준")
    lines.append("")
    lines.append("### PD vs CVA (감별 진단)")
    lines.append("- **획기적 개선**: 논문 0.657 → 우리 0.934")
    lines.append("- **Sensitivity 94.2%**: PD 환자 정확 분류")
    lines.append("- **Specificity 81.9%**: CVA 환자 정확 분류")
    lines.append("- **임상 활용**: 웨어러블 센서 기반 신경질환 감별 가능성 입증")

    save_path = VISUALIZATIONS_PATH / 'PERFORMANCE_SUMMARY.md'
    save_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"[OK] Performance summary table saved: {save_path}")


def main():
    print("=" * 60)
    print("Performance Comparison Visualization")
    print("=" * 60)
    print("")

    print("1. ROC-AUC comparison...")
    plot_auc_comparison()

    print("2. Balanced Accuracy comparison...")
    plot_balanced_acc_comparison()

    print("3. Sensitivity/Specificity...")
    plot_sensitivity_specificity()

    print("4. Improvement heatmap...")
    plot_improvement_heatmap()

    print("5. Performance summary table...")
    create_summary_table()

    print("")
    print("=" * 60)
    print(f"All visualizations completed!")
    print(f"Saved to: {VISUALIZATIONS_PATH}")
    print("=" * 60)
    print("")
    print("Generated files:")
    print("  - auc_comparison.png")
    print("  - balanced_acc_comparison.png")
    print("  - sensitivity_specificity.png")
    print("  - improvement_heatmap.png")
    print("  - PERFORMANCE_SUMMARY.md")


if __name__ == '__main__':
    main()
