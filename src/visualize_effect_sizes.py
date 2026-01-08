"""
Effect Size 심층 분석 및 시각화
Cohen's d effect size 해석 및 임상적 의미
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Paths
RESULTS_DIR = Path("D:/gait_wearable_sensor/results/data_statistics")
OUTPUT_DIR = Path("D:/gait_wearable_sensor/results/effect_size_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

TASKS = ['OA_Screening', 'PD_Screening', 'CVA_Detection']
CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

def interpret_effect_size(d):
    """Cohen's d 해석"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible (무시할만함)", "#e8e8e8"
    elif abs_d < 0.5:
        return "Small (작음)", "#a8d5ba"
    elif abs_d < 0.8:
        return "Medium (중간)", "#ffd966"
    elif abs_d < 1.2:
        return "Large (큼)", "#ff9966"
    else:
        return "Very Large (매우 큼)", "#ff6666"


def load_all_effect_sizes():
    """모든 태스크의 Effect Size 로드"""
    all_data = {}

    for task in TASKS:
        json_file = RESULTS_DIR / f'{task}_data_statistics.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            all_data[task] = data['channel_importance']

    return all_data


def create_effect_size_heatmap(all_data):
    """Effect Size Heatmap 생성"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 데이터 준비
    effect_matrix = []
    for task in TASKS:
        row = [all_data[task][ch]['effect_size'] for ch in CHANNELS]
        effect_matrix.append(row)

    effect_matrix = np.array(effect_matrix)

    # Heatmap
    sns.heatmap(effect_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
               center=0, vmin=-1.5, vmax=1.5,
               xticklabels=CHANNELS,
               yticklabels=[t.replace('_', ' ') for t in TASKS],
               cbar_kws={'label': 'Effect Size (Cohen\'s d)'},
               linewidths=0.5, linecolor='gray')

    ax.set_title('Effect Size Heatmap - 모든 태스크', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task', fontsize=12, fontweight='bold')

    # 해석 가이드 추가
    interpretation = (
        "해석 가이드:\n"
        "  |d| < 0.2: Negligible (무시할만함)\n"
        "  0.2 ≤ |d| < 0.5: Small (작음)\n"
        "  0.5 ≤ |d| < 0.8: Medium (중간)\n"
        "  0.8 ≤ |d| < 1.2: Large (큼)\n"
        "  |d| ≥ 1.2: Very Large (매우 큼)"
    )
    fig.text(0.02, 0.02, interpretation, fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'effect_size_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Heatmap saved: {OUTPUT_DIR / 'effect_size_heatmap.png'}")
    plt.close()


def create_effect_size_ranking(all_data):
    """Effect Size 순위 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Channel Importance Ranking by Effect Size', fontsize=16, fontweight='bold')

    for idx, task in enumerate(TASKS):
        ax = axes[idx]

        # 데이터 준비
        channels = []
        effect_sizes = []
        colors = []
        interpretations = []

        for ch in CHANNELS:
            effect_size = all_data[task][ch]['effect_size']
            interpretation, color = interpret_effect_size(effect_size)

            channels.append(ch)
            effect_sizes.append(abs(effect_size))
            colors.append(color)
            interpretations.append(interpretation.split(' ')[0])

        # 정렬
        sorted_indices = np.argsort(effect_sizes)[::-1]
        channels = [channels[i] for i in sorted_indices]
        effect_sizes = [effect_sizes[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        interpretations = [interpretations[i] for i in sorted_indices]

        # Bar plot
        bars = ax.barh(channels, effect_sizes, color=colors, alpha=0.8, edgecolor='black')

        # 값 표시
        for i, (bar, es, interp) in enumerate(zip(bars, effect_sizes, interpretations)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {es:.3f} ({interp})',
                   ha='left', va='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('|Effect Size|', fontweight='bold')
        ax.set_title(task.replace('_', ' '), fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim([0, max(effect_sizes) * 1.3])

        # 기준선
        ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=1.2, color='darkred', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'effect_size_ranking.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Ranking saved: {OUTPUT_DIR / 'effect_size_ranking.png'}")
    plt.close()


def create_clinical_interpretation(all_data):
    """임상적 해석 차트"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Clinical Interpretation of Effect Sizes', fontsize=16, fontweight='bold')

    # 1. Accelerometer vs Gyroscope 비교
    ax = axes[0, 0]
    acc_gyro_comparison = []

    for task in TASKS:
        acc_effects = [abs(all_data[task][f'Acc_{axis}']['effect_size']) for axis in ['X', 'Y', 'Z']]
        gyr_effects = [abs(all_data[task][f'Gyr_{axis}']['effect_size']) for axis in ['X', 'Y', 'Z']]

        acc_gyro_comparison.append({
            'task': task.replace('_', ' '),
            'Accelerometer': np.mean(acc_effects),
            'Gyroscope': np.mean(gyr_effects)
        })

    df_acc_gyr = pd.DataFrame(acc_gyro_comparison)
    x = np.arange(len(TASKS))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_acc_gyr['Accelerometer'], width, label='Accelerometer (가속도)', color='#66b3ff', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_acc_gyr['Gyroscope'], width, label='Gyroscope (각속도)', color='#ff9999', alpha=0.8)

    ax.set_ylabel('Mean |Effect Size|', fontweight='bold')
    ax.set_title('Accelerometer vs Gyroscope Importance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', ' ') for t in TASKS])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. X, Y, Z축 비교
    ax = axes[0, 1]
    axis_comparison = {task: {'X': [], 'Y': [], 'Z': []} for task in TASKS}

    for task in TASKS:
        for sensor_type in ['Acc', 'Gyr']:
            for axis in ['X', 'Y', 'Z']:
                effect = abs(all_data[task][f'{sensor_type}_{axis}']['effect_size'])
                axis_comparison[task][axis].append(effect)

    # 평균
    axis_means = {task: {axis: np.mean(vals) for axis, vals in axes_data.items()}
                  for task, axes_data in axis_comparison.items()}

    x = np.arange(len(TASKS))
    width = 0.25

    for i, axis in enumerate(['X', 'Y', 'Z']):
        values = [axis_means[task][axis] for task in TASKS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=f'{axis}-axis', alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Mean |Effect Size|', fontweight='bold')
    ax.set_title('Importance by Axis (X, Y, Z)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', ' ') for t in TASKS])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. 질환별 가장 중요한 채널
    ax = axes[1, 0]

    top_channels = {}
    for task in TASKS:
        effects = [(ch, abs(all_data[task][ch]['effect_size'])) for ch in CHANNELS]
        effects.sort(key=lambda x: x[1], reverse=True)
        top_channels[task] = effects[:3]

    y_pos = np.arange(len(TASKS))

    for i, task in enumerate(TASKS):
        text = f"{task.replace('_', ' ')}:\n"
        for rank, (ch, es) in enumerate(top_channels[task], 1):
            interp, _ = interpret_effect_size(es)
            text += f"  {rank}. {ch}: {es:.3f} ({interp.split()[0]})\n"

        ax.text(0.05, i, text.strip(), va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax.set_ylim([-0.5, len(TASKS) - 0.5])
    ax.set_xlim([0, 1])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Top 3 Most Important Channels per Task', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 4. 임상적 의미 요약
    ax = axes[1, 1]

    clinical_text = """
    임상적 해석 요약:

    OA (골관절염):
    • Gyr_X (각속도 X축): -0.536 (Large)
    • 해석: 발목/무릎 관절의 회전 제한
    • 증상: 보행 시 발 회전 감소, 보폭 감소
    • 모니터링: 각속도 센서 중심

    PD (파킨슨병):
    • Acc_Y (가속도 Y축): -1.420 (Very Large)
    • 해석: 수직 운동 감소 (Shuffling gait)
    • 증상: 발을 들지 못함, 질질 끄는 걸음
    • 모니터링: 수직 가속도 + Freezing 감지

    CVA (뇌졸중):
    • Acc_Y (가속도 Y축): -1.236 (Very Large)
    • 해석: 편측마비로 인한 수직 운동 저하
    • 증상: 한쪽 다리 힘 약화, Drop foot
    • 모니터링: 수직 가속도 + 좌우 비대칭

    핵심 인사이트:
    • OA: 회전 패턴 (Gyroscope)
    • PD/CVA: 수직 운동 (Acc_Y)
    • 모두: Y축이 가장 중요 (보행의 수직 성분)
    """

    ax.text(0.05, 0.95, clinical_text, va='top', fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clinical_interpretation.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Clinical interpretation saved: {OUTPUT_DIR / 'clinical_interpretation.png'}")
    plt.close()


def create_effect_size_summary_table(all_data):
    """Effect Size 요약 테이블 생성"""
    summary_data = []

    for task in TASKS:
        for ch in CHANNELS:
            effect_size = all_data[task][ch]['effect_size']
            interpretation, _ = interpret_effect_size(effect_size)

            summary_data.append({
                'Task': task.replace('_', ' '),
                'Channel': ch,
                'Effect_Size': effect_size,
                '|Effect_Size|': abs(effect_size),
                'Interpretation': interpretation,
                'T_statistic': all_data[task][ch]['t_stat'],
                'p_value': all_data[task][ch]['p_value'],
                'Significant': 'Yes' if all_data[task][ch]['p_value'] < 0.05 else 'No'
            })

    df = pd.DataFrame(summary_data)

    # 정렬: 태스크별로 Effect Size 내림차순
    df = df.sort_values(['Task', '|Effect_Size|'], ascending=[True, False])

    # CSV 저장
    csv_file = OUTPUT_DIR / 'effect_size_summary.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"[OK] Summary table saved: {csv_file}")

    # 콘솔 출력
    print("\n" + "="*100)
    print("Effect Size Summary Table")
    print("="*100)

    for task in TASKS:
        print(f"\n{task.replace('_', ' ')}:")
        task_df = df[df['Task'] == task.replace('_', ' ')][
            ['Channel', 'Effect_Size', 'Interpretation', 'Significant']
        ]
        print(task_df.to_string(index=False))

    print("\n" + "="*100)

    return df


def main():
    print("="*80)
    print("Effect Size 심층 분석")
    print("="*80)

    # 데이터 로드
    all_data = load_all_effect_sizes()

    # 1. Effect Size Heatmap
    print("\n[1/4] Creating Effect Size Heatmap...")
    create_effect_size_heatmap(all_data)

    # 2. Effect Size Ranking
    print("\n[2/4] Creating Effect Size Ranking...")
    create_effect_size_ranking(all_data)

    # 3. Clinical Interpretation
    print("\n[3/4] Creating Clinical Interpretation...")
    create_clinical_interpretation(all_data)

    # 4. Summary Table
    print("\n[4/4] Creating Summary Table...")
    create_effect_size_summary_table(all_data)

    print("\n" + "="*80)
    print("[DONE] Effect Size analysis complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
