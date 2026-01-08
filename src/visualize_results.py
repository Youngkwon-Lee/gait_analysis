"""
HPC 훈련 결과 시각화 스크립트

ROC Curve, Confusion Matrix, 성능 비교표 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_PATH = Path(__file__).parent.parent
RESULTS_PATH = BASE_PATH / 'results'
VISUALIZATIONS_PATH = BASE_PATH / 'visualizations'
VISUALIZATIONS_PATH.mkdir(exist_ok=True)

# 베이스라인 논문 결과 (arXiv:2503.05708)
BASELINE_RESULTS = {
    'PD_Screening': {'auc': 0.821, 'balanced_acc': 0.639},
    'OA_Screening': {'auc': 0.990, 'balanced_acc': 0.942},
    'CVA_Detection': {'auc': 0.950, 'balanced_acc': 0.747},
    'PD_vs_CVA': {'auc': 0.657, 'balanced_acc': 0.607}
}

# Task 이름 매핑
TASK_NAMES = {
    'PD_Screening': 'PD Screening (HS vs PD)',
    'OA_Screening': 'OA Screening (HS vs HOA)',
    'CVA_Detection': 'CVA Detection (HS vs CVA)',
    'PD_vs_CVA': 'PD vs CVA'
}


def load_results():
    """HPC 훈련 결과 CSV 파일 로드"""
    results = {}

    csv_files = sorted(RESULTS_PATH.glob('dl_baseline_results_*.csv'))

    if not csv_files:
        print("❌ 결과 CSV 파일을 찾을 수 없습니다.")
        print(f"경로: {RESULTS_PATH}")
        print("\nHPC에서 결과 파일을 다운로드하세요:")
        print("bash scripts/download_hpc_results.sh")
        return None

    print(f"✅ {len(csv_files)}개 결과 파일 발견")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Task 이름 추출
        task_name = df['task'].iloc[0]
        results[task_name] = df

        print(f"  - {task_name}: {csv_file.name}")

    return results


def plot_roc_curves(results):
    """ROC Curve 시각화 (4개 Task 한 그래프에)"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (task_name, df) in enumerate(results.items()):
        ax = axes[idx]

        # ROC 데이터가 있는지 확인
        if 'fpr' in df.columns and 'tpr' in df.columns:
            # JSON 형식으로 저장된 리스트 파싱
            fpr = json.loads(df['fpr'].iloc[0])
            tpr = json.loads(df['tpr'].iloc[0])
            roc_auc = df['roc_auc'].iloc[0]

            # ROC Curve 그리기
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'우리 모델 (AUC = {roc_auc:.3f})')

            # 베이스라인 비교선 (AUC만 표시)
            baseline_auc = BASELINE_RESULTS[task_name]['auc']
            ax.axhline(y=baseline_auc, color='blue', linestyle='--', lw=1.5,
                      label=f'논문 (AUC = {baseline_auc:.3f})')
        else:
            # ROC 데이터 없으면 빈 그래프
            ax.text(0.5, 0.5, 'ROC 데이터 없음', ha='center', va='center')

        # 대각선 (랜덤 분류기)
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)

        # 그래프 설정
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{TASK_NAMES[task_name]}', fontsize=13, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    save_path = VISUALIZATIONS_PATH / 'roc_curves_all_tasks.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ ROC Curves 저장: {save_path}")

    plt.close()


def plot_confusion_matrices(results):
    """Confusion Matrix 시각화 (4개 Task)"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (task_name, df) in enumerate(results.items()):
        ax = axes[idx]

        # Confusion Matrix 데이터 확인
        if 'tn' in df.columns and 'fp' in df.columns:
            tn = df['tn'].iloc[0]
            fp = df['fp'].iloc[0]
            fn = df['fn'].iloc[0]
            tp = df['tp'].iloc[0]

            cm = np.array([[tn, fp], [fn, tp]])

            # Heatmap 그리기
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'},
                       xticklabels=['Class 0', 'Class 1'],
                       yticklabels=['Class 0', 'Class 1'])

            # 정확도 계산
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # 제목에 성능 지표 추가
            title = f'{TASK_NAMES[task_name]}\n'
            title += f'Acc: {accuracy:.3f} | Sens: {sensitivity:.3f} | Spec: {specificity:.3f}'

        else:
            # 데이터 없으면 빈 그래프
            ax.text(0.5, 0.5, 'Confusion Matrix 데이터 없음',
                   ha='center', va='center')
            title = TASK_NAMES[task_name]

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()

    # 저장
    save_path = VISUALIZATIONS_PATH / 'confusion_matrices_all_tasks.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion Matrices 저장: {save_path}")

    plt.close()


def plot_performance_comparison(results):
    """베이스라인 대비 성능 비교 막대그래프"""

    # 데이터 준비
    tasks = []
    our_auc = []
    baseline_auc = []
    our_acc = []
    baseline_acc = []

    for task_name, df in results.items():
        tasks.append(TASK_NAMES[task_name])
        our_auc.append(df['roc_auc'].iloc[0])
        baseline_auc.append(BASELINE_RESULTS[task_name]['auc'])
        our_acc.append(df['balanced_accuracy'].iloc[0])
        baseline_acc.append(BASELINE_RESULTS[task_name]['balanced_acc'])

    # 2개 서브플롯 (AUC, Balanced Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(tasks))
    width = 0.35

    # AUC 비교
    bars1 = ax1.bar(x - width/2, our_auc, width, label='우리 결과', color='#FF6B6B')
    bars2 = ax1.bar(x + width/2, baseline_auc, width, label='논문 (Baseline)', color='#4ECDC4')

    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    ax1.set_title('ROC-AUC 비교', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=15, ha='right')
    ax1.legend(fontsize=11)
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')

    # 막대 위에 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    # Balanced Accuracy 비교
    bars3 = ax2.bar(x - width/2, our_acc, width, label='우리 결과', color='#FF6B6B')
    bars4 = ax2.bar(x + width/2, baseline_acc, width, label='논문 (Baseline)', color='#4ECDC4')

    ax2.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Balanced Accuracy 비교', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=15, ha='right')
    ax2.legend(fontsize=11)
    ax2.set_ylim([0.5, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')

    # 막대 위에 값 표시
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 저장
    save_path = VISUALIZATIONS_PATH / 'performance_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 성능 비교 그래프 저장: {save_path}")

    plt.close()


def create_performance_table(results):
    """성능 지표 상세 비교표 생성 (마크다운)"""

    table = []
    table.append("# 성능 지표 상세 비교")
    table.append("")
    table.append("| Task | Metric | 우리 결과 | 논문 (Baseline) | 개선도 | 등급 |")
    table.append("|------|--------|-----------|-----------------|--------|------|")

    for task_name, df in results.items():
        our_auc = df['roc_auc'].iloc[0]
        our_acc = df['balanced_accuracy'].iloc[0]
        baseline_auc = BASELINE_RESULTS[task_name]['auc']
        baseline_acc = BASELINE_RESULTS[task_name]['balanced_acc']

        auc_diff = our_auc - baseline_auc
        acc_diff = our_acc - baseline_acc

        # 개선도에 따른 등급
        auc_grade = "🔥🔥🔥" if auc_diff > 0.2 else "🔥🔥" if auc_diff > 0.1 else "✅" if auc_diff > 0 else "⚠️"
        acc_grade = "🔥🔥🔥" if acc_diff > 0.2 else "🔥🔥" if acc_diff > 0.1 else "✅" if acc_diff > 0 else "⚠️"

        # AUC 행
        table.append(f"| {TASK_NAMES[task_name]} | **ROC-AUC** | **{our_auc:.3f}** | {baseline_auc:.3f} | **{auc_diff:+.1%}** | {auc_grade} |")

        # Balanced Accuracy 행
        table.append(f"| {TASK_NAMES[task_name]} | **Balanced Acc** | **{our_acc:.3f}** | {baseline_acc:.3f} | **{acc_diff:+.1%}** | {acc_grade} |")

        # Sensitivity/Specificity 행 (우리 결과만)
        if 'sensitivity' in df.columns:
            sensitivity = df['sensitivity'].iloc[0]
            specificity = df['specificity'].iloc[0]
            table.append(f"| {TASK_NAMES[task_name]} | Sensitivity | {sensitivity:.3f} | - | - | - |")
            table.append(f"| {TASK_NAMES[task_name]} | Specificity | {specificity:.3f} | - | - | - |")

    table.append("")
    table.append("## 개선도 등급")
    table.append("- 🔥🔥🔥: >20% 개선")
    table.append("- 🔥🔥: 10-20% 개선")
    table.append("- ✅: 0-10% 개선")
    table.append("- ⚠️: 논문보다 낮음")

    # 저장
    save_path = VISUALIZATIONS_PATH / 'PERFORMANCE_COMPARISON_TABLE.md'
    save_path.write_text('\n'.join(table), encoding='utf-8')
    print(f"✅ 성능 비교표 저장: {save_path}")

    # 터미널 출력
    print("\n" + '\n'.join(table))


def main():
    print("=" * 60)
    print("HPC 훈련 결과 시각화 스크립트")
    print("=" * 60)

    # 결과 파일 로드
    results = load_results()

    if results is None:
        return

    print("\n" + "=" * 60)
    print("시각화 생성 중...")
    print("=" * 60)

    # 1. ROC Curves
    print("\n1. ROC Curves 생성 중...")
    plot_roc_curves(results)

    # 2. Confusion Matrices
    print("\n2. Confusion Matrices 생성 중...")
    plot_confusion_matrices(results)

    # 3. Performance Comparison
    print("\n3. 성능 비교 그래프 생성 중...")
    plot_performance_comparison(results)

    # 4. Performance Table
    print("\n4. 성능 비교표 생성 중...")
    create_performance_table(results)

    print("\n" + "=" * 60)
    print(f"✅ 모든 시각화 완료!")
    print(f"저장 경로: {VISUALIZATIONS_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
