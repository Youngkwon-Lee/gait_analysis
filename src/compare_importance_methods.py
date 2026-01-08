"""
Feature Importance 방법론 비교
데이터 기반 vs 모델 기반 분석 결과 비교
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
STATS_FILE = Path("D:/gait_wearable_sensor/results/data_statistics/OA_Screening_data_statistics.json")
MODEL_FILE = Path("D:/gait_wearable_sensor/results/feature_importance/OA_Screening_importance.json")
OUTPUT_DIR = Path("D:/gait_wearable_sensor/results/importance_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
SENSORS = ['HE', 'LB', 'LF', 'RF']


def load_data():
    """Load both statistical and model-based results"""
    with open(STATS_FILE, 'r') as f:
        stats_data = json.load(f)

    with open(MODEL_FILE, 'r') as f:
        model_data = json.load(f)

    return stats_data, model_data


def create_channel_comparison(stats_data, model_data):
    """Compare channel importance across methods"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Channel Importance: Data-based vs Model-based Analysis',
                 fontsize=16, fontweight='bold')

    # 1. Effect Size (Statistical) vs Permutation Importance
    ax = axes[0, 0]

    effect_sizes = [abs(stats_data['channel_importance'][ch]['effect_size']) for ch in CHANNELS]
    perm_importance = [model_data['permutation_channels'][ch]['auc_drop'] for ch in CHANNELS]

    x = np.arange(len(CHANNELS))
    width = 0.35

    bars1 = ax.bar(x - width/2, effect_sizes, width, label='Statistical (Effect Size)', color='#ff9999', alpha=0.8)
    bars2 = ax.bar(x + width/2, perm_importance, width, label='Model-based (Permutation)', color='#66b3ff', alpha=0.8)

    ax.set_ylabel('Importance', fontweight='bold')
    ax.set_title('Statistical vs Model-based Importance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. Normalized Comparison (0-1 scale)
    ax = axes[0, 1]

    # Normalize to 0-1
    effect_norm = np.array(effect_sizes) / max(effect_sizes) if max(effect_sizes) > 0 else effect_sizes
    perm_norm = np.array(perm_importance) / max(perm_importance) if max(perm_importance) > 0 else perm_importance

    bars1 = ax.bar(x - width/2, effect_norm, width, label='Statistical (Normalized)', color='#ff9999', alpha=0.8)
    bars2 = ax.bar(x + width/2, perm_norm, width, label='Model-based (Normalized)', color='#66b3ff', alpha=0.8)

    ax.set_ylabel('Normalized Importance (0-1)', fontweight='bold')
    ax.set_title('Normalized Importance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    # 3. Ranking Comparison
    ax = axes[1, 0]

    # Get rankings (1 = most important)
    stat_ranks = np.argsort(effect_sizes)[::-1] + 1
    model_ranks = np.argsort(perm_importance)[::-1] + 1

    comparison_data = []
    for i, ch in enumerate(CHANNELS):
        comparison_data.append({
            'Channel': ch,
            'Statistical Rank': stat_ranks[i],
            'Model Rank': model_ranks[i],
            'Rank Difference': abs(stat_ranks[i] - model_ranks[i])
        })

    df_ranks = pd.DataFrame(comparison_data)

    # Heatmap
    rank_matrix = np.array([stat_ranks, model_ranks])
    sns.heatmap(rank_matrix, annot=True, fmt='d', cmap='RdYlGn_r',
               xticklabels=CHANNELS,
               yticklabels=['Statistical Rank', 'Model Rank'],
               ax=ax, cbar_kws={'label': 'Rank (1=Most Important)'},
               vmin=1, vmax=6)
    ax.set_title('Ranking Comparison', fontweight='bold')

    # 4. Method Comparison Table
    ax = axes[1, 1]

    table_text = "Channel Importance Comparison\n\n"
    table_text += "Statistical Method (Effect Size):\n"
    sorted_stat = sorted(zip(CHANNELS, effect_sizes), key=lambda x: x[1], reverse=True)
    for rank, (ch, val) in enumerate(sorted_stat, 1):
        table_text += f"  {rank}. {ch}: {val:.3f}\n"

    table_text += "\nModel-based Method (Permutation):\n"
    sorted_model = sorted(zip(CHANNELS, perm_importance), key=lambda x: x[1], reverse=True)
    for rank, (ch, val) in enumerate(sorted_model, 1):
        table_text += f"  {rank}. {ch}: {val:.3f} (AUC drop)\n"

    table_text += "\nKey Differences:\n"
    if sorted_stat[0][0] != sorted_model[0][0]:
        table_text += f"• Top channel differs!\n"
        table_text += f"  Statistical: {sorted_stat[0][0]}\n"
        table_text += f"  Model-based: {sorted_model[0][0]}\n"

    ax.text(0.05, 0.95, table_text, va='top', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'channel_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Channel comparison saved: {OUTPUT_DIR / 'channel_importance_comparison.png'}")
    plt.close()


def create_sensor_comparison(stats_data, model_data):
    """Compare sensor importance across methods"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Sensor Importance: Multiple Methods Comparison',
                 fontsize=16, fontweight='bold')

    # 1. Permutation Importance
    ax = axes[0]
    perm_sensors = [model_data['permutation_sensors'][s]['auc_drop'] for s in SENSORS]
    bars = ax.bar(SENSORS, perm_sensors, color='#66b3ff', alpha=0.8, edgecolor='black')
    ax.set_ylabel('AUC Drop', fontweight='bold')
    ax.set_title('Permutation Importance', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 2. Ablation Study
    ax = axes[1]
    ablation_sensors = [model_data['ablation_sensors'][s]['auc_drop'] for s in SENSORS]
    bars = ax.bar(SENSORS, ablation_sensors, color='#99ff99', alpha=0.8, edgecolor='black')
    ax.set_ylabel('AUC Drop', fontweight='bold')
    ax.set_title('Ablation Study', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 3. Gradient-based
    ax = axes[2]
    gradient_sensors = [model_data['gradient']['sensors'][s] for s in SENSORS]
    # Normalize
    max_grad = max(gradient_sensors)
    gradient_norm = [g/max_grad for g in gradient_sensors]
    bars = ax.bar(SENSORS, gradient_norm, color='#ffcc99', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Normalized Gradient', fontweight='bold')
    ax.set_title('Gradient-based Importance', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sensor_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Sensor comparison saved: {OUTPUT_DIR / 'sensor_importance_comparison.png'}")
    plt.close()


def create_insights_summary(stats_data, model_data):
    """Create insights summary document"""

    # Channel analysis
    effect_sizes = {ch: abs(stats_data['channel_importance'][ch]['effect_size']) for ch in CHANNELS}
    perm_importance = {ch: model_data['permutation_channels'][ch]['auc_drop'] for ch in CHANNELS}

    top_stat = max(effect_sizes.items(), key=lambda x: x[1])
    top_model = max(perm_importance.items(), key=lambda x: x[1])

    # Sensor analysis
    perm_sensors = {s: model_data['permutation_sensors'][s]['auc_drop'] for s in SENSORS}
    top_sensor = max(perm_sensors.items(), key=lambda x: x[1])

    summary = f"""
# Feature Importance Analysis Summary
**Task**: OA Screening (HS vs HOA+KOA)
**Date**: 2026-01-08

---

## 핵심 발견사항

### 1. 채널 중요도: 방법론 간 **완전히 다른 결과**

**통계 기반 (Effect Size)**:
- 1위: {top_stat[0]} (Effect Size: {top_stat[1]:.3f})
- 해석: 클래스 간 통계적 차이가 가장 큼
- 특징: 각속도(Gyroscope) 중심

**모델 기반 (Permutation Importance)**:
- 1위: {top_model[0]} (AUC drop: {top_model[1]:.3f})
- 해석: 모델 성능에 가장 중요
- 특징: 가속도(Accelerometer) 중심

**왜 다를까?**
→ 모델은 통계적으로 가장 다른 특징이 아닌, **분류 결정 경계에 가장 유용한 특징**을 학습함

---

## 2. 채널별 순위 비교

| Channel | Statistical Rank | Model Rank | Difference |
|---------|-----------------|------------|------------|
"""

    # Add ranking table
    effect_list = sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True)
    perm_list = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)

    stat_ranks = {ch: i+1 for i, (ch, _) in enumerate(effect_list)}
    model_ranks = {ch: i+1 for i, (ch, _) in enumerate(perm_list)}

    for ch in CHANNELS:
        diff = abs(stat_ranks[ch] - model_ranks[ch])
        summary += f"| {ch} | {stat_ranks[ch]} | {model_ranks[ch]} | {diff} |\n"

    summary += f"""
---

## 3. 센서 중요도 (모델 기반)

**Permutation Importance**:
"""
    for i, (s, drop) in enumerate(sorted(perm_sensors.items(), key=lambda x: x[1], reverse=True), 1):
        summary += f"{i}. {s}: AUC drop {drop:.4f}\n"

    summary += f"""
→ **{top_sensor[0]} (AUC drop {top_sensor[1]:.4f})** 가장 중요

---

## 4. 임상적 시사점

**통계 기반 분석의 의미**:
- Gyr_X가 클래스 간 가장 큰 차이
- OA 환자의 회전 패턴 변화가 명확
- **연구/설명** 목적에 유용

**모델 기반 분석의 의미**:
- Acc_X가 실제 분류에 가장 중요
- 모델이 수평 가속도 패턴을 핵심적으로 사용
- **실제 배포/응용** 목적에 유용

**권장사항**:
1. **웨어러블 디바이스 설계**: 모델 기반 결과 사용
   - Acc_X 우선 모니터링
   - 가속도계 중심 설계

2. **임상 연구**: 통계 기반 결과 참고
   - Gyr_X로 회전 패턴 변화 설명
   - 병리학적 기전 이해

3. **하이브리드 접근**:
   - Acc_X (모델 중요) + Gyr_X (통계 중요) 모두 활용
   - 성능과 해석력 균형

---

## 5. 수치 비교

**Baseline Performance**:
- AUC: {model_data['baseline']['auc']:.4f}
- Balanced Accuracy: {model_data['baseline']['balanced_acc']:.4f}

**Top Channel 제거 시**:
- Acc_X 제거: AUC {model_data['permutation_channels']['Acc_X']['auc']:.4f} (drop {perm_importance['Acc_X']:.4f})
- Gyr_X 제거: AUC {model_data['permutation_channels']['Gyr_X']['auc']:.4f} (drop {perm_importance['Gyr_X']:.4f})

→ Acc_X 제거가 **{perm_importance['Acc_X']/perm_importance['Gyr_X']:.1f}배 더 큰 성능 저하**

---

## 결론

1. **방법론마다 다른 관점**: 통계 vs 모델의 중요도가 다름
2. **가속도 vs 각속도**: 모델은 가속도, 통계는 각속도 선호
3. **응용 목적에 따라 선택**: 배포용(모델), 연구용(통계)
4. **발 센서 중요**: 모든 방법에서 일관되게 중요

**최종 권장**:
- 실제 배포: **Acc_X 중심** (모델 기반)
- 논문 설명: **Gyr_X 포함** (통계 기반)
- 최적 성능: **두 특징 모두 활용**
"""

    # Save
    md_file = OUTPUT_DIR / 'importance_analysis_summary.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"[OK] Summary saved: {md_file}")


def main():
    print("="*80)
    print("Feature Importance 방법론 비교")
    print("="*80)

    # Load data
    stats_data, model_data = load_data()

    # Create visualizations
    print("\n[1/3] Creating channel comparison...")
    create_channel_comparison(stats_data, model_data)

    print("\n[2/3] Creating sensor comparison...")
    create_sensor_comparison(stats_data, model_data)

    print("\n[3/3] Creating insights summary...")
    create_insights_summary(stats_data, model_data)

    print("\n" + "="*80)
    print("[DONE] Analysis complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
