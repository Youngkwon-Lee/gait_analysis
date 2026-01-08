"""
Confusion Analysis for OA Screening Model
Analyzes False Positive vs False Negative and finds optimal threshold

Phase 1-2 from NEXT_ANALYSIS_PLAN.md
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, f1_score, matthews_corrcoef
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Import from analyze_errors
import sys
sys.path.append(str(Path(__file__).parent))
from analyze_errors import (
    Config, MultiStreamAttentionCNN, GaitDataset, load_oa_screening_data
)

# ============================================================================
# Confusion Analysis
# ============================================================================

class ConfusionAnalyzer:
    """Analyze confusion matrix and find optimal threshold"""

    def __init__(self, model_path, task_name='OA_Screening'):
        self.task_name = task_name
        self.device = Config.DEVICE
        Config.OUTPUT_PATH = Path(os.environ.get('OUTPUT_PATH', 'D:/gait_wearable_sensor/results/confusion_analysis'))
        Config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = MultiStreamAttentionCNN(
            num_sensors=len(Config.SENSORS),
            in_channels=len(Config.CHANNELS),
            hidden_dim=64,
            num_heads=4,
            dropout=0.3
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("Model loaded successfully")

        # Load data
        print("\nLoading data...")
        _, _, _, test_trials, test_labels, test_subjects = load_oa_screening_data()

        self.test_dataset = GaitDataset(
            test_trials, test_labels,
            Config.WINDOW_SIZE, Config.STRIDE
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False
        )

        print(f"Test dataset: {len(self.test_dataset)} windows")

    def get_predictions(self):
        """Get all predictions on test set"""
        print("\nGetting predictions...")

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_probs), np.array(all_labels)

    def find_optimal_threshold(self, probs, labels):
        """Find optimal threshold using multiple criteria"""

        thresholds = np.linspace(0.01, 0.99, 99)
        results = []

        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            cm = confusion_matrix(labels, preds)

            tn, fp, fn, tp = cm.ravel()

            # Metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, True Positive Rate
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = f1_score(labels, preds)
            balanced_acc = (sensitivity + specificity) / 2
            mcc = matthews_corrcoef(labels, preds)

            # Youden's Index (J = Sensitivity + Specificity - 1)
            youden_index = sensitivity + specificity - 1

            results.append({
                'threshold': thresh,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1,
                'balanced_accuracy': balanced_acc,
                'youden_index': youden_index,
                'mcc': mcc,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            })

        df = pd.DataFrame(results)

        # Find optimal thresholds by different criteria
        optimal_thresholds = {
            'default': {
                'threshold': 0.5,
                'reason': 'Standard classification threshold'
            },
            'youden': {
                'threshold': float(df.loc[df['youden_index'].idxmax(), 'threshold']),
                'reason': 'Maximizes (Sensitivity + Specificity - 1)'
            },
            'f1': {
                'threshold': float(df.loc[df['f1_score'].idxmax(), 'threshold']),
                'reason': 'Maximizes F1 Score'
            },
            'balanced_acc': {
                'threshold': float(df.loc[df['balanced_accuracy'].idxmax(), 'threshold']),
                'reason': 'Maximizes Balanced Accuracy'
            },
            'mcc': {
                'threshold': float(df.loc[df['mcc'].idxmax(), 'threshold']),
                'reason': 'Maximizes Matthews Correlation Coefficient'
            },
            'high_sensitivity': {
                'threshold': float(df[df['sensitivity'] >= 0.95].loc[df[df['sensitivity'] >= 0.95]['specificity'].idxmax(), 'threshold']) if len(df[df['sensitivity'] >= 0.95]) > 0 else 0.3,
                'reason': 'Maintains ≥95% sensitivity (catch most OA patients)'
            },
            'high_specificity': {
                'threshold': float(df[df['specificity'] >= 0.95].loc[df[df['specificity'] >= 0.95]['sensitivity'].idxmax(), 'threshold']) if len(df[df['specificity'] >= 0.95]) > 0 else 0.7,
                'reason': 'Maintains ≥95% specificity (minimize false alarms)'
            }
        }

        return df, optimal_thresholds

    def analyze_threshold_impact(self, probs, labels, optimal_thresholds):
        """Analyze impact of different thresholds"""

        print("\n" + "="*80)
        print("OPTIMAL THRESHOLD ANALYSIS")
        print("="*80)

        results = {}

        for name, info in optimal_thresholds.items():
            thresh = info['threshold']
            preds = (probs >= thresh).astype(int)
            cm = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

            print(f"\n{name.upper()} (Threshold: {thresh:.3f})")
            print(f"  Reason: {info['reason']}")
            print(f"  Sensitivity (Recall): {sensitivity:.4f} - {tp}/{tp+fn} OA patients detected")
            print(f"  Specificity: {specificity:.4f} - {tn}/{tn+fp} healthy correctly identified")
            print(f"  PPV (Precision): {ppv:.4f} - {tp}/{tp+fp} positive predictions correct")
            print(f"  NPV: {npv:.4f} - {tn}/{tn+fn} negative predictions correct")
            print(f"  Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

            results[name] = {
                'threshold': float(thresh),
                'reason': info['reason'],
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'ppv': float(ppv),
                'npv': float(npv),
                'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
            }

        return results

    def visualize_confusion_analysis(self, probs, labels, threshold_df, optimal_thresholds, threshold_results):
        """Create comprehensive confusion analysis visualizations"""

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. ROC Curve with optimal points
        ax1 = fig.add_subplot(gs[0, :2])
        fpr, tpr, roc_thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)

        ax1.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={auc:.4f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

        # Mark optimal points
        colors = {'youden': 'red', 'f1': 'blue', 'balanced_acc': 'green',
                 'high_sensitivity': 'orange', 'high_specificity': 'purple'}

        for name in ['youden', 'f1', 'balanced_acc']:
            if name in threshold_results:
                sens = threshold_results[name]['sensitivity']
                spec = threshold_results[name]['specificity']
                fpr_point = 1 - spec
                ax1.plot(fpr_point, sens, 'o', markersize=10, color=colors[name],
                        label=f'{name.upper()} ({threshold_results[name]["threshold"]:.3f})')

        ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
        ax1.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
        ax1.set_title('ROC Curve with Optimal Thresholds', fontweight='bold', fontsize=13)
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)

        # 2. Precision-Recall Curve
        ax2 = fig.add_subplot(gs[0, 2])
        precision, recall, pr_thresholds = precision_recall_curve(labels, probs)

        ax2.plot(recall, precision, linewidth=2)
        ax2.set_xlabel('Recall (Sensitivity)', fontweight='bold')
        ax2.set_ylabel('Precision (PPV)', fontweight='bold')
        ax2.set_title('Precision-Recall Curve', fontweight='bold', fontsize=13)
        ax2.grid(alpha=0.3)

        # 3. Sensitivity vs Specificity
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(threshold_df['threshold'], threshold_df['sensitivity'],
                label='Sensitivity (Recall)', linewidth=2, color='blue')
        ax3.plot(threshold_df['threshold'], threshold_df['specificity'],
                label='Specificity', linewidth=2, color='red')

        # Mark optimal points
        for name in ['youden', 'high_sensitivity', 'high_specificity']:
            if name in optimal_thresholds:
                thresh = optimal_thresholds[name]['threshold']
                ax3.axvline(thresh, linestyle='--', alpha=0.5, label=f'{name} ({thresh:.2f})')

        ax3.set_xlabel('Threshold', fontweight='bold')
        ax3.set_ylabel('Value', fontweight='bold')
        ax3.set_title('Sensitivity vs Specificity', fontweight='bold', fontsize=13)
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Youden's Index
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(threshold_df['threshold'], threshold_df['youden_index'],
                linewidth=2, color='green')
        max_idx = threshold_df['youden_index'].idxmax()
        max_thresh = threshold_df.loc[max_idx, 'threshold']
        max_youden = threshold_df.loc[max_idx, 'youden_index']
        ax4.plot(max_thresh, max_youden, 'ro', markersize=10,
                label=f'Max: {max_youden:.3f} @ {max_thresh:.3f}')
        ax4.set_xlabel('Threshold', fontweight='bold')
        ax4.set_ylabel("Youden's Index (J)", fontweight='bold')
        ax4.set_title("Youden's Index (Sensitivity + Specificity - 1)", fontweight='bold', fontsize=13)
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. F1 Score
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(threshold_df['threshold'], threshold_df['f1_score'],
                linewidth=2, color='purple')
        max_idx = threshold_df['f1_score'].idxmax()
        max_thresh = threshold_df.loc[max_idx, 'threshold']
        max_f1 = threshold_df.loc[max_idx, 'f1_score']
        ax5.plot(max_thresh, max_f1, 'ro', markersize=10,
                label=f'Max: {max_f1:.3f} @ {max_thresh:.3f}')
        ax5.set_xlabel('Threshold', fontweight='bold')
        ax5.set_ylabel('F1 Score', fontweight='bold')
        ax5.set_title('F1 Score vs Threshold', fontweight='bold', fontsize=13)
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. Error counts
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.plot(threshold_df['threshold'], threshold_df['fp'],
                label='False Positive (건강→OA)', linewidth=2, color='red')
        ax6.plot(threshold_df['threshold'], threshold_df['fn'],
                label='False Negative (OA→건강)', linewidth=2, color='orange')
        ax6.set_xlabel('Threshold', fontweight='bold')
        ax6.set_ylabel('Count', fontweight='bold')
        ax6.set_title('Error Counts vs Threshold', fontweight='bold', fontsize=13)
        ax6.legend()
        ax6.grid(alpha=0.3)

        # 7. Confusion matrices at different thresholds
        threshold_names = ['high_sensitivity', 'youden', 'default', 'high_specificity']
        for idx, name in enumerate(threshold_names):
            ax = fig.add_subplot(gs[2, idx+1] if idx < 2 else gs[3, idx-2])

            if name == 'default':
                cm_data = threshold_results[name]['confusion_matrix']
            elif name in threshold_results:
                cm_data = threshold_results[name]['confusion_matrix']
            else:
                continue

            cm = np.array([[cm_data['tn'], cm_data['fp']],
                          [cm_data['fn'], cm_data['tp']]])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['건강', 'OA'],
                       yticklabels=['건강', 'OA'],
                       cbar=False)

            thresh_val = threshold_results[name]['threshold']
            sens = threshold_results[name]['sensitivity']
            spec = threshold_results[name]['specificity']

            ax.set_title(f'{name.replace("_", " ").title()}\n'
                        f'Threshold: {thresh_val:.3f}\n'
                        f'Sens: {sens:.3f}, Spec: {spec:.3f}',
                        fontweight='bold', fontsize=11)

        # 8. Summary table
        ax_summary = fig.add_subplot(gs[3, 2:])
        ax_summary.axis('off')

        summary_text = """
임상적 고려사항

1. False Negative (FN) vs False Positive (FP):
   • FN (OA → 건강 오진): 환자 놓침 → 치료 지연 → 더 위험!
   • FP (건강 → OA 오진): 추가 검사 → 비용 증가 → 덜 위험

2. 권장 Threshold:
   • 일반 스크리닝: Youden's Index ({})
     → Sensitivity와 Specificity 균형

   • 안전 우선 (환자 놓치지 않기): High Sensitivity ({})
     → Sensitivity ≥95%, FN 최소화

   • 정밀 진단: High Specificity ({})
     → Specificity ≥95%, FP 최소화

3. Threshold 선택 기준:
   • 초기 스크리닝 → High Sensitivity (환자 놓치지 않기)
   • 확진 검사 → High Specificity (정확한 진단)
   • 연구용 → Youden's Index (균형잡힌 성능)

4. Trade-off:
   • Threshold ↓ → Sensitivity ↑, Specificity ↓ (FP 증가)
   • Threshold ↑ → Sensitivity ↓, Specificity ↑ (FN 증가)
""".format(
            threshold_results['youden']['threshold'],
            threshold_results['high_sensitivity']['threshold'],
            threshold_results['high_specificity']['threshold']
        )

        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle(f'{self.task_name} - Confusion Analysis & Threshold Optimization',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(Config.OUTPUT_PATH / f'{self.task_name}_confusion_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n[OK] Visualization saved: {Config.OUTPUT_PATH / f'{self.task_name}_confusion_analysis.png'}")
        plt.close()


def main():
    print("="*80)
    print("Confusion Analysis - Phase 1-2")
    print("="*80)

    # Model path
    model_path = Config.MODEL_PATH / 'OA_Screening_best.pth'

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    # Create analyzer
    analyzer = ConfusionAnalyzer(model_path, task_name='OA_Screening')

    # Get predictions
    probs, labels = analyzer.get_predictions()

    # Find optimal thresholds
    print("\n[1/3] Finding optimal thresholds...")
    threshold_df, optimal_thresholds = analyzer.find_optimal_threshold(probs, labels)

    # Analyze threshold impact
    print("\n[2/3] Analyzing threshold impact...")
    threshold_results = analyzer.analyze_threshold_impact(probs, labels, optimal_thresholds)

    # Save results
    output_file = Config.OUTPUT_PATH / 'OA_Screening_confusion_analysis.json'
    results_to_save = {
        'optimal_thresholds': optimal_thresholds,
        'threshold_results': threshold_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved: {output_file}")

    # Create visualizations
    print("\n[3/3] Creating visualizations...")
    analyzer.visualize_confusion_analysis(probs, labels, threshold_df,
                                         optimal_thresholds, threshold_results)

    print("\n" + "="*80)
    print("[DONE] Confusion Analysis Complete!")
    print(f"Output directory: {Config.OUTPUT_PATH}")
    print("="*80)


if __name__ == '__main__':
    main()
