"""
Error Analysis for OA Screening Model
Identifies and analyzes misclassified cases to find improvement opportunities

Phase 1-1 from NEXT_ANALYSIS_PLAN.md
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths - adjust for HPC or local
    BASE_PATH = Path(os.environ.get('DATA_PATH', 'D:/gait_wearable_sensor/dataset/data'))
    OUTPUT_PATH = Path(os.environ.get('OUTPUT_PATH', 'D:/gait_wearable_sensor/results/error_analysis'))
    MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'D:/gait_wearable_sensor/models'))

    SENSORS = ['HE', 'LB', 'LF', 'RF']
    CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
    SAMPLE_RATE = 100
    WINDOW_SIZE = 300
    STRIDE = 150

    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Architecture (same as training)
# ============================================================================

class MultiStreamAttentionCNN(nn.Module):
    """Multi-Stream Attention CNN (arXiv 2511.02047)"""

    def __init__(self, num_sensors=4, num_channels=6, window_size=300):
        super().__init__()
        self.num_sensors = num_sensors

        # Per-sensor stream (identical architecture)
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        # Attention mechanism
        pooled_size = window_size // 4
        self.attention = nn.Sequential(
            nn.Linear(128 * pooled_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Fusion and classification
        self.fc1 = nn.Linear(128 * num_sensors, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batch_size = x.size(0)
        sensor_features = []

        # Process each sensor stream
        for sensor_idx in range(self.num_sensors):
            sensor_data = x[:, sensor_idx, :, :]  # (batch, channels, time)

            # Convolutional layers
            h = F.relu(self.bn1(self.conv1(sensor_data)))
            h = self.pool(h)
            h = F.relu(self.bn2(self.conv2(h)))
            h = self.pool(h)
            h = self.dropout(h)

            # Flatten
            h = h.view(batch_size, -1)

            # Attention weight
            attn_weight = torch.sigmoid(self.attention(h))
            h = h * attn_weight

            sensor_features.append(h)

        # Concatenate sensor features
        fused = torch.cat(sensor_features, dim=1)

        # Classification head
        out = F.relu(self.fc1(fused))
        out = self.dropout(out)
        out = self.fc2(out)

        return out.squeeze(1)


# ============================================================================
# Dataset
# ============================================================================

class GaitDataset(Dataset):
    """Gait signals dataset with windowing"""

    def __init__(self, trials, labels, subjects, window_size=300, stride=150):
        self.trials = trials
        self.labels = labels
        self.subjects = subjects
        self.window_size = window_size
        self.stride = stride

        # Create windows
        self.windows = []
        self.window_labels = []
        self.window_subjects = []
        self.window_trial_indices = []

        for trial_idx, (trial_path, label, subject) in enumerate(zip(trials, labels, subjects)):
            trial_data = self._load_trial(trial_path)

            # Sliding window
            num_windows = (trial_data.shape[-1] - window_size) // stride + 1
            for i in range(num_windows):
                start = i * stride
                end = start + window_size
                window = trial_data[:, :, start:end]

                self.windows.append(window)
                self.window_labels.append(label)
                self.window_subjects.append(subject)
                self.window_trial_indices.append(trial_idx)

    def _load_trial(self, trial_path):
        """Load trial data from disk"""
        data = np.load(trial_path)
        return torch.FloatTensor(data)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.window_labels[idx], \
               self.window_subjects[idx], self.window_trial_indices[idx]


def load_oa_screening_data():
    """Load OA Screening data (same as training)"""

    # Class 0: HS
    hs_path = Config.BASE_PATH / 'healthy/HS'
    hs_subjects = sorted([s for s in hs_path.glob('*') if s.is_dir()])

    class0_trials = []
    for subject_dir in hs_subjects:
        subject_id = subject_dir.name
        trials = sorted([t for t in subject_dir.glob('*') if t.is_dir()])
        for trial_dir in trials:
            npy_file = trial_dir / f"{trial_dir.name}_processed.npy"
            if npy_file.exists():
                class0_trials.append((str(npy_file), 0, subject_id))

    # Class 1: HOA + KOA
    class1_trials = []
    for cohort in ['HOA', 'KOA']:
        cohort_path = Config.BASE_PATH / f'ortho/{cohort}'
        if not cohort_path.exists():
            continue
        subjects = sorted([s for s in cohort_path.glob('*') if s.is_dir()])
        for subject_dir in subjects:
            subject_id = subject_dir.name
            trials = sorted([t for t in subject_dir.glob('*') if t.is_dir()])
            for trial_dir in trials:
                npy_file = trial_dir / f"{trial_dir.name}_processed.npy"
                if npy_file.exists():
                    class1_trials.append((str(npy_file), 1, subject_id))

    print(f"Loaded Class 0 (HS): {len(class0_trials)} trials")
    print(f"Loaded Class 1 (OA): {len(class1_trials)} trials")

    # Combine
    all_trials = class0_trials + class1_trials

    # Extract components
    trial_paths = [t[0] for t in all_trials]
    labels = [t[1] for t in all_trials]
    subjects = [t[2] for t in all_trials]

    # Subject-wise split
    unique_subjects = list(set(subjects))
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, random_state=Config.SEED,
        stratify=[labels[subjects.index(s)] for s in unique_subjects]
    )

    # Split trials
    train_idx = [i for i, s in enumerate(subjects) if s in train_subjects]
    test_idx = [i for i, s in enumerate(subjects) if s in test_subjects]

    train_trials = [trial_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_subjects = [subjects[i] for i in train_idx]

    test_trials = [trial_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_subjects = [subjects[i] for i in test_idx]

    print(f"\nTrain: {len(train_trials)} trials, {len(set(train_subjects))} subjects")
    print(f"Test: {len(test_trials)} trials, {len(set(test_subjects))} subjects")

    return train_trials, train_labels, train_subjects, \
           test_trials, test_labels, test_subjects


# ============================================================================
# Error Analysis
# ============================================================================

class ErrorAnalyzer:
    """Analyze prediction errors to find improvement opportunities"""

    def __init__(self, model_path, task_name='OA_Screening'):
        self.task_name = task_name
        self.device = Config.DEVICE

        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = MultiStreamAttentionCNN(
            num_sensors=len(Config.SENSORS),
            num_channels=len(Config.CHANNELS),
            window_size=Config.WINDOW_SIZE
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
            test_trials, test_labels, test_subjects,
            Config.WINDOW_SIZE, Config.STRIDE
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False
        )

        print(f"Test dataset: {len(self.test_dataset)} windows")

    def get_predictions(self):
        """Get all predictions on test set"""
        print("\nGetting predictions...")

        all_preds = []
        all_probs = []
        all_labels = []
        all_subjects = []
        all_trial_indices = []

        with torch.no_grad():
            for inputs, labels, subjects, trial_indices in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_subjects.extend(subjects)
                all_trial_indices.extend(trial_indices.cpu().numpy())

        return {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs),
            'labels': np.array(all_labels),
            'subjects': all_subjects,
            'trial_indices': np.array(all_trial_indices)
        }

    def analyze_errors(self):
        """Comprehensive error analysis"""
        results = self.get_predictions()

        preds = results['predictions']
        probs = results['probabilities']
        labels = results['labels']
        subjects = results['subjects']

        # Overall metrics
        auc = roc_auc_score(labels, probs)
        acc = balanced_accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        print("\n" + "="*80)
        print("OVERALL PERFORMANCE")
        print("="*80)
        print(f"AUC: {auc:.4f}")
        print(f"Balanced Accuracy: {acc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

        # Identify errors
        errors = preds != labels
        num_errors = errors.sum()
        error_rate = num_errors / len(labels)

        print(f"\nTotal Errors: {num_errors}/{len(labels)} ({error_rate*100:.2f}%)")

        # False Positives (건강 → OA로 오판)
        fp_mask = (labels == 0) & (preds == 1)
        fp_probs = probs[fp_mask]
        fp_subjects = [subjects[i] for i in range(len(subjects)) if fp_mask[i]]

        print(f"\nFalse Positives (건강 → OA 오판): {fp_mask.sum()}")
        if len(fp_probs) > 0:
            print(f"  Mean probability: {fp_probs.mean():.4f}")
            print(f"  Confidence range: [{fp_probs.min():.4f}, {fp_probs.max():.4f}]")
            print(f"  Unique subjects: {len(set(fp_subjects))}")

        # False Negatives (OA → 건강으로 오판)
        fn_mask = (labels == 1) & (preds == 0)
        fn_probs = probs[fn_mask]
        fn_subjects = [subjects[i] for i in range(len(subjects)) if fn_mask[i]]

        print(f"\nFalse Negatives (OA → 건강 오판): {fn_mask.sum()}")
        if len(fn_probs) > 0:
            print(f"  Mean probability: {fn_probs.mean():.4f}")
            print(f"  Confidence range: [{fn_probs.min():.4f}, {fn_probs.max():.4f}]")
            print(f"  Unique subjects: {len(set(fn_subjects))}")

        # True Positives/Negatives
        tp_mask = (labels == 1) & (preds == 1)
        tn_mask = (labels == 0) & (preds == 0)

        tp_probs = probs[tp_mask]
        tn_probs = probs[tn_mask]

        print(f"\nTrue Positives (OA 정확): {tp_mask.sum()}")
        if len(tp_probs) > 0:
            print(f"  Mean probability: {tp_probs.mean():.4f}")

        print(f"\nTrue Negatives (건강 정확): {tn_mask.sum()}")
        if len(tn_probs) > 0:
            print(f"  Mean probability: {tn_probs.mean():.4f}")

        return {
            'overall': {
                'auc': float(auc),
                'balanced_accuracy': float(acc),
                'confusion_matrix': cm.tolist(),
                'total_errors': int(num_errors),
                'error_rate': float(error_rate)
            },
            'false_positives': {
                'count': int(fp_mask.sum()),
                'mean_prob': float(fp_probs.mean()) if len(fp_probs) > 0 else 0,
                'prob_range': [float(fp_probs.min()), float(fp_probs.max())] if len(fp_probs) > 0 else [0, 0],
                'unique_subjects': len(set(fp_subjects))
            },
            'false_negatives': {
                'count': int(fn_mask.sum()),
                'mean_prob': float(fn_probs.mean()) if len(fn_probs) > 0 else 0,
                'prob_range': [float(fn_probs.min()), float(fn_probs.max())] if len(fn_probs) > 0 else [0, 0],
                'unique_subjects': len(set(fn_subjects))
            },
            'true_positives': {
                'count': int(tp_mask.sum()),
                'mean_prob': float(tp_probs.mean()) if len(tp_probs) > 0 else 0
            },
            'true_negatives': {
                'count': int(tn_mask.sum()),
                'mean_prob': float(tn_probs.mean()) if len(tn_probs) > 0 else 0
            },
            'raw_results': results
        }

    def visualize_errors(self, analysis_results):
        """Create comprehensive error visualizations"""
        results = analysis_results['raw_results']

        preds = results['predictions']
        probs = results['probabilities']
        labels = results['labels']

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['건강 (예측)', 'OA (예측)'],
                   yticklabels=['건강 (실제)', 'OA (실제)'])
        ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=12)

        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax2.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontweight='bold', fontsize=12)
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        precision, recall, _ = precision_recall_curve(labels, probs)
        ax3.plot(recall, precision, linewidth=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
        ax3.grid(alpha=0.3)

        # 4. Probability Distribution by Class
        ax4 = fig.add_subplot(gs[1, 0])

        tp_mask = (labels == 1) & (preds == 1)
        tn_mask = (labels == 0) & (preds == 0)
        fp_mask = (labels == 0) & (preds == 1)
        fn_mask = (labels == 1) & (preds == 0)

        ax4.hist(probs[tn_mask], bins=50, alpha=0.5, label='TN (건강 정확)', color='green')
        ax4.hist(probs[tp_mask], bins=50, alpha=0.5, label='TP (OA 정확)', color='blue')
        ax4.hist(probs[fp_mask], bins=50, alpha=0.5, label='FP (건강→OA 오판)', color='red')
        ax4.hist(probs[fn_mask], bins=50, alpha=0.5, label='FN (OA→건강 오판)', color='orange')
        ax4.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Count')
        ax4.set_title('Prediction Probability Distribution', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. Error Type Comparison
        ax5 = fig.add_subplot(gs[1, 1])
        error_counts = [
            analysis_results['false_positives']['count'],
            analysis_results['false_negatives']['count']
        ]
        bars = ax5.bar(['False Positive\\n(건강→OA)', 'False Negative\\n(OA→건강)'],
                      error_counts, color=['red', 'orange'], alpha=0.7)
        ax5.set_ylabel('Count', fontweight='bold')
        ax5.set_title('Error Type Comparison', fontweight='bold', fontsize=12)
        ax5.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        # 6. Probability Box Plot
        ax6 = fig.add_subplot(gs[1, 2])

        data_to_plot = [
            probs[tn_mask],
            probs[fn_mask],
            probs[fp_mask],
            probs[tp_mask]
        ]
        labels_plot = ['TN\\n(건강 정확)', 'FN\\n(OA→건강 오판)',
                      'FP\\n(건강→OA 오판)', 'TP\\n(OA 정확)']
        colors = ['green', 'orange', 'red', 'blue']

        bp = ax6.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax6.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        ax6.set_ylabel('Predicted Probability', fontweight='bold')
        ax6.set_title('Probability Distribution by Outcome', fontweight='bold', fontsize=12)
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)

        # 7. Summary Statistics Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')

        summary_text = f"""
Error Analysis Summary

Overall Performance:
  • AUC: {analysis_results['overall']['auc']:.4f}
  • Balanced Accuracy: {analysis_results['overall']['balanced_accuracy']:.4f}
  • Total Errors: {analysis_results['overall']['total_errors']} ({analysis_results['overall']['error_rate']*100:.2f}%)

False Positives (건강 → OA 오판):
  • Count: {analysis_results['false_positives']['count']}
  • Mean Probability: {analysis_results['false_positives']['mean_prob']:.4f}
  • Unique Subjects: {analysis_results['false_positives']['unique_subjects']}

False Negatives (OA → 건강 오판):
  • Count: {analysis_results['false_negatives']['count']}
  • Mean Probability: {analysis_results['false_negatives']['mean_prob']:.4f}
  • Unique Subjects: {analysis_results['false_negatives']['unique_subjects']}

Key Insights:
  1. 어느 에러가 더 많은가? {'FP' if analysis_results['false_positives']['count'] > analysis_results['false_negatives']['count'] else 'FN'}
  2. 에러의 확신도는? FP: {analysis_results['false_positives']['mean_prob']:.2f}, FN: {analysis_results['false_negatives']['mean_prob']:.2f}
  3. 개선 방향: {'FN 줄이기 (환자 놓치지 않기)' if analysis_results['false_negatives']['count'] > analysis_results['false_positives']['count'] else 'FP 줄이기 (오진 방지)'}
"""

        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle(f'{self.task_name} - Error Analysis',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(Config.OUTPUT_PATH / f'{self.task_name}_error_analysis.png',
                   dpi=300, bbox_inches='tight')
        print(f"\n[OK] Visualization saved: {Config.OUTPUT_PATH / f'{self.task_name}_error_analysis.png'}")
        plt.close()


def main():
    print("="*80)
    print("Error Analysis - Phase 1-1")
    print("="*80)

    # Model path
    model_path = Config.MODEL_PATH / 'OA_Screening_best.pth'

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    # Create analyzer
    analyzer = ErrorAnalyzer(model_path, task_name='OA_Screening')

    # Run analysis
    print("\n[1/3] Running error analysis...")
    analysis_results = analyzer.analyze_errors()

    # Save results
    print("\n[2/3] Saving results...")
    output_file = Config.OUTPUT_PATH / 'OA_Screening_error_analysis.json'

    # Remove raw_results for JSON serialization
    results_to_save = {k: v for k, v in analysis_results.items() if k != 'raw_results'}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved: {output_file}")

    # Create visualizations
    print("\n[3/3] Creating visualizations...")
    analyzer.visualize_errors(analysis_results)

    print("\n" + "="*80)
    print("[DONE] Error Analysis Complete!")
    print(f"Output directory: {Config.OUTPUT_PATH}")
    print("="*80)


if __name__ == '__main__':
    main()
