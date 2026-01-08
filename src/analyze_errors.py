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
    # Use CPU to avoid GPU memory issues during analysis
    DEVICE = torch.device('cpu')

    # Task definitions (same as train_baseline_hpc.py)
    TASKS = {
        'PD_Screening': {'class0': ('HS', 'healthy'), 'class1': ('PD', 'neuro')},
        'OA_Screening': {'class0': ('HS', 'healthy'), 'class1': [('HOA', 'ortho'), ('KOA', 'ortho')]},
        'CVA_Detection': {'class0': ('HS', 'healthy'), 'class1': ('CVA', 'neuro')},
        'PD_vs_CVA': {'class0': ('PD', 'neuro'), 'class1': ('CVA', 'neuro')}
    }

Config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Architecture (same as training)
# ============================================================================

class SensorStream(nn.Module):
    """CNN stream for single sensor"""

    def __init__(self, in_channels=6, hidden_dim=64, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, channels, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)

        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention for sensor fusion"""

    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, num_sensors, hidden_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return x


class MultiStreamAttentionCNN(nn.Module):
    """Multi-Stream Attention CNN for Gait Classification"""

    def __init__(self, num_sensors=4, in_channels=6, hidden_dim=64,
                 num_heads=4, dropout=0.3):
        super().__init__()

        # Per-sensor CNN streams
        self.streams = nn.ModuleList([
            SensorStream(in_channels, hidden_dim, dropout)
            for _ in range(num_sensors)
        ])

        # Sensor fusion with attention
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_sensors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch, num_sensors, channels, time)
        batch_size = x.shape[0]

        # Process each sensor stream
        sensor_features = []
        for i, stream in enumerate(self.streams):
            sensor_x = x[:, i]  # (batch, channels, time)
            sensor_feat = stream(sensor_x)  # (batch, hidden_dim)
            sensor_features.append(sensor_feat)

        # Stack sensors: (batch, num_sensors, hidden_dim)
        x = torch.stack(sensor_features, dim=1)

        # Apply attention
        x = self.attention(x)  # (batch, num_sensors, hidden_dim)

        # Flatten and classify
        x = x.view(batch_size, -1)  # (batch, num_sensors * hidden_dim)
        x = self.fc(x)

        return x.squeeze(-1)


# ============================================================================
# Dataset (same as train_baseline_hpc.py)
# ============================================================================

def get_trial_paths(cohort, group):
    """Get all trial paths for a cohort"""
    cohort_path = Config.BASE_PATH / group / cohort

    if not cohort_path.exists():
        print(f"Warning: {cohort_path} does not exist")
        return []

    trials = []
    for meta_file in cohort_path.rglob("*_meta.json"):
        trial_path = meta_file.parent

        with open(meta_file, 'r') as f:
            meta = json.load(f)

        subject = meta.get('subject', 'unknown')
        trials.append((trial_path, subject))

    return trials


class GaitDataset(Dataset):
    """Gait signals dataset (same as train_baseline_hpc.py)"""

    def __init__(self, trial_paths, labels, window_size=300, stride=150, augment=False):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment

        # Load all trials and create sliding windows
        self.samples = []
        self.labels = []

        for trial_path, label in zip(trial_paths, labels):
            trial_data = self._load_trial_data(trial_path)
            if trial_data is None:
                continue

            # Sliding window
            for i in range(0, trial_data.shape[0] - window_size + 1, stride):
                window = trial_data[i:i + window_size]
                self.samples.append(window)
                self.labels.append(label)

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        print(f"  Created {len(self.samples)} windows from {len(trial_paths)} trials")

    def _load_trial_data(self, trial_path):
        """Load trial data from directory"""
        try:
            # Load each sensor's data
            sensors_data = {}
            for sensor in Config.SENSORS:
                # Correct filename pattern: {trial_name}_raw_data_{sensor}.txt
                sensor_file = trial_path / f"{trial_path.name}_raw_data_{sensor}.txt"
                if not sensor_file.exists():
                    print(f"Warning: {sensor_file} not found")
                    return None

                # Load sensor data: skip header row
                # Columns: PacketCounter(0), Acc_X(1), Acc_Y(2), Acc_Z(3), Gyr_X(4), Gyr_Y(5), Gyr_Z(6), Mag_X(7), Mag_Y(8), Mag_Z(9)
                data = np.loadtxt(sensor_file, skiprows=1)
                # Take columns 1-6 (Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z), exclude PacketCounter and Mag
                sensors_data[sensor] = data[:, 1:7]

            # Stack sensors: (time, sensors, channels)
            trial_data = np.stack([sensors_data[s] for s in Config.SENSORS], axis=1)
            return trial_data

        except Exception as e:
            print(f"Warning: Failed to load {trial_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]  # (window_size, 4, 6)
        y = self.labels[idx]

        # Transpose to (sensors, channels, time) for CNN
        x = np.transpose(x, (1, 2, 0))  # (4, 6, window_size)

        if self.augment and np.random.rand() > 0.5:
            # Random time shift
            shift = np.random.randint(-20, 20)
            x = np.roll(x, shift, axis=2)

            # Random noise
            noise = np.random.randn(*x.shape).astype(np.float32) * 0.01
            x = x + noise

        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)


def load_oa_screening_data():
    """Load OA Screening data (same as train_baseline_hpc.py)"""
    task_config = Config.TASKS['OA_Screening']

    # Load trials for class 0 (HS)
    class0_config = task_config['class0']
    cohort0, group0 = class0_config
    print(f"\nLoading {cohort0} (class 0)...")
    trials0 = get_trial_paths(cohort0, group0)
    print(f"  Found {len(trials0)} trials")

    # Load trials for class 1 (HOA + KOA)
    class1_config = task_config['class1']
    trials1 = []
    for cohort, group in class1_config:
        print(f"\nLoading {cohort} (class 1)...")
        cohort_trials = get_trial_paths(cohort, group)
        trials1.extend(cohort_trials)
        print(f"  Found {len(cohort_trials)} trials")
    print(f"  Total class 1: {len(trials1)} trials")

    # Get unique subjects
    subjects0 = list(set([t[1] for t in trials0]))
    subjects1 = list(set([t[1] for t in trials1]))

    # Subject-wise split
    train_subjects0, test_subjects0 = train_test_split(
        subjects0, test_size=0.2, random_state=Config.SEED
    )
    train_subjects1, test_subjects1 = train_test_split(
        subjects1, test_size=0.2, random_state=Config.SEED
    )

    # Split trials by subject
    train_trials0 = [t for t in trials0 if t[1] in train_subjects0]
    test_trials0 = [t for t in trials0 if t[1] in test_subjects0]
    train_trials1 = [t for t in trials1 if t[1] in train_subjects1]
    test_trials1 = [t for t in trials1 if t[1] in test_subjects1]

    # Combine train and test
    train_paths = [t[0] for t in train_trials0 + train_trials1]
    train_labels = [0] * len(train_trials0) + [1] * len(train_trials1)
    train_subjects = [t[1] for t in train_trials0 + train_trials1]

    test_paths = [t[0] for t in test_trials0 + test_trials1]
    test_labels = [0] * len(test_trials0) + [1] * len(test_trials1)
    test_subjects = [t[1] for t in test_trials0 + test_trials1]

    print(f"\nTrain: {len(train_paths)} trials, {len(set(train_subjects))} subjects")
    print(f"Test: {len(test_paths)} trials, {len(set(test_subjects))} subjects")

    return train_paths, train_labels, train_subjects, \
           test_paths, test_labels, test_subjects


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
            Config.WINDOW_SIZE, Config.STRIDE,
            augment=False
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

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs),
            'labels': np.array(all_labels)
        }

    def analyze_errors(self):
        """Comprehensive error analysis"""
        results = self.get_predictions()

        preds = results['predictions']
        probs = results['probabilities']
        labels = results['labels']

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

        print(f"\nFalse Positives (건강 → OA 오판): {fp_mask.sum()}")
        if len(fp_probs) > 0:
            print(f"  Mean probability: {fp_probs.mean():.4f}")
            print(f"  Confidence range: [{fp_probs.min():.4f}, {fp_probs.max():.4f}]")

        # False Negatives (OA → 건강으로 오판)
        fn_mask = (labels == 1) & (preds == 0)
        fn_probs = probs[fn_mask]

        print(f"\nFalse Negatives (OA → 건강 오판): {fn_mask.sum()}")
        if len(fn_probs) > 0:
            print(f"  Mean probability: {fn_probs.mean():.4f}")
            print(f"  Confidence range: [{fn_probs.min():.4f}, {fn_probs.max():.4f}]")

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
                'prob_range': [float(fp_probs.min()), float(fp_probs.max())] if len(fp_probs) > 0 else [0, 0]
            },
            'false_negatives': {
                'count': int(fn_mask.sum()),
                'mean_prob': float(fn_probs.mean()) if len(fn_probs) > 0 else 0,
                'prob_range': [float(fn_probs.min()), float(fn_probs.max())] if len(fn_probs) > 0 else [0, 0]
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
