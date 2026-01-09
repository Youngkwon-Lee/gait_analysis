"""
Phase 2-2: Sensor Importance Analysis
Identify which sensors (HE, LB, LF, RF) contribute most to OA detection

Goals:
1. Measure performance drop when each sensor is removed (ablation study)
2. Analyze sensor combinations (all, 3-sensor, 2-sensor, 1-sensor)
3. Calculate sensor importance scores
4. Identify minimum sensor set for acceptable performance
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration with environment variable support"""
    BASE_PATH = Path(os.environ.get('DATA_PATH', 'D:/gait_wearable_sensor/dataset/data'))
    MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'D:/gait_wearable_sensor/models'))
    OUTPUT_PATH = Path(os.environ.get('OUTPUT_PATH', 'D:/gait_wearable_sensor/results/sensor_importance'))

    # Create output directory
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # Data parameters
    SENSORS = ['HE', 'LB', 'LF', 'RF']  # 4 IMU sensors
    SENSOR_NAMES = {
        'HE': 'Heel',
        'LB': 'Left Back',
        'LF': 'Left Front',
        'RF': 'Right Front'
    }
    CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']  # NO Magnetometer
    WINDOW_SIZE = 300  # 3 seconds @ 100Hz
    OVERLAP = 150  # 50% overlap
    SEED = 42

    # Tasks
    TASKS = {
        'OA_Screening': {
            'class0': ('HS', 'healthy'),
            'class1': [('HOA', 'ortho'), ('KOA', 'ortho')]
        }
    }

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Model Architecture (EXACT COPY from train_baseline_hpc.py)
# ============================================================================

import torch.nn.functional as F

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
            hidden_dim,
            num_heads,
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
    """Multi-stream CNN with attention-based sensor fusion"""

    def __init__(self, num_sensors=4, in_channels=6, hidden_dim=64,
                 num_classes=1, num_heads=4, dropout=0.3):
        super().__init__()

        # Separate CNN stream for each sensor
        self.streams = nn.ModuleList([
            SensorStream(in_channels, hidden_dim, dropout)
            for _ in range(num_sensors)
        ])

        # Attention-based fusion
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_sensors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, sensor_mask=None):
        # x: (batch, num_sensors, in_channels, time)
        # sensor_mask: (batch, num_sensors) - 1 for keep, 0 for ablate
        batch_size = x.shape[0]
        sensor_features = []

        for i, stream in enumerate(self.streams):
            sensor_x = x[:, i]  # (batch, channels, time)
            sensor_feat = stream(sensor_x)  # (batch, hidden_dim)

            # Apply sensor mask if provided (ablation)
            if sensor_mask is not None:
                mask = sensor_mask[:, i].unsqueeze(1)  # (batch, 1)
                sensor_feat = sensor_feat * mask

            sensor_features.append(sensor_feat)

        # Stack sensor features
        x = torch.stack(sensor_features, dim=1)  # (batch, num_sensors, hidden_dim)

        # Apply attention
        x = self.attention(x)  # (batch, num_sensors, hidden_dim)

        # Flatten and classify
        x = x.view(batch_size, -1)  # (batch, num_sensors * hidden_dim)
        x = self.fc(x)

        return x.squeeze(-1)

# ============================================================================
# Dataset (EXACT COPY from analyze_temporal.py)
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

        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            subject = meta.get('subject', 'unknown')
            trials.append((trial_path, subject))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Skipping {meta_file} - corrupted or empty JSON: {e}")
            continue

    return trials


class GaitDataset(Dataset):
    """Gait signals dataset (same as train_baseline_hpc.py)"""

    def __init__(self, trial_paths, labels, subjects=None, window_size=300, stride=150, augment=False):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment

        # Load all trials and create sliding windows
        self.samples = []
        self.labels = []

        for idx, (trial_path, label) in enumerate(zip(trial_paths, labels)):
            trial_data = self._load_trial_data(trial_path)
            if trial_data is None:
                continue

            # Sliding window
            for i in range(0, trial_data.shape[0] - window_size + 1, stride):
                window = trial_data[i:i + window_size]  # (window_size, 4, 6)

                # Normalize per window (same as train_baseline_hpc.py)
                mean = window.mean(axis=0, keepdims=True)
                std = window.std(axis=0, keepdims=True) + 1e-8
                window = (window - mean) / std

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
                    return None

                # Load sensor data: skip header row
                # Columns: PacketCounter(0), Acc_X(1), Acc_Y(2), Acc_Z(3), Gyr_X(4), Gyr_Y(5), Gyr_Z(6), Mag_X(7), Mag_Y(8), Mag_Z(9)
                data = np.loadtxt(sensor_file, skiprows=1)

                # Extract only Acc + Gyr (columns 1-6), skip Mag
                sensors_data[sensor] = data[:, 1:7]  # (n_samples, 6)

            # Stack sensors: (n_samples, 4, 6)
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
# Sensor Importance Analyzer
# ============================================================================

class SensorImportanceAnalyzer:
    """Analyze sensor importance via ablation study"""

    def __init__(self, model_path, task_name='OA_Screening'):
        self.task_name = task_name
        self.device = Config.DEVICE

        # Load model
        print(f"\n[Model] Loading from {model_path}")
        self.model = MultiStreamAttentionCNN(
            num_sensors=4,
            in_channels=6,  # Acc + Gyr (no Mag)
            hidden_dim=64,
            num_classes=1,
            num_heads=4,
            dropout=0.3
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"[Model] Loaded successfully, using {self.device}")

        # Load test dataset
        print("\n[Dataset] Loading test data...")
        _, _, _, test_paths, test_labels, test_subjects = load_oa_screening_data()
        self.dataset = GaitDataset(
            test_paths,
            test_labels,
            test_subjects,
            window_size=Config.WINDOW_SIZE,
            stride=Config.OVERLAP,
            augment=False
        )
        self.loader = DataLoader(self.dataset, batch_size=32, shuffle=False)

    def evaluate_with_sensor_mask(self, sensor_mask):
        """
        Evaluate model performance with specific sensors ablated

        Args:
            sensor_mask: List of sensor indices to KEEP (e.g., [0,1,2] = keep HE, LB, LF, ablate RF)

        Returns:
            dict with predictions, probabilities, labels, metrics
        """
        all_preds = []
        all_probs = []
        all_labels = []

        # Create mask tensor: 1 for keep, 0 for ablate
        mask = torch.zeros(4, device=self.device)
        for idx in sensor_mask:
            mask[idx] = 1.0

        with torch.no_grad():
            for inputs, labels in self.loader:
                inputs = inputs.to(self.device)
                labels = labels.cpu().numpy()

                # Expand mask for batch
                batch_mask = mask.unsqueeze(0).expand(inputs.shape[0], -1)

                # Forward with sensor mask
                outputs = self.model(inputs, sensor_mask=batch_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        return {
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels,
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'auc': auc
        }

    def run_ablation_study(self):
        """Run complete ablation study"""
        print("\n" + "="*80)
        print("SENSOR IMPORTANCE ANALYSIS - ABLATION STUDY")
        print("="*80)

        results = {}

        # 1. All sensors (baseline)
        print("\n[1/15] All 4 sensors (baseline)...")
        results['all'] = self.evaluate_with_sensor_mask([0, 1, 2, 3])
        print(f"  AUC: {results['all']['auc']:.4f}, Acc: {results['all']['accuracy']:.4f}")

        # 2. Remove one sensor at a time (3-sensor combinations)
        print("\n[2-5] Remove each sensor (leave-one-out)...")
        sensor_names = Config.SENSORS
        for i, sensor in enumerate(sensor_names):
            mask = [j for j in range(4) if j != i]
            print(f"  Remove {sensor}: ", end='')
            results[f'without_{sensor}'] = self.evaluate_with_sensor_mask(mask)
            print(f"AUC: {results[f'without_{sensor}']['auc']:.4f}, " +
                  f"Acc: {results[f'without_{sensor}']['accuracy']:.4f}")

        # 3. Two-sensor combinations
        print("\n[6-11] Two-sensor combinations...")
        two_combos = list(combinations(range(4), 2))
        for combo in two_combos:
            combo_names = [sensor_names[i] for i in combo]
            combo_str = '+'.join(combo_names)
            print(f"  {combo_str}: ", end='')
            results[f'combo_{combo_str}'] = self.evaluate_with_sensor_mask(list(combo))
            print(f"AUC: {results[f'combo_{combo_str}']['auc']:.4f}, " +
                  f"Acc: {results[f'combo_{combo_str}']['accuracy']:.4f}")

        # 4. Single sensors
        print("\n[12-15] Single sensors...")
        for i, sensor in enumerate(sensor_names):
            print(f"  {sensor} only: ", end='')
            results[f'only_{sensor}'] = self.evaluate_with_sensor_mask([i])
            print(f"AUC: {results[f'only_{sensor}']['auc']:.4f}, " +
                  f"Acc: {results[f'only_{sensor}']['accuracy']:.4f}")

        return results

    def calculate_sensor_importance(self, results):
        """Calculate importance score for each sensor"""
        baseline_auc = results['all']['auc']
        sensor_names = Config.SENSORS

        importance_scores = {}
        for i, sensor in enumerate(sensor_names):
            # Importance = performance drop when sensor is removed
            without_auc = results[f'without_{sensor}']['auc']
            importance = baseline_auc - without_auc
            importance_scores[sensor] = {
                'importance': importance,
                'baseline_auc': baseline_auc,
                'without_auc': without_auc,
                'relative_drop': (importance / baseline_auc) * 100
            }

        return importance_scores

    def save_results(self, results, importance_scores):
        """Save results to JSON"""
        output = {
            'task': self.task_name,
            'total_windows': len(self.dataset),
            'ablation_results': {},
            'sensor_importance': importance_scores
        }

        # Convert results to serializable format
        for key, val in results.items():
            output['ablation_results'][key] = {
                'accuracy': float(val['accuracy']),
                'balanced_accuracy': float(val['balanced_accuracy']),
                'auc': float(val['auc'])
            }

        output_file = Config.OUTPUT_PATH / f'{self.task_name}_sensor_importance.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n[OK] Results saved: {output_file}")

    def visualize_results(self, results, importance_scores):
        """Create comprehensive visualizations"""
        print("\n[Visualization] Creating plots...")

        fig = plt.figure(figsize=(16, 12))

        # 1. Sensor importance bar plot
        ax1 = plt.subplot(2, 3, 1)
        sensors = list(importance_scores.keys())
        importance_vals = [importance_scores[s]['importance'] for s in sensors]
        bars = ax1.bar(sensors, importance_vals, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Sensor', fontsize=12)
        ax1.set_ylabel('Importance Score (AUC drop)', fontsize=12)
        ax1.set_title('Sensor Importance (Leave-One-Out)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Annotate bars
        for bar, val in zip(bars, importance_vals):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=10)

        # 2. AUC comparison: All vs Without each sensor
        ax2 = plt.subplot(2, 3, 2)
        baseline = results['all']['auc']
        without_aucs = [results[f'without_{s}']['auc'] for s in sensors]
        x = np.arange(len(sensors))
        width = 0.35
        ax2.bar(x - width/2, [baseline]*len(sensors), width, label='All sensors', alpha=0.7)
        ax2.bar(x + width/2, without_aucs, width, label='Without sensor', alpha=0.7)
        ax2.set_xlabel('Removed Sensor', fontsize=12)
        ax2.set_ylabel('AUC', fontsize=12)
        ax2.set_title('Performance Drop Analysis', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sensors)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. Single sensor performance
        ax3 = plt.subplot(2, 3, 3)
        single_aucs = [results[f'only_{s}']['auc'] for s in sensors]
        bars = ax3.bar(sensors, single_aucs, color='coral', alpha=0.7)
        ax3.axhline(y=baseline, color='green', linestyle='--', label='All sensors')
        ax3.set_xlabel('Sensor', fontsize=12)
        ax3.set_ylabel('AUC', fontsize=12)
        ax3.set_title('Individual Sensor Performance', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # Annotate
        for bar, val in zip(bars, single_aucs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9)

        # 4. Heatmap: All combinations
        ax4 = plt.subplot(2, 3, 4)
        # Create matrix: rows = number of sensors, cols = specific combinations
        combo_data = []
        combo_labels = []

        # All sensors
        combo_data.append(results['all']['auc'])
        combo_labels.append('All (4)')

        # 3-sensor combinations
        for s in sensors:
            combo_data.append(results[f'without_{s}']['auc'])
            combo_labels.append(f'w/o {s}')

        # 2-sensor combinations (select few)
        two_combos = list(combinations(range(4), 2))[:4]  # First 4
        for combo in two_combos:
            combo_names = [sensors[i] for i in combo]
            combo_str = '+'.join(combo_names)
            if f'combo_{combo_str}' in results:
                combo_data.append(results[f'combo_{combo_str}']['auc'])
                combo_labels.append(combo_str)

        # Single sensors
        for s in sensors:
            combo_data.append(results[f'only_{s}']['auc'])
            combo_labels.append(f'{s} only')

        # Plot as horizontal bar
        y_pos = np.arange(len(combo_labels))
        bars = ax4.barh(y_pos, combo_data, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(combo_labels, fontsize=9)
        ax4.set_xlabel('AUC', fontsize=12)
        ax4.set_title('All Sensor Combinations', fontsize=14, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)

        # Color by performance
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=min(combo_data), vmax=max(combo_data))
        for bar, val in zip(bars, combo_data):
            bar.set_color(cmap(norm(val)))

        # 5. Relative importance (percentage)
        ax5 = plt.subplot(2, 3, 5)
        relative_drops = [importance_scores[s]['relative_drop'] for s in sensors]
        bars = ax5.bar(sensors, relative_drops, color='darkred', alpha=0.7)
        ax5.set_xlabel('Sensor', fontsize=12)
        ax5.set_ylabel('Performance Drop (%)', fontsize=12)
        ax5.set_title('Relative Importance (%)', fontsize=14, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

        # Annotate
        for bar, val in zip(bars, relative_drops):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%',
                    ha='center', va='bottom', fontsize=10)

        # 6. Summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        summary_data = []
        for s in sensors:
            summary_data.append([
                Config.SENSOR_NAMES[s],
                f"{importance_scores[s]['importance']:.4f}",
                f"{importance_scores[s]['relative_drop']:.2f}%",
                f"{results[f'only_{s}']['auc']:.4f}"
            ])

        table = ax6.table(
            cellText=summary_data,
            colLabels=['Sensor', 'Importance', 'Drop %', 'Solo AUC'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax6.set_title('Sensor Importance Summary', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        output_file = Config.OUTPUT_PATH / f'{self.task_name}_sensor_importance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Visualization saved: {output_file}")

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("Phase 2-2: Sensor Importance Analysis")
    print("="*80)

    # Initialize analyzer
    model_path = Config.MODEL_PATH / 'OA_Screening_best.pth'
    analyzer = SensorImportanceAnalyzer(model_path, task_name='OA_Screening')

    # Run ablation study
    results = analyzer.run_ablation_study()

    # Calculate importance scores
    importance_scores = analyzer.calculate_sensor_importance(results)

    # Print summary
    print("\n" + "="*80)
    print("SENSOR IMPORTANCE SUMMARY")
    print("="*80)
    print(f"\nBaseline (All 4 sensors): AUC = {results['all']['auc']:.4f}")
    print("\nImportance Ranking (by AUC drop when removed):")
    sorted_sensors = sorted(importance_scores.items(),
                           key=lambda x: x[1]['importance'],
                           reverse=True)
    for rank, (sensor, scores) in enumerate(sorted_sensors, 1):
        print(f"  {rank}. {Config.SENSOR_NAMES[sensor]} ({sensor}): " +
              f"Importance = {scores['importance']:.4f} " +
              f"({scores['relative_drop']:.2f}% drop)")

    # Save results
    analyzer.save_results(results, importance_scores)

    # Create visualizations
    analyzer.visualize_results(results, importance_scores)

    print("\n" + "="*80)
    print("[DONE] Sensor Importance Analysis Complete!")
    print(f"Output directory: {Config.OUTPUT_PATH}")
    print("="*80)

if __name__ == '__main__':
    main()
