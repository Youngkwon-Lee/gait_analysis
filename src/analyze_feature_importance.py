"""
Phase 2-3: Feature (Channel) Importance Analysis
Identify which channels (Acc_X/Y/Z, Gyr_X/Y/Z) within each sensor are most critical

Goals:
1. Answer the LF anomaly question: Which LF channels drive 0.9523 AUC?
2. Identify most important channels across all sensors
3. Compare Accelerometer vs Gyroscope importance
4. Analyze X/Y/Z axis importance
5. Provide channel-level recommendations for sensor design
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

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration with environment variable support"""
    BASE_PATH = Path(os.environ.get('DATA_PATH', 'D:/gait_wearable_sensor/dataset/data'))
    MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'D:/gait_wearable_sensor/models'))
    OUTPUT_PATH = Path(os.environ.get('OUTPUT_PATH', 'D:/gait_wearable_sensor/results/feature_importance'))

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

    def forward(self, x, channel_mask=None):
        # x: (batch, num_sensors, in_channels, time)
        # channel_mask: (batch, num_sensors, in_channels) - 1 for keep, 0 for ablate
        batch_size = x.shape[0]
        sensor_features = []

        for i, stream in enumerate(self.streams):
            sensor_x = x[:, i]  # (batch, channels, time)

            # Apply channel mask if provided (ablation)
            if channel_mask is not None:
                mask = channel_mask[:, i]  # (batch, in_channels)
                # Expand mask to match time dimension
                mask = mask.unsqueeze(-1)  # (batch, in_channels, 1)
                sensor_x = sensor_x * mask

            sensor_feat = stream(sensor_x)  # (batch, hidden_dim)
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
# Feature (Channel) Importance Analyzer
# ============================================================================

class FeatureImportanceAnalyzer:
    """Analyze channel importance via ablation study"""

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

    def evaluate_with_channel_mask(self, channel_mask):
        """
        Evaluate model performance with specific channels ablated

        Args:
            channel_mask: (4, 6) numpy array - 1 for keep, 0 for ablate
                         Shape: [num_sensors, num_channels]

        Returns:
            dict with predictions, probabilities, labels, metrics
        """
        all_preds = []
        all_probs = []
        all_labels = []

        # Create mask tensor: (4, 6)
        mask = torch.tensor(channel_mask, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for inputs, labels in self.loader:
                inputs = inputs.to(self.device)
                labels = labels.cpu().numpy()

                # Expand mask for batch
                batch_mask = mask.unsqueeze(0).expand(inputs.shape[0], -1, -1)

                # Forward with channel mask
                outputs = self.model(inputs, channel_mask=batch_mask)
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
        """Run complete channel ablation study"""
        print("\n" + "="*80)
        print("FEATURE (CHANNEL) IMPORTANCE ANALYSIS - ABLATION STUDY")
        print("="*80)

        results = {}
        total_tests = 1 + 24 + 2 + 8  # baseline + leave-one-out + sensor_types + per-sensor

        # 1. All channels (baseline)
        test_num = 1
        print(f"\n[{test_num}/{total_tests}] All 24 channels (baseline)...")
        mask_all = np.ones((4, 6))
        results['all'] = self.evaluate_with_channel_mask(mask_all)
        print(f"  AUC: {results['all']['auc']:.4f}, Acc: {results['all']['accuracy']:.4f}")

        # 2. Leave-one-channel-out (24 tests)
        print(f"\n[{test_num+1}-{test_num+24}] Leave-one-channel-out (24 tests)...")
        test_num += 1
        for sensor_idx, sensor in enumerate(Config.SENSORS):
            for channel_idx, channel in enumerate(Config.CHANNELS):
                mask = np.ones((4, 6))
                mask[sensor_idx, channel_idx] = 0

                key = f'without_{sensor}_{channel}'
                results[key] = self.evaluate_with_channel_mask(mask)

                print(f"  [{test_num}/{total_tests}] Remove {sensor}_{channel}: " +
                      f"AUC: {results[key]['auc']:.4f}")
                test_num += 1

        # 3. Accelerometer-only vs Gyroscope-only
        print(f"\n[{test_num}-{test_num+1}] Sensor type analysis...")

        # Acc-only (channels 0,1,2)
        mask_acc = np.zeros((4, 6))
        mask_acc[:, 0:3] = 1
        results['acc_only'] = self.evaluate_with_channel_mask(mask_acc)
        print(f"  [{test_num}/{total_tests}] Accelerometer only: " +
              f"AUC: {results['acc_only']['auc']:.4f}")
        test_num += 1

        # Gyr-only (channels 3,4,5)
        mask_gyr = np.zeros((4, 6))
        mask_gyr[:, 3:6] = 1
        results['gyr_only'] = self.evaluate_with_channel_mask(mask_gyr)
        print(f"  [{test_num}/{total_tests}] Gyroscope only: " +
              f"AUC: {results['gyr_only']['auc']:.4f}")
        test_num += 1

        # 4. Per-sensor channel analysis (focus on LF and HE)
        print(f"\n[{test_num}-{total_tests}] Per-sensor channel analysis...")

        for sensor_idx, sensor in enumerate(Config.SENSORS):
            # Sensor Acc-only
            mask = np.zeros((4, 6))
            mask[sensor_idx, 0:3] = 1
            key = f'{sensor}_acc_only'
            results[key] = self.evaluate_with_channel_mask(mask)
            print(f"  [{test_num}/{total_tests}] {sensor} Accelerometer only: " +
                  f"AUC: {results[key]['auc']:.4f}")
            test_num += 1

            # Sensor Gyr-only
            mask = np.zeros((4, 6))
            mask[sensor_idx, 3:6] = 1
            key = f'{sensor}_gyr_only'
            results[key] = self.evaluate_with_channel_mask(mask)
            print(f"  [{test_num}/{total_tests}] {sensor} Gyroscope only: " +
                  f"AUC: {results[key]['auc']:.4f}")
            test_num += 1

        return results

    def calculate_channel_importance(self, results):
        """Calculate importance score for each channel"""
        baseline_auc = results['all']['auc']

        importance_scores = {}

        for sensor_idx, sensor in enumerate(Config.SENSORS):
            for channel_idx, channel in enumerate(Config.CHANNELS):
                key = f'without_{sensor}_{channel}'
                without_auc = results[key]['auc']
                importance = baseline_auc - without_auc

                channel_name = f'{sensor}_{channel}'
                importance_scores[channel_name] = {
                    'sensor': sensor,
                    'channel': channel,
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
            'channel_importance': importance_scores,
            'sensor_type_comparison': {
                'acc_only': {
                    'auc': float(results['acc_only']['auc']),
                    'accuracy': float(results['acc_only']['accuracy'])
                },
                'gyr_only': {
                    'auc': float(results['gyr_only']['auc']),
                    'accuracy': float(results['gyr_only']['accuracy'])
                }
            }
        }

        # Convert results to serializable format
        for key, val in results.items():
            output['ablation_results'][key] = {
                'accuracy': float(val['accuracy']),
                'balanced_accuracy': float(val['balanced_accuracy']),
                'auc': float(val['auc'])
            }

        output_file = Config.OUTPUT_PATH / f'{self.task_name}_feature_importance.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n[OK] Results saved: {output_file}")

    def visualize_results(self, results, importance_scores):
        """Create comprehensive visualizations"""
        print("\n[Visualization] Creating plots...")

        fig = plt.figure(figsize=(20, 14))

        # 1. Channel importance heatmap (4×6)
        ax1 = plt.subplot(3, 3, 1)
        importance_matrix = np.zeros((4, 6))
        for channel_name, scores in importance_scores.items():
            sensor = scores['sensor']
            channel = scores['channel']
            sensor_idx = Config.SENSORS.index(sensor)
            channel_idx = Config.CHANNELS.index(channel)
            importance_matrix[sensor_idx, channel_idx] = scores['importance']

        sns.heatmap(importance_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=Config.CHANNELS, yticklabels=Config.SENSORS,
                   cbar_kws={'label': 'Importance (AUC drop)'}, ax=ax1)
        ax1.set_title('Channel Importance Heatmap', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Channel', fontsize=12)
        ax1.set_ylabel('Sensor', fontsize=12)

        # 2. Top 10 most important channels
        ax2 = plt.subplot(3, 3, 2)
        sorted_channels = sorted(importance_scores.items(),
                                key=lambda x: x[1]['importance'],
                                reverse=True)[:10]
        channel_names = [x[0] for x in sorted_channels]
        channel_vals = [x[1]['importance'] for x in sorted_channels]

        bars = ax2.barh(range(len(channel_names)), channel_vals, alpha=0.7)
        ax2.set_yticks(range(len(channel_names)))
        ax2.set_yticklabels(channel_names, fontsize=9)
        ax2.set_xlabel('Importance (AUC drop)', fontsize=12)
        ax2.set_title('Top 10 Most Important Channels', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()

        # Color gradient
        cmap = plt.cm.Reds
        norm = plt.Normalize(vmin=min(channel_vals), vmax=max(channel_vals))
        for bar, val in zip(bars, channel_vals):
            bar.set_color(cmap(norm(val)))

        # 3. Accelerometer vs Gyroscope comparison
        ax3 = plt.subplot(3, 3, 3)
        sensor_types = ['All\nChannels', 'Acc\nOnly', 'Gyr\nOnly']
        aucs = [
            results['all']['auc'],
            results['acc_only']['auc'],
            results['gyr_only']['auc']
        ]
        colors = ['green', 'blue', 'orange']
        bars = ax3.bar(sensor_types, aucs, color=colors, alpha=0.7)
        ax3.set_ylabel('AUC', fontsize=12)
        ax3.set_title('Accelerometer vs Gyroscope', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0.85, 1.0])

        # Annotate
        for bar, val in zip(bars, aucs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 4. LF channel breakdown (answer the anomaly question!)
        ax4 = plt.subplot(3, 3, 4)
        lf_channels = [f'LF_{ch}' for ch in Config.CHANNELS]
        lf_importance = [importance_scores[ch]['importance'] for ch in lf_channels]

        bars = ax4.bar(Config.CHANNELS, lf_importance, color='steelblue', alpha=0.7)
        ax4.set_xlabel('Channel', fontsize=12)
        ax4.set_ylabel('Importance (AUC drop)', fontsize=12)
        ax4.set_title('LF Sensor Channel Breakdown', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # Annotate
        for bar, val in zip(bars, lf_importance):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9)

        # 5. HE channel breakdown
        ax5 = plt.subplot(3, 3, 5)
        he_channels = [f'HE_{ch}' for ch in Config.CHANNELS]
        he_importance = [importance_scores[ch]['importance'] for ch in he_channels]

        bars = ax5.bar(Config.CHANNELS, he_importance, color='coral', alpha=0.7)
        ax5.set_xlabel('Channel', fontsize=12)
        ax5.set_ylabel('Importance (AUC drop)', fontsize=12)
        ax5.set_title('HE Sensor Channel Breakdown', fontsize=14, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

        # Annotate
        for bar, val in zip(bars, he_importance):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9)

        # 6. Per-sensor Acc vs Gyr
        ax6 = plt.subplot(3, 3, 6)
        x = np.arange(len(Config.SENSORS))
        width = 0.35

        acc_aucs = [results[f'{s}_acc_only']['auc'] for s in Config.SENSORS]
        gyr_aucs = [results[f'{s}_gyr_only']['auc'] for s in Config.SENSORS]

        ax6.bar(x - width/2, acc_aucs, width, label='Accelerometer', alpha=0.7)
        ax6.bar(x + width/2, gyr_aucs, width, label='Gyroscope', alpha=0.7)
        ax6.set_xlabel('Sensor', fontsize=12)
        ax6.set_ylabel('AUC', fontsize=12)
        ax6.set_title('Per-Sensor: Acc vs Gyr', fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(Config.SENSORS)
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)

        # 7. Axis importance (X, Y, Z) - aggregate across sensors
        ax7 = plt.subplot(3, 3, 7)
        axis_importance = {'Acc_X': 0, 'Acc_Y': 0, 'Acc_Z': 0,
                          'Gyr_X': 0, 'Gyr_Y': 0, 'Gyr_Z': 0}

        for channel_name, scores in importance_scores.items():
            channel = scores['channel']
            axis_importance[channel] += scores['importance']

        # Average across 4 sensors
        axis_importance = {k: v/4 for k, v in axis_importance.items()}

        colors_axis = ['blue', 'blue', 'blue', 'orange', 'orange', 'orange']
        bars = ax7.bar(axis_importance.keys(), axis_importance.values(),
                      color=colors_axis, alpha=0.7)
        ax7.set_xlabel('Channel', fontsize=12)
        ax7.set_ylabel('Average Importance', fontsize=12)
        ax7.set_title('Axis Importance (Averaged)', fontsize=14, fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

        # 8. LF vs HE importance comparison
        ax8 = plt.subplot(3, 3, 8)
        lf_total = sum(lf_importance)
        he_total = sum(he_importance)

        comparison = ['LF Total', 'HE Total']
        totals = [lf_total, he_total]
        bars = ax8.bar(comparison, totals, color=['steelblue', 'coral'], alpha=0.7)
        ax8.set_ylabel('Total Importance', fontsize=12)
        ax8.set_title('LF vs HE: Total Channel Importance', fontsize=14, fontweight='bold')
        ax8.grid(axis='y', alpha=0.3)

        # Annotate
        for bar, val in zip(bars, totals):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 9. Summary table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_data = []

        # Top 5 channels
        top5 = sorted(importance_scores.items(),
                     key=lambda x: x[1]['importance'],
                     reverse=True)[:5]

        summary_data.append(['=== Top 5 Channels ===', '', ''])
        for i, (name, scores) in enumerate(top5, 1):
            summary_data.append([
                f"{i}. {name}",
                f"{scores['importance']:.4f}",
                f"{scores['relative_drop']:.2f}%"
            ])

        summary_data.append(['', '', ''])
        summary_data.append(['=== Sensor Types ===', 'AUC', ''])
        summary_data.append(['All channels', f"{results['all']['auc']:.4f}", ''])
        summary_data.append(['Acc only', f"{results['acc_only']['auc']:.4f}", ''])
        summary_data.append(['Gyr only', f"{results['gyr_only']['auc']:.4f}", ''])

        table = ax9.table(
            cellText=summary_data,
            colLabels=['Channel', 'Importance', 'Drop %'],
            cellLoc='left',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax9.set_title('Summary', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        output_file = Config.OUTPUT_PATH / f'{self.task_name}_feature_importance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Visualization saved: {output_file}")

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("Phase 2-3: Feature (Channel) Importance Analysis")
    print("="*80)

    # Initialize analyzer
    model_path = Config.MODEL_PATH / 'OA_Screening_best.pth'
    analyzer = FeatureImportanceAnalyzer(model_path, task_name='OA_Screening')

    # Run ablation study
    results = analyzer.run_ablation_study()

    # Calculate importance scores
    importance_scores = analyzer.calculate_channel_importance(results)

    # Print summary
    print("\n" + "="*80)
    print("FEATURE (CHANNEL) IMPORTANCE SUMMARY")
    print("="*80)
    print(f"\nBaseline (All 24 channels): AUC = {results['all']['auc']:.4f}")

    print("\n=== Top 10 Most Important Channels ===")
    sorted_channels = sorted(importance_scores.items(),
                            key=lambda x: x[1]['importance'],
                            reverse=True)[:10]
    for rank, (channel, scores) in enumerate(sorted_channels, 1):
        print(f"  {rank:2d}. {channel:15s}: " +
              f"Importance = {scores['importance']:.4f} " +
              f"({scores['relative_drop']:.2f}% drop)")

    print("\n=== Sensor Type Comparison ===")
    print(f"  Accelerometer only: AUC = {results['acc_only']['auc']:.4f}")
    print(f"  Gyroscope only:     AUC = {results['gyr_only']['auc']:.4f}")

    print("\n=== LF Anomaly Investigation ===")
    print("  LF channel breakdown:")
    lf_channels = [f'LF_{ch}' for ch in Config.CHANNELS]
    for ch in lf_channels:
        print(f"    {ch:12s}: Importance = {importance_scores[ch]['importance']:.4f}")

    # Save results
    analyzer.save_results(results, importance_scores)

    # Create visualizations
    analyzer.visualize_results(results, importance_scores)

    print("\n" + "="*80)
    print("[DONE] Feature Importance Analysis Complete!")
    print(f"Output directory: {Config.OUTPUT_PATH}")
    print("="*80)

if __name__ == '__main__':
    main()
