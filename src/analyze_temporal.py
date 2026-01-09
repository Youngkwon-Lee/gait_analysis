"""
Phase 2-1: Temporal Analysis
Analyze temporal patterns within gait windows to identify when errors occur

Goals:
1. Identify which part of 3-second window causes FN/FP errors
2. Analyze gait phase contributions (Heel Strike, Mid-Stance, Toe-Off, Swing)
3. Generate temporal activation patterns
4. Save GT, predictions, and temporal patterns for each window
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration with environment variable support"""
    BASE_PATH = Path(os.environ.get('DATA_PATH', 'D:/gait_wearable_sensor/dataset/data'))
    MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'D:/gait_wearable_sensor/models'))
    OUTPUT_PATH = Path(os.environ.get('OUTPUT_PATH', 'D:/gait_wearable_sensor/results/temporal_analysis'))

    # Create output directory
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # Data parameters
    SENSORS = ['HE', 'LB', 'LF', 'RF']  # 4 IMU sensors
    CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']  # NO Magnetometer
    WINDOW_SIZE = 300  # 3 seconds @ 100Hz
    OVERLAP = 150  # 50% overlap
    SEED = 42

    # Temporal analysis parameters
    SUB_WINDOW_SIZE = 50  # 0.5 seconds @ 100Hz
    SUB_WINDOW_STRIDE = 25  # 0.25 second stride

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

    def __init__(self, num_sensors=4, in_channels=6, hidden_dim=64, num_classes=1, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_sensors = num_sensors

        # Per-sensor CNN streams
        self.streams = nn.ModuleList([
            SensorStream(in_channels, hidden_dim, dropout)
            for _ in range(num_sensors)
        ])

        # Multi-head self-attention for sensor fusion
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_sensors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, num_sensors, in_channels, time)
        batch_size = x.shape[0]
        sensor_features = []

        for i, stream in enumerate(self.streams):
            sensor_x = x[:, i]  # (batch, channels, time)
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
# Dataset (EXACT COPY from analyze_errors.py)
# ============================================================================

from sklearn.model_selection import train_test_split

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

        # Store metadata for each window
        self.window_metadata = []

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

                # Store metadata
                self.window_metadata.append({
                    'trial_idx': idx,
                    'trial_path': str(trial_path),
                    'subject': subjects[idx] if subjects else 'unknown',
                    'window_start': i,
                    'label': label
                })

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
        metadata = self.window_metadata[idx]

        # Transpose to (sensors, channels, time) for CNN
        x = np.transpose(x, (1, 2, 0))  # (4, 6, window_size)

        if self.augment and np.random.rand() > 0.5:
            # Random time shift
            shift = np.random.randint(-20, 20)
            x = np.roll(x, shift, axis=2)

            # Random noise
            noise = np.random.randn(*x.shape).astype(np.float32) * 0.01
            x = x + noise

        return torch.tensor(x), torch.tensor(y, dtype=torch.float32), metadata


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
# Temporal Analyzer
# ============================================================================

def collate_fn_with_metadata(batch):
    """Custom collate function to handle metadata properly"""
    inputs, labels, metadata = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    # Keep metadata as list of dicts
    return inputs, labels, list(metadata)


class TemporalAnalyzer:
    """Analyze temporal patterns within windows"""

    def __init__(self, model_path, task_name='OA_Screening'):
        self.task_name = task_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn_with_metadata
        )

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns for all windows"""
        print("\n" + "="*80)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*80)

        results = []

        with torch.no_grad():
            for idx, (inputs, labels, metadata) in enumerate(tqdm(self.loader, desc="Analyzing windows")):
                inputs = inputs.to(self.device)
                label = float(labels[0].cpu().numpy())

                # metadata is a list of dicts when batch_size=1
                meta = metadata if isinstance(metadata, dict) else metadata[0]

                # Full window prediction
                outputs = self.model(inputs)
                prob = float(torch.sigmoid(outputs)[0].cpu().numpy())
                pred = int(prob >= 0.5)

                # Temporal sub-window analysis
                temporal_probs = self._analyze_sub_windows(inputs[0])

                # Gait phase analysis
                phase_probs = self._analyze_gait_phases(temporal_probs)

                # Store results
                result = {
                    'window_id': idx,
                    'trial_path': str(meta['trial_path']),
                    'subject': meta['subject'],
                    'window_start': meta['window_start'],
                    'label': int(label),
                    'prediction': pred,
                    'probability': prob,
                    'correct': pred == int(label),
                    'error_type': self._get_error_type(pred, int(label)),
                    'temporal_probabilities': [float(p) for p in temporal_probs],
                    'gait_phase_probabilities': {k: float(v) for k, v in phase_probs.items()},
                    'temporal_variance': float(np.var(temporal_probs)),
                    'temporal_trend': self._calculate_trend(temporal_probs)
                }

                results.append(result)

        return results

    def _analyze_sub_windows(self, window_data):
        """Analyze sub-windows within a 3-second window"""
        # window_data shape: (4, 6, 300)
        sub_window_size = Config.SUB_WINDOW_SIZE
        stride = Config.SUB_WINDOW_STRIDE
        window_length = window_data.shape[2]  # Time dimension is now axis 2

        temporal_probs = []

        for start in range(0, window_length - sub_window_size + 1, stride):
            end = start + sub_window_size

            # Extract sub-window and pad to full size
            sub_window = torch.zeros(4, 6, 300, device=self.device)
            sub_window[:, :, start:end] = window_data[:, :, start:end]

            # Predict
            sub_window = sub_window.unsqueeze(0)  # Add batch dimension -> (1, 4, 6, 300)
            outputs = self.model(sub_window)
            prob = torch.sigmoid(outputs).cpu().numpy()[0]
            temporal_probs.append(prob)

        return temporal_probs

    def _analyze_gait_phases(self, temporal_probs):
        """Analyze probabilities by gait phase"""
        num_sub_windows = len(temporal_probs)

        # Define gait phases as percentage of gait cycle
        phases = {
            'heel_strike': (0, 0.2),      # 0-20%: Initial contact
            'mid_stance': (0.2, 0.5),      # 20-50%: Loading response + mid-stance
            'toe_off': (0.5, 0.6),         # 50-60%: Terminal stance + pre-swing
            'swing': (0.6, 1.0)            # 60-100%: Swing phase
        }

        phase_probs = {}

        for phase_name, (start_pct, end_pct) in phases.items():
            start_idx = int(start_pct * num_sub_windows)
            end_idx = int(end_pct * num_sub_windows)

            if end_idx > start_idx:
                phase_prob = np.mean(temporal_probs[start_idx:end_idx])
            else:
                phase_prob = 0.0

            phase_probs[phase_name] = phase_prob

        return phase_probs

    def _calculate_trend(self, temporal_probs):
        """Calculate temporal trend (increasing, decreasing, stable)"""
        if len(temporal_probs) < 2:
            return 'stable'

        # Linear regression slope
        x = np.arange(len(temporal_probs))
        slope = np.polyfit(x, temporal_probs, 1)[0]

        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    def _get_error_type(self, pred, label):
        """Classify error type"""
        if pred == label:
            return 'TP' if label == 1 else 'TN'
        else:
            return 'FP' if label == 0 else 'FN'

    def visualize_temporal_patterns(self, results):
        """Create comprehensive temporal visualizations"""
        print("\n" + "="*80)
        print("GENERATING TEMPORAL VISUALIZATIONS")
        print("="*80)

        # Filter error cases
        fn_results = [r for r in results if r['error_type'] == 'FN']
        fp_results = [r for r in results if r['error_type'] == 'FP']
        tp_results = [r for r in results if r['error_type'] == 'TP']
        tn_results = [r for r in results if r['error_type'] == 'TN']

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # 1. FN Temporal Patterns
        if fn_results:
            ax1 = fig.add_subplot(gs[0, :])
            for r in fn_results:
                time_points = np.linspace(0, 3, len(r['temporal_probabilities']))
                ax1.plot(time_points, r['temporal_probabilities'],
                        marker='o', alpha=0.7, linewidth=2,
                        label=f"Window {r['window_id']} (prob={r['probability']:.3f})")
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('OA Probability', fontsize=12)
            ax1.set_title('False Negatives: Temporal Probability Evolution', fontweight='bold', fontsize=14)
            ax1.legend(loc='best')
            ax1.grid(alpha=0.3)

        # 2. FP Temporal Patterns
        if fp_results:
            ax2 = fig.add_subplot(gs[1, :])
            for r in fp_results:
                time_points = np.linspace(0, 3, len(r['temporal_probabilities']))
                ax2.plot(time_points, r['temporal_probabilities'],
                        marker='s', alpha=0.7, linewidth=2,
                        label=f"Window {r['window_id']} (prob={r['probability']:.3f})")
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('OA Probability', fontsize=12)
            ax2.set_title('False Positives: Temporal Probability Evolution', fontweight='bold', fontsize=14)
            ax2.legend(loc='best')
            ax2.grid(alpha=0.3)

        # 3. Gait Phase Comparison: FN vs TP
        ax3 = fig.add_subplot(gs[2, 0])
        if fn_results and tp_results:
            phases = ['heel_strike', 'mid_stance', 'toe_off', 'swing']
            fn_phase_avg = {p: np.mean([r['gait_phase_probabilities'][p] for r in fn_results]) for p in phases}
            tp_phase_avg = {p: np.mean([r['gait_phase_probabilities'][p] for r in tp_results[:10]]) for p in phases}

            x = np.arange(len(phases))
            width = 0.35
            ax3.bar(x - width/2, [fn_phase_avg[p] for p in phases], width, label='FN', alpha=0.8, color='red')
            ax3.bar(x + width/2, [tp_phase_avg[p] for p in phases], width, label='TP (sample)', alpha=0.8, color='green')
            ax3.set_xlabel('Gait Phase', fontsize=11)
            ax3.set_ylabel('Average Probability', fontsize=11)
            ax3.set_title('Gait Phase Comparison: FN vs TP', fontweight='bold', fontsize=12)
            ax3.set_xticks(x)
            ax3.set_xticklabels(['Heel\nStrike', 'Mid\nStance', 'Toe\nOff', 'Swing'], fontsize=9)
            ax3.legend()
            ax3.grid(alpha=0.3, axis='y')

        # 4. Gait Phase Comparison: FP vs TN
        ax4 = fig.add_subplot(gs[2, 1])
        if fp_results and tn_results:
            phases = ['heel_strike', 'mid_stance', 'toe_off', 'swing']
            fp_phase_avg = {p: np.mean([r['gait_phase_probabilities'][p] for r in fp_results]) for p in phases}
            tn_phase_avg = {p: np.mean([r['gait_phase_probabilities'][p] for r in tn_results[:10]]) for p in phases}

            x = np.arange(len(phases))
            width = 0.35
            ax4.bar(x - width/2, [fp_phase_avg[p] for p in phases], width, label='FP', alpha=0.8, color='orange')
            ax4.bar(x + width/2, [tn_phase_avg[p] for p in phases], width, label='TN (sample)', alpha=0.8, color='blue')
            ax4.set_xlabel('Gait Phase', fontsize=11)
            ax4.set_ylabel('Average Probability', fontsize=11)
            ax4.set_title('Gait Phase Comparison: FP vs TN', fontweight='bold', fontsize=12)
            ax4.set_xticks(x)
            ax4.set_xticklabels(['Heel\nStrike', 'Mid\nStance', 'Toe\nOff', 'Swing'], fontsize=9)
            ax4.legend()
            ax4.grid(alpha=0.3, axis='y')

        # 5. Temporal Variance Distribution
        ax5 = fig.add_subplot(gs[2, 2])
        variances_by_type = {
            'TP': [r['temporal_variance'] for r in tp_results],
            'TN': [r['temporal_variance'] for r in tn_results],
            'FP': [r['temporal_variance'] for r in fp_results],
            'FN': [r['temporal_variance'] for r in fn_results]
        }
        data_for_box = [v for v in variances_by_type.values() if len(v) > 0]
        labels_for_box = [k for k, v in variances_by_type.items() if len(v) > 0]
        ax5.boxplot(data_for_box, tick_labels=labels_for_box)
        ax5.set_ylabel('Temporal Variance', fontsize=11)
        ax5.set_title('Temporal Variance by Error Type', fontweight='bold', fontsize=12)
        ax5.grid(alpha=0.3, axis='y')

        # 6. Temporal Trend Distribution
        ax6 = fig.add_subplot(gs[3, 0])
        trends_by_type = {}
        for error_type in ['TP', 'TN', 'FP', 'FN']:
            results_type = [r for r in results if r['error_type'] == error_type]
            trend_counts = {
                'increasing': sum(1 for r in results_type if r['temporal_trend'] == 'increasing'),
                'stable': sum(1 for r in results_type if r['temporal_trend'] == 'stable'),
                'decreasing': sum(1 for r in results_type if r['temporal_trend'] == 'decreasing')
            }
            trends_by_type[error_type] = trend_counts

        # Stacked bar chart
        trend_types = ['increasing', 'stable', 'decreasing']
        bottom = np.zeros(len(trends_by_type))
        for trend in trend_types:
            values = [trends_by_type[et][trend] for et in trends_by_type.keys()]
            ax6.bar(list(trends_by_type.keys()), values, bottom=bottom, label=trend, alpha=0.8)
            bottom += values
        ax6.set_ylabel('Count', fontsize=11)
        ax6.set_title('Temporal Trend Distribution', fontweight='bold', fontsize=12)
        ax6.legend(title='Trend')
        ax6.grid(alpha=0.3, axis='y')

        # 7-8. Sample TP and TN patterns
        ax7 = fig.add_subplot(gs[3, 1])
        for r in tp_results[:5]:
            time_points = np.linspace(0, 3, len(r['temporal_probabilities']))
            ax7.plot(time_points, r['temporal_probabilities'], alpha=0.5, linewidth=1.5)
        ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax7.set_xlabel('Time (seconds)', fontsize=11)
        ax7.set_ylabel('OA Probability', fontsize=11)
        ax7.set_title('Sample TP Temporal Patterns', fontweight='bold', fontsize=12)
        ax7.grid(alpha=0.3)

        ax8 = fig.add_subplot(gs[3, 2])
        for r in tn_results[:5]:
            time_points = np.linspace(0, 3, len(r['temporal_probabilities']))
            ax8.plot(time_points, r['temporal_probabilities'], alpha=0.5, linewidth=1.5)
        ax8.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax8.set_xlabel('Time (seconds)', fontsize=11)
        ax8.set_ylabel('OA Probability', fontsize=11)
        ax8.set_title('Sample TN Temporal Patterns', fontweight='bold', fontsize=12)
        ax8.grid(alpha=0.3)

        plt.suptitle('Temporal Pattern Analysis - Phase 2-1',
                    fontsize=18, fontweight='bold', y=0.995)

        output_file = Config.OUTPUT_PATH / f'{self.task_name}_temporal_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Visualization saved: {output_file}")
        plt.close()

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("Phase 2-1: Temporal Analysis")
    print("="*80)

    # Model path
    model_path = Config.MODEL_PATH / 'OA_Screening_best.pth'

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    # Create analyzer
    analyzer = TemporalAnalyzer(model_path, task_name='OA_Screening')

    # Run temporal analysis
    print("\n[1/3] Analyzing temporal patterns...")
    results = analyzer.analyze_temporal_patterns()

    # Save results
    print("\n[2/3] Saving results...")
    output_file = Config.OUTPUT_PATH / 'OA_Screening_temporal_analysis.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total = len(results)
    fn_count = sum(1 for r in results if r['error_type'] == 'FN')
    fp_count = sum(1 for r in results if r['error_type'] == 'FP')

    print(f"\nTotal windows analyzed: {total}")
    print(f"False Negatives: {fn_count}")
    print(f"False Positives: {fp_count}")

    if fn_count > 0:
        print(f"\nFN Windows:")
        for r in results:
            if r['error_type'] == 'FN':
                print(f"  Window {r['window_id']}: {r['cohort']} (prob={r['probability']:.4f})")
                print(f"    Gait phases: HS={r['gait_phase_probabilities']['heel_strike']:.3f}, "
                      f"MS={r['gait_phase_probabilities']['mid_stance']:.3f}, "
                      f"TO={r['gait_phase_probabilities']['toe_off']:.3f}, "
                      f"SW={r['gait_phase_probabilities']['swing']:.3f}")
                print(f"    Temporal trend: {r['temporal_trend']}, variance: {r['temporal_variance']:.6f}")

    if fp_count > 0:
        print(f"\nFP Windows:")
        for r in results:
            if r['error_type'] == 'FP':
                print(f"  Window {r['window_id']}: {r['cohort']} (prob={r['probability']:.4f})")
                print(f"    Gait phases: HS={r['gait_phase_probabilities']['heel_strike']:.3f}, "
                      f"MS={r['gait_phase_probabilities']['mid_stance']:.3f}, "
                      f"TO={r['gait_phase_probabilities']['toe_off']:.3f}, "
                      f"SW={r['gait_phase_probabilities']['swing']:.3f}")
                print(f"    Temporal trend: {r['temporal_trend']}, variance: {r['temporal_variance']:.6f}")

    # Create visualizations
    print("\n[3/3] Creating visualizations...")
    analyzer.visualize_temporal_patterns(results)

    print("\n" + "="*80)
    print("[DONE] Temporal Analysis Complete!")
    print(f"Output directory: {Config.OUTPUT_PATH}")
    print("="*80)

if __name__ == '__main__':
    main()
