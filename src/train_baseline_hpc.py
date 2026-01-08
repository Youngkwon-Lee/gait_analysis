"""
Multi-Stream Attention CNN for Gait Classification
HPC Training Script - Based on arXiv 2511.02047
Excludes Magnetometer features to avoid sensor confound

Usage on HPC:
    nohup python -u train_baseline_hpc.py --task PD_Screening > logs/pd.log 2>&1 &
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

# Ensure unbuffered output for HPC logs
sys.stdout.reconfigure(line_buffering=True)

print("="*60)
print("Multi-Stream Attention CNN - HPC Training")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths - adjust for HPC
    BASE_PATH = Path(os.environ.get('DATA_PATH', 'D:/gait_wearable_sensor/dataset/data'))
    OUTPUT_PATH = Path(os.environ.get('OUTPUT_PATH', 'D:/gait_wearable_sensor/results'))
    MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'D:/gait_wearable_sensor/models'))

    # Data
    SENSORS = ['HE', 'LB', 'LF', 'RF']  # 4 IMU sensors
    CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']  # NO Magnetometer
    SAMPLE_RATE = 100  # Hz
    WINDOW_SIZE = 300  # 3 seconds
    STRIDE = 150       # 50% overlap

    # Model
    INPUT_CHANNELS = 6  # 6 channels per sensor (Acc + Gyr)
    NUM_SENSORS = 4
    HIDDEN_DIM = 64
    NUM_HEADS = 4
    DROPOUT = 0.3

    # Training
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001

    # Tasks
    TASKS = {
        'PD_Screening': {'class0': ('HS', 'healthy'), 'class1': ('PD', 'neuro')},
        'OA_Screening': {'class0': ('HS', 'healthy'), 'class1': [('HOA', 'ortho'), ('KOA', 'ortho')]},  # HOA + KOA combined
        'CVA_Detection': {'class0': ('HS', 'healthy'), 'class1': ('CVA', 'neuro')},
        'PD_vs_CVA': {'class0': ('PD', 'neuro'), 'class1': ('CVA', 'neuro')}
    }

    # Paper baseline results
    BASELINE = {
        'PD_Screening': {'roc_auc': 0.821, 'balanced_acc': 0.639},
        'OA_Screening': {'roc_auc': 0.990, 'balanced_acc': 0.942},
        'CVA_Detection': {'roc_auc': 0.950, 'balanced_acc': 0.747},
        'PD_vs_CVA': {'roc_auc': 0.657, 'balanced_acc': 0.607}
    }


# ============================================================================
# Dataset
# ============================================================================

class GaitDataset(Dataset):
    """Gait signal dataset with windowing"""

    def __init__(self, trial_paths, window_size=300, stride=150, augment=False):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.samples = []
        self.labels = []
        self.subjects = []

        for trial_path, label, subject in trial_paths:
            windows = self._extract_windows(trial_path)
            if windows is not None:
                for w in windows:
                    self.samples.append(w)
                    self.labels.append(label)
                    self.subjects.append(subject)

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels)

        print(f"  Dataset: {len(self.samples)} windows from {len(set(self.subjects))} subjects")

    def _extract_windows(self, trial_path):
        """Extract sliding windows from a trial"""
        trial_name = trial_path.name

        try:
            # Load all sensors
            all_data = []
            min_len = float('inf')

            for sensor in Config.SENSORS:
                sensor_file = trial_path / f"{trial_name}_raw_data_{sensor}.txt"
                if not sensor_file.exists():
                    return None

                df = pd.read_csv(sensor_file, sep='\t')
                sensor_data = df[Config.CHANNELS].values.astype(np.float32)
                all_data.append(sensor_data)
                min_len = min(min_len, len(sensor_data))

            # Truncate to minimum length
            all_data = [d[:min_len] for d in all_data]

            # Stack sensors: (time, sensors, channels)
            data = np.stack(all_data, axis=1)  # (time, 4, 6)

            # Extract windows
            windows = []
            for start in range(0, len(data) - self.window_size + 1, self.stride):
                window = data[start:start + self.window_size]  # (window_size, 4, 6)

                # Normalize per window
                mean = window.mean(axis=0, keepdims=True)
                std = window.std(axis=0, keepdims=True) + 1e-8
                window = (window - mean) / std

                windows.append(window)

            return windows if windows else None

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


# ============================================================================
# Model: Multi-Stream Attention CNN
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
# Training Functions
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


def create_dataloaders(task_name):
    """Create train/test dataloaders with subject-wise split"""
    task_config = Config.TASKS[task_name]

    # Load trials for class 0
    class0_config = task_config['class0']
    if isinstance(class0_config, list):
        # Multiple cohorts
        trials0 = []
        for cohort, group in class0_config:
            print(f"\nLoading {cohort} (class 0)...")
            cohort_trials = get_trial_paths(cohort, group)
            trials0.extend(cohort_trials)
            print(f"  Found {len(cohort_trials)} trials")
        print(f"  Total class 0: {len(trials0)} trials")
    else:
        # Single cohort
        cohort0, group0 = class0_config
        print(f"\nLoading {cohort0} (class 0)...")
        trials0 = get_trial_paths(cohort0, group0)
        print(f"  Found {len(trials0)} trials")

    # Load trials for class 1
    class1_config = task_config['class1']
    if isinstance(class1_config, list):
        # Multiple cohorts (e.g., HOA + KOA)
        trials1 = []
        for cohort, group in class1_config:
            print(f"\nLoading {cohort} (class 1)...")
            cohort_trials = get_trial_paths(cohort, group)
            trials1.extend(cohort_trials)
            print(f"  Found {len(cohort_trials)} trials")
        print(f"  Total class 1: {len(trials1)} trials")
    else:
        # Single cohort
        cohort1, group1 = class1_config
        print(f"\nLoading {cohort1} (class 1)...")
        trials1 = get_trial_paths(cohort1, group1)
        print(f"  Found {len(trials1)} trials")

    # Get unique subjects
    subjects0 = list(set([t[1] for t in trials0]))
    subjects1 = list(set([t[1] for t in trials1]))

    # Subject-wise split
    train_subj0, test_subj0 = train_test_split(subjects0, test_size=0.2, random_state=42)
    train_subj1, test_subj1 = train_test_split(subjects1, test_size=0.2, random_state=42)

    # Assign trials to train/test
    train_trials = []
    test_trials = []

    for trial_path, subject in trials0:
        if subject in train_subj0:
            train_trials.append((trial_path, 0, subject))
        else:
            test_trials.append((trial_path, 0, subject))

    for trial_path, subject in trials1:
        if subject in train_subj1:
            train_trials.append((trial_path, 1, subject))
        else:
            test_trials.append((trial_path, 1, subject))

    print(f"\nTrain: {len(train_trials)} trials ({len(train_subj0)+len(train_subj1)} subjects)")
    print(f"Test: {len(test_trials)} trials ({len(test_subj0)+len(test_subj1)} subjects)")

    # Create datasets
    train_dataset = GaitDataset(train_trials, augment=True)
    test_dataset = GaitDataset(test_trials, augment=False)

    # Calculate class weights
    train_labels = train_dataset.labels
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / (n_pos + 1e-8)

    print(f"Class balance - Neg: {n_neg}, Pos: {n_pos}, Weight: {pos_weight:.2f}")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, test_loader, pos_weight


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(x)
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model and return detailed metrics"""
    from sklearn.metrics import roc_curve

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Performance metrics
    roc_auc = roc_auc_score(all_labels, all_preds)
    predictions = (all_preds > 0.5).astype(int)
    bal_acc = balanced_accuracy_score(all_labels, predictions)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

    return {
        'roc_auc': roc_auc,
        'balanced_acc': bal_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        # Confusion matrix
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        # ROC curve (convert to list for JSON serialization)
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        # Raw predictions (for detailed analysis)
        'predictions': all_preds.tolist(),
        'true_labels': all_labels.tolist()
    }


def train_task(task_name, device):
    """Train model for a specific task"""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    # Create dataloaders
    train_loader, test_loader, pos_weight = create_dataloaders(task_name)

    # Create model
    model = MultiStreamAttentionCNN(
        num_sensors=Config.NUM_SENSORS,
        in_channels=Config.INPUT_CHANNELS,
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_auc = 0
    best_metrics = None

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()

        metrics = evaluate(model, test_loader, device)

        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_metrics = metrics.copy()

            # Save best model
            Config.MODEL_PATH.mkdir(exist_ok=True, parents=True)
            model_file = Config.MODEL_PATH / f"{task_name}_best.pt"
            torch.save(model.state_dict(), model_file)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                  f"AUC: {metrics['roc_auc']:.3f} | Acc: {metrics['balanced_acc']:.3f}")

    # Final evaluation
    print(f"\n--- Best Results for {task_name} ---")
    print(f"ROC-AUC:      {best_metrics['roc_auc']:.3f} (Paper: {Config.BASELINE[task_name]['roc_auc']:.3f})")
    print(f"Balanced Acc: {best_metrics['balanced_acc']:.3f} (Paper: {Config.BASELINE[task_name]['balanced_acc']:.3f})")
    print(f"Sensitivity:  {best_metrics['sensitivity']:.3f}")
    print(f"Specificity:  {best_metrics['specificity']:.3f}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'PD_Screening', 'OA_Screening',
                                'CVA_Detection', 'PD_vs_CVA'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("\nWARNING: Running on CPU - training will be slow!")

    # Create output directory
    Config.OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # Run tasks
    all_results = []

    if args.task == 'all':
        tasks = list(Config.TASKS.keys())
    else:
        tasks = [args.task]

    for task in tasks:
        result = train_task(task, device)
        result['task'] = task
        all_results.append(result)

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print(f"\n{'Task':<20} {'ROC-AUC':<20} {'Balanced Acc':<20}")
    print(f"{'':<20} {'Ours / Paper':<20} {'Ours / Paper':<20}")
    print("-"*60)

    for r in all_results:
        task = r['task']
        paper = Config.BASELINE[task]
        print(f"{task:<20} {r['roc_auc']:.3f} / {paper['roc_auc']:.3f}      "
              f"{r['balanced_acc']:.3f} / {paper['balanced_acc']:.3f}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save summary as CSV (excluding large arrays)
    summary_results = []
    for r in all_results:
        summary = {
            'task': r['task'],
            'roc_auc': r['roc_auc'],
            'balanced_acc': r['balanced_acc'],
            'sensitivity': r['sensitivity'],
            'specificity': r['specificity'],
            'tn': r['tn'],
            'fp': r['fp'],
            'fn': r['fn'],
            'tp': r['tp']
        }
        summary_results.append(summary)

    results_df = pd.DataFrame(summary_results)
    csv_file = Config.OUTPUT_PATH / f'dl_baseline_results_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\nSummary saved to: {csv_file}")

    # Save detailed results as JSON (includes ROC curve, predictions)
    json_file = Config.OUTPUT_PATH / f'dl_baseline_detailed_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
