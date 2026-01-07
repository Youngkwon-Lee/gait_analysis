"""
Baseline Reproduction: Multi-Stream Attention CNN for Gait Classification
Based on: "Attention-Based Sensor Optimization and Automated Dataset Auditing" (arXiv 2511.02047)

Tasks:
1. PD Screening (PD vs Healthy)
2. OA Screening (HOA vs Healthy)
3. CVA Detection (CVA vs Healthy)
4. PD vs CVA (Differential Diagnosis)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_curve, auc, matthews_corrcoef,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BASE_PATH = Path(r"D:\gait_wearable_sensor\dataset\data")
OUTPUT_PATH = Path(r"D:\gait_wearable_sensor\results")
OUTPUT_PATH.mkdir(exist_ok=True)

# Sensor configuration
SENSORS = ['HE', 'LB', 'LF', 'RF']
CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
SAMPLING_FREQ = 100  # Hz

# ====================
# Data Loading
# ====================

def load_trial_data(trial_path: Path, max_length: int = 3000):
    """Load sensor data from a trial directory"""
    trial_name = trial_path.name

    # Load metadata
    meta_file = trial_path / f"{trial_name}_meta.json"
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    # Load sensor data
    sensor_data = {}
    for sensor in SENSORS:
        sensor_file = trial_path / f"{trial_name}_raw_data_{sensor}.txt"
        if sensor_file.exists():
            df = pd.read_csv(sensor_file, sep='\t')
            # Select only the channels we need
            data = df[CHANNELS].values
            sensor_data[sensor] = data

    if len(sensor_data) != 4:
        return None, None

    # Find minimum length across sensors
    min_len = min(len(sensor_data[s]) for s in SENSORS)
    min_len = min(min_len, max_length)

    # Stack sensors: (4, 9, time) - use float32 for memory efficiency
    stacked = np.stack([sensor_data[s][:min_len].T.astype(np.float32) for s in SENSORS], axis=0)

    return stacked, meta

def load_cohort_data(cohort: str, group: str, max_trials: int = None):
    """Load all trials from a cohort"""
    cohort_path = BASE_PATH / group / cohort

    if not cohort_path.exists():
        print(f"Warning: {cohort_path} does not exist")
        return [], []

    all_data = []
    all_meta = []

    # Find all trial directories
    meta_files = list(cohort_path.rglob("*_meta.json"))

    if max_trials:
        meta_files = meta_files[:max_trials]

    for meta_file in meta_files:
        trial_path = meta_file.parent
        data, meta = load_trial_data(trial_path)

        if data is not None:
            all_data.append(data)
            all_meta.append(meta)

    return all_data, all_meta

def prepare_binary_task(cohort1: str, group1: str, cohort2: str, group2: str,
                        window_size: int = 500, stride: int = 250):
    """Prepare data for binary classification task"""

    print(f"\nLoading {cohort1} vs {cohort2}...")

    # Load data
    data1, meta1 = load_cohort_data(cohort1, group1)
    data2, meta2 = load_cohort_data(cohort2, group2)

    print(f"  {cohort1}: {len(data1)} trials")
    print(f"  {cohort2}: {len(data2)} trials")

    # Extract subject IDs
    subjects1 = [m['subject'] for m in meta1]
    subjects2 = [m['subject'] for m in meta2]

    # Create windows with subject tracking
    X1, subj1 = create_windows(data1, subjects1, window_size, stride)
    X2, subj2 = create_windows(data2, subjects2, window_size, stride)

    # Labels
    y1 = np.zeros(len(X1))
    y2 = np.ones(len(X2))

    # Combine
    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    subjects = subj1 + subj2

    print(f"  Total windows: {len(X)} (Class 0: {len(X1)}, Class 1: {len(X2)})")

    return X, y, subjects

def create_windows(data_list, subjects_list, window_size: int, stride: int):
    """Create overlapping windows from variable-length trials"""
    windows = []
    window_subjects = []

    for data, subject in zip(data_list, subjects_list):
        # data shape: (4, 9, time)
        time_len = data.shape[2]

        for start in range(0, time_len - window_size + 1, stride):
            window = data[:, :, start:start + window_size].astype(np.float32)
            windows.append(window)
            window_subjects.append(subject)

    return np.array(windows, dtype=np.float32), window_subjects

def subject_stratified_split(X, y, subjects, test_size=0.2, val_size=0.1):
    """Split data by subject (not by sample) to prevent data leakage"""

    unique_subjects = list(set(subjects))
    subject_labels = {}

    # Get majority label for each subject
    for subj in unique_subjects:
        subj_mask = [s == subj for s in subjects]
        subj_labels = y[subj_mask]
        subject_labels[subj] = int(np.round(np.mean(subj_labels)))

    # Split subjects
    subj_array = np.array(unique_subjects)
    subj_y = np.array([subject_labels[s] for s in unique_subjects])

    # First split: train+val vs test
    subj_trainval, subj_test = train_test_split(
        subj_array, test_size=test_size, stratify=subj_y, random_state=SEED
    )

    # Second split: train vs val
    subj_trainval_y = np.array([subject_labels[s] for s in subj_trainval])
    subj_train, subj_val = train_test_split(
        subj_trainval, test_size=val_size/(1-test_size),
        stratify=subj_trainval_y, random_state=SEED
    )

    # Create masks
    train_mask = np.array([s in subj_train for s in subjects])
    val_mask = np.array([s in subj_val for s in subjects])
    test_mask = np.array([s in subj_test for s in subjects])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Subject split - Train: {len(subj_train)}, Val: {len(subj_val)}, Test: {len(subj_test)}")
    print(f"  Sample split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ====================
# Dataset & DataLoader
# ====================

class GaitDataset(Dataset):
    def __init__(self, X, y, normalize=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

        if normalize:
            # Z-score normalization per sensor per channel
            self.X = self._normalize(self.X)

    def _normalize(self, X):
        # X shape: (N, 4, 9, time)
        mean = X.mean(dim=(0, 3), keepdim=True)
        std = X.std(dim=(0, 3), keepdim=True) + 1e-8
        return (X - mean) / std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====================
# Model Architecture
# ====================

class SensorCNN(nn.Module):
    """1D CNN branch for a single sensor"""
    def __init__(self, in_channels=9, hidden_dim=64):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, channels, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = x.squeeze(-1)

        return x  # (batch, hidden_dim*4)

class MultiStreamAttentionCNN(nn.Module):
    """Multi-stream CNN with sensor attention"""
    def __init__(self, num_sensors=4, in_channels=9, hidden_dim=64, num_classes=1):
        super().__init__()

        self.num_sensors = num_sensors
        feature_dim = hidden_dim * 4  # Output dim of each sensor CNN

        # Create CNN branch for each sensor
        self.sensor_cnns = nn.ModuleList([
            SensorCNN(in_channels, hidden_dim) for _ in range(num_sensors)
        ])

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        # x: (batch, num_sensors, channels, time)
        batch_size = x.size(0)

        # Process each sensor
        sensor_features = []
        for i, cnn in enumerate(self.sensor_cnns):
            sensor_input = x[:, i, :, :]  # (batch, channels, time)
            features = cnn(sensor_input)  # (batch, feature_dim)
            sensor_features.append(features)

        # Stack: (batch, num_sensors, feature_dim)
        sensor_features = torch.stack(sensor_features, dim=1)

        # Compute attention weights
        attn_scores = self.attention(sensor_features)  # (batch, num_sensors, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, num_sensors, 1)

        # Weighted sum
        context = (sensor_features * attn_weights).sum(dim=1)  # (batch, feature_dim)

        # Classification
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights.squeeze(-1)

        return logits

# ====================
# Training
# ====================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X).squeeze()

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_attn = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits, attn = model(X, return_attention=True)
            logits = logits.squeeze()

            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
            all_attn.extend(attn.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels), np.array(all_attn)

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    """Compute comprehensive metrics"""
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {}
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_prob)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    metrics['pr_auc'] = auc(recall, precision)

    metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics

# ====================
# Main Training Loop
# ====================

def run_task(task_name, cohort1, group1, cohort2, group2,
             epochs=100, batch_size=32, lr=1e-3, window_size=500):
    """Run a single binary classification task"""

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    # Prepare data
    X, y, subjects = prepare_binary_task(cohort1, group1, cohort2, group2,
                                          window_size=window_size, stride=window_size//2)

    # Subject-wise split
    X_train, X_val, X_test, y_train, y_val, y_test = subject_stratified_split(
        X, y, subjects, test_size=0.2, val_size=0.1
    )

    # Create datasets
    train_dataset = GaitDataset(X_train, y_train)
    val_dataset = GaitDataset(X_val, y_val)
    test_dataset = GaitDataset(X_test, y_test)

    # Compute class weight for imbalanced data
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(DEVICE)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = MultiStreamAttentionCNN(num_sensors=4, in_channels=9, hidden_dim=64).to(DEVICE)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_auc = 0
    best_model_state = None
    patience = 15
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, DEVICE)

        val_auc = roc_auc_score(val_labels, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_preds, test_labels, test_attn = evaluate(model, test_loader, criterion, DEVICE)

    # Compute final metrics
    metrics = compute_metrics(test_labels, test_preds)

    # Sensor attention weights
    mean_attn = test_attn.mean(axis=0)
    sensor_importance = {SENSORS[i]: mean_attn[i] for i in range(4)}

    print(f"\n--- Test Results ---")
    print(f"ROC-AUC:       {metrics['roc_auc']:.3f}")
    print(f"PR-AUC:        {metrics['pr_auc']:.3f}")
    print(f"Balanced Acc:  {metrics['balanced_acc']:.3f}")
    print(f"MCC:           {metrics['mcc']:.3f}")
    print(f"Sensitivity:   {metrics['sensitivity']:.3f}")
    print(f"Specificity:   {metrics['specificity']:.3f}")
    print(f"\nSensor Importance:")
    for sensor, weight in sorted(sensor_importance.items(), key=lambda x: -x[1]):
        print(f"  {sensor}: {weight*100:.1f}%")

    return {
        'task': task_name,
        'metrics': metrics,
        'sensor_importance': sensor_importance,
        'history': history
    }

def main():
    print("="*60)
    print("Baseline Reproduction: Multi-Stream Attention CNN")
    print("="*60)

    # Configuration
    EPOCHS = 50  # Reduced for faster training
    BATCH_SIZE = 32
    LR = 1e-3
    WINDOW_SIZE = 300  # 3 seconds at 100Hz (reduced for memory)

    results = []

    # Task 1: PD Screening (PD vs Healthy)
    result1 = run_task(
        task_name="PD_Screening",
        cohort1="HS", group1="healthy",
        cohort2="PD", group2="neuro",
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, window_size=WINDOW_SIZE
    )
    results.append(result1)

    # Task 2: OA Screening (HOA vs Healthy)
    result2 = run_task(
        task_name="OA_Screening",
        cohort1="HS", group1="healthy",
        cohort2="HOA", group2="ortho",
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, window_size=WINDOW_SIZE
    )
    results.append(result2)

    # Task 3: CVA Detection (CVA vs Healthy)
    result3 = run_task(
        task_name="CVA_Detection",
        cohort1="HS", group1="healthy",
        cohort2="CVA", group2="neuro",
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, window_size=WINDOW_SIZE
    )
    results.append(result3)

    # Task 4: PD vs CVA (Differential Diagnosis)
    result4 = run_task(
        task_name="PD_vs_CVA",
        cohort1="PD", group1="neuro",
        cohort2="CVA", group2="neuro",
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, window_size=WINDOW_SIZE
    )
    results.append(result4)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Comparison with Baseline Paper")
    print("="*60)

    baseline_results = {
        'PD_Screening': {'roc_auc': 0.821, 'balanced_acc': 0.639},
        'OA_Screening': {'roc_auc': 0.990, 'balanced_acc': 0.942},
        'CVA_Detection': {'roc_auc': 0.950, 'balanced_acc': 0.747},
        'PD_vs_CVA': {'roc_auc': 0.657, 'balanced_acc': 0.607}
    }

    print(f"\n{'Task':<20} {'ROC-AUC':<15} {'Balanced Acc':<15}")
    print(f"{'':<20} {'Ours/Paper':<15} {'Ours/Paper':<15}")
    print("-"*50)

    for result in results:
        task = result['task']
        our_auc = result['metrics']['roc_auc']
        our_acc = result['metrics']['balanced_acc']
        paper_auc = baseline_results[task]['roc_auc']
        paper_acc = baseline_results[task]['balanced_acc']

        print(f"{task:<20} {our_auc:.3f}/{paper_auc:.3f}    {our_acc:.3f}/{paper_acc:.3f}")

    # Save results
    results_df = pd.DataFrame([
        {
            'task': r['task'],
            **r['metrics'],
            **{f'attn_{k}': v for k, v in r['sensor_importance'].items()}
        }
        for r in results
    ])
    results_df.to_csv(OUTPUT_PATH / 'baseline_results.csv', index=False)
    print(f"\nResults saved to: {OUTPUT_PATH / 'baseline_results.csv'}")

if __name__ == "__main__":
    main()
