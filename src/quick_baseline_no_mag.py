"""
Quick Baseline Test WITHOUT Magnetometer
- Excludes Mag_X, Mag_Y, Mag_Z to check for data leakage
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("Starting Quick Baseline Test (NO MAGNETOMETER)...")

BASE_PATH = Path(r"D:\gait_wearable_sensor\dataset\data")
OUTPUT_PATH = Path(r"D:\gait_wearable_sensor\results")
OUTPUT_PATH.mkdir(exist_ok=True)

SENSORS = ['HE', 'LB', 'LF', 'RF']
# EXCLUDE MAGNETOMETER - only use Accelerometer and Gyroscope
CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

def load_trial_features(trial_path: Path):
    """Extract handcrafted features from a trial (NO MAGNETOMETER)"""
    trial_name = trial_path.name

    # Load metadata
    meta_file = trial_path / f"{trial_name}_meta.json"
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    features = {}

    # Load and extract features from each sensor
    for sensor in SENSORS:
        sensor_file = trial_path / f"{trial_name}_raw_data_{sensor}.txt"
        if not sensor_file.exists():
            return None, None

        df = pd.read_csv(sensor_file, sep='\t')

        for channel in CHANNELS:
            data = df[channel].values
            prefix = f"{sensor}_{channel}"

            # Statistical features
            features[f"{prefix}_mean"] = np.mean(data)
            features[f"{prefix}_std"] = np.std(data)
            features[f"{prefix}_min"] = np.min(data)
            features[f"{prefix}_max"] = np.max(data)
            features[f"{prefix}_range"] = np.max(data) - np.min(data)
            features[f"{prefix}_rms"] = np.sqrt(np.mean(data**2))

            # Percentiles
            features[f"{prefix}_p25"] = np.percentile(data, 25)
            features[f"{prefix}_p75"] = np.percentile(data, 75)
            features[f"{prefix}_iqr"] = features[f"{prefix}_p75"] - features[f"{prefix}_p25"]

    # Gait-specific features from metadata
    left_events = meta.get('leftGaitEvents', [])
    if len(left_events) > 1:
        stride_times = [(left_events[i+1][0] - left_events[i][0]) / 100
                        for i in range(len(left_events)-1)]
        features['stride_time_mean'] = np.mean(stride_times)
        features['stride_time_std'] = np.std(stride_times)
        features['stride_time_cv'] = np.std(stride_times) / np.mean(stride_times) if np.mean(stride_times) > 0 else 0
        features['cadence'] = 60 / np.mean(stride_times) if np.mean(stride_times) > 0 else 0
        features['num_strides'] = len(left_events)
    else:
        features['stride_time_mean'] = 0
        features['stride_time_std'] = 0
        features['stride_time_cv'] = 0
        features['cadence'] = 0
        features['num_strides'] = 0

    return features, meta

def load_cohort_features(cohort: str, group: str):
    """Load features from all trials in a cohort"""
    cohort_path = BASE_PATH / group / cohort

    if not cohort_path.exists():
        print(f"Warning: {cohort_path} does not exist")
        return [], [], []

    all_features = []
    all_subjects = []

    meta_files = list(cohort_path.rglob("*_meta.json"))

    for meta_file in meta_files:
        trial_path = meta_file.parent
        features, meta = load_trial_features(trial_path)

        if features is not None:
            all_features.append(features)
            all_subjects.append(meta['subject'])

    return all_features, all_subjects

def run_binary_classification(cohort1, group1, cohort2, group2, task_name):
    """Run binary classification with Random Forest"""

    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"{'='*50}")

    # Load features
    print(f"Loading {cohort1}...")
    feat1, subj1 = load_cohort_features(cohort1, group1)
    print(f"  {cohort1}: {len(feat1)} trials")

    print(f"Loading {cohort2}...")
    feat2, subj2 = load_cohort_features(cohort2, group2)
    print(f"  {cohort2}: {len(feat2)} trials")

    # Convert to DataFrame
    df1 = pd.DataFrame(feat1)
    df1['label'] = 0
    df1['subject'] = subj1

    df2 = pd.DataFrame(feat2)
    df2['label'] = 1
    df2['subject'] = subj2

    df = pd.concat([df1, df2], ignore_index=True)

    # Handle missing values
    df = df.fillna(0)

    # Subject-wise split
    unique_subjects = df['subject'].unique()
    subject_labels = df.groupby('subject')['label'].first().to_dict()

    subj_y = np.array([subject_labels[s] for s in unique_subjects])

    subj_train, subj_test = train_test_split(
        unique_subjects, test_size=0.2, stratify=subj_y, random_state=42
    )

    train_mask = df['subject'].isin(subj_train)
    test_mask = df['subject'].isin(subj_test)

    feature_cols = [c for c in df.columns if c not in ['label', 'subject']]

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, 'label'].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, 'label'].values

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Train subjects: {len(subj_train)}, Test subjects: {len(subj_test)}")
    print(f"Feature count: {len(feature_cols)} (NO MAGNETOMETER)")

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    print("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n--- Results ---")
    print(f"ROC-AUC:      {roc_auc:.3f}")
    print(f"Balanced Acc: {bal_acc:.3f}")
    print(f"Sensitivity:  {sensitivity:.3f}")
    print(f"Specificity:  {specificity:.3f}")

    # Feature importance (top 10)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return {
        'task': task_name,
        'roc_auc': roc_auc,
        'balanced_acc': bal_acc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def main():
    print("="*60)
    print("Quick Baseline: Random Forest (NO MAGNETOMETER)")
    print("="*60)

    results = []

    # Task 1: PD Screening
    r1 = run_binary_classification('HS', 'healthy', 'PD', 'neuro', 'PD_Screening')
    results.append(r1)

    # Task 2: OA Screening (HOA vs Healthy)
    r2 = run_binary_classification('HS', 'healthy', 'HOA', 'ortho', 'OA_Screening')
    results.append(r2)

    # Task 3: CVA Detection
    r3 = run_binary_classification('HS', 'healthy', 'CVA', 'neuro', 'CVA_Detection')
    results.append(r3)

    # Task 4: PD vs CVA
    r4 = run_binary_classification('PD', 'neuro', 'CVA', 'neuro', 'PD_vs_CVA')
    results.append(r4)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (NO MAGNETOMETER)")
    print("="*60)

    # Baseline paper results for comparison
    baseline = {
        'PD_Screening': {'roc_auc': 0.821, 'balanced_acc': 0.639},
        'OA_Screening': {'roc_auc': 0.990, 'balanced_acc': 0.942},
        'CVA_Detection': {'roc_auc': 0.950, 'balanced_acc': 0.747},
        'PD_vs_CVA': {'roc_auc': 0.657, 'balanced_acc': 0.607}
    }

    print(f"\n{'Task':<20} {'ROC-AUC':<20} {'Balanced Acc':<20}")
    print(f"{'':<20} {'Ours / Paper':<20} {'Ours / Paper':<20}")
    print("-"*60)

    for r in results:
        task = r['task']
        our_auc = r['roc_auc']
        our_acc = r['balanced_acc']
        paper_auc = baseline[task]['roc_auc']
        paper_acc = baseline[task]['balanced_acc']

        print(f"{task:<20} {our_auc:.3f} / {paper_auc:.3f}      {our_acc:.3f} / {paper_acc:.3f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH / 'quick_baseline_no_mag_results.csv', index=False)
    print(f"\nResults saved to: {OUTPUT_PATH / 'quick_baseline_no_mag_results.csv'}")

if __name__ == "__main__":
    main()
