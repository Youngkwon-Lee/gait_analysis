"""
Data-based Feature Importance Analysis
Statistical analysis of sensor and channel importance without trained model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

# Configuration
BASE_PATH = Path("D:/gait_wearable_sensor/dataset/data")
OUTPUT_DIR = Path("D:/gait_wearable_sensor/results/data_statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = ['HE', 'LB', 'LF', 'RF']
CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']

TASKS = {
    'OA_Screening': {
        'class0': [('HS', 'healthy')],
        'class1': [('HOA', 'ortho'), ('KOA', 'ortho')]
    },
    'PD_Screening': {
        'class0': [('HS', 'healthy')],
        'class1': [('PD', 'neuro')]
    },
    'CVA_Detection': {
        'class0': [('HS', 'healthy')],
        'class1': [('CVA', 'neuro')]
    }
}


def load_trial_data(trial_path):
    """Load all sensor data from a trial"""
    trial_name = trial_path.name
    trial_data = {}

    try:
        for sensor in SENSORS:
            sensor_file = trial_path / f"{trial_name}_raw_data_{sensor}.txt"
            if not sensor_file.exists():
                return None

            df = pd.read_csv(sensor_file, sep='\t')
            trial_data[sensor] = df[CHANNELS].values

        return trial_data

    except Exception as e:
        print(f"Warning: Failed to load {trial_path}: {e}")
        return None


def extract_statistical_features(data):
    """Extract statistical features from time series data"""
    features = {}

    # Basic statistics
    features['mean'] = np.mean(data, axis=0)
    features['std'] = np.std(data, axis=0)
    features['min'] = np.min(data, axis=0)
    features['max'] = np.max(data, axis=0)
    features['range'] = features['max'] - features['min']

    # Percentiles
    features['q25'] = np.percentile(data, 25, axis=0)
    features['q50'] = np.percentile(data, 50, axis=0)
    features['q75'] = np.percentile(data, 75, axis=0)

    # Higher-order moments
    features['skewness'] = stats.skew(data, axis=0)
    features['kurtosis'] = stats.kurtosis(data, axis=0)

    # Energy
    features['rms'] = np.sqrt(np.mean(data**2, axis=0))
    features['energy'] = np.sum(data**2, axis=0)

    return features


def get_trial_paths(cohorts_list, group):
    """Get trial paths for given cohorts"""
    all_trials = []

    for cohort in cohorts_list:
        cohort_path = BASE_PATH / group / cohort
        if not cohort_path.exists():
            print(f"Warning: {cohort_path} not found")
            continue

        for meta_file in cohort_path.rglob("*_meta.json"):
            trial_path = meta_file.parent
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            all_trials.append((trial_path, meta.get('subject_code')))

    return all_trials


def analyze_task(task_name):
    """Analyze statistical differences for a task"""
    print("="*80)
    print(f"Data Statistics Analysis: {task_name}")
    print("="*80)

    task_config = TASKS[task_name]

    # Load class 0 data
    class0_cohorts = [c[0] for c in task_config['class0']]
    class0_group = task_config['class0'][0][1]
    class0_trials = get_trial_paths(class0_cohorts, class0_group)
    print(f"\nClass 0: {class0_cohorts} - {len(class0_trials)} trials")

    # Load class 1 data
    class1_cohorts = [c[0] for c in task_config['class1']]
    class1_group = task_config['class1'][0][1]
    class1_trials = get_trial_paths(class1_cohorts, class1_group)
    print(f"Class 1: {class1_cohorts} - {len(class1_trials)} trials")

    # Extract features
    print("\nExtracting statistical features...")

    class0_features = {sensor: [] for sensor in SENSORS}
    class1_features = {sensor: [] for sensor in SENSORS}

    # Process class 0
    for trial_path, subject in class0_trials[:50]:  # Limit to 50 for speed
        trial_data = load_trial_data(trial_path)
        if trial_data is None:
            continue

        for sensor in SENSORS:
            features = extract_statistical_features(trial_data[sensor])
            # Flatten all features into vector
            feature_vec = np.concatenate([features[k].flatten() for k in
                                         ['mean', 'std', 'range', 'rms', 'skewness', 'kurtosis']])
            class0_features[sensor].append(feature_vec)

    # Process class 1
    for trial_path, subject in class1_trials[:50]:  # Limit to 50 for speed
        trial_data = load_trial_data(trial_path)
        if trial_data is None:
            continue

        for sensor in SENSORS:
            features = extract_statistical_features(trial_data[sensor])
            feature_vec = np.concatenate([features[k].flatten() for k in
                                         ['mean', 'std', 'range', 'rms', 'skewness', 'kurtosis']])
            class1_features[sensor].append(feature_vec)

    # Convert to arrays
    for sensor in SENSORS:
        class0_features[sensor] = np.array(class0_features[sensor])
        class1_features[sensor] = np.array(class1_features[sensor])

    print(f"Processed {len(class0_features['HE'])} class 0 trials, {len(class1_features['HE'])} class 1 trials")

    # Statistical testing (t-test) for each sensor
    print("\nStatistical Tests (per sensor):")
    sensor_importance = {}

    for sensor in SENSORS:
        if len(class0_features[sensor]) == 0 or len(class1_features[sensor]) == 0:
            continue

        # Mean difference across all features
        mean_diff = np.abs(class0_features[sensor].mean(axis=0) - class1_features[sensor].mean(axis=0)).mean()

        # T-test for each feature dimension
        t_stats = []
        p_values = []
        for i in range(class0_features[sensor].shape[1]):
            t, p = stats.ttest_ind(class0_features[sensor][:, i],
                                   class1_features[sensor][:, i])
            t_stats.append(abs(t))
            p_values.append(p)

        mean_t = np.mean(t_stats)
        mean_p = np.mean(p_values)
        significant = np.sum(np.array(p_values) < 0.05)

        sensor_importance[sensor] = {
            'mean_diff': float(mean_diff),
            'mean_t_stat': float(mean_t),
            'mean_p_value': float(mean_p),
            'num_significant': int(significant),
            'total_features': len(p_values)
        }

        print(f"  {sensor}:")
        print(f"    Mean difference: {mean_diff:.4f}")
        print(f"    Mean |t-stat|: {mean_t:.2f}")
        print(f"    Significant features: {significant}/{len(p_values)} (p<0.05)")

    # Random Forest Feature Importance
    print("\nRandom Forest Feature Importance:")
    rf_importance = {}

    for sensor in SENSORS:
        if len(class0_features[sensor]) == 0 or len(class1_features[sensor]) == 0:
            continue

        # Prepare data
        X = np.vstack([class0_features[sensor], class1_features[sensor]])
        y = np.array([0]*len(class0_features[sensor]) + [1]*len(class1_features[sensor]))

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train RF
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_scaled, y)

        accuracy = rf.score(X_scaled, y)
        feature_importances = rf.feature_importances_

        rf_importance[sensor] = {
            'accuracy': float(accuracy),
            'mean_importance': float(feature_importances.mean()),
            'max_importance': float(feature_importances.max())
        }

        print(f"  {sensor}:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Mean feature importance: {feature_importances.mean():.4f}")

    # Channel-wise analysis
    print("\nChannel-wise Analysis:")
    channel_importance = analyze_channels_importance(class0_trials[:30], class1_trials[:30])

    # Save results
    results = {
        'task': task_name,
        'sensor_statistics': sensor_importance,
        'sensor_rf': rf_importance,
        'channel_importance': channel_importance
    }

    json_file = OUTPUT_DIR / f'{task_name}_data_statistics.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved: {json_file}")

    # Plot results
    plot_statistics_results(results, task_name)

    return results


def analyze_channels_importance(class0_trials, class1_trials):
    """Analyze importance of individual channels across all sensors"""
    print("  Analyzing channel importance...")

    channel_stats = {ch: {'class0': [], 'class1': []} for ch in CHANNELS}

    # Collect channel data
    for trial_path, subject in class0_trials:
        trial_data = load_trial_data(trial_path)
        if trial_data is None:
            continue

        for sensor in SENSORS:
            data = trial_data[sensor]
            for i, ch in enumerate(CHANNELS):
                # Use RMS as representative metric
                rms = np.sqrt(np.mean(data[:, i]**2))
                channel_stats[ch]['class0'].append(rms)

    for trial_path, subject in class1_trials:
        trial_data = load_trial_data(trial_path)
        if trial_data is None:
            continue

        for sensor in SENSORS:
            data = trial_data[sensor]
            for i, ch in enumerate(CHANNELS):
                rms = np.sqrt(np.mean(data[:, i]**2))
                channel_stats[ch]['class1'].append(rms)

    # Statistical tests
    channel_importance = {}

    for ch in CHANNELS:
        if len(channel_stats[ch]['class0']) == 0 or len(channel_stats[ch]['class1']) == 0:
            continue

        class0_vals = np.array(channel_stats[ch]['class0'])
        class1_vals = np.array(channel_stats[ch]['class1'])

        t, p = stats.ttest_ind(class0_vals, class1_vals)
        effect_size = (class1_vals.mean() - class0_vals.mean()) / np.sqrt(
            (class0_vals.std()**2 + class1_vals.std()**2) / 2
        )

        channel_importance[ch] = {
            't_stat': float(abs(t)),
            'p_value': float(p),
            'effect_size': float(effect_size),
            'class0_mean': float(class0_vals.mean()),
            'class1_mean': float(class1_vals.mean())
        }

        print(f"    {ch}: t={abs(t):.2f}, p={p:.4f}, effect_size={effect_size:.3f}")

    return channel_importance


def plot_statistics_results(results, task_name):
    """Plot statistical analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Data Statistics Analysis - {task_name}', fontsize=14, fontweight='bold')

    sensors = SENSORS
    channels = CHANNELS

    # 1. Sensor Statistical Importance
    ax = axes[0, 0]
    sensor_stats = results['sensor_statistics']
    mean_diffs = [sensor_stats[s]['mean_diff'] for s in sensors]
    bars = ax.bar(sensors, mean_diffs, color='#ff9999', alpha=0.8)
    ax.set_ylabel('Mean Difference', fontweight='bold')
    ax.set_title('Statistical Difference (Mean)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')

    # 2. Sensor RF Accuracy
    ax = axes[0, 1]
    sensor_rf = results['sensor_rf']
    accuracies = [sensor_rf[s]['accuracy'] for s in sensors]
    bars = ax.bar(sensors, accuracies, color='#66b3ff', alpha=0.8)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Random Forest Accuracy (per sensor)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')

    # 3. Channel Effect Size
    ax = axes[1, 0]
    channel_imp = results['channel_importance']
    effect_sizes = [abs(channel_imp[ch]['effect_size']) for ch in channels]
    bars = ax.bar(channels, effect_sizes, color='#99ff99', alpha=0.8)
    ax.set_ylabel('|Effect Size|', fontweight='bold')
    ax.set_title('Channel Discriminative Power', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # 4. Sensor Comparison Heatmap
    ax = axes[1, 1]
    mean_diffs_array = np.array(mean_diffs)
    max_diff = max(mean_diffs) if max(mean_diffs) > 0 else 1.0
    t_stats = [sensor_stats[s]['mean_t_stat'] for s in sensors]
    max_t = max(t_stats) if max(t_stats) > 0 else 1.0

    comparison_data = np.array([
        mean_diffs_array / max_diff,  # Normalize
        accuracies,
        np.array(t_stats) / max_t
    ])

    sns.heatmap(comparison_data, annot=True, fmt='.3f', cmap='YlOrRd',
               xticklabels=sensors,
               yticklabels=['Mean Diff', 'RF Accuracy', 'T-stat'],
               ax=ax, cbar_kws={'label': 'Importance (normalized)'})
    ax.set_title('Sensor Importance Comparison', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{task_name}_data_statistics.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Visualization saved: {OUTPUT_DIR / f'{task_name}_data_statistics.png'}")
    plt.close()


def main():
    """Main analysis"""
    print("Data-based Feature Importance Analysis")
    print("="*80)

    # Analyze all tasks
    for task_name in TASKS.keys():
        results = analyze_task(task_name)
        print()

    print("="*80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
