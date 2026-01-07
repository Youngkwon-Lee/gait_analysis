"""
Gait Wearable Sensor Dataset - All Cohorts Analysis
Comprehensive visualization and statistics for ML planning
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 9
plt.rcParams['figure.figsize'] = (16, 12)

BASE_PATH = Path(r"D:\gait_wearable_sensor\dataset\data")
OUTPUT_PATH = Path(r"D:\gait_wearable_sensor")

# Cohort definitions
COHORTS = {
    'HS': {'group': 'healthy', 'name': 'Healthy', 'color': '#2ecc71'},
    'PD': {'group': 'neuro', 'name': 'Parkinson Disease', 'color': '#e74c3c'},
    'CVA': {'group': 'neuro', 'name': 'Stroke (CVA)', 'color': '#9b59b6'},
    'CIPN': {'group': 'neuro', 'name': 'Chemo Neuropathy', 'color': '#f39c12'},
    'RIL': {'group': 'neuro', 'name': 'Radiation Leuko.', 'color': '#1abc9c'},
    'HOA': {'group': 'ortho', 'name': 'Hip Osteoarthritis', 'color': '#3498db'},
    'KOA': {'group': 'ortho', 'name': 'Knee Osteoarthritis', 'color': '#34495e'},
    'ACL': {'group': 'ortho', 'name': 'ACL Injury', 'color': '#e67e22'},
}

def load_all_metadata():
    """Load metadata from all trials"""
    all_meta = []

    for cohort_key, cohort_info in COHORTS.items():
        group = cohort_info['group']
        cohort_path = BASE_PATH / group / cohort_key

        if not cohort_path.exists():
            continue

        meta_files = list(cohort_path.rglob("*_meta.json"))

        for meta_file in meta_files:
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                meta['cohort'] = cohort_key
                meta['trial_path'] = str(meta_file.parent)
                all_meta.append(meta)
            except Exception as e:
                print(f"Error loading {meta_file}: {e}")

    return pd.DataFrame(all_meta)

def calculate_gait_features(meta):
    """Calculate gait features from metadata"""
    features = {}

    # Left gait events
    left_events = meta.get('leftGaitEvents', [])
    right_events = meta.get('rightGaitEvents', [])
    freq = meta.get('freq', 100)

    if len(left_events) > 1:
        # Stride times (time between consecutive toe-offs)
        stride_times = [(left_events[i+1][0] - left_events[i][0]) / freq
                        for i in range(len(left_events)-1)]
        features['stride_time_mean'] = np.mean(stride_times)
        features['stride_time_std'] = np.std(stride_times)
        features['stride_time_cv'] = np.std(stride_times) / np.mean(stride_times) * 100 if np.mean(stride_times) > 0 else 0

        # Swing times (time from toe-off to heel-strike)
        swing_times = [(event[1] - event[0]) / freq for event in left_events]
        features['swing_time_mean'] = np.mean(swing_times)
        features['swing_time_std'] = np.std(swing_times)

        # Cadence
        features['cadence'] = 60 / np.mean(stride_times) if np.mean(stride_times) > 0 else 0

        # Number of strides
        features['num_strides'] = len(left_events)

    # Asymmetry (if both left and right events available)
    if len(left_events) > 1 and len(right_events) > 1:
        left_stride = np.mean([(left_events[i+1][0] - left_events[i][0]) / freq
                               for i in range(len(left_events)-1)])
        right_stride = np.mean([(right_events[i+1][0] - right_events[i][0]) / freq
                                for i in range(len(right_events)-1)])
        features['asymmetry'] = abs(left_stride - right_stride) / ((left_stride + right_stride) / 2) * 100

    return features

def plot_cohort_comparison(df, output_path):
    """Create comprehensive cohort comparison plot"""

    fig = plt.figure(figsize=(18, 14))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    cohort_order = ['HS', 'PD', 'CVA', 'CIPN', 'RIL', 'HOA', 'KOA', 'ACL']
    colors = [COHORTS[c]['color'] for c in cohort_order if c in df['cohort'].unique()]
    cohorts_present = [c for c in cohort_order if c in df['cohort'].unique()]

    # 1. Trial count by cohort
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df.groupby('cohort').size().reindex(cohorts_present)
    bars = ax1.bar(range(len(cohorts_present)), counts.values, color=colors)
    ax1.set_xticks(range(len(cohorts_present)))
    ax1.set_xticklabels(cohorts_present, rotation=45)
    ax1.set_ylabel('Number of Trials')
    ax1.set_title('Trials per Cohort')
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontsize=8)

    # 2. Age distribution by cohort
    ax2 = fig.add_subplot(gs[0, 1])
    age_data = [pd.to_numeric(df[df['cohort'] == c]['age'], errors='coerce').dropna().values for c in cohorts_present]
    age_data = [arr if len(arr) > 0 else np.array([0]) for arr in age_data]
    bp = ax2.boxplot(age_data, labels=cohorts_present, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('Age (years)')
    ax2.set_title('Age Distribution')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Gender distribution
    ax3 = fig.add_subplot(gs[0, 2])
    gender_data = df.groupby(['cohort', 'gender']).size().unstack(fill_value=0)
    gender_data = gender_data.reindex(cohorts_present)
    x = np.arange(len(cohorts_present))
    width = 0.35
    ax3.bar(x - width/2, gender_data.get('M', [0]*len(cohorts_present)), width, label='Male', color='#3498db')
    ax3.bar(x + width/2, gender_data.get('F', [0]*len(cohorts_present)), width, label='Female', color='#e91e63')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cohorts_present, rotation=45)
    ax3.set_ylabel('Count')
    ax3.set_title('Gender Distribution')
    ax3.legend()

    # 4. Stride Time by cohort
    ax4 = fig.add_subplot(gs[1, 0])
    stride_data = [pd.to_numeric(df[df['cohort'] == c]['stride_time_mean'], errors='coerce').dropna().values for c in cohorts_present]
    stride_data = [arr if len(arr) > 0 else np.array([0]) for arr in stride_data]
    bp4 = ax4.boxplot(stride_data, labels=cohorts_present, patch_artist=True)
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_ylabel('Stride Time (s)')
    ax4.set_title('Stride Time Distribution')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=1.1, color='green', linestyle='--', alpha=0.5, label='Healthy norm')

    # 5. Cadence by cohort
    ax5 = fig.add_subplot(gs[1, 1])
    cadence_data = [pd.to_numeric(df[df['cohort'] == c]['cadence'], errors='coerce').dropna().values for c in cohorts_present]
    cadence_data = [arr if len(arr) > 0 else np.array([0]) for arr in cadence_data]
    bp5 = ax5.boxplot(cadence_data, labels=cohorts_present, patch_artist=True)
    for patch, color in zip(bp5['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax5.set_ylabel('Cadence (steps/min)')
    ax5.set_title('Cadence Distribution')
    ax5.tick_params(axis='x', rotation=45)

    # 6. Stride Time Variability (CV)
    ax6 = fig.add_subplot(gs[1, 2])
    cv_data = [pd.to_numeric(df[df['cohort'] == c]['stride_time_cv'], errors='coerce').dropna().values for c in cohorts_present]
    cv_data = [arr if len(arr) > 0 else np.array([0]) for arr in cv_data]
    bp6 = ax6.boxplot(cv_data, labels=cohorts_present, patch_artist=True)
    for patch, color in zip(bp6['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax6.set_ylabel('Stride Time CV (%)')
    ax6.set_title('Gait Variability (Higher = Less Stable)')
    ax6.tick_params(axis='x', rotation=45)

    # 7. Visual Gait Assessment
    ax7 = fig.add_subplot(gs[2, 0])
    vga_data = [pd.to_numeric(df[df['cohort'] == c]['visualGaitAssessment'], errors='coerce').dropna().values for c in cohorts_present]
    # Filter out empty arrays
    vga_data = [arr if len(arr) > 0 else np.array([0]) for arr in vga_data]
    bp7 = ax7.boxplot(vga_data, labels=cohorts_present, patch_artist=True)
    for patch, color in zip(bp7['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax7.set_ylabel('Visual Gait Score')
    ax7.set_title('Clinical Visual Assessment (0=Normal)')
    ax7.tick_params(axis='x', rotation=45)

    # 8. Asymmetry
    ax8 = fig.add_subplot(gs[2, 1])
    asym_data = [pd.to_numeric(df[df['cohort'] == c]['asymmetry'], errors='coerce').dropna().values for c in cohorts_present]
    asym_data = [arr if len(arr) > 0 else np.array([0]) for arr in asym_data]
    bp8 = ax8.boxplot(asym_data, labels=cohorts_present, patch_artist=True)
    for patch, color in zip(bp8['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax8.set_ylabel('Asymmetry (%)')
    ax8.set_title('Left-Right Asymmetry')
    ax8.tick_params(axis='x', rotation=45)

    # 9. Summary table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary_text = "=== Dataset Summary ===\n\n"
    for cohort in cohorts_present:
        cohort_df = df[df['cohort'] == cohort]
        summary_text += f"{cohort} ({COHORTS[cohort]['name']}):\n"
        summary_text += f"  Trials: {len(cohort_df)}, "
        summary_text += f"Subjects: {cohort_df['subject'].nunique()}\n"
        summary_text += f"  Age: {cohort_df['age'].mean():.1f}±{cohort_df['age'].std():.1f}\n"
        summary_text += f"  Stride: {cohort_df['stride_time_mean'].mean():.2f}s, "
        summary_text += f"CV: {cohort_df['stride_time_cv'].mean():.1f}%\n\n"

    ax9.text(0.05, 0.95, summary_text, fontsize=8, fontfamily='monospace',
             verticalalignment='top', transform=ax9.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Gait Wearable Sensor Dataset - All Cohorts Analysis', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_sample_signals(df, output_path):
    """Plot sample signals from each cohort"""

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle('Sample Gait Signals by Cohort (Left Foot Gyr_Y)', fontsize=14, fontweight='bold')

    cohort_order = ['HS', 'PD', 'CVA', 'HOA', 'CIPN', 'KOA', 'RIL', 'ACL']

    for idx, cohort in enumerate(cohort_order):
        if cohort not in df['cohort'].unique():
            continue

        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        # Get first trial for this cohort
        sample = df[df['cohort'] == cohort].iloc[0]
        trial_path = Path(sample['trial_path'])
        trial_name = trial_path.name

        # Load LF sensor data
        lf_file = trial_path / f"{trial_name}_raw_data_LF.txt"
        if lf_file.exists():
            sensor_data = pd.read_csv(lf_file, sep='\t')
            time = np.arange(len(sensor_data)) / sample['freq']

            ax.plot(time, sensor_data['Gyr_Y'], color=COHORTS[cohort]['color'], alpha=0.8, linewidth=0.6)

            # Mark U-turn
            if 'uturnBoundaries' in sample and sample['uturnBoundaries']:
                ax.axvspan(sample['uturnBoundaries'][0]/sample['freq'],
                          sample['uturnBoundaries'][1]/sample['freq'],
                          alpha=0.2, color='orange')

        title = f"{cohort}: {COHORTS[cohort]['name']}\n"
        title += f"Age {sample['age']}, {sample['gender']}"
        if pd.notna(sample.get('evaluationScoreValue')):
            title += f", Score: {sample['evaluationScoreValue']}"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel('Gyr_Y (rad/s)')

        if row == 3:
            ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_ml_statistics(df):
    """Generate statistics for ML planning"""

    print("\n" + "="*60)
    print("ML TRAINING DATA STATISTICS")
    print("="*60)

    # Class distribution
    print("\n### Class Distribution ###")
    print(df.groupby(['group', 'cohort']).size())

    # Subject counts
    print("\n### Subject Counts ###")
    for cohort in df['cohort'].unique():
        subjects = df[df['cohort'] == cohort]['subject'].nunique()
        trials = len(df[df['cohort'] == cohort])
        print(f"{cohort}: {subjects} subjects, {trials} trials ({trials/subjects:.1f} trials/subject)")

    # Feature statistics
    print("\n### Feature Statistics ###")
    features = ['stride_time_mean', 'stride_time_cv', 'cadence', 'asymmetry', 'swing_time_mean']
    for feat in features:
        if feat in df.columns:
            print(f"\n{feat}:")
            print(df.groupby('cohort')[feat].agg(['mean', 'std', 'min', 'max']).round(3))

    # Missing data
    print("\n### Missing Data ###")
    print(df[features].isnull().sum())

    return df

def main():
    print("="*60)
    print("Loading all metadata...")
    print("="*60)

    df = load_all_metadata()
    print(f"Loaded {len(df)} trials from {df['cohort'].nunique()} cohorts")

    # Calculate gait features for each trial
    print("\nCalculating gait features...")
    features_list = []
    for _, row in df.iterrows():
        features = calculate_gait_features(row.to_dict())
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    df = pd.concat([df, features_df], axis=1)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_cohort_comparison(df, OUTPUT_PATH / "all_cohorts_comparison.png")
    plot_sample_signals(df, OUTPUT_PATH / "all_cohorts_signals.png")

    # Generate statistics
    df = generate_ml_statistics(df)

    # Save processed dataframe
    df.to_csv(OUTPUT_PATH / "dataset_summary.csv", index=False)
    print(f"\nSaved summary to: {OUTPUT_PATH / 'dataset_summary.csv'}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    main()
