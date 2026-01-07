"""
Gait Wearable Sensor Dataset - Sample Visualization
Compares Healthy vs Parkinson's Disease gait patterns
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (14, 10)

BASE_PATH = Path(r"D:\gait_wearable_sensor\dataset\data")

def load_trial(trial_path: Path):
    """Load a single trial's data and metadata"""
    trial_name = trial_path.name

    # Load metadata
    meta_file = trial_path / f"{trial_name}_meta.json"
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    # Load raw sensor data (Left Foot - most informative for gait)
    sensors = {}
    for sensor in ['LF', 'RF', 'LB', 'HE']:
        sensor_file = trial_path / f"{trial_name}_raw_data_{sensor}.txt"
        if sensor_file.exists():
            sensors[sensor] = pd.read_csv(sensor_file, sep='\t')

    return meta, sensors

def plot_comparison(healthy_path: Path, pd_path: Path, output_path: Path):
    """Compare healthy vs PD gait patterns"""

    # Load data
    hs_meta, hs_sensors = load_trial(healthy_path)
    pd_meta, pd_sensors = load_trial(pd_path)

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Gait Signal Comparison: Healthy vs Parkinson\'s Disease', fontsize=14, fontweight='bold')

    # Use minimum length across sensors for consistency
    def get_min_len(sensors):
        return min(len(sensors[k]) for k in sensors.keys())

    hs_min_len = get_min_len(hs_sensors)
    pd_min_len = get_min_len(pd_sensors)

    # Time axis (100Hz sampling)
    hs_time = np.arange(hs_min_len) / 100  # seconds
    pd_time = np.arange(pd_min_len) / 100

    # Color scheme
    hs_color = '#2ecc71'  # Green for healthy
    pd_color = '#e74c3c'  # Red for PD

    # --- Row 1: Left Foot Angular Velocity (Gyr_Y - sagittal plane) ---
    ax1 = axes[0, 0]
    ax1.plot(hs_time, hs_sensors['LF']['Gyr_Y'].values[:hs_min_len], color=hs_color, alpha=0.8, linewidth=0.8)
    ax1.set_title(f"Healthy (HS_1) - Left Foot Gyr_Y\nAge: {hs_meta['age']}, Gender: {hs_meta['gender']}")
    ax1.set_ylabel('Angular Vel (rad/s)')
    ax1.axvspan(hs_meta['uturnBoundaries'][0]/100, hs_meta['uturnBoundaries'][1]/100,
                alpha=0.2, color='orange', label='U-turn')
    ax1.legend(loc='upper right')

    ax2 = axes[0, 1]
    ax2.plot(pd_time, pd_sensors['LF']['Gyr_Y'].values[:pd_min_len], color=pd_color, alpha=0.8, linewidth=0.8)
    ax2.set_title(f"Parkinson (PD_1) - Left Foot Gyr_Y\nAge: {pd_meta['age']}, UPDRS: {pd_meta['evaluationScoreValue']}")
    ax2.set_ylabel('Angular Vel (rad/s)')
    ax2.axvspan(pd_meta['uturnBoundaries'][0]/100, pd_meta['uturnBoundaries'][1]/100,
                alpha=0.2, color='orange', label='U-turn')
    ax2.legend(loc='upper right')

    # --- Row 2: Lower Back Acceleration (vertical - Acc_Z) ---
    ax3 = axes[1, 0]
    ax3.plot(hs_time, hs_sensors['LB']['Acc_Z'].values[:hs_min_len], color=hs_color, alpha=0.8, linewidth=0.8)
    ax3.set_title("Healthy - Lower Back Acc_Z (Vertical)")
    ax3.set_ylabel('Acceleration (m/s²)')

    ax4 = axes[1, 1]
    ax4.plot(pd_time, pd_sensors['LB']['Acc_Z'].values[:pd_min_len], color=pd_color, alpha=0.8, linewidth=0.8)
    ax4.set_title("Parkinson - Lower Back Acc_Z (Vertical)")
    ax4.set_ylabel('Acceleration (m/s²)')

    # --- Row 3: Gait Events Visualization (zoomed) ---
    # Focus on a few gait cycles (5-15 seconds)
    zoom_start_hs, zoom_end_hs = 5, 12
    zoom_start_pd, zoom_end_pd = 12, 25

    # Get LF data with proper length
    hs_lf_data = hs_sensors['LF']['Gyr_Y'].values[:hs_min_len]
    pd_lf_data = pd_sensors['LF']['Gyr_Y'].values[:pd_min_len]

    ax5 = axes[2, 0]
    mask_hs = (hs_time >= zoom_start_hs) & (hs_time <= zoom_end_hs)
    ax5.plot(hs_time[mask_hs], hs_lf_data[mask_hs],
             color=hs_color, alpha=0.8, linewidth=1.2)
    # Mark gait events
    for event in hs_meta['leftGaitEvents']:
        toe_off, heel_strike = event
        if zoom_start_hs*100 <= toe_off <= zoom_end_hs*100:
            ax5.axvline(toe_off/100, color='blue', alpha=0.5, linestyle='--', linewidth=0.8)
            ax5.axvline(heel_strike/100, color='red', alpha=0.5, linestyle='-', linewidth=0.8)
    ax5.set_title("Healthy - Gait Cycles (Zoomed)\nBlue=Toe-off, Red=Heel-strike")
    ax5.set_ylabel('Angular Vel (rad/s)')

    ax6 = axes[2, 1]
    mask_pd = (pd_time >= zoom_start_pd) & (pd_time <= zoom_end_pd)
    ax6.plot(pd_time[mask_pd], pd_lf_data[mask_pd],
             color=pd_color, alpha=0.8, linewidth=1.2)
    for event in pd_meta['leftGaitEvents']:
        toe_off, heel_strike = event
        if zoom_start_pd*100 <= toe_off <= zoom_end_pd*100:
            ax6.axvline(toe_off/100, color='blue', alpha=0.5, linestyle='--', linewidth=0.8)
            ax6.axvline(heel_strike/100, color='red', alpha=0.5, linestyle='-', linewidth=0.8)
    ax6.set_title("Parkinson - Gait Cycles (Zoomed)\nBlue=Toe-off, Red=Heel-strike")
    ax6.set_ylabel('Angular Vel (rad/s)')

    # --- Row 4: Summary Statistics ---
    ax7 = axes[3, 0]
    ax7.axis('off')

    # Calculate gait parameters
    hs_stride_times = []
    for i in range(1, len(hs_meta['leftGaitEvents'])):
        stride = (hs_meta['leftGaitEvents'][i][0] - hs_meta['leftGaitEvents'][i-1][0]) / 100
        hs_stride_times.append(stride)

    pd_stride_times = []
    for i in range(1, len(pd_meta['leftGaitEvents'])):
        stride = (pd_meta['leftGaitEvents'][i][0] - pd_meta['leftGaitEvents'][i-1][0]) / 100
        pd_stride_times.append(stride)

    stats_text = f"""
    === HEALTHY (HS_1) ===
    Age: {hs_meta['age']} | Gender: {hs_meta['gender']} | BMI: {hs_meta['BMI']}
    Trial Duration: {len(hs_sensors['LF'])/100:.1f} sec
    Total Strides: {len(hs_meta['leftGaitEvents'])}
    Stride Time: {np.mean(hs_stride_times):.3f} ± {np.std(hs_stride_times):.3f} sec
    Cadence: {60/np.mean(hs_stride_times):.1f} steps/min
    Stride Variability (CV): {np.std(hs_stride_times)/np.mean(hs_stride_times)*100:.1f}%
    """
    ax7.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor=hs_color, alpha=0.2))

    ax8 = axes[3, 1]
    ax8.axis('off')

    stats_text_pd = f"""
    === PARKINSON (PD_1) ===
    Age: {pd_meta['age']} | Gender: {pd_meta['gender']} | BMI: {pd_meta['BMI']}
    UPDRS III: {pd_meta['evaluationScoreValue']}/108
    Trial Duration: {len(pd_sensors['LF'])/100:.1f} sec
    Total Strides: {len(pd_meta['leftGaitEvents'])}
    Stride Time: {np.mean(pd_stride_times):.3f} ± {np.std(pd_stride_times):.3f} sec
    Cadence: {60/np.mean(pd_stride_times):.1f} steps/min
    Stride Variability (CV): {np.std(pd_stride_times)/np.mean(pd_stride_times)*100:.1f}%
    """
    ax8.text(0.1, 0.5, stats_text_pd, fontsize=11, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor=pd_color, alpha=0.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {output_path}")

def plot_all_sensors(trial_path: Path, output_path: Path):
    """Plot all 4 sensors for a single trial"""
    meta, sensors = load_trial(trial_path)

    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    fig.suptitle(f"All Sensors - {meta['subject']} ({meta['pathology']})\n"
                 f"Age: {meta['age']}, Gender: {meta['gender']}", fontsize=12, fontweight='bold')

    sensor_names = ['HE (Head)', 'LB (Lower Back)', 'LF (Left Foot)', 'RF (Right Foot)']
    sensor_keys = ['HE', 'LB', 'LF', 'RF']
    colors = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c']

    for i, (name, key, color) in enumerate(zip(sensor_names, sensor_keys, colors)):
        if key not in sensors:
            continue

        # Use sensor-specific time array
        sensor_len = len(sensors[key])
        time = np.arange(sensor_len) / 100

        # Accelerometer
        axes[i, 0].plot(time, sensors[key]['Acc_X'], alpha=0.7, label='X')
        axes[i, 0].plot(time, sensors[key]['Acc_Y'], alpha=0.7, label='Y')
        axes[i, 0].plot(time, sensors[key]['Acc_Z'], alpha=0.7, label='Z')
        axes[i, 0].set_ylabel(f'{name}\nAcc (m/s²)')
        axes[i, 0].legend(loc='upper right', fontsize=8)
        if i == 0:
            axes[i, 0].set_title('Accelerometer')

        # Gyroscope
        axes[i, 1].plot(time, sensors[key]['Gyr_X'], alpha=0.7, label='X')
        axes[i, 1].plot(time, sensors[key]['Gyr_Y'], alpha=0.7, label='Y')
        axes[i, 1].plot(time, sensors[key]['Gyr_Z'], alpha=0.7, label='Z')
        axes[i, 1].set_ylabel('Gyr (rad/s)')
        if i == 0:
            axes[i, 1].set_title('Gyroscope')

        # Magnetometer
        axes[i, 2].plot(time, sensors[key]['Mag_X'], alpha=0.7, label='X')
        axes[i, 2].plot(time, sensors[key]['Mag_Y'], alpha=0.7, label='Y')
        axes[i, 2].plot(time, sensors[key]['Mag_Z'], alpha=0.7, label='Z')
        axes[i, 2].set_ylabel('Mag (a.u.)')
        if i == 0:
            axes[i, 2].set_title('Magnetometer')

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Sample paths
    healthy_sample = BASE_PATH / "healthy" / "HS" / "HS_1" / "HS_1_1"
    pd_sample = BASE_PATH / "neuro" / "PD" / "PD_1" / "PD_1_1"

    output_dir = Path(r"D:\gait_wearable_sensor")

    print("=" * 60)
    print("Gait Wearable Sensor Dataset - Visualization")
    print("=" * 60)

    # 1. Compare Healthy vs PD
    print("\n[1] Comparing Healthy vs Parkinson's Disease...")
    plot_comparison(healthy_sample, pd_sample, output_dir / "comparison_HS_vs_PD.png")

    # 2. All sensors for healthy subject
    print("\n[2] Plotting all sensors for Healthy subject...")
    plot_all_sensors(healthy_sample, output_dir / "all_sensors_healthy.png")

    # 3. All sensors for PD subject
    print("\n[3] Plotting all sensors for Parkinson's subject...")
    plot_all_sensors(pd_sample, output_dir / "all_sensors_PD.png")

    print("\n" + "=" * 60)
    print("Done! Check the output images in:", output_dir)
