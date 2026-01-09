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
    WINDOW_SIZE = 300  # 3 seconds @ 100Hz
    OVERLAP = 150  # 50% overlap

    # Temporal analysis parameters
    SUB_WINDOW_SIZE = 50  # 0.5 seconds @ 100Hz
    SUB_WINDOW_STRIDE = 25  # 0.25 second stride

    # Model parameters
    NUM_SENSORS = 4
    NUM_CHANNELS = 9
    NUM_CLASSES = 1

# ============================================================================
# Model Architecture (copied from train_baseline_hpc.py)
# ============================================================================

class StreamAttention(nn.Module):
    """Attention mechanism for sensor streams"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, length = x.shape
        weights = self.attention(x).unsqueeze(-1)
        return x * weights

class MultiStreamAttentionCNN(nn.Module):
    """Multi-Stream Attention CNN for OA Screening"""
    def __init__(self, num_sensors=4, num_channels=9, num_classes=1):
        super().__init__()
        self.num_sensors = num_sensors

        # Per-sensor CNN streams
        self.sensor_streams = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_channels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),

                StreamAttention(128),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            for _ in range(num_sensors)
        ])

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(256 * num_sensors, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        sensor_features = []

        for i, stream in enumerate(self.sensor_streams):
            sensor_data = x[:, i*9:(i+1)*9, :]
            features = stream(sensor_data)
            sensor_features.append(features.view(batch_size, -1))

        fused = torch.cat(sensor_features, dim=1)
        output = self.fusion(fused)
        return output

# ============================================================================
# Dataset
# ============================================================================

class GaitDataset(Dataset):
    """Gait dataset for temporal analysis"""
    def __init__(self, data_path, task_name='OA_Screening', split='test'):
        self.data_path = Path(data_path)
        self.task_name = task_name
        self.split = split

        # Task configuration
        self.task_config = {
            'OA_Screening': {
                'positive': ['Pathological'],
                'negative': ['Healthy']
            }
        }

        self.sensors = ['L-ANKLE', 'L-FOOT', 'R-ANKLE', 'R-FOOT']
        self.data, self.labels, self.metadata = self._load_data()

        print(f"\n[Dataset] {split} set: {len(self.data)} windows")
        print(f"  Positive (OA): {sum(self.labels)}")
        print(f"  Negative (Healthy): {len(self.labels) - sum(self.labels)}")

    def _load_data(self):
        """Load data with metadata tracking"""
        all_data = []
        all_labels = []
        all_metadata = []

        config = self.task_config[self.task_name]

        # Process each group
        for group_type, groups in [('positive', config['positive']),
                                     ('negative', config['negative'])]:
            label = 1 if group_type == 'positive' else 0

            for group in groups:
                group_path = self.data_path / group
                if not group_path.exists():
                    continue

                cohorts = sorted([d for d in group_path.iterdir() if d.is_dir()])

                # Split: 60% train, 20% val, 20% test
                n_cohorts = len(cohorts)
                train_end = int(0.6 * n_cohorts)
                val_end = int(0.8 * n_cohorts)

                if self.split == 'train':
                    cohorts = cohorts[:train_end]
                elif self.split == 'val':
                    cohorts = cohorts[train_end:val_end]
                else:  # test
                    cohorts = cohorts[val_end:]

                # Load cohort data
                for cohort_path in cohorts:
                    cohort_name = cohort_path.name

                    # Load sensor data
                    sensor_data = []
                    for sensor in self.sensors:
                        file_path = cohort_path / f'_raw_data_{sensor}.txt'
                        if not file_path.exists():
                            break
                        data = np.loadtxt(file_path, skiprows=1)
                        sensor_data.append(data)

                    if len(sensor_data) != len(self.sensors):
                        continue

                    sensor_data = np.array(sensor_data)

                    # Create windows
                    num_samples = sensor_data.shape[1]
                    windows = self._create_windows(sensor_data, num_samples)

                    # Add to dataset with metadata
                    for window_idx, window in enumerate(windows):
                        all_data.append(window)
                        all_labels.append(label)
                        all_metadata.append({
                            'group': group,
                            'cohort': cohort_name,
                            'window_idx': window_idx,
                            'label': label
                        })

        return np.array(all_data), np.array(all_labels), all_metadata

    def _create_windows(self, sensor_data, num_samples):
        """Create sliding windows"""
        windows = []
        window_size = Config.WINDOW_SIZE
        overlap = Config.OVERLAP
        stride = window_size - overlap

        for start in range(0, num_samples - window_size + 1, stride):
            end = start + window_size
            window = sensor_data[:, :, start:end]

            # Normalize per window
            window = (window - window.mean(axis=2, keepdims=True)) / (window.std(axis=2, keepdims=True) + 1e-8)

            # Reshape: (4 sensors, 9 channels, 300 samples) -> (36, 300)
            window = window.reshape(-1, window_size)
            windows.append(window)

        return windows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor([self.labels[idx]]), self.metadata[idx]

# ============================================================================
# Temporal Analyzer
# ============================================================================

class TemporalAnalyzer:
    """Analyze temporal patterns within windows"""

    def __init__(self, model_path, task_name='OA_Screening'):
        self.task_name = task_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        print(f"\n[Model] Loading from {model_path}")
        self.model = MultiStreamAttentionCNN(
            num_sensors=Config.NUM_SENSORS,
            num_channels=Config.NUM_CHANNELS,
            num_classes=Config.NUM_CLASSES
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
        self.dataset = GaitDataset(Config.BASE_PATH, task_name=task_name, split='test')
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns for all windows"""
        print("\n" + "="*80)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*80)

        results = []

        with torch.no_grad():
            for idx, (inputs, labels, metadata) in enumerate(tqdm(self.loader, desc="Analyzing windows")):
                inputs = inputs.to(self.device)
                labels = labels.cpu().numpy()[0, 0]
                metadata = metadata

                # Full window prediction
                outputs = self.model(inputs)
                prob = torch.sigmoid(outputs).cpu().numpy()[0, 0]
                pred = int(prob >= 0.5)

                # Temporal sub-window analysis
                temporal_probs = self._analyze_sub_windows(inputs[0])

                # Gait phase analysis
                phase_probs = self._analyze_gait_phases(temporal_probs)

                # Store results
                result = {
                    'window_id': idx,
                    'group': metadata['group'][0],
                    'cohort': metadata['cohort'][0],
                    'cohort_window_idx': metadata['window_idx'][0],
                    'label': int(labels),
                    'prediction': pred,
                    'probability': float(prob),
                    'correct': pred == int(labels),
                    'error_type': self._get_error_type(pred, int(labels)),
                    'temporal_probabilities': [float(p) for p in temporal_probs],
                    'gait_phase_probabilities': {k: float(v) for k, v in phase_probs.items()},
                    'temporal_variance': float(np.var(temporal_probs)),
                    'temporal_trend': self._calculate_trend(temporal_probs)
                }

                results.append(result)

        return results

    def _analyze_sub_windows(self, window_data):
        """Analyze sub-windows within a 3-second window"""
        # window_data shape: (36, 300)
        sub_window_size = Config.SUB_WINDOW_SIZE
        stride = Config.SUB_WINDOW_STRIDE
        window_length = window_data.shape[1]

        temporal_probs = []

        for start in range(0, window_length - sub_window_size + 1, stride):
            end = start + sub_window_size

            # Extract sub-window and pad to full size
            sub_window = torch.zeros(36, 300, device=self.device)
            sub_window[:, start:end] = window_data[:, start:end]

            # Predict
            sub_window = sub_window.unsqueeze(0)  # Add batch dimension
            outputs = self.model(sub_window)
            prob = torch.sigmoid(outputs).cpu().numpy()[0, 0]
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
