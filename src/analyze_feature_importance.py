"""
Feature Importance Analysis for Multi-Stream Attention CNN
Analyzes which sensors and channels are most important for classification
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import model and dataset from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_baseline_hpc import (
    Config, MultiStreamAttentionCNN, GaitDataset,
    get_trial_paths, create_dataloaders
)


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods"""

    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.eval()

        self.sensors = Config.SENSORS
        self.channels = Config.CHANNELS

    def get_baseline_performance(self):
        """Get baseline performance without any perturbation"""
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs)

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        auc = roc_auc_score(all_labels, all_preds)
        acc = balanced_accuracy_score(all_labels, (all_preds > 0.5).astype(int))

        return {'auc': auc, 'balanced_acc': acc}

    def permutation_importance_sensors(self):
        """Calculate importance by permuting each sensor"""
        print("\n" + "="*80)
        print("Permutation Importance - Sensors")
        print("="*80)

        baseline = self.get_baseline_performance()
        print(f"\nBaseline Performance:")
        print(f"  AUC: {baseline['auc']:.4f}")
        print(f"  Balanced Acc: {baseline['balanced_acc']:.4f}")

        importance = {}

        for sensor_idx, sensor_name in enumerate(self.sensors):
            print(f"\nPermuting sensor: {sensor_name} (index {sensor_idx})...")

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Permute this sensor's data across batch
                    inputs_perm = inputs.clone()
                    perm_indices = torch.randperm(inputs.shape[0])
                    inputs_perm[:, sensor_idx] = inputs[perm_indices, sensor_idx]

                    outputs = self.model(inputs_perm)
                    probs = torch.sigmoid(outputs)

                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            auc = roc_auc_score(all_labels, all_preds)
            acc = balanced_accuracy_score(all_labels, (all_preds > 0.5).astype(int))

            importance[sensor_name] = {
                'auc': auc,
                'balanced_acc': acc,
                'auc_drop': baseline['auc'] - auc,
                'acc_drop': baseline['balanced_acc'] - acc
            }

            print(f"  AUC: {auc:.4f} (drop: {baseline['auc'] - auc:.4f})")
            print(f"  Balanced Acc: {acc:.4f} (drop: {baseline['balanced_acc'] - acc:.4f})")

        return importance

    def permutation_importance_channels(self):
        """Calculate importance by permuting each channel across all sensors"""
        print("\n" + "="*80)
        print("Permutation Importance - Channels")
        print("="*80)

        baseline = self.get_baseline_performance()

        importance = {}

        for channel_idx, channel_name in enumerate(self.channels):
            print(f"\nPermuting channel: {channel_name} (index {channel_idx})...")

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Permute this channel across all sensors
                    inputs_perm = inputs.clone()
                    perm_indices = torch.randperm(inputs.shape[0])
                    inputs_perm[:, :, channel_idx, :] = inputs[perm_indices, :, channel_idx, :]

                    outputs = self.model(inputs_perm)
                    probs = torch.sigmoid(outputs)

                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            auc = roc_auc_score(all_labels, all_preds)
            acc = balanced_accuracy_score(all_labels, (all_preds > 0.5).astype(int))

            importance[channel_name] = {
                'auc': auc,
                'balanced_acc': acc,
                'auc_drop': baseline['auc'] - auc,
                'acc_drop': baseline['balanced_acc'] - acc
            }

            print(f"  AUC: {auc:.4f} (drop: {baseline['auc'] - auc:.4f})")
            print(f"  Balanced Acc: {acc:.4f} (drop: {baseline['balanced_acc'] - acc:.4f})")

        return importance

    def ablation_study_sensors(self):
        """Test performance when removing each sensor (set to zero)"""
        print("\n" + "="*80)
        print("Ablation Study - Sensors")
        print("="*80)

        baseline = self.get_baseline_performance()
        print(f"\nBaseline Performance:")
        print(f"  AUC: {baseline['auc']:.4f}")
        print(f"  Balanced Acc: {baseline['balanced_acc']:.4f}")

        ablation = {}

        for sensor_idx, sensor_name in enumerate(self.sensors):
            print(f"\nRemoving sensor: {sensor_name} (set to zero)...")

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero out this sensor
                    inputs_ablated = inputs.clone()
                    inputs_ablated[:, sensor_idx] = 0

                    outputs = self.model(inputs_ablated)
                    probs = torch.sigmoid(outputs)

                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            auc = roc_auc_score(all_labels, all_preds)
            acc = balanced_accuracy_score(all_labels, (all_preds > 0.5).astype(int))

            ablation[sensor_name] = {
                'auc': auc,
                'balanced_acc': acc,
                'auc_drop': baseline['auc'] - auc,
                'acc_drop': baseline['balanced_acc'] - acc
            }

            print(f"  AUC: {auc:.4f} (drop: {baseline['auc'] - auc:.4f})")
            print(f"  Balanced Acc: {acc:.4f} (drop: {baseline['balanced_acc'] - acc:.4f})")

        return ablation

    def gradient_based_importance(self, num_samples=100):
        """Calculate importance using gradients (integrated gradients approximation)"""
        print("\n" + "="*80)
        print("Gradient-based Importance")
        print("="*80)

        # Sample random test examples
        self.model.train()  # Need gradients

        sensor_gradients = np.zeros((len(self.sensors),))
        channel_gradients = np.zeros((len(self.channels),))

        sample_count = 0

        for inputs, labels in self.test_loader:
            if sample_count >= num_samples:
                break

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs.requires_grad = True

            # Forward pass
            outputs = self.model(inputs)
            probs = torch.sigmoid(outputs)

            # Backward pass for each sample
            for i in range(min(inputs.shape[0], num_samples - sample_count)):
                if probs[i] > 0.5:
                    target = probs[i]
                else:
                    target = 1 - probs[i]

                self.model.zero_grad()
                if inputs.grad is not None:
                    inputs.grad.zero_()

                target.backward(retain_graph=True)

                # Get gradients
                grad = inputs.grad[i].abs().cpu().numpy()  # (sensors, channels, time)

                # Average over time dimension
                grad_mean = grad.mean(axis=2)  # (sensors, channels)

                # Accumulate
                sensor_gradients += grad_mean.sum(axis=1)  # Sum over channels
                channel_gradients += grad_mean.sum(axis=0)  # Sum over sensors

                sample_count += 1

        # Normalize
        sensor_gradients /= sample_count
        channel_gradients /= sample_count

        self.model.eval()

        sensor_importance = {
            sensor: float(grad)
            for sensor, grad in zip(self.sensors, sensor_gradients)
        }

        channel_importance = {
            channel: float(grad)
            for channel, grad in zip(self.channels, channel_gradients)
        }

        print(f"\nProcessed {sample_count} samples")
        print("\nSensor Gradient Importance:")
        for sensor, grad in sensor_importance.items():
            print(f"  {sensor}: {grad:.6f}")

        print("\nChannel Gradient Importance:")
        for channel, grad in channel_importance.items():
            print(f"  {channel}: {grad:.6f}")

        return {'sensors': sensor_importance, 'channels': channel_importance}


def plot_importance_results(results, task_name, output_dir):
    """Visualize feature importance results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Feature Importance Analysis - {task_name}', fontsize=16, fontweight='bold')

    sensors = Config.SENSORS
    channels = Config.CHANNELS

    # 1. Permutation Importance - Sensors (AUC Drop)
    ax = axes[0, 0]
    perm_sensor = results['permutation_sensors']
    sensor_drops = [perm_sensor[s]['auc_drop'] for s in sensors]
    bars = ax.bar(sensors, sensor_drops, color='#ff9999', alpha=0.8)
    ax.set_ylabel('AUC Drop', fontweight='bold')
    ax.set_title('Permutation Importance - Sensors', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 2. Permutation Importance - Channels (AUC Drop)
    ax = axes[0, 1]
    perm_channel = results['permutation_channels']
    channel_drops = [perm_channel[c]['auc_drop'] for c in channels]
    bars = ax.bar(channels, channel_drops, color='#66b3ff', alpha=0.8)
    ax.set_ylabel('AUC Drop', fontweight='bold')
    ax.set_title('Permutation Importance - Channels', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 3. Ablation Study - Sensors (AUC Drop)
    ax = axes[0, 2]
    ablation = results['ablation_sensors']
    ablation_drops = [ablation[s]['auc_drop'] for s in sensors]
    bars = ax.bar(sensors, ablation_drops, color='#99ff99', alpha=0.8)
    ax.set_ylabel('AUC Drop', fontweight='bold')
    ax.set_title('Ablation Study - Sensors', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 4. Gradient-based - Sensors
    ax = axes[1, 0]
    grad_sensor = results['gradient']['sensors']
    grad_sensor_vals = [grad_sensor[s] for s in sensors]
    # Normalize to 0-1
    max_grad = max(grad_sensor_vals)
    grad_sensor_norm = [v/max_grad for v in grad_sensor_vals]
    bars = ax.bar(sensors, grad_sensor_norm, color='#ffcc99', alpha=0.8)
    ax.set_ylabel('Normalized Gradient', fontweight='bold')
    ax.set_title('Gradient-based Importance - Sensors', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 5. Gradient-based - Channels
    ax = axes[1, 1]
    grad_channel = results['gradient']['channels']
    grad_channel_vals = [grad_channel[c] for c in channels]
    # Normalize to 0-1
    max_grad = max(grad_channel_vals)
    grad_channel_norm = [v/max_grad for v in grad_channel_vals]
    bars = ax.bar(channels, grad_channel_norm, color='#cc99ff', alpha=0.8)
    ax.set_ylabel('Normalized Gradient', fontweight='bold')
    ax.set_title('Gradient-based Importance - Channels', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 6. Comparison heatmap (all methods)
    ax = axes[1, 2]

    # Create comparison matrix (methods × sensors)
    comparison_data = np.array([
        sensor_drops,
        ablation_drops,
        grad_sensor_norm
    ])

    sns.heatmap(comparison_data, annot=True, fmt='.3f', cmap='YlOrRd',
               xticklabels=sensors,
               yticklabels=['Permutation', 'Ablation', 'Gradient'],
               ax=ax, cbar_kws={'label': 'Importance'})
    ax.set_title('Sensor Importance - All Methods', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / f'{task_name}_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Visualization saved: {output_dir / f'{task_name}_feature_importance.png'}")
    plt.close()


def analyze_task(task_name, device):
    """Analyze feature importance for a specific task"""
    print("\n" + "="*80)
    print(f"Feature Importance Analysis: {task_name}")
    print("="*80)

    # Load model
    model_path = Config.MODEL_PATH / f'{task_name}_best.pt'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return None

    print(f"\nLoading model from: {model_path}")

    model = MultiStreamAttentionCNN(
        num_sensors=len(Config.SENSORS),
        in_channels=len(Config.CHANNELS),
        hidden_dim=64,
        num_heads=4,
        dropout=0.3
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Format: {'model_state_dict': ..., 'epoch': ..., 'best_auc': ...}
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        best_auc = checkpoint.get('best_auc', 0.0)
        print(f"Model loaded (epoch {epoch}, AUC: {best_auc:.4f})")
    else:
        # Format: state_dict directly
        model.load_state_dict(checkpoint)
        print(f"Model loaded successfully")

    model.eval()

    # Load test data
    print("\nLoading test data...")
    _, test_loader = create_dataloaders(task_name)

    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(model, test_loader, device)

    # Run all analyses
    results = {
        'task': task_name,
        'baseline': analyzer.get_baseline_performance(),
        'permutation_sensors': analyzer.permutation_importance_sensors(),
        'permutation_channels': analyzer.permutation_importance_channels(),
        'ablation_sensors': analyzer.ablation_study_sensors(),
        'gradient': analyzer.gradient_based_importance(num_samples=100)
    }

    # Save results
    output_dir = Config.OUTPUT_PATH / 'feature_importance'
    output_dir.mkdir(exist_ok=True)

    json_file = output_dir / f'{task_name}_importance.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved: {json_file}")

    # Plot results
    plot_importance_results(results, task_name, output_dir)

    return results


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Feature Importance Analysis')
    parser.add_argument('--task', type=str, default='OA_Screening',
                       choices=['PD_Screening', 'OA_Screening', 'CVA_Detection', 'PD_vs_CVA', 'all'],
                       help='Task to analyze')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')

    args = parser.parse_args()

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Analyze task(s)
    if args.task == 'all':
        tasks = ['PD_Screening', 'OA_Screening', 'CVA_Detection', 'PD_vs_CVA']
    else:
        tasks = [args.task]

    all_results = {}
    for task in tasks:
        results = analyze_task(task, device)
        if results:
            all_results[task] = results

    print("\n" + "="*80)
    print("Feature Importance Analysis Complete!")
    print("="*80)

    # Print summary
    print("\nSummary - Most Important Features:")
    print("-" * 80)

    for task, results in all_results.items():
        print(f"\n{task}:")

        # Top sensors by permutation importance
        perm_sensors = results['permutation_sensors']
        sorted_sensors = sorted(perm_sensors.items(),
                               key=lambda x: x[1]['auc_drop'],
                               reverse=True)
        print(f"  Top Sensor: {sorted_sensors[0][0]} (AUC drop: {sorted_sensors[0][1]['auc_drop']:.4f})")

        # Top channels by permutation importance
        perm_channels = results['permutation_channels']
        sorted_channels = sorted(perm_channels.items(),
                                key=lambda x: x[1]['auc_drop'],
                                reverse=True)
        print(f"  Top Channel: {sorted_channels[0][0]} (AUC drop: {sorted_channels[0][1]['auc_drop']:.4f})")


if __name__ == '__main__':
    main()
