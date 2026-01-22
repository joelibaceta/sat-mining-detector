#!/usr/bin/env python3
"""
Compare multiple model architectures for temporal change detection.
Trains all models and generates a comparison report.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from train_new import train_model


def compare_models(dataset_dir, output_base_dir, epochs=150, batch_size=16):
    """
    Train and compare multiple model architectures.
    
    Args:
        dataset_dir: Path to dataset
        output_base_dir: Base directory for outputs
        epochs: Number of epochs
        batch_size: Batch size
    """
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Models to compare
    models_config = [
        {
            'name': 'custom_cnn',
            'kwargs': {}
        },
        {
            'name': 'efficientnet',
            'kwargs': {'freeze_backbone': True}
        },
        {
            'name': 'resnet18',
            'kwargs': {}
        }
    ]
    
    results = {}
    
    print("="*70)
    print("MODEL COMPARISON EXPERIMENT")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Output base: {output_base_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Models to train: {[m['name'] for m in models_config]}")
    print()
    
    # Train each model
    for i, config in enumerate(models_config, 1):
        model_name = config['name']
        model_kwargs = config['kwargs']
        
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL {i}/{len(models_config)}: {model_name.upper()}")
        print(f"{'='*70}\n")
        
        output_dir = output_base_dir / f"{timestamp}_{model_name}"
        
        try:
            _, history, metrics = train_model(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                model_name=model_name,
                epochs=epochs,
                batch_size=batch_size,
                **model_kwargs
            )
            
            # Load full results
            with open(output_dir / 'training_results.json', 'r') as f:
                full_results = json.load(f)
            
            results[model_name] = {
                'output_dir': str(output_dir),
                'metrics': metrics,
                'full_results': full_results,
                'success': True
            }
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            results[model_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Generate comparison report
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)
    
    comparison_file = output_base_dir / f"{timestamp}_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison plots
    plot_comparison(results, output_base_dir / f"{timestamp}_comparison.png")
    
    # Create comparison table
    create_comparison_table(results, output_base_dir / f"{timestamp}_comparison.txt")
    
    print(f"\n✅ Comparison complete!")
    print(f"Results saved to: {output_base_dir}")
    print(f"Comparison file: {comparison_file}")


def plot_comparison(results, output_file):
    """Create comparison plots for all models."""
    
    successful_models = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_models) == 0:
        print("⚠️  No successful models to compare")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    models = list(successful_models.keys())
    class_names = ['mining_emergence', 'human_activity', 'natural_change']
    
    # 1. Overall Accuracy Comparison
    val_accs = [r['full_results']['best_val_acc'] for r in successful_models.values()]
    test_f1s = [r['metrics']['macro']['f1_score'] for r in successful_models.values()]
    params = [r['full_results']['model_params'] / 1000 for r in successful_models.values()]
    
    x = range(len(models))
    axes[0, 0].bar(x, val_accs, alpha=0.7, label='Val Accuracy')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Best Validation Accuracy', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1-Score Comparison
    axes[0, 1].bar(x, test_f1s, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Macro F1-Score')
    axes[0, 1].set_title('Test Set Macro F1-Score', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Parameters vs Performance
    axes[1, 0].scatter(params, test_f1s, s=200, alpha=0.7)
    for i, model in enumerate(models):
        axes[1, 0].annotate(model, (params[i], test_f1s[i]), 
                           fontsize=9, ha='center', va='bottom')
    axes[1, 0].set_xlabel('Parameters (K)')
    axes[1, 0].set_ylabel('Macro F1-Score')
    axes[1, 0].set_title('Efficiency: Params vs Performance', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Per-Class F1-Score Heatmap
    f1_matrix = []
    for model in models:
        f1_scores = [successful_models[model]['metrics']['per_class'][cn]['f1_score'] 
                    for cn in class_names]
        f1_matrix.append(f1_scores)
    
    im = axes[1, 1].imshow(f1_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_xticks(range(len(class_names)))
    axes[1, 1].set_yticks(range(len(models)))
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(models)
    axes[1, 1].set_title('Per-Class F1-Score Heatmap', fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(class_names)):
            axes[1, 1].text(j, i, f'{f1_matrix[i][j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved: {output_file}")


def create_comparison_table(results, output_file):
    """Create a text comparison table."""
    
    successful_models = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_models) == 0:
        return
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall metrics table
        f.write("OVERALL METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'Params':>10} {'Val Acc':>10} {'Test F1':>10} {'ROC AUC':>10}\n")
        f.write("-"*80 + "\n")
        
        for model_name, result in successful_models.items():
            params = result['full_results']['model_params']
            val_acc = result['full_results']['best_val_acc']
            test_f1 = result['metrics']['macro']['f1_score']
            roc_auc = result['metrics']['macro']['roc_auc']
            
            f.write(f"{model_name:<20} {params:>10,} {val_acc:>10.4f} {test_f1:>10.4f} {roc_auc:>10.4f}\n")
        
        f.write("\n\n")
        
        # Per-class metrics
        class_names = ['mining_emergence', 'human_activity', 'natural_change']
        
        for class_name in class_names:
            f.write(f"PER-CLASS METRICS: {class_name.upper()}\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Model':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'ROC AUC':>12}\n")
            f.write("-"*80 + "\n")
            
            for model_name, result in successful_models.items():
                metrics = result['metrics']['per_class'][class_name]
                f.write(f"{model_name:<20} {metrics['precision']:>12.4f} "
                       f"{metrics['recall']:>12.4f} {metrics['f1_score']:>12.4f} "
                       f"{metrics['roc_auc']:>12.4f}\n")
            
            f.write("\n")
    
    print(f"Comparison table saved: {output_file}")
    
    # Also print to console
    with open(output_file, 'r') as f:
        print("\n" + f.read())


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare model architectures')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for comparison results')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    
    args = parser.parse_args()
    
    compare_models(
        dataset_dir=args.dataset_dir,
        output_base_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
