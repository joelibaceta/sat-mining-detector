#!/usr/bin/env python3
"""
Training script for temporal change detection models.
Supports multiple architectures: improved_cnn, efficientnet, resnet18
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models import get_model
from utils import (evaluate_model, plot_training_history, 
                   plot_confusion_matrix, focal_loss)


class DataGenerator(keras.utils.Sequence):
    """Data generator for temporal pairs with 3 classes."""
    
    def __init__(self, csv_path: str, batch_size: int = 16, 
                 shuffle: bool = True, augment: bool = False, num_classes: int = 3):
        self.df = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes
        self.csv_dir = Path(csv_path).parent
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._load_batch(batch_indices)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_batch(self, batch_indices):
        X_batch = []
        y_batch = []
        
        for i in batch_indices:
            row = self.df.iloc[i]
            t0_path = self.csv_dir / row['t0_path']
            t1_path = self.csv_dir / row['t1_path']
            
            if t0_path.exists() and t1_path.exists():
                t0 = np.load(t0_path).astype(np.float32)
                t1 = np.load(t1_path).astype(np.float32)
                
                # Normalize to [-1, 1]
                t0 = (t0 - 0.5) * 2
                t1 = (t1 - 0.5) * 2
                
                # Compute difference
                diff = t1 - t0
                
                # Optional augmentation
                if self.augment and np.random.rand() > 0.5:
                    # Random flip horizontal
                    if np.random.rand() > 0.5:
                        diff = np.flip(diff, axis=1)
                    # Random flip vertical
                    if np.random.rand() > 0.5:
                        diff = np.flip(diff, axis=0)
                    # Random brightness
                    if np.random.rand() > 0.5:
                        diff = diff * np.random.uniform(0.9, 1.1)
                
                X_batch.append(diff)
                # One-hot encode label
                label = int(row['label'])
                y_one_hot = np.zeros(self.num_classes)
                y_one_hot[label] = 1
                y_batch.append(y_one_hot)
        
        return np.array(X_batch), np.array(y_batch)


def train_model(dataset_dir, output_dir, model_name='custom_cnn',
                epochs=150, batch_size=16, learning_rate=0.0001, 
                patience=40, use_focal_loss=False, **model_kwargs):
    """
    Train a temporal change detection model.
    
    Args:
        dataset_dir: Path to dataset directory
        output_dir: Path to save outputs
        model_name: Model architecture ('custom_cnn', 'efficientnet', 'resnet18')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        patience: Early stopping patience
        use_focal_loss: Whether to use focal loss
        **model_kwargs: Additional arguments for model (e.g., freeze_backbone)
    """
    
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['mining_emergence', 'human_activity', 'natural_change']
    num_classes = 3
    
    print("="*70)
    print(f"TRAINING {model_name.upper()} MODEL")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_name}")
    print(f"Classes: {class_names}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")
    print(f"Use focal loss: {use_focal_loss}")
    
    # Create data generators
    print("\nLoading data...")
    train_gen = DataGenerator(
        dataset_dir / 'dataset_train.csv',
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        num_classes=num_classes
    )
    
    val_gen = DataGenerator(
        dataset_dir / 'dataset_val.csv',
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        num_classes=num_classes
    )
    
    test_gen = DataGenerator(
        dataset_dir / 'dataset_test.csv',
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        num_classes=num_classes
    )
    
    print(f"Train batches: {len(train_gen)} ({len(train_gen) * batch_size} samples)")
    print(f"Val batches: {len(val_gen)} ({len(val_gen) * batch_size} samples)")
    print(f"Test batches: {len(test_gen)} ({len(test_gen) * batch_size} samples)")
    
    # Build model
    print("\nBuilding model...")
    model = get_model(model_name, input_shape=(48, 48, 6), 
                     num_classes=num_classes, **model_kwargs)
    
    # Choose loss function
    if use_focal_loss:
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        loss_name = 'focal_loss'
    else:
        loss_fn = 'categorical_crossentropy'
        loss_name = 'categorical_crossentropy'
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    print(f"\nLoss: {loss_name}")
    
    # Save model summary
    with open(output_dir / 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            output_dir / 'best_model.keras',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\nStarting training...")
    print("-"*70)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot history
    print("\nSaving training plots...")
    plot_training_history(history, output_dir)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    y_true, y_pred, cm, metrics_summary = evaluate_model(
        model, test_gen, class_names, output_dir
    )
    
    # Calculate per-class accuracies
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'dataset': str(dataset_dir),
        'classes': class_names,
        'num_classes': num_classes,
        'loss_function': loss_name,
        'epochs_trained': len(history.history['loss']),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'patience': patience,
        'model_params': int(model.count_params()),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
        'best_val_acc': float(max(history.history['val_accuracy'])),
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': {
            class_names[i]: float(per_class_acc[i]) 
            for i in range(num_classes)
        },
        'metrics': metrics_summary,
        'model_kwargs': model_kwargs
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"\nPer-class test metrics:")
    for class_name in class_names:
        metrics = metrics_summary['per_class'][class_name]
        print(f"\n{class_name}:")
        print(f"  Accuracy:  {per_class_acc[class_names.index(class_name)]:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.3f}")
    
    print(f"\nMacro averages:")
    print(f"  Precision: {metrics_summary['macro']['precision']:.3f}")
    print(f"  Recall:    {metrics_summary['macro']['recall']:.3f}")
    print(f"  F1-Score:  {metrics_summary['macro']['f1_score']:.3f}")
    print(f"  ROC AUC:   {metrics_summary['macro']['roc_auc']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return model, history, metrics_summary


def main():
    parser = argparse.ArgumentParser(
        description='Train temporal change detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Dataset directory with train/val/test CSVs')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for model and results')
    parser.add_argument('--model', type=str, default='custom_cnn',
                        choices=['custom_cnn', 'efficientnet', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=40,
                        help='Early stopping patience')
    parser.add_argument('--use-focal-loss', action='store_true',
                        help='Use focal loss instead of categorical crossentropy')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone for transfer learning (EfficientNet only)')
    
    args = parser.parse_args()
    
    model_kwargs = {}
    if args.model == 'efficientnet':
        model_kwargs['freeze_backbone'] = args.freeze_backbone
    
    train_model(
        args.dataset_dir,
        args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        use_focal_loss=args.use_focal_loss,
        **model_kwargs
    )


if __name__ == '__main__':
    main()
