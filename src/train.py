#!/usr/bin/env python3
"""
Improved 3-class model for mining change detection.
Classes: mining_emergence, human_activity, natural_change
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from datetime import datetime
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')


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


def channel_attention(x, ratio=8):
    """Channel attention module"""
    channels = x.shape[-1]
    
    # Global average pooling
    avg_pool = layers.GlobalAveragePooling2D()(x)
    
    # Shared MLP
    dense1 = layers.Dense(channels // ratio, activation='relu')
    dense2 = layers.Dense(channels, activation='sigmoid')
    
    avg_out = dense2(dense1(avg_pool))
    
    # Reshape and multiply
    attention = layers.Reshape((1, 1, channels))(avg_out)
    return layers.Multiply()([x, attention])


def build_improved_model_3class(input_shape=(48, 48, 6), num_classes=3):
    """
    Improved model with ~20K parameters.
    Features: channel attention, better architecture, focal loss support.
    """
    from tensorflow.keras.regularizers import l2
    
    inputs = layers.Input(shape=input_shape)
    
    # First conv block with attention
    x = layers.Conv2D(32, 3, padding='same', 
                      kernel_regularizer=l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = channel_attention(x, ratio=4)
    x = layers.Dropout(0.3)(x)
    
    # Second conv block
    x = layers.Conv2D(48, 3, padding='same',
                      kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)
    
    # Third conv block with attention
    x = layers.Conv2D(64, 3, padding='same',
                      kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = channel_attention(x, ratio=8)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer (3 classes)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='improved_3class_detector')
    return model


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance.
    Helps the model focus on hard examples.
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        focal = alpha * weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    
    return loss_fn


def plot_training_history(history, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], linewidth=2, color='orange')
        axes[1, 0].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy difference (overfitting indicator)
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    acc_diff = train_acc - val_acc
    
    axes[1, 1].plot(acc_diff, linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Overfitting Indicator (Train - Val Acc)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Normalized Confusion Matrix')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm_norm[i, j]:{fmt}}\n({cm[i, j]})',
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true_one_hot, y_pred_proba, class_names, output_dir):
    """Plot ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green']
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Multi-class', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curves(y_true_one_hot, y_pred_proba, class_names, output_dir):
    """Plot Precision-Recall curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green']
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_true_one_hot[:, i], 
                                                       y_pred_proba[:, i])
        avg_precision = average_precision_score(y_true_one_hot[:, i], 
                                                y_pred_proba[:, i])
        
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{class_name} (AP = {avg_precision:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model, generator, class_names, output_dir=None):
    """Evaluate model with comprehensive metrics."""
    print("\nEvaluating model...")
    
    # Collect predictions
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for i in range(len(generator)):
        X, y = generator[i]
        preds = model.predict(X, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        y_pred_proba.extend(preds)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # One-hot encode y_true for ROC/PR curves
    num_classes = len(class_names)
    y_true_one_hot = np.zeros((len(y_true), num_classes))
    for i, label in enumerate(y_true):
        y_true_one_hot[i, label] = 1
    
    # ============= BASIC METRICS =============
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    
    # ============= CONFUSION MATRIX =============
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"{'':>20} {' '.join(f'{c:>15}' for c in class_names)}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>20} {' '.join(f'{v:>15}' for v in row)}")
    
    # ============= PER-CLASS METRICS =============
    print("\n" + "="*70)
    print("PER-CLASS DETAILED METRICS")
    print("="*70)
    
    per_class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        # Binary classification metrics for this class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true_one_hot[:, i], y_pred_proba[:, i])
        except:
            roc_auc = 0.0
        
        # Average Precision
        try:
            avg_precision = average_precision_score(y_true_one_hot[:, i], 
                                                    y_pred_proba[:, i])
        except:
            avg_precision = 0.0
        
        per_class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'support': int(np.sum(y_true == i))
        }
        
        print(f"\n{class_name}:")
        print(f"  Precision:        {precision:.3f}")
        print(f"  Recall:           {recall:.3f}")
        print(f"  F1-Score:         {f1:.3f}")
        print(f"  ROC AUC:          {roc_auc:.3f}")
        print(f"  Avg Precision:    {avg_precision:.3f}")
        print(f"  Support:          {np.sum(y_true == i)}")
    
    # ============= MACRO METRICS =============
    print("\n" + "="*70)
    print("MACRO METRICS (average across classes)")
    print("="*70)
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        macro_roc_auc = roc_auc_score(y_true_one_hot, y_pred_proba, 
                                      average='macro', multi_class='ovr')
    except:
        macro_roc_auc = 0.0
    
    print(f"Macro Precision:   {macro_precision:.3f}")
    print(f"Macro Recall:      {macro_recall:.3f}")
    print(f"Macro F1-Score:    {macro_f1:.3f}")
    print(f"Macro ROC AUC:     {macro_roc_auc:.3f}")
    
    # ============= WEIGHTED METRICS =============
    print("\n" + "="*70)
    print("WEIGHTED METRICS (weighted by support)")
    print("="*70)
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        weighted_roc_auc = roc_auc_score(y_true_one_hot, y_pred_proba,
                                         average='weighted', multi_class='ovr')
    except:
        weighted_roc_auc = 0.0
    
    print(f"Weighted Precision: {weighted_precision:.3f}")
    print(f"Weighted Recall:    {weighted_recall:.3f}")
    print(f"Weighted F1-Score:  {weighted_f1:.3f}")
    print(f"Weighted ROC AUC:   {weighted_roc_auc:.3f}")
    
    # ============= PLOT CURVES =============
    if output_dir:
        print("\nGenerating metric plots...")
        plot_roc_curves(y_true_one_hot, y_pred_proba, class_names, output_dir)
        plot_precision_recall_curves(y_true_one_hot, y_pred_proba, class_names, output_dir)
    
    metrics_summary = {
        'per_class': per_class_metrics,
        'macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1_score': float(macro_f1),
            'roc_auc': float(macro_roc_auc)
        },
        'weighted': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1_score': float(weighted_f1),
            'roc_auc': float(weighted_roc_auc)
        }
    }
    
    return y_true, y_pred, cm, metrics_summary


def train_model(dataset_dir, output_dir, epochs=150, batch_size=16, 
                learning_rate=0.0001, patience=40, use_focal_loss=False):
    """Train the improved 3-class model."""
    
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['mining_emergence', 'human_activity', 'natural_change']
    num_classes = 3
    
    print("="*70)
    print("TRAINING IMPROVED 3-CLASS TEMPORAL CHANGE DETECTOR")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
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
    model = build_improved_model_3class(input_shape=(48, 48, 6), num_classes=num_classes)
    
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
    
    print(f"\nModel: {model.name}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Loss: {loss_name}")
    
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
    y_true, y_pred, cm, metrics_summary = evaluate_model(model, test_gen, class_names, output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, output_dir)
    
    # Calculate per-class accuracies
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
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
        'metrics': metrics_summary
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
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
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train improved 3-class temporal change detector')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Dataset directory with train/val/test CSVs')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for model and results')
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
    
    args = parser.parse_args()
    
    train_model(
        args.dataset_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        use_focal_loss=args.use_focal_loss
    )


if __name__ == '__main__':
    main()
