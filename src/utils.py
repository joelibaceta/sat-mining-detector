#!/usr/bin/env python3
"""
Utility functions for training, evaluation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score)


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
    """
    Evaluate model with comprehensive metrics.
    
    Args:
        model: Trained Keras model
        generator: Data generator
        class_names: List of class names
        output_dir: Directory to save plots (optional)
        
    Returns:
        tuple: (y_true, y_pred, confusion_matrix, metrics_summary)
    """
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
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true_one_hot[:, i], y_pred_proba[:, i])
        except:
            roc_auc = 0.0
        
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
        plot_confusion_matrix(cm, class_names, output_dir)
    
    # ============= SUMMARY =============
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


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance.
    Helps the model focus on hard examples.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balance parameter
        
    Returns:
        Loss function
    """
    import tensorflow as tf
    
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        focal = alpha * weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    
    return loss_fn
