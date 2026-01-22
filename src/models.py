#!/usr/bin/env python3
"""
Model architectures for temporal change detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2


def channel_attention(x, ratio=8):
    """Channel attention module."""
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


class CustomCNN:
    """
    Custom CNN with channel attention (~47K parameters).
    Lightweight architecture designed for temporal change detection.
    """
    
    @staticmethod
    def build(input_shape=(48, 48, 6), num_classes=3):
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
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='custom_cnn')
        return model


class EfficientNetBasic:
    """
    EfficientNetB0 with custom classification head.
    Pre-trained on ImageNet, fine-tuned for change detection.
    """
    
    @staticmethod
    def build(input_shape=(48, 48, 6), num_classes=3, freeze_backbone=True):
        inputs = layers.Input(shape=input_shape)
        
        # Split into two 3-channel groups if input has 6 channels
        if input_shape[-1] == 6:
            # First 3 channels (RGB-like)
            x1 = layers.Lambda(lambda x: x[:, :, :, :3])(inputs)
            # Last 3 channels (IR-like)
            x2 = layers.Lambda(lambda x: x[:, :, :, 3:])(inputs)
            
            # Process each group separately with EfficientNet
            base_model1 = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(48, 48, 3),
                pooling='avg'
            )
            base_model2 = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(48, 48, 3),
                pooling='avg'
            )
            
            if freeze_backbone:
                base_model1.trainable = False
                base_model2.trainable = False
            
            features1 = base_model1(x1)
            features2 = base_model2(x2)
            
            # Concatenate features
            x = layers.Concatenate()([features1, features2])
        else:
            # Direct processing for 3 channels
            base_model = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape,
                pooling='avg'
            )
            if freeze_backbone:
                base_model.trainable = False
            x = base_model(inputs)
        
        # Classification head
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='efficientnet_basic')
        return model


class ResNet18:
    """
    ResNet18 architecture for temporal change detection.
    Adapted for small input size (48x48).
    """
    
    @staticmethod
    def _residual_block(x, filters, stride=1, downsample=None):
        """Basic residual block."""
        identity = x
        
        # First conv
        out = layers.Conv2D(filters, 3, strides=stride, padding='same',
                           kernel_regularizer=l2(0.01))(x)
        out = layers.BatchNormalization()(out)
        out = layers.Activation('relu')(out)
        
        # Second conv
        out = layers.Conv2D(filters, 3, padding='same',
                           kernel_regularizer=l2(0.01))(out)
        out = layers.BatchNormalization()(out)
        
        # Downsample identity if needed
        if downsample is not None:
            identity = downsample(x)
        
        # Add and activate
        out = layers.Add()([out, identity])
        out = layers.Activation('relu')(out)
        
        return out
    
    @staticmethod
    def _make_layer(x, filters, blocks, stride=1):
        """Create a layer with multiple residual blocks."""
        downsample = None
        
        # Downsample if stride != 1 or channels change
        if stride != 1 or x.shape[-1] != filters:
            downsample = lambda inp: layers.Conv2D(
                filters, 1, strides=stride, padding='same',
                kernel_regularizer=l2(0.01)
            )(layers.BatchNormalization()(inp))
        
        # First block (may downsample)
        x = ResNet18._residual_block(x, filters, stride, downsample)
        
        # Remaining blocks
        for _ in range(1, blocks):
            x = ResNet18._residual_block(x, filters)
        
        return x
    
    @staticmethod
    def build(input_shape=(48, 48, 6), num_classes=3):
        inputs = layers.Input(shape=input_shape)
        
        # Initial conv (adapted for 48x48, using smaller stride)
        x = layers.Conv2D(64, 7, strides=1, padding='same',
                         kernel_regularizer=l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # ResNet layers: [2, 2, 2, 2] blocks
        x = ResNet18._make_layer(x, 64, blocks=2, stride=1)
        x = layers.Dropout(0.2)(x)
        
        x = ResNet18._make_layer(x, 128, blocks=2, stride=2)
        x = layers.Dropout(0.3)(x)
        
        x = ResNet18._make_layer(x, 256, blocks=2, stride=2)
        x = layers.Dropout(0.4)(x)
        
        x = ResNet18._make_layer(x, 512, blocks=2, stride=2)
        x = layers.Dropout(0.5)(x)
        
        # Global average pooling and classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='resnet18')
        return model


def get_model(model_name='improved_cnn', input_shape=(48, 48, 6), num_classes=3, **kwargs):
    """
    Factory function to get a model by name.
    
    Args:
        model_name: One of 'improved_cnn', 'efficientnet', 'resnet18'
        input_shape: Input shape tuple
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific models
        
    Returns:
        Compiled Keras model
    """
    models = {
        'custom_cnn': CustomCNN,
        'efficientnet': EfficientNetBasic,
        'resnet18': ResNet18
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    print(f"\nBuilding model: {model_name}")
    model = models[model_name].build(input_shape=input_shape, num_classes=num_classes, **kwargs)
    
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == '__main__':
    # Test all models
    print("="*70)
    print("TESTING MODEL ARCHITECTURES")
    print("="*70)
    
    for model_name in ['custom_cnn', 'efficientnet', 'resnet18']:
        print(f"\n{model_name.upper()}:")
        print("-"*70)
        model = get_model(model_name)
        model.summary()
        print()
