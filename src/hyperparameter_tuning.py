# =============================================================================
# HYPERPARAMETER TUNING MODULE
# =============================================================================
# Módulo compartido para validación de hiperparámetros con Keras Tuner.
# Implementa búsqueda de refinamiento (no exploratoria) para validar
# que los hiperparámetros actuales son óptimos o cercanos al óptimo.
#
# Estrategia: Búsqueda enfocada con rangos estrechos (±20-30% de valores actuales)
# Tiempo estimado: 30-45 min por modelo con 10 trials × 15 epochs
#
# Referencias:
# - Li et al. (2018): Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
# - Bergstra & Bengio (2012): Random Search for Hyper-Parameter Optimization
# =============================================================================

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path


# =============================================================================
# TUNER CONFIGURATION
# =============================================================================

TUNER_CONFIG = {
    'max_trials': 10,           # Número máximo de combinaciones a probar
    'executions_per_trial': 1,  # Ejecuciones por trial (1 para velocidad)
    'epochs_per_trial': 15,     # Epochs por trial (suficiente para convergencia inicial)
    'early_stopping_patience': 5,  # Cortar trials malos rápidamente
    'directory': 'tuner_results',
}


# =============================================================================
# CNN BASELINE TUNER
# =============================================================================

def build_cnn_baseline_tuner(hp, img_height=150, img_width=150, num_classes=8):
    """
    Build CNN Baseline model with tunable hyperparameters.
    
    Rangos de búsqueda enfocados (±20-30% de valores actuales):
    - Learning rate: [5e-4, 2e-3] (actual: 1e-3)
    - Dropout: [0.4, 0.7] (actual: 0.6)
    - L2 reg: [1e-4, 5e-4] (actual: 3e-4)
    - Dense units: [128, 256, 512] (actual: 256)
    - Spatial dropout: [0.1, 0.2] (actual: 0.15)
    """
    # Hyperparameters to tune
    learning_rate = hp.Float('learning_rate', min_value=5e-4, max_value=2e-3, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.4, max_value=0.7, step=0.1)
    l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=5e-4, sampling='log')
    dense_units = hp.Choice('dense_units', values=[128, 256, 512])
    spatial_dropout = hp.Float('spatial_dropout', min_value=0.1, max_value=0.2, step=0.05)
    
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = inputs
    
    # Convolutional blocks (fixed architecture, tunable regularization)
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3, 3), padding='same', 
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.SpatialDropout2D(spatial_dropout)(x)
        x = layers.Dropout(0.25)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# VGG19 TUNER
# =============================================================================

def build_vgg_tuner(hp, img_height=224, img_width=224, num_classes=8):
    """
    Build VGG19 transfer learning model with tunable hyperparameters.
    
    Rangos de búsqueda enfocados:
    - Learning rate: [5e-5, 3e-4] (actual: 1e-4)
    - Dropout: [0.2, 0.5] (actual: 0.3)
    - Dense units: [64, 128, 256] (actual: 128)
    - L2 reg: [5e-5, 3e-4] (actual: 1e-4)
    """
    from tensorflow.keras.applications import VGG19
    
    learning_rate = hp.Float('learning_rate', min_value=5e-5, max_value=3e-4, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    dense_units = hp.Choice('dense_units', values=[64, 128, 256])
    l2_reg = hp.Float('l2_reg', min_value=5e-5, max_value=3e-4, sampling='log')
    
    base_model = VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3)
    )
    
    # Freeze all except block5
    base_model.trainable = True
    for layer in base_model.layers:
        if 'block5' not in layer.name:
            layer.trainable = False
    
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# RESNET50 TUNER
# =============================================================================

def build_resnet_tuner(hp, img_height=224, img_width=224, num_classes=8):
    """
    Build ResNet50 transfer learning model with tunable hyperparameters.
    
    Rangos de búsqueda enfocados:
    - Learning rate: [1e-5, 1e-4] (actual: 5e-5)
    - Dropout: [0.2, 0.5] (actual: 0.3)
    - Dense units: [512, 1024, 2048] (actual: 1024)
    - L2 reg: [5e-4, 2e-3] (actual: 1e-3)
    """
    from tensorflow.keras.applications import ResNet50
    
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    dense_units = hp.Choice('dense_units', values=[512, 1024, 2048])
    l2_reg = hp.Float('l2_reg', min_value=5e-4, max_value=2e-3, sampling='log')
    
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3)
    )
    
    # Name-based freezing from conv5_block1_out
    base_model.trainable = True
    fine_tune_from = None
    for i, layer in enumerate(base_model.layers):
        if layer.name == 'conv5_block1_out':
            fine_tune_from = i
            break
    
    if fine_tune_from:
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
    
    # Freeze all BatchNormalization
    for layer in base_model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# EFFICIENTNET TUNER
# =============================================================================

def build_efficientnet_tuner(hp, img_height=224, img_width=224, num_classes=8):
    """
    Build EfficientNetB0 transfer learning model with tunable hyperparameters.
    
    Rangos de búsqueda enfocados:
    - Learning rate: [1e-5, 1e-4] (actual: 5e-5)
    - Dropout: [0.2, 0.5] (actual: 0.3)
    - Dense units: [256, 512, 1024] (actual: 512)
    - L2 reg: [5e-4, 2e-3] (actual: 1e-3)
    """
    from tensorflow.keras.applications import EfficientNetB0
    
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    dense_units = hp.Choice('dense_units', values=[256, 512, 1024])
    l2_reg = hp.Float('l2_reg', min_value=5e-4, max_value=2e-3, sampling='log')
    
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3)
    )
    
    # Name-based freezing
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    
    trainable_blocks = ['block7', 'block6', 'top_conv', 'top_bn']
    for layer in base_model.layers:
        if any(block in layer.name for block in trainable_blocks):
            layer.trainable = True
    
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# TUNER EXECUTION HELPER
# =============================================================================

def run_tuner(
    build_fn,
    train_dataset,
    val_dataset,
    project_name,
    max_trials=10,
    epochs_per_trial=15,
    directory='tuner_results'
):
    """
    Execute Keras Tuner search with Bayesian optimization.
    
    Args:
        build_fn: Model building function with hp parameter
        train_dataset: Training tf.data.Dataset
        val_dataset: Validation tf.data.Dataset
        project_name: Name for tuner project directory
        max_trials: Maximum number of hyperparameter combinations
        epochs_per_trial: Maximum epochs per trial
        directory: Base directory for tuner results
    
    Returns:
        tuner: Fitted Keras Tuner object
        best_hps: Best hyperparameters found
    """
    # Create tuner
    tuner = kt.BayesianOptimization(
        build_fn,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory=directory,
        project_name=project_name,
        overwrite=True
    )
    
    # Callbacks for efficient search
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Run search
    print(f"\n{'='*60}")
    print(f"KERAS TUNER - {project_name}")
    print(f"{'='*60}")
    print(f"Max trials: {max_trials}")
    print(f"Epochs per trial: {epochs_per_trial}")
    print(f"Estimated time: {max_trials * 3}-{max_trials * 5} minutes")
    print(f"{'='*60}\n")
    
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_per_trial,
        callbacks=callbacks,
        verbose=1
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return tuner, best_hps


def print_tuner_results(tuner, best_hps, current_config):
    """
    Print comparison between tuner results and current configuration.
    
    Args:
        tuner: Fitted Keras Tuner object
        best_hps: Best hyperparameters from tuner
        current_config: Dictionary with current hyperparameter values
    """
    print(f"\n{'='*60}")
    print("TUNER RESULTS - HYPERPARAMETER COMPARISON")
    print(f"{'='*60}")
    print(f"{'Parameter':<20} {'Current':<15} {'Tuner Best':<15} {'Status':<10}")
    print(f"{'-'*60}")
    
    for param, current_val in current_config.items():
        try:
            tuner_val = best_hps.get(param)
            if tuner_val is not None:
                # Determine if values are similar (within 30%)
                if isinstance(current_val, (int, float)) and isinstance(tuner_val, (int, float)):
                    ratio = tuner_val / current_val if current_val != 0 else float('inf')
                    if 0.7 <= ratio <= 1.3:
                        status = "✓ Similar"
                    else:
                        status = "→ Adjust"
                else:
                    status = "✓ Match" if current_val == tuner_val else "→ Adjust"
                
                print(f"{param:<20} {str(current_val):<15} {str(tuner_val):<15} {status:<10}")
        except:
            pass
    
    print(f"{'='*60}")
    
    # Best trial summary
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    print(f"\nBest Trial Validation Accuracy: {best_trial.score:.4f}")
    print(f"{'='*60}\n")
