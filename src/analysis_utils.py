"""
Analysis Utilities for Histopathology Classification
=====================================================

This module provides utilities for model analysis, metrics calculation,
and visualization.

AUDIT FIXES:
- 7.2.1: Fixed get_flops() function
- Improved error handling and documentation

Author: Refactored by MLOps Engineer
Date: November 2025
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


def get_flops(model, batch_size: int = 1, verbose: bool = True) -> int:
    """
    Calculate FLOPs (Floating Point Operations) for a Keras model.
    
    AUDIT FIX 7.2.1: Fixed the original implementation that failed with:
    "Warning: Could not calculate FLOPs: 'tuple' object has no attribute 'graph'"
    
    The issue was that convert_variables_to_constants_v2_as_graph returns a tuple
    in newer TensorFlow versions, not a single object.
    
    Args:
        model: Keras model to analyze
        batch_size: Batch size for FLOPs calculation
        verbose: Print detailed information
        
    Returns:
        Total FLOPs (int), or 0 if calculation fails
    """
    try:
        # Get input shape
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Create concrete function
        @tf.function
        def forward_pass(x):
            return model(x, training=False)
        
        # Build concrete function with explicit input spec
        input_spec = tf.TensorSpec(
            shape=(batch_size,) + input_shape[1:],
            dtype=tf.float32
        )
        concrete_func = forward_pass.get_concrete_function(input_spec)
        
        # Convert to frozen graph
        # AUDIT FIX: Handle tuple return value from convert_variables_to_constants_v2_as_graph
        result = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        # Handle both tuple and single return value
        if isinstance(result, tuple):
            frozen_func, graph_def = result
        else:
            frozen_func = result
        
        # Get the graph
        if hasattr(frozen_func, 'graph'):
            graph = frozen_func.graph
        else:
            # Fallback: use the concrete function's graph
            graph = concrete_func.graph
        
        # Calculate FLOPs using profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
        flops_info = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_meta,
            cmd='op',
            options=opts
        )
        
        total_flops = flops_info.total_float_ops
        
        if verbose and total_flops > 0:
            gflops = total_flops / 1e9
            print(f"FLOPs: {total_flops:,} ({gflops:.2f} GFLOPs)")
        
        return total_flops
        
    except Exception as e:
        if verbose:
            print(f"Warning: Could not calculate FLOPs using profiler: {e}")
        
        # Fallback: Estimate FLOPs from model layers
        return _estimate_flops_fallback(model, verbose)


def _estimate_flops_fallback(model, verbose: bool = True) -> int:
    """
    Estimate FLOPs by analyzing model layers (fallback method).
    
    This provides a rough estimate when the TF profiler fails.
    """
    total_flops = 0
    
    try:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                # FLOPs ≈ 2 * K_h * K_w * C_in * C_out * H_out * W_out
                config = layer.get_config()
                kernel_size = config['kernel_size']
                filters = config['filters']
                
                if layer.output_shape is not None:
                    output_shape = layer.output_shape
                    if isinstance(output_shape, list):
                        output_shape = output_shape[0]
                    
                    h_out = output_shape[1] if output_shape[1] else 1
                    w_out = output_shape[2] if output_shape[2] else 1
                    c_in = layer.input_shape[-1] if layer.input_shape[-1] else 1
                    
                    layer_flops = 2 * kernel_size[0] * kernel_size[1] * c_in * filters * h_out * w_out
                    total_flops += layer_flops
                    
            elif isinstance(layer, tf.keras.layers.Dense):
                # FLOPs ≈ 2 * input_dim * output_dim
                units = layer.units
                input_dim = layer.input_shape[-1] if layer.input_shape[-1] else 1
                layer_flops = 2 * input_dim * units
                total_flops += layer_flops
        
        if verbose:
            if total_flops > 0:
                gflops = total_flops / 1e9
                print(f"FLOPs (estimated): {total_flops:,} ({gflops:.2f} GFLOPs)")
            else:
                print("Warning: Could not estimate FLOPs")
        
        return total_flops
        
    except Exception as e:
        if verbose:
            print(f"Warning: FLOPs estimation failed: {e}")
        return 0

def track_peak_memory():
    """
    Returns the peak GPU memory usage in MB.
    """
    try:
        # Reset peak memory stats at the beginning of tracking if needed, 
        # but here we just return the peak since the start of the program or last reset.
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        return memory_info['peak'] / (1024 ** 2)
    except Exception as e:
        print(f"Warning: Could not track GPU memory (running on CPU?): {e}")
        return 0

def measure_inference_time(model, sample_input, num_runs=100, warmup=10):
    """
    Measures average inference time in milliseconds.
    """
    # Warmup
    print(f"Warming up for {warmup} runs...")
    for _ in range(warmup):
        _ = model(sample_input, training=False)
    
    print(f"Measuring inference time over {num_runs} runs...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(sample_input, training=False)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000 # ms
    return avg_time

def calculate_metrics(y_true, y_pred_classes):
    """
    Calculates Accuracy, Precision, Recall, F1-Score.
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred_classes),
        'Precision': precision_score(y_true, y_pred_classes, average='weighted'),
        'Recall': recall_score(y_true, y_pred_classes, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred_classes, average='weighted')
    }

def plot_roc_curve(y_true, y_pred_proba, classes, figsize=(10, 8)):
    """
    Plots ROC Curve and calculates AUC for each class.
    """
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # One-hot encode y_true if it's not already
    if len(y_true.shape) == 1:
        y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
    else:
        y_true_bin = y_true

    plt.figure(figsize=figsize)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_learning_curves(history):
    """
    Plots Loss and Accuracy curves from Keras history.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_confusion_matrix(y_true, y_pred_classes, classes, figsize=(16, 7), save_path=None):
    """
    Plots two Confusion Matrix heatmaps side by side:
    - Left: Absolute values (counts)
    - Right: Normalized percentages (row-wise)
    
    Args:
        y_true: True labels
        y_pred_classes: Predicted labels
        classes: List of class names
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    # Compute confusion matrices
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Simplify class names for display (remove numeric prefix)
    display_classes = [c.split('_')[-1] if '_' in c else c for c in classes]
    
    # Create annotations with all values visible
    # For counts matrix
    annot_counts = np.array([[str(val) for val in row] for row in cm])
    
    # For percentage matrix  
    annot_pct = np.array([[f'{val:.1f}' for val in row] for row in cm_normalized])
    
    # Left plot: Absolute values
    sns.heatmap(
        cm, 
        annot=annot_counts, 
        fmt='', 
        cmap='Blues', 
        xticklabels=display_classes, 
        yticklabels=display_classes,
        ax=axes[0],
        annot_kws={'size': 10, 'weight': 'bold'},
        cbar_kws={'shrink': 0.8},
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    # Fix text color for visibility (white on dark, black on light)
    for text in axes[0].texts:
        text.set_color('white' if float(text.get_text()) > cm.max() * 0.5 else 'black')
    
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('True', fontsize=11)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=9)
    
    # Right plot: Normalized percentages
    sns.heatmap(
        cm_normalized, 
        annot=annot_pct, 
        fmt='', 
        cmap='Blues', 
        xticklabels=display_classes, 
        yticklabels=display_classes,
        ax=axes[1],
        annot_kws={'size': 10, 'weight': 'bold'},
        cbar_kws={'shrink': 0.8},
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    # Fix text color for visibility
    for text in axes[1].texts:
        text.set_color('white' if float(text.get_text()) > 50 else 'black')
    
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('True', fontsize=11)
    axes[1].set_title('Confusion Matrix (% per class)', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


# =============================================================================
# AUDIT FIX 7.3: STATISTICAL TESTS (Re-exported from statistical_tests.py)
# =============================================================================

# Import statistical tests for convenience
try:
    from statistical_tests import (
        mcnemar_test,
        mcnemar_test_multiple,
        bootstrap_ci,
        bootstrap_compare_models,
        cohens_d,
        cohens_d_from_accuracy,
        interpret_cohens_d,
        comprehensive_model_comparison,
        generate_statistical_report
    )
except ImportError:
    # Fallback if statistical_tests.py is not available
    pass


# =============================================================================
# COMPREHENSIVE EVALUATION FUNCTION
# =============================================================================

def evaluate_model_comprehensive(
    model,
    test_dataset,
    class_names,
    model_name: str = "Model",
    save_dir: str = None,
    compare_predictions: dict = None,
    verbose: bool = True
):
    """
    Perform comprehensive model evaluation with all metrics and statistical tests.
    
    AUDIT FIX: Combines all evaluation metrics in one function:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix
    - ROC curves
    - Computational efficiency (FLOPs, inference time)
    - Statistical comparison with other models (if provided)
    
    Args:
        model: Trained Keras model
        test_dataset: tf.data.Dataset for evaluation
        class_names: List of class names
        model_name: Name for reporting
        save_dir: Directory to save results (optional)
        compare_predictions: Dict of {model_name: predictions} for statistical comparison
        verbose: Print detailed results
        
    Returns:
        Dictionary with all evaluation results
    """
    from pathlib import Path
    
    results = {
        'model_name': model_name,
        'classification_metrics': {},
        'per_class_metrics': {},
        'efficiency_metrics': {},
        'statistical_comparison': None
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION: {model_name}")
        print(f"{'='*60}")
    
    # Generate predictions
    if verbose:
        print("\nGenerating predictions...")
    
    y_true = []
    y_pred_proba = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    
    # Classification metrics
    if verbose:
        print("\n--- Classification Metrics ---")
    
    results['classification_metrics'] = calculate_metrics(y_true, y_pred_classes)
    
    if verbose:
        for metric, value in results['classification_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    # Per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    results['per_class_metrics'] = report
    
    # Efficiency metrics
    if verbose:
        print("\n--- Efficiency Metrics ---")
    
    flops = get_flops(model, verbose=False)
    
    sample_input = tf.random.normal((1, 224, 224, 3))
    inference_time = measure_inference_time(model, sample_input, num_runs=50, warmup=10)
    
    peak_memory = track_peak_memory()
    
    results['efficiency_metrics'] = {
        'flops': flops,
        'flops_gflops': flops / 1e9 if flops > 0 else 0,
        'inference_time_ms': inference_time,
        'throughput_imgs_per_sec': 1000 / inference_time if inference_time > 0 else 0,
        'peak_memory_mb': peak_memory,
        'total_params': model.count_params()
    }
    
    if verbose:
        print(f"  FLOPs: {results['efficiency_metrics']['flops']:,} ({results['efficiency_metrics']['flops_gflops']:.2f} GFLOPs)")
        print(f"  Inference time: {inference_time:.2f} ms")
        print(f"  Throughput: {results['efficiency_metrics']['throughput_imgs_per_sec']:.1f} img/s")
    
    # Statistical comparison (if other predictions provided)
    if compare_predictions is not None:
        if verbose:
            print("\n--- Statistical Comparison ---")
        
        # Add current model predictions
        all_predictions = compare_predictions.copy()
        all_predictions[model_name] = y_pred_classes
        
        results['statistical_comparison'] = comprehensive_model_comparison(
            y_true,
            all_predictions,
            verbose=verbose
        )
    
    # Save results
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        import pandas as pd
        pred_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred_classes,
            **{f'prob_class_{i}': y_pred_proba[:, i] for i in range(len(class_names))}
        })
        pred_df.to_csv(save_path / 'predictions.csv', index=False)
        
        # Save metrics
        import json
        metrics_to_save = {
            'classification_metrics': results['classification_metrics'],
            'efficiency_metrics': {k: float(v) for k, v in results['efficiency_metrics'].items()}
        }
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        if verbose:
            print(f"\nResults saved to: {save_path}")
    
    # Store predictions for return
    results['y_true'] = y_true
    results['y_pred_classes'] = y_pred_classes
    results['y_pred_proba'] = y_pred_proba
    
    return results
