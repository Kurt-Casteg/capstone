"""
Statistical Tests for Model Comparison
======================================

This module implements statistical tests for rigorous model comparison
as required by the technical audit (Section 7.3 and README §8.4).

AUDIT ISSUES RESOLVED:
- 7.3.1: McNemar's Test for paired model comparison
- 7.3.2: Bootstrap Confidence Intervals (95%)
- 7.3.3: Cohen's d (effect size)
- 7.3.4: Bonferroni correction for multiple comparisons

References:
- McNemar, Q. (1947). "Note on the sampling error of the difference between 
  correlated proportions or percentages." Psychometrika, 12(2), 153-157.
- Efron, B., & Tibshirani, R. J. (1994). "An Introduction to the Bootstrap."
  Chapman and Hall/CRC.
- Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences."
  Lawrence Erlbaum Associates.

Author: MLOps Engineer
Date: November 2025
Compatibility: Python 3.8+, NumPy, SciPy
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional, Any
import warnings


# =============================================================================
# AUDIT FIX 7.3.1: McNEMAR'S TEST
# =============================================================================

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    correction: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform McNemar's test to compare two classifiers.
    
    AUDIT FIX 7.3.1: Implements McNemar's test as required by README §8.4.
    
    McNemar's test is used to determine if there is a statistically significant
    difference between the predictions of two classifiers on the same dataset.
    It only considers samples where the two classifiers disagree.
    
    The test constructs a 2x2 contingency table:
    
                        Model B Correct    Model B Incorrect
    Model A Correct         n00                 n01 (b)
    Model A Incorrect       n10 (c)             n11
    
    H0: The two classifiers have the same error rate (b = c)
    H1: The two classifiers have different error rates (b ≠ c)
    
    Reference:
        McNemar, Q. (1947). "Note on the sampling error of the difference 
        between correlated proportions or percentages." Psychometrika.
    
    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from model A
        y_pred_b: Predictions from model B
        correction: Apply continuity correction (Edwards' correction)
        verbose: Print detailed results
        
    Returns:
        Dictionary with test results:
        - statistic: McNemar's chi-squared statistic
        - p_value: Two-sided p-value
        - contingency_table: 2x2 contingency table
        - significant: Whether the difference is significant (p < 0.05)
        - interpretation: Human-readable interpretation
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    
    # Check shapes
    if not (y_true.shape == y_pred_a.shape == y_pred_b.shape):
        raise ValueError("All arrays must have the same shape")
    
    # Determine correct/incorrect predictions
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    
    # Build contingency table
    # n00: Both correct, n01: A correct B incorrect (b)
    # n10: A incorrect B correct (c), n11: Both incorrect
    n00 = np.sum(correct_a & correct_b)      # Both correct
    n01 = np.sum(correct_a & ~correct_b)     # A correct, B incorrect (b)
    n10 = np.sum(~correct_a & correct_b)     # A incorrect, B correct (c)
    n11 = np.sum(~correct_a & ~correct_b)    # Both incorrect
    
    contingency_table = np.array([[n00, n01], [n10, n11]])
    
    # b and c are the discordant pairs
    b = n01  # A correct, B incorrect
    c = n10  # A incorrect, B correct
    
    # McNemar's statistic
    if b + c == 0:
        # No discordant pairs - models are identical on this dataset
        statistic = 0.0
        p_value = 1.0
    else:
        if correction:
            # Edwards' continuity correction
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
        else:
            statistic = (b - c) ** 2 / (b + c)
        
        # Chi-squared distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    # Determine significance
    significant = p_value < 0.05
    
    # Interpretation
    if not significant:
        interpretation = "No significant difference between models (p >= 0.05)"
    elif b > c:
        interpretation = f"Model A significantly better than Model B (p = {p_value:.4f})"
    else:
        interpretation = f"Model B significantly better than Model A (p = {p_value:.4f})"
    
    results = {
        'statistic': statistic,
        'p_value': p_value,
        'contingency_table': contingency_table,
        'b_discordant': b,  # A correct, B incorrect
        'c_discordant': c,  # A incorrect, B correct
        'significant': significant,
        'interpretation': interpretation,
        'correction_applied': correction
    }
    
    if verbose:
        print("\n" + "="*60)
        print("McNEMAR'S TEST RESULTS")
        print("="*60)
        print(f"\nContingency Table:")
        print(f"                    Model B Correct  Model B Incorrect")
        print(f"  Model A Correct        {n00:5d}            {n01:5d} (b)")
        print(f"  Model A Incorrect      {n10:5d} (c)        {n11:5d}")
        print(f"\nDiscordant pairs: b={b}, c={c}")
        print(f"McNemar's statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Continuity correction: {'Yes' if correction else 'No'}")
        print(f"\nConclusion: {interpretation}")
        print("="*60)
    
    return results


def mcnemar_test_multiple(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    correction: bool = True,
    alpha: float = 0.05,
    bonferroni: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform McNemar's test for multiple model comparisons with Bonferroni correction.
    
    AUDIT FIX 7.3.4: Implements Bonferroni correction for multiple comparisons.
    
    Args:
        y_true: Ground truth labels
        predictions: Dictionary of {model_name: predictions}
        correction: Apply continuity correction
        alpha: Significance level (default 0.05)
        bonferroni: Apply Bonferroni correction
        verbose: Print detailed results
        
    Returns:
        Dictionary with pairwise comparison results
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)
    n_comparisons = n_models * (n_models - 1) // 2
    
    # Bonferroni-corrected alpha
    alpha_corrected = alpha / n_comparisons if bonferroni else alpha
    
    results = {
        'pairwise_comparisons': {},
        'n_comparisons': n_comparisons,
        'alpha_original': alpha,
        'alpha_corrected': alpha_corrected,
        'bonferroni_applied': bonferroni
    }
    
    if verbose:
        print("\n" + "="*70)
        print("PAIRWISE MODEL COMPARISONS (McNemar's Test)")
        print("="*70)
        if bonferroni:
            print(f"Bonferroni correction applied: α = {alpha} / {n_comparisons} = {alpha_corrected:.4f}")
        print()
    
    # Perform pairwise comparisons
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_a = model_names[i]
            model_b = model_names[j]
            
            comparison_key = f"{model_a}_vs_{model_b}"
            
            test_result = mcnemar_test(
                y_true,
                predictions[model_a],
                predictions[model_b],
                correction=correction,
                verbose=False
            )
            
            # Adjust significance based on corrected alpha
            test_result['significant_corrected'] = test_result['p_value'] < alpha_corrected
            
            results['pairwise_comparisons'][comparison_key] = test_result
            
            if verbose:
                sig_marker = "***" if test_result['significant_corrected'] else ""
                print(f"{model_a} vs {model_b}:")
                print(f"  χ² = {test_result['statistic']:.4f}, p = {test_result['p_value']:.4f} {sig_marker}")
                print(f"  Discordant: b={test_result['b_discordant']}, c={test_result['c_discordant']}")
                print()
    
    if verbose:
        print("*** Significant after Bonferroni correction")
        print("="*70)
    
    return results


# =============================================================================
# AUDIT FIX 7.3.2: BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    AUDIT FIX 7.3.2: Implements Bootstrap CI as required by README §8.4.
    
    Bootstrap resampling provides non-parametric confidence intervals
    for any metric without assuming a specific distribution.
    
    Reference:
        Efron, B., & Tibshirani, R. J. (1994). "An Introduction to the Bootstrap."
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        verbose: Print results
        
    Returns:
        Dictionary with:
        - point_estimate: Original metric value
        - ci_lower: Lower bound of CI
        - ci_upper: Upper bound of CI
        - std_error: Bootstrap standard error
        - bootstrap_distribution: Array of bootstrap estimates
    """
    np.random.seed(random_state)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)
    
    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)
    
    # Bootstrap resampling
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric on bootstrap sample
        estimate = metric_fn(y_true_boot, y_pred_boot)
        bootstrap_estimates.append(estimate)
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # Calculate confidence interval (percentile method)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)
    
    # Standard error
    std_error = np.std(bootstrap_estimates, ddof=1)
    
    results = {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'std_error': std_error,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'bootstrap_distribution': bootstrap_estimates
    }
    
    if verbose:
        print(f"\nBootstrap {confidence_level*100:.0f}% CI:")
        print(f"  Point estimate: {point_estimate:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Standard error: {std_error:.4f}")
    
    return results


def bootstrap_compare_models(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare two models using bootstrap confidence intervals for the difference.
    
    This tests whether the difference in performance is significantly different
    from zero by checking if the CI for the difference includes zero.
    
    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from model A
        y_pred_b: Predictions from model B
        metric_fn: Metric function (y_true, y_pred) -> scalar
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_state: Random seed
        verbose: Print results
        
    Returns:
        Dictionary with comparison results
    """
    np.random.seed(random_state)
    
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n_samples = len(y_true)
    
    # Point estimates
    metric_a = metric_fn(y_true, y_pred_a)
    metric_b = metric_fn(y_true, y_pred_b)
    diff_point = metric_a - metric_b
    
    # Bootstrap the difference
    diff_bootstrap = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_a_boot = y_pred_a[indices]
        y_pred_b_boot = y_pred_b[indices]
        
        metric_a_boot = metric_fn(y_true_boot, y_pred_a_boot)
        metric_b_boot = metric_fn(y_true_boot, y_pred_b_boot)
        diff_bootstrap.append(metric_a_boot - metric_b_boot)
    
    diff_bootstrap = np.array(diff_bootstrap)
    
    # Confidence interval for difference
    alpha = 1 - confidence_level
    ci_lower = np.percentile(diff_bootstrap, alpha / 2 * 100)
    ci_upper = np.percentile(diff_bootstrap, (1 - alpha / 2) * 100)
    
    # Significance: CI doesn't include zero
    significant = not (ci_lower <= 0 <= ci_upper)
    
    results = {
        'metric_a': metric_a,
        'metric_b': metric_b,
        'difference': diff_point,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': significant,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap
    }
    
    if verbose:
        print(f"\nBootstrap Model Comparison ({confidence_level*100:.0f}% CI):")
        print(f"  Model A: {metric_a:.4f}")
        print(f"  Model B: {metric_b:.4f}")
        print(f"  Difference (A - B): {diff_point:.4f}")
        print(f"  95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
        if significant:
            winner = "Model A" if diff_point > 0 else "Model B"
            print(f"  Conclusion: {winner} is significantly better (CI excludes 0)")
        else:
            print(f"  Conclusion: No significant difference (CI includes 0)")
    
    return results


# =============================================================================
# AUDIT FIX 7.3.3: COHEN'S D (EFFECT SIZE)
# =============================================================================

def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True
) -> float:
    """
    Calculate Cohen's d effect size.
    
    AUDIT FIX 7.3.3: Implements Cohen's d as required by README §8.4.
    
    Cohen's d measures the standardized difference between two means.
    
    Interpretation (Cohen, 1988):
    - |d| < 0.2: Negligible
    - 0.2 <= |d| < 0.5: Small
    - 0.5 <= |d| < 0.8: Medium
    - |d| >= 0.8: Large
    
    Reference:
        Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences."
    
    Args:
        group1: First group of values
        group2: Second group of values
        pooled: Use pooled standard deviation (default True)
        
    Returns:
        Cohen's d value
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    if pooled:
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std
    else:
        # Use group1's standard deviation as reference
        d = (mean1 - mean2) / np.sqrt(var1)
    
    return d


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    direction = "positive" if d > 0 else "negative" if d < 0 else "zero"
    
    return f"{magnitude} effect ({direction})"


def cohens_d_from_accuracy(
    acc_a: float,
    acc_b: float,
    n_samples: int,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calculate Cohen's d for accuracy difference using normal approximation.
    
    For binary outcomes (correct/incorrect), we can estimate effect size
    from accuracy values using the arcsine transformation.
    
    Args:
        acc_a: Accuracy of model A
        acc_b: Accuracy of model B
        n_samples: Number of test samples
        verbose: Print results
        
    Returns:
        Dictionary with effect size results
    """
    # Arcsine transformation for proportions
    # Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    h = 2 * np.arcsin(np.sqrt(acc_a)) - 2 * np.arcsin(np.sqrt(acc_b))
    
    # Standard error for the difference
    se = np.sqrt((1 / n_samples) + (1 / n_samples))
    
    interpretation = interpret_cohens_d(h)
    
    results = {
        'cohens_h': h,
        'acc_a': acc_a,
        'acc_b': acc_b,
        'difference': acc_a - acc_b,
        'interpretation': interpretation,
        'n_samples': n_samples
    }
    
    if verbose:
        print(f"\nCohen's h (effect size for proportions):")
        print(f"  Model A accuracy: {acc_a:.4f}")
        print(f"  Model B accuracy: {acc_b:.4f}")
        print(f"  Difference: {acc_a - acc_b:+.4f}")
        print(f"  Cohen's h: {h:.4f}")
        print(f"  Interpretation: {interpretation}")
    
    return results


# =============================================================================
# COMPREHENSIVE MODEL COMPARISON
# =============================================================================

def comprehensive_model_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    model_accuracies: Optional[Dict[str, float]] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical comparison of multiple models.
    
    This function combines all statistical tests required by the audit:
    - McNemar's test with Bonferroni correction
    - Bootstrap confidence intervals
    - Cohen's d effect sizes
    
    Args:
        y_true: Ground truth labels
        predictions: Dictionary of {model_name: predictions}
        model_accuracies: Optional dictionary of {model_name: accuracy}
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        verbose: Print detailed results
        
    Returns:
        Comprehensive comparison results
    """
    from sklearn.metrics import accuracy_score
    
    y_true = np.asarray(y_true)
    model_names = list(predictions.keys())
    n_samples = len(y_true)
    
    # Calculate accuracies if not provided
    if model_accuracies is None:
        model_accuracies = {
            name: accuracy_score(y_true, predictions[name])
            for name in model_names
        }
    
    results = {
        'model_accuracies': model_accuracies,
        'n_samples': n_samples,
        'n_models': len(model_names),
        'mcnemar_results': None,
        'bootstrap_results': {},
        'effect_sizes': {}
    }
    
    if verbose:
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*70)
        print(f"\nModels: {', '.join(model_names)}")
        print(f"Test samples: {n_samples}")
        print(f"Significance level: α = {alpha}")
        
        print("\n--- Model Accuracies ---")
        for name, acc in model_accuracies.items():
            print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)")
    
    # 1. McNemar's tests with Bonferroni correction
    if verbose:
        print("\n" + "-"*70)
        print("1. McNEMAR'S TESTS (with Bonferroni correction)")
        print("-"*70)
    
    results['mcnemar_results'] = mcnemar_test_multiple(
        y_true, predictions,
        correction=True,
        alpha=alpha,
        bonferroni=True,
        verbose=verbose
    )
    
    # 2. Bootstrap CIs for each model
    if verbose:
        print("\n" + "-"*70)
        print("2. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
        print("-"*70)
    
    for name in model_names:
        results['bootstrap_results'][name] = bootstrap_ci(
            y_true, predictions[name],
            metric_fn=accuracy_score,
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            verbose=verbose
        )
    
    # 3. Effect sizes (Cohen's h)
    if verbose:
        print("\n" + "-"*70)
        print("3. EFFECT SIZES (Cohen's h)")
        print("-"*70)
    
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i+1:]:
            key = f"{name_a}_vs_{name_b}"
            results['effect_sizes'][key] = cohens_d_from_accuracy(
                model_accuracies[name_a],
                model_accuracies[name_b],
                n_samples,
                verbose=verbose
            )
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)
    
    return results


# =============================================================================
# SUMMARY REPORT GENERATION
# =============================================================================

def generate_statistical_report(
    comparison_results: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a formatted statistical report from comparison results.
    
    Args:
        comparison_results: Results from comprehensive_model_comparison()
        output_path: Optional path to save the report
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Model accuracies
    lines.append("MODEL PERFORMANCE SUMMARY")
    lines.append("-" * 40)
    for name, acc in comparison_results['model_accuracies'].items():
        ci = comparison_results['bootstrap_results'].get(name, {})
        ci_str = ""
        if ci:
            ci_str = f" [95% CI: {ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        lines.append(f"  {name}: {acc:.4f}{ci_str}")
    lines.append("")
    
    # McNemar's tests
    lines.append("PAIRWISE COMPARISONS (McNemar's Test)")
    lines.append("-" * 40)
    mcnemar = comparison_results['mcnemar_results']
    lines.append(f"Bonferroni-corrected α: {mcnemar['alpha_corrected']:.4f}")
    lines.append("")
    
    for key, result in mcnemar['pairwise_comparisons'].items():
        sig = "***" if result['significant_corrected'] else ""
        lines.append(f"  {key}:")
        lines.append(f"    χ² = {result['statistic']:.4f}, p = {result['p_value']:.4f} {sig}")
    lines.append("")
    
    # Effect sizes
    lines.append("EFFECT SIZES (Cohen's h)")
    lines.append("-" * 40)
    for key, result in comparison_results['effect_sizes'].items():
        lines.append(f"  {key}:")
        lines.append(f"    h = {result['cohens_h']:.4f} ({result['interpretation']})")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("*** Significant after Bonferroni correction")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    print("Testing Statistical Tests Module...")
    print("="*60)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 750
    n_classes = 8
    
    # Simulated ground truth
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Simulated predictions with different accuracies
    # Model A: ~95% accuracy
    y_pred_a = y_true.copy()
    flip_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    y_pred_a[flip_indices] = (y_pred_a[flip_indices] + 1) % n_classes
    
    # Model B: ~90% accuracy
    y_pred_b = y_true.copy()
    flip_indices = np.random.choice(n_samples, int(n_samples * 0.10), replace=False)
    y_pred_b[flip_indices] = (y_pred_b[flip_indices] + 1) % n_classes
    
    # Model C: ~92% accuracy
    y_pred_c = y_true.copy()
    flip_indices = np.random.choice(n_samples, int(n_samples * 0.08), replace=False)
    y_pred_c[flip_indices] = (y_pred_c[flip_indices] + 1) % n_classes
    
    predictions = {
        'Model_A': y_pred_a,
        'Model_B': y_pred_b,
        'Model_C': y_pred_c
    }
    
    # Run comprehensive comparison
    print("\n1. Testing McNemar's Test...")
    mcnemar_result = mcnemar_test(y_true, y_pred_a, y_pred_b, verbose=True)
    
    print("\n2. Testing Bootstrap CI...")
    from sklearn.metrics import accuracy_score
    bootstrap_result = bootstrap_ci(y_true, y_pred_a, accuracy_score, n_bootstrap=500, verbose=True)
    
    print("\n3. Testing Cohen's d...")
    acc_a = accuracy_score(y_true, y_pred_a)
    acc_b = accuracy_score(y_true, y_pred_b)
    effect_result = cohens_d_from_accuracy(acc_a, acc_b, n_samples, verbose=True)
    
    print("\n4. Testing Comprehensive Comparison...")
    results = comprehensive_model_comparison(y_true, predictions, verbose=True)
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("="*60)
