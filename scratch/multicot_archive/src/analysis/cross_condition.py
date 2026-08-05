"""
Cross-Condition Analysis

Compares importance patterns and accuracy across different language conditions
(e.g., native vs english_thinking).

Key analyses:
- Anchor invariance: Do the same sentences become anchors across conditions?
- Position normalization: Compare importance curves across different-length traces
- Statistical testing: Paired t-tests, bootstrap confidence intervals
"""

import math
import statistics
from dataclasses import dataclass, asdict
from typing import Optional
import random

from .importance import AnchorResult, FullImportanceResult


@dataclass
class CrossConditionComparison:
    """
    Comparison results between two conditions.

    Contains accuracy differences, importance correlations, and position statistics.
    """
    condition_1: str
    condition_2: str

    # Accuracy comparison
    accuracy_1: float
    accuracy_2: float
    accuracy_diff: float  # condition_1 - condition_2
    accuracy_diff_pvalue: Optional[float] = None

    # Importance comparison (for matched problems)
    importance_correlation: Optional[float] = None
    importance_diff_mean: Optional[float] = None
    importance_diff_std: Optional[float] = None

    # Position statistics
    mean_anchor_position_1: Optional[float] = None  # Normalized [0,1]
    mean_anchor_position_2: Optional[float] = None
    position_correlation: Optional[float] = None

    # Anchor invariance
    anchor_overlap_ratio: Optional[float] = None  # Fraction of anchors that match

    # Sample sizes
    n_problems_1: int = 0
    n_problems_2: int = 0
    n_matched_problems: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def compare_conditions(
    results_1: list[FullImportanceResult],
    results_2: list[FullImportanceResult],
    condition_1_name: str = "condition_1",
    condition_2_name: str = "condition_2",
) -> CrossConditionComparison:
    """
    Compare importance and accuracy between two conditions.

    Args:
        results_1: Results from condition 1
        results_2: Results from condition 2
        condition_1_name: Name for condition 1
        condition_2_name: Name for condition 2

    Returns:
        CrossConditionComparison with all comparison metrics
    """
    # Compute mean baseline accuracies
    accuracy_1 = sum(r.baseline_accuracy for r in results_1) / len(results_1) if results_1 else 0
    accuracy_2 = sum(r.baseline_accuracy for r in results_2) / len(results_2) if results_2 else 0

    # Match problems by problem_id
    results_1_by_problem = {r.problem_id: r for r in results_1}
    results_2_by_problem = {r.problem_id: r for r in results_2}

    matched_problems = set(results_1_by_problem.keys()) & set(results_2_by_problem.keys())

    # Initialize comparison
    comparison = CrossConditionComparison(
        condition_1=condition_1_name,
        condition_2=condition_2_name,
        accuracy_1=accuracy_1,
        accuracy_2=accuracy_2,
        accuracy_diff=accuracy_1 - accuracy_2,
        n_problems_1=len(results_1),
        n_problems_2=len(results_2),
        n_matched_problems=len(matched_problems),
    )

    if not matched_problems:
        return comparison

    # Compute importance correlation and differences for matched problems
    importance_diffs = []
    position_1_list = []
    position_2_list = []
    overlap_counts = []

    for problem_id in matched_problems:
        r1 = results_1_by_problem[problem_id]
        r2 = results_2_by_problem[problem_id]

        # Get top anchor positions (normalized)
        if r1.top_anchor_idx is not None and r1.num_sentences > 1:
            pos_1 = r1.top_anchor_idx / (r1.num_sentences - 1)
            position_1_list.append(pos_1)

        if r2.top_anchor_idx is not None and r2.num_sentences > 1:
            pos_2 = r2.top_anchor_idx / (r2.num_sentences - 1)
            position_2_list.append(pos_2)

        # Compare importance scores
        if r1.mean_importance is not None and r2.mean_importance is not None:
            importance_diffs.append(r1.mean_importance - r2.mean_importance)

        # Compute anchor overlap
        anchors_1 = {
            ar.sentence_idx for ar in r1.anchor_results
            if ar.is_candidate and ar.importance_score is not None and ar.importance_score > 0
        }
        anchors_2 = {
            ar.sentence_idx for ar in r2.anchor_results
            if ar.is_candidate and ar.importance_score is not None and ar.importance_score > 0
        }

        if anchors_1 and anchors_2:
            overlap = len(anchors_1 & anchors_2)
            union = len(anchors_1 | anchors_2)
            overlap_counts.append(overlap / union if union > 0 else 0)

    # Compute aggregate statistics
    if importance_diffs:
        comparison.importance_diff_mean = statistics.mean(importance_diffs)
        if len(importance_diffs) > 1:
            comparison.importance_diff_std = statistics.stdev(importance_diffs)

    if position_1_list:
        comparison.mean_anchor_position_1 = statistics.mean(position_1_list)
    if position_2_list:
        comparison.mean_anchor_position_2 = statistics.mean(position_2_list)

    if position_1_list and position_2_list and len(position_1_list) == len(position_2_list):
        comparison.position_correlation = _pearson_correlation(position_1_list, position_2_list)

    if overlap_counts:
        comparison.anchor_overlap_ratio = statistics.mean(overlap_counts)

    # Paired t-test for accuracy difference
    if matched_problems and len(matched_problems) >= 5:
        acc_1_matched = [results_1_by_problem[p].baseline_accuracy for p in matched_problems]
        acc_2_matched = [results_2_by_problem[p].baseline_accuracy for p in matched_problems]
        comparison.accuracy_diff_pvalue = paired_ttest_importance(acc_1_matched, acc_2_matched)["pvalue"]

    return comparison


def position_normalize_importance(
    results: list[FullImportanceResult],
    num_bins: int = 10,
) -> list[float]:
    """
    Normalize importance scores by position to enable cross-trace comparison.

    Maps sentence positions to [0, 1] and bins importance scores.

    Args:
        results: List of FullImportanceResult
        num_bins: Number of position bins

    Returns:
        List of mean importance per bin (length = num_bins)
    """
    bins = [[] for _ in range(num_bins)]

    for result in results:
        n = result.num_sentences
        if n < 2:
            continue

        for ar in result.anchor_results:
            if ar.importance_score is None:
                continue

            # Normalize position to [0, 1]
            norm_pos = ar.sentence_idx / (n - 1)

            # Assign to bin
            bin_idx = min(int(norm_pos * num_bins), num_bins - 1)
            bins[bin_idx].append(ar.importance_score)

    # Compute mean per bin
    return [
        statistics.mean(b) if b else 0.0
        for b in bins
    ]


def aggregate_position_curves(
    all_results: list[list[FullImportanceResult]],
    num_bins: int = 10,
) -> tuple[list[float], list[float]]:
    """
    Aggregate position curves across multiple result sets.

    Returns mean and std curves for visualization.

    Args:
        all_results: List of result lists (e.g., multiple runs or conditions)
        num_bins: Number of position bins

    Returns:
        Tuple of (mean_curve, std_curve) each of length num_bins
    """
    curves = [position_normalize_importance(results, num_bins) for results in all_results]

    mean_curve = []
    std_curve = []

    for bin_idx in range(num_bins):
        bin_values = [c[bin_idx] for c in curves]
        mean_curve.append(statistics.mean(bin_values) if bin_values else 0.0)
        std_curve.append(statistics.stdev(bin_values) if len(bin_values) > 1 else 0.0)

    return mean_curve, std_curve


def compute_anchor_invariance(
    c1_results: list[FullImportanceResult],
    c2_results: list[FullImportanceResult],
    num_bins: int = 10,
) -> dict:
    """
    Compute anchor invariance between two conditions.

    Measures how consistent the importance patterns are across conditions.

    Args:
        c1_results: Results from condition 1
        c2_results: Results from condition 2
        num_bins: Number of position bins for curve comparison

    Returns:
        Dict with invariance metrics
    """
    # Position curves
    curve_1 = position_normalize_importance(c1_results, num_bins)
    curve_2 = position_normalize_importance(c2_results, num_bins)

    # Correlation between curves
    curve_correlation = _pearson_correlation(curve_1, curve_2)

    # Match by problem and compute per-problem invariance
    c1_by_problem = {r.problem_id: r for r in c1_results}
    c2_by_problem = {r.problem_id: r for r in c2_results}

    matched = set(c1_by_problem.keys()) & set(c2_by_problem.keys())

    top_anchor_matches = 0
    position_diffs = []

    for problem_id in matched:
        r1 = c1_by_problem[problem_id]
        r2 = c2_by_problem[problem_id]

        # Check if top anchor is at similar position
        if r1.top_anchor_idx is not None and r2.top_anchor_idx is not None:
            if r1.num_sentences > 1 and r2.num_sentences > 1:
                pos_1 = r1.top_anchor_idx / (r1.num_sentences - 1)
                pos_2 = r2.top_anchor_idx / (r2.num_sentences - 1)

                position_diffs.append(abs(pos_1 - pos_2))

                # Consider a match if within same decile
                if abs(pos_1 - pos_2) < 0.1:
                    top_anchor_matches += 1

    return {
        "curve_correlation": curve_correlation,
        "curve_1": curve_1,
        "curve_2": curve_2,
        "num_matched_problems": len(matched),
        "top_anchor_match_rate": top_anchor_matches / len(matched) if matched else 0,
        "mean_position_diff": statistics.mean(position_diffs) if position_diffs else None,
        "std_position_diff": statistics.stdev(position_diffs) if len(position_diffs) > 1 else None,
    }


def paired_ttest_importance(
    values_1: list[float],
    values_2: list[float],
) -> dict:
    """
    Perform paired t-test on importance/accuracy values.

    Args:
        values_1: Values from condition 1
        values_2: Values from condition 2 (must be same length, paired)

    Returns:
        Dict with t-statistic, p-value, mean difference, and confidence interval
    """
    if len(values_1) != len(values_2):
        raise ValueError("Lists must have same length for paired test")

    n = len(values_1)
    if n < 2:
        return {
            "t_statistic": None,
            "pvalue": None,
            "mean_diff": None,
            "ci_95": None,
        }

    # Compute differences
    diffs = [v1 - v2 for v1, v2 in zip(values_1, values_2)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if n > 1 else 0

    # t-statistic
    se = std_diff / math.sqrt(n) if std_diff > 0 else 0
    t_stat = mean_diff / se if se > 0 else 0

    # Approximate p-value using normal distribution (for large n)
    # For small n, this is an approximation
    pvalue = 2 * (1 - _normal_cdf(abs(t_stat)))

    # 95% confidence interval
    t_critical = 1.96  # Approximation for large n
    ci_margin = t_critical * se
    ci_95 = (mean_diff - ci_margin, mean_diff + ci_margin)

    return {
        "t_statistic": t_stat,
        "pvalue": pvalue,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_95": ci_95,
        "n": n,
    }


def bootstrap_confidence_interval(
    values: list[float],
    statistic_fn=statistics.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> dict:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: Sample values
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        seed: Random seed for reproducibility

    Returns:
        Dict with estimate, CI bounds, and bootstrap samples
    """
    if seed is not None:
        random.seed(seed)

    n = len(values)
    if n == 0:
        return {
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "se": None,
        }

    # Point estimate
    estimate = statistic_fn(values)

    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(n)]
        bootstrap_stats.append(statistic_fn(sample))

    # Sort for percentile calculation
    bootstrap_stats.sort()

    # Confidence interval
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    ci_lower = bootstrap_stats[lower_idx]
    ci_upper = bootstrap_stats[upper_idx]

    # Standard error
    se = statistics.stdev(bootstrap_stats) if len(bootstrap_stats) > 1 else 0

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
