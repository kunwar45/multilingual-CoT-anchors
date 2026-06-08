"""
Metrics for Multilingual Chain-of-Thought Analysis

Computes:
- Accuracy across conditions
- Anchor concentration (how focused importance is)
- Position analysis (where anchors occur)
"""

from dataclasses import dataclass
from typing import Optional
import statistics

from .importance import AnchorResult


@dataclass
class AccuracyResult:
    """Accuracy computation result."""

    total: int
    correct: int
    accuracy: float
    by_problem: dict[str, float]  # problem_id -> accuracy


def compute_accuracy(
    rollouts: list[dict],
    group_by: str = "problem_id",
) -> AccuracyResult:
    """
    Compute accuracy from rollouts.

    Args:
        rollouts: List of rollout dicts with 'correct' and group_by fields
        group_by: Field to group by for per-group accuracy

    Returns:
        AccuracyResult with overall and per-group accuracy
    """
    total = len(rollouts)
    correct = sum(1 for r in rollouts if r.get("correct", False))

    # Group by field
    groups: dict[str, list[bool]] = {}
    for r in rollouts:
        key = r.get(group_by, "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(r.get("correct", False))

    by_group = {
        key: sum(vals) / len(vals) if vals else 0
        for key, vals in groups.items()
    }

    return AccuracyResult(
        total=total,
        correct=correct,
        accuracy=correct / total if total > 0 else 0,
        by_problem=by_group,
    )


def compute_anchor_concentration(
    results: list[AnchorResult],
    top_k: int = 3,
) -> dict:
    """
    Compute how concentrated importance is into top anchors.

    Higher concentration means a few sentences do most of the work.

    Args:
        results: List of AnchorResult objects
        top_k: Number of top anchors to consider

    Returns:
        Dict with concentration metrics
    """
    # Get importance scores (use sensitivity if importance not computed)
    scores = []
    for r in results:
        if r.importance_score is not None:
            scores.append((r.sentence_idx, r.importance_score))
        else:
            scores.append((r.sentence_idx, r.sensitivity_score))

    if not scores:
        return {
            "concentration_ratio": 0,
            "gini_coefficient": 0,
            "top_k_fraction": 0,
            "top_anchor_indices": [],
        }

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    total_score = sum(s for _, s in scores)
    top_k_score = sum(s for _, s in scores[:top_k])

    # Concentration ratio: fraction of importance in top k
    concentration_ratio = top_k_score / total_score if total_score > 0 else 0

    # Gini coefficient (inequality measure)
    all_scores = sorted([s for _, s in scores])
    n = len(all_scores)
    if n > 0 and sum(all_scores) > 0:
        cumulative = 0
        gini_sum = 0
        for i, score in enumerate(all_scores):
            cumulative += score
            gini_sum += cumulative
        gini = (n + 1 - 2 * gini_sum / cumulative) / n if cumulative > 0 else 0
    else:
        gini = 0

    # Fraction of sentences that are "anchors" (above mean)
    mean_score = total_score / len(scores) if scores else 0
    above_mean = sum(1 for _, s in scores if s > mean_score)
    anchor_fraction = above_mean / len(scores) if scores else 0

    return {
        "concentration_ratio": concentration_ratio,
        "gini_coefficient": gini,
        "anchor_fraction": anchor_fraction,
        "top_k_fraction": top_k_score / total_score if total_score > 0 else 0,
        "top_anchor_indices": [idx for idx, _ in scores[:top_k]],
    }


def compute_position_stats(
    results: list[AnchorResult],
) -> dict:
    """
    Analyze where anchors occur in the trace.

    Args:
        results: List of AnchorResult objects

    Returns:
        Dict with position statistics
    """
    if not results:
        return {}

    num_sentences = len(results)

    # Get scores
    scores = []
    for r in results:
        if r.importance_score is not None:
            scores.append(r.importance_score)
        else:
            scores.append(r.sensitivity_score)

    # Weighted average position
    total_weight = sum(scores)
    if total_weight > 0:
        weighted_pos = sum(r.sentence_idx * s for r, s in zip(results, scores)) / total_weight
        relative_weighted_pos = weighted_pos / (num_sentences - 1) if num_sentences > 1 else 0
    else:
        weighted_pos = 0
        relative_weighted_pos = 0

    # Position of max score
    max_idx = max(range(len(scores)), key=lambda i: scores[i])
    relative_max_pos = max_idx / (num_sentences - 1) if num_sentences > 1 else 0

    # Early vs late distribution
    mid = num_sentences // 2
    early_score = sum(scores[:mid])
    late_score = sum(scores[mid:])
    early_late_ratio = early_score / late_score if late_score > 0 else float("inf")

    return {
        "num_sentences": num_sentences,
        "weighted_mean_position": weighted_pos,
        "relative_weighted_position": relative_weighted_pos,
        "max_score_position": max_idx,
        "relative_max_position": relative_max_pos,
        "early_late_ratio": early_late_ratio,
        "early_total_score": early_score,
        "late_total_score": late_score,
    }


def compare_conditions(
    results_by_condition: dict[str, list[dict]],
) -> dict:
    """
    Compare metrics across language conditions.

    Args:
        results_by_condition: Dict mapping condition name to rollout results

    Returns:
        Comparison statistics
    """
    comparisons = {}

    for condition, rollouts in results_by_condition.items():
        acc = compute_accuracy(rollouts)
        comparisons[condition] = {
            "accuracy": acc.accuracy,
            "total_rollouts": acc.total,
            "correct": acc.correct,
        }

    # Pairwise comparisons
    conditions = list(results_by_condition.keys())
    pairwise = {}

    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1 :]:
            diff = comparisons[c1]["accuracy"] - comparisons[c2]["accuracy"]
            pairwise[f"{c1}_vs_{c2}"] = {
                "accuracy_difference": diff,
                "c1_accuracy": comparisons[c1]["accuracy"],
                "c2_accuracy": comparisons[c2]["accuracy"],
            }

    return {
        "by_condition": comparisons,
        "pairwise": pairwise,
    }
