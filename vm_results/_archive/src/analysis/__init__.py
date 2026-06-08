from .segmentation import segment_cot, Sentence, compute_sentence_stats
from .importance import (
    compute_sensitivity_scores,
    select_candidate_anchors,
    compute_anchor_importance,
    compute_anchor_importance_full,
    AnchorResult,
    ResamplingConfig,
    FullImportanceResult,
    save_importance_results,
    load_importance_results,
    save_full_importance_results,
    load_full_importance_results,
    PLANNING_PHRASES,
    REASONING_PHRASES,
    CHECKING_PHRASES,
    CONCLUSION_PHRASES,
)
from .metrics import compute_accuracy, compute_anchor_concentration, compute_position_stats
from .cross_condition import (
    CrossConditionComparison,
    compare_conditions,
    position_normalize_importance,
    aggregate_position_curves,
    compute_anchor_invariance,
    paired_ttest_importance,
    bootstrap_confidence_interval,
)

__all__ = [
    # Segmentation
    "segment_cot",
    "Sentence",
    "compute_sentence_stats",
    # Importance
    "compute_sensitivity_scores",
    "select_candidate_anchors",
    "compute_anchor_importance",
    "compute_anchor_importance_full",
    "AnchorResult",
    "ResamplingConfig",
    "FullImportanceResult",
    "save_importance_results",
    "load_importance_results",
    "save_full_importance_results",
    "load_full_importance_results",
    # Phrase dictionaries
    "PLANNING_PHRASES",
    "REASONING_PHRASES",
    "CHECKING_PHRASES",
    "CONCLUSION_PHRASES",
    # Metrics
    "compute_accuracy",
    "compute_anchor_concentration",
    "compute_position_stats",
    # Cross-condition analysis
    "CrossConditionComparison",
    "compare_conditions",
    "position_normalize_importance",
    "aggregate_position_curves",
    "compute_anchor_invariance",
    "paired_ttest_importance",
    "bootstrap_confidence_interval",
]
