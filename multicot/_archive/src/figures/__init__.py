"""
Figure Generation Module

Provides visualization functions for MultiCoT analysis results.

Modules:
- importance_curves: Importance by position, anchor invariance
- accuracy_charts: Accuracy comparisons across conditions
- statistical_plots: Effect sizes, bootstrap CIs
"""

from .importance_curves import (
    plot_anchor_invariance,
    plot_importance_by_position,
    plot_position_comparison,
)
from .accuracy_charts import (
    plot_accuracy_comparison,
    plot_accuracy_by_language,
    plot_accuracy_heatmap,
)
from .statistical_plots import (
    plot_effect_sizes,
    plot_bootstrap_ci,
    plot_paired_comparison,
)

__all__ = [
    # Importance curves
    "plot_anchor_invariance",
    "plot_importance_by_position",
    "plot_position_comparison",
    # Accuracy charts
    "plot_accuracy_comparison",
    "plot_accuracy_by_language",
    "plot_accuracy_heatmap",
    # Statistical plots
    "plot_effect_sizes",
    "plot_bootstrap_ci",
    "plot_paired_comparison",
]
