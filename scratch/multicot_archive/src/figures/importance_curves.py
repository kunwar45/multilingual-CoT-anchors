"""
Importance Curve Visualizations

Plots for visualizing importance patterns across chain-of-thought traces:
- Importance by position curves
- Anchor invariance across conditions
- Top-k anchor highlighting
"""

from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Default style settings
STYLE = {
    "figure.figsize": (8, 6),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


def _apply_style():
    """Apply default style settings."""
    plt.rcParams.update(STYLE)


def plot_anchor_invariance(
    curve_1: list[float],
    curve_2: list[float],
    condition_1_name: str = "Condition 1",
    condition_2_name: str = "Condition 2",
    title: str = "Anchor Invariance Across Conditions",
    xlabel: str = "Normalized Position in CoT",
    ylabel: str = "Mean Importance",
    correlation: Optional[float] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot anchor invariance comparison between two conditions.

    Shows importance curves for both conditions with correlation annotation.

    Args:
        curve_1: Importance curve for condition 1 (binned)
        curve_2: Importance curve for condition 2 (binned)
        condition_1_name: Label for condition 1
        condition_2_name: Label for condition 2
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        correlation: Pearson correlation to display (optional)
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    num_bins = len(curve_1)
    x = np.linspace(0, 1, num_bins)

    # Plot curves
    ax.plot(x, curve_1, 'b-', linewidth=2, marker='o', markersize=6, label=condition_1_name)
    ax.plot(x, curve_2, 'r--', linewidth=2, marker='s', markersize=6, label=condition_2_name)

    # Fill between (shaded region)
    ax.fill_between(x, curve_1, curve_2, alpha=0.2, color='gray')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add correlation annotation
    if correlation is not None:
        ax.annotate(
            f"r = {correlation:.3f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_importance_by_position(
    importance_scores: list[float],
    sentence_texts: Optional[list[str]] = None,
    top_k: int = 3,
    title: str = "Importance by Position",
    xlabel: str = "Sentence Index",
    ylabel: str = "Importance Score",
    highlight_color: str = '#e74c3c',
    normal_color: str = '#3498db',
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot importance scores as a bar chart with top-k highlighted.

    Args:
        importance_scores: List of importance scores per sentence
        sentence_texts: Optional list of sentence texts for hover/annotation
        top_k: Number of top anchors to highlight
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        highlight_color: Color for highlighted bars
        normal_color: Color for normal bars
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    n = len(importance_scores)
    x = np.arange(n)

    # Find top-k indices
    top_indices = set(
        sorted(range(n), key=lambda i: importance_scores[i], reverse=True)[:top_k]
    )

    # Create color array
    colors = [highlight_color if i in top_indices else normal_color for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 6))

    bars = ax.bar(x, importance_scores, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)

    if n <= 20:
        ax.set_xticklabels([str(i) for i in x], rotation=0)
    else:
        # Show every 5th label for readability
        ax.set_xticklabels([str(i) if i % 5 == 0 else '' for i in x])

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Legend
    highlight_patch = mpatches.Patch(color=highlight_color, label=f'Top {top_k} Anchors')
    normal_patch = mpatches.Patch(color=normal_color, label='Other Sentences')
    ax.legend(handles=[highlight_patch, normal_patch], loc='upper right')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_position_comparison(
    results_by_condition: dict[str, list[float]],
    title: str = "Importance by Position Across Conditions",
    xlabel: str = "Normalized Position",
    ylabel: str = "Mean Importance",
    colors: Optional[dict[str, str]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare position curves across multiple conditions.

    Args:
        results_by_condition: Dict mapping condition name to importance curve
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Optional dict mapping condition name to color
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    if colors is None:
        # Default color palette
        color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        colors = {
            cond: color_palette[i % len(color_palette)]
            for i, cond in enumerate(results_by_condition.keys())
        }

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ['o', 's', '^', 'D', 'v']
    linestyles = ['-', '--', '-.', ':', '-']

    for i, (condition, curve) in enumerate(results_by_condition.items()):
        num_bins = len(curve)
        x = np.linspace(0, 1, num_bins)
        ax.plot(
            x, curve,
            color=colors.get(condition, f'C{i}'),
            linewidth=2,
            marker=markers[i % len(markers)],
            markersize=6,
            linestyle=linestyles[i % len(linestyles)],
            label=condition,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_importance_heatmap(
    importance_matrix: list[list[float]],
    problem_ids: list[str],
    num_bins: int = 10,
    title: str = "Importance Heatmap",
    xlabel: str = "Normalized Position",
    ylabel: str = "Problem",
    cmap: str = "RdYlBu_r",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot heatmap of importance across problems and positions.

    Args:
        importance_matrix: 2D list [problem][position] of importance values
        problem_ids: List of problem identifiers for y-axis
        num_bins: Number of position bins
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap name
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(12, max(6, len(problem_ids) * 0.3)))

    # Convert to numpy array for imshow
    matrix = np.array(importance_matrix)

    im = ax.imshow(matrix, aspect='auto', cmap=cmap)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Importance')

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # X ticks (position bins)
    ax.set_xticks(np.linspace(0, num_bins - 1, 5))
    ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    # Y ticks (problem IDs)
    if len(problem_ids) <= 30:
        ax.set_yticks(range(len(problem_ids)))
        ax.set_yticklabels(problem_ids)
    else:
        # Show every Nth for large datasets
        step = max(1, len(problem_ids) // 20)
        ax.set_yticks(range(0, len(problem_ids), step))
        ax.set_yticklabels(problem_ids[::step])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig
