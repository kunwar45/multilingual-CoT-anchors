"""
Accuracy Chart Visualizations

Plots for comparing accuracy across conditions and languages:
- Bar charts with error bars
- Grouped comparisons
- Heatmaps
"""

from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _apply_style():
    """Apply default style settings."""
    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
    })


def plot_accuracy_comparison(
    results_by_condition: dict[str, dict],
    title: str = "Accuracy by Condition",
    xlabel: str = "Condition",
    ylabel: str = "Accuracy",
    colors: Optional[dict[str, str]] = None,
    show_error_bars: bool = True,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot accuracy comparison across conditions with error bars.

    Args:
        results_by_condition: Dict mapping condition name to dict with 'accuracy', 'ci_lower', 'ci_upper'
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Optional dict mapping condition name to color
        show_error_bars: Whether to show confidence interval error bars
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    conditions = list(results_by_condition.keys())
    accuracies = [results_by_condition[c].get('accuracy', 0) for c in conditions]

    if colors is None:
        color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        colors = {c: color_palette[i % len(color_palette)] for i, c in enumerate(conditions)}

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.5), 6))

    x = np.arange(len(conditions))
    bar_colors = [colors.get(c, f'C{i}') for i, c in enumerate(conditions)]

    if show_error_bars:
        # Compute error bar values
        ci_lower = [results_by_condition[c].get('ci_lower', accuracies[i]) for i, c in enumerate(conditions)]
        ci_upper = [results_by_condition[c].get('ci_upper', accuracies[i]) for i, c in enumerate(conditions)]
        yerr = [
            [acc - cl for acc, cl in zip(accuracies, ci_lower)],
            [cu - acc for acc, cu in zip(accuracies, ci_upper)],
        ]
        ax.bar(x, accuracies, color=bar_colors, edgecolor='white', linewidth=1,
               yerr=yerr, capsize=5, error_kw={'linewidth': 2})
    else:
        ax.bar(x, accuracies, color=bar_colors, edgecolor='white', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for i, (acc, xi) in enumerate(zip(accuracies, x)):
        ax.annotate(
            f'{acc:.1%}',
            xy=(xi, acc),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            fontsize=10,
            fontweight='bold',
        )

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_accuracy_by_language(
    results: dict[str, dict[str, float]],
    title: str = "Accuracy by Language and Condition",
    xlabel: str = "Language",
    ylabel: str = "Accuracy",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot grouped bar chart of accuracy by language and condition.

    Args:
        results: Dict mapping language to dict of condition -> accuracy
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    languages = list(results.keys())
    conditions = list(next(iter(results.values())).keys())
    n_conditions = len(conditions)

    fig, ax = plt.subplots(figsize=(max(10, len(languages) * 1.5), 6))

    x = np.arange(len(languages))
    width = 0.8 / n_conditions

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    for i, condition in enumerate(conditions):
        accuracies = [results[lang].get(condition, 0) for lang in languages]
        offset = (i - n_conditions / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            accuracies,
            width * 0.9,
            label=condition,
            color=colors[i % len(colors)],
            edgecolor='white',
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_accuracy_heatmap(
    accuracy_matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "Accuracy Heatmap",
    xlabel: str = "Condition",
    ylabel: str = "Language",
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annotate: bool = True,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot heatmap of accuracy across languages and conditions.

    Args:
        accuracy_matrix: 2D list [row][col] of accuracy values
        row_labels: Labels for rows (e.g., languages)
        col_labels: Labels for columns (e.g., conditions)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        annotate: Whether to show values in cells
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.5), max(6, len(row_labels) * 0.6)))

    matrix = np.array(accuracy_matrix)

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy')

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = matrix[i, j]
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.1%}', ha='center', va='center',
                        color=text_color, fontsize=10, fontweight='bold')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_accuracy_delta(
    baseline_results: dict[str, float],
    comparison_results: dict[str, float],
    baseline_name: str = "Baseline",
    comparison_name: str = "Comparison",
    title: str = "Accuracy Change",
    xlabel: str = "Language/Problem",
    ylabel: str = "Accuracy Change",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot accuracy change between two conditions.

    Args:
        baseline_results: Dict mapping key to baseline accuracy
        comparison_results: Dict mapping key to comparison accuracy
        baseline_name: Name of baseline condition
        comparison_name: Name of comparison condition
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    keys = list(set(baseline_results.keys()) & set(comparison_results.keys()))
    keys.sort()

    deltas = [comparison_results[k] - baseline_results[k] for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.5), 6))

    x = np.arange(len(keys))
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]

    ax.bar(x, deltas, color=colors, edgecolor='white', linewidth=0.5)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({comparison_name} - {baseline_name})")
    ax.set_xticks(x)

    if len(keys) <= 20:
        ax.set_xticklabels(keys, rotation=45, ha='right')
    else:
        step = max(1, len(keys) // 20)
        ax.set_xticklabels([k if i % step == 0 else '' for i, k in enumerate(keys)], rotation=45, ha='right')

    ax.grid(True, alpha=0.3, axis='y')

    # Add mean delta line
    mean_delta = np.mean(deltas)
    ax.axhline(y=mean_delta, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {mean_delta:+.1%}')
    ax.legend(loc='upper right')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig
