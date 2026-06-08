"""
Statistical Plot Visualizations

Plots for statistical analysis results:
- Effect size forest plots
- Bootstrap confidence interval histograms
- Paired comparison plots
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


def plot_effect_sizes(
    comparisons: list[dict],
    title: str = "Effect Sizes",
    xlabel: str = "Effect Size (Accuracy Difference)",
    ylabel: str = "Comparison",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot forest plot of effect sizes with confidence intervals.

    Args:
        comparisons: List of dicts with keys:
            - 'name': Comparison name
            - 'effect': Effect size (e.g., accuracy difference)
            - 'ci_lower': Lower CI bound
            - 'ci_upper': Upper CI bound
            - 'pvalue': Optional p-value
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    n = len(comparisons)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))

    y_positions = np.arange(n)

    for i, comp in enumerate(comparisons):
        effect = comp['effect']
        ci_lower = comp.get('ci_lower', effect)
        ci_upper = comp.get('ci_upper', effect)
        pvalue = comp.get('pvalue')

        # Determine color based on significance
        if pvalue is not None and pvalue < 0.05:
            color = '#2ecc71' if effect > 0 else '#e74c3c'
        else:
            color = '#95a5a6'

        # Plot error bar
        ax.errorbar(
            effect, i,
            xerr=[[effect - ci_lower], [ci_upper - effect]],
            fmt='o',
            markersize=8,
            color=color,
            capsize=5,
            capthick=2,
            elinewidth=2,
        )

        # Add p-value annotation
        if pvalue is not None:
            if pvalue < 0.001:
                p_text = "p<0.001"
            elif pvalue < 0.01:
                p_text = f"p={pvalue:.3f}"
            else:
                p_text = f"p={pvalue:.2f}"
            ax.annotate(
                p_text,
                xy=(ci_upper + 0.01, i),
                fontsize=9,
                color='gray',
                va='center',
            )

    # Reference line at 0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([c['name'] for c in comparisons])

    # Invert y-axis so first comparison is at top
    ax.invert_yaxis()

    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_bootstrap_ci(
    bootstrap_samples: list[float],
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    title: str = "Bootstrap Distribution",
    xlabel: str = "Statistic Value",
    ylabel: str = "Frequency",
    n_bins: int = 50,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot bootstrap distribution with confidence interval.

    Args:
        bootstrap_samples: List of bootstrap sample statistics
        estimate: Point estimate
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        n_bins: Number of histogram bins
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(bootstrap_samples, bins=n_bins, color='#3498db', alpha=0.7, edgecolor='white')

    # CI boundaries
    ax.axvline(x=ci_lower, color='#e74c3c', linestyle='--', linewidth=2, label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax.axvline(x=ci_upper, color='#e74c3c', linestyle='--', linewidth=2)

    # Shaded CI region
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='#e74c3c')

    # Point estimate
    ax.axvline(x=estimate, color='#2ecc71', linestyle='-', linewidth=2, label=f'Estimate: {estimate:.4f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_paired_comparison(
    values_1: list[float],
    values_2: list[float],
    labels: Optional[list[str]] = None,
    condition_1_name: str = "Condition 1",
    condition_2_name: str = "Condition 2",
    title: str = "Paired Comparison",
    xlabel: str = "Observation",
    ylabel: str = "Value",
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot paired comparison with connecting lines.

    Args:
        values_1: Values for condition 1
        values_2: Values for condition 2
        labels: Optional labels for each pair
        condition_1_name: Name for condition 1
        condition_2_name: Name for condition 2
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    if len(values_1) != len(values_2):
        raise ValueError("Both value lists must have the same length")

    n = len(values_1)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 6))

    # X positions for the two conditions
    x1 = np.zeros(n)
    x2 = np.ones(n)

    # Plot connecting lines (before points so they're behind)
    for i, (v1, v2) in enumerate(zip(values_1, values_2)):
        color = '#2ecc71' if v2 > v1 else '#e74c3c' if v2 < v1 else '#95a5a6'
        ax.plot([0, 1], [v1, v2], color=color, alpha=0.5, linewidth=1)

    # Plot points
    ax.scatter(x1, values_1, color='#3498db', s=50, zorder=5, label=condition_1_name)
    ax.scatter(x2, values_2, color='#e74c3c', s=50, zorder=5, label=condition_2_name)

    # Axis configuration
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([condition_1_name, condition_2_name])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add mean lines
    mean_1 = np.mean(values_1)
    mean_2 = np.mean(values_2)
    ax.axhline(y=mean_1, xmin=0.1, xmax=0.3, color='#3498db', linestyle='--', linewidth=2)
    ax.axhline(y=mean_2, xmin=0.7, xmax=0.9, color='#e74c3c', linestyle='--', linewidth=2)

    # Add mean annotations
    ax.annotate(f'Mean: {mean_1:.3f}', xy=(-0.15, mean_1), fontsize=10, color='#3498db')
    ax.annotate(f'Mean: {mean_2:.3f}', xy=(1.05, mean_2), fontsize=10, color='#e74c3c')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_scatter_with_regression(
    x: list[float],
    y: list[float],
    labels: Optional[list[str]] = None,
    title: str = "Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    show_regression: bool = True,
    show_identity: bool = True,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Scatter plot with optional regression line.

    Args:
        x: X values
        y: Y values
        labels: Optional labels for each point
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_regression: Whether to show regression line
        show_identity: Whether to show y=x line
        output_path: Path to save figure (optional)
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x, y, color='#3498db', s=50, alpha=0.7)

    # Identity line (y = x)
    if show_identity:
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='y = x')

    # Regression line
    if show_regression and len(x) > 1:
        # Simple linear regression
        x_arr = np.array(x)
        y_arr = np.array(y)
        slope = np.cov(x_arr, y_arr)[0, 1] / np.var(x_arr)
        intercept = np.mean(y_arr) - slope * np.mean(x_arr)

        x_range = np.array([min(x), max(x)])
        y_pred = slope * x_range + intercept
        ax.plot(x_range, y_pred, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')

        # Correlation
        corr = np.corrcoef(x_arr, y_arr)[0, 1]
        ax.annotate(
            f'r = {corr:.3f}',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig
