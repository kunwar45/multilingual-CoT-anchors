#!/usr/bin/env python3
"""
Generate all figures for a MultiCoT run.

Takes importance results and generates publication-ready figures.

Usage:
    python scripts/generate_figures.py --importance runs/r1/importance/full_importance.json \
        --output runs/r1/figures

    # Compare two conditions
    python scripts/generate_figures.py \
        --importance-1 runs/r1/importance/full_importance.json \
        --importance-2 runs/r2/importance/full_importance.json \
        --output runs/comparison/figures
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.importance import load_full_importance_results, FullImportanceResult
from analysis.cross_condition import (
    compare_conditions,
    position_normalize_importance,
    compute_anchor_invariance,
    bootstrap_confidence_interval,
)
from figures import (
    plot_anchor_invariance,
    plot_importance_by_position,
    plot_position_comparison,
    plot_accuracy_comparison,
    plot_accuracy_by_language,
    plot_accuracy_heatmap,
    plot_effect_sizes,
    plot_bootstrap_ci,
    plot_paired_comparison,
)


def load_results(path: Path) -> list[FullImportanceResult]:
    """Load importance results from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [FullImportanceResult.from_dict(d) for d in data]


def generate_single_condition_figures(
    results: list[FullImportanceResult],
    output_dir: Path,
    condition_name: str = "results",
):
    """Generate figures for a single condition."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures for {condition_name}...")

    # 1. Overall importance by position curve
    print("  - Position importance curve...")
    curve = position_normalize_importance(results, num_bins=10)
    plot_importance_by_position(
        importance_scores=curve,
        top_k=3,
        title=f"Mean Importance by Position ({condition_name})",
        output_path=output_dir / "importance_by_position.png",
        show=False,
    )

    # 2. Accuracy by language
    print("  - Accuracy by language...")
    acc_by_lang = {}
    for r in results:
        if r.language not in acc_by_lang:
            acc_by_lang[r.language] = {"correct": 0, "total": 0}
        acc_by_lang[r.language]["total"] += 1
        if r.baseline_accuracy > 0.5:
            acc_by_lang[r.language]["correct"] += 1

    acc_results = {
        lang: {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
            "n": data["total"],
        }
        for lang, data in acc_by_lang.items()
    }

    if len(acc_results) > 1:
        plot_accuracy_comparison(
            results_by_condition=acc_results,
            title=f"Accuracy by Language ({condition_name})",
            xlabel="Language",
            output_path=output_dir / "accuracy_by_language.png",
            show=False,
        )

    # 3. Example rollout importance plot (first rollout with candidates)
    print("  - Example rollout importance...")
    for r in results:
        if r.num_candidates > 0:
            scores = [
                ar.importance_score if ar.importance_score is not None else ar.sensitivity_score
                for ar in r.anchor_results
            ]
            plot_importance_by_position(
                importance_scores=scores,
                top_k=min(3, r.num_candidates),
                title=f"Importance by Position (Problem {r.problem_id})",
                xlabel="Sentence Index",
                output_path=output_dir / f"example_importance_{r.problem_id}.png",
                show=False,
            )
            break

    print(f"  Saved figures to {output_dir}")


def generate_comparison_figures(
    results_1: list[FullImportanceResult],
    results_2: list[FullImportanceResult],
    output_dir: Path,
    condition_1_name: str = "Condition 1",
    condition_2_name: str = "Condition 2",
):
    """Generate comparison figures for two conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparison figures...")

    # 1. Anchor invariance
    print("  - Anchor invariance...")
    invariance = compute_anchor_invariance(results_1, results_2, num_bins=10)
    plot_anchor_invariance(
        curve_1=invariance["curve_1"],
        curve_2=invariance["curve_2"],
        condition_1_name=condition_1_name,
        condition_2_name=condition_2_name,
        title="Anchor Invariance Across Conditions",
        correlation=invariance["curve_correlation"],
        output_path=output_dir / "anchor_invariance.png",
        show=False,
    )

    # 2. Position comparison
    print("  - Position comparison...")
    plot_position_comparison(
        results_by_condition={
            condition_1_name: invariance["curve_1"],
            condition_2_name: invariance["curve_2"],
        },
        title="Importance by Position Comparison",
        output_path=output_dir / "position_comparison.png",
        show=False,
    )

    # 3. Accuracy comparison
    print("  - Accuracy comparison...")
    comparison = compare_conditions(
        results_1, results_2,
        condition_1_name, condition_2_name,
    )

    # Bootstrap CI for accuracy difference
    matched_1 = {r.problem_id: r.baseline_accuracy for r in results_1}
    matched_2 = {r.problem_id: r.baseline_accuracy for r in results_2}
    matched_ids = set(matched_1.keys()) & set(matched_2.keys())

    if matched_ids:
        diffs = [matched_1[pid] - matched_2[pid] for pid in matched_ids]
        ci_result = bootstrap_confidence_interval(diffs, n_bootstrap=1000, seed=42)

        acc_results = {
            condition_1_name: {
                "accuracy": comparison.accuracy_1,
                "ci_lower": comparison.accuracy_1 - 0.05,  # Placeholder
                "ci_upper": comparison.accuracy_1 + 0.05,
            },
            condition_2_name: {
                "accuracy": comparison.accuracy_2,
                "ci_lower": comparison.accuracy_2 - 0.05,
                "ci_upper": comparison.accuracy_2 + 0.05,
            },
        }

        plot_accuracy_comparison(
            results_by_condition=acc_results,
            title="Accuracy Comparison",
            output_path=output_dir / "accuracy_comparison.png",
            show=False,
        )

    # 4. Effect size plot
    print("  - Effect sizes...")
    comparisons = [
        {
            "name": f"{condition_1_name} vs {condition_2_name} (Accuracy)",
            "effect": comparison.accuracy_diff,
            "ci_lower": comparison.accuracy_diff - 0.05,  # Placeholder
            "ci_upper": comparison.accuracy_diff + 0.05,
            "pvalue": comparison.accuracy_diff_pvalue,
        },
    ]

    if comparison.importance_diff_mean is not None:
        comparisons.append({
            "name": "Mean Importance Difference",
            "effect": comparison.importance_diff_mean,
            "ci_lower": comparison.importance_diff_mean - (comparison.importance_diff_std or 0.05),
            "ci_upper": comparison.importance_diff_mean + (comparison.importance_diff_std or 0.05),
            "pvalue": None,
        })

    plot_effect_sizes(
        comparisons=comparisons,
        title="Effect Sizes",
        output_path=output_dir / "effect_sizes.png",
        show=False,
    )

    # 5. Paired comparison of baseline accuracies
    print("  - Paired comparison...")
    if matched_ids:
        values_1 = [matched_1[pid] for pid in sorted(matched_ids)]
        values_2 = [matched_2[pid] for pid in sorted(matched_ids)]

        plot_paired_comparison(
            values_1=values_1,
            values_2=values_2,
            condition_1_name=condition_1_name,
            condition_2_name=condition_2_name,
            title="Paired Accuracy Comparison",
            ylabel="Baseline Accuracy",
            output_path=output_dir / "paired_comparison.png",
            show=False,
        )

    # 6. Save summary statistics
    print("  - Summary statistics...")
    summary = {
        "comparison": comparison.to_dict(),
        "anchor_invariance": {
            "curve_correlation": invariance["curve_correlation"],
            "top_anchor_match_rate": invariance["top_anchor_match_rate"],
            "mean_position_diff": invariance["mean_position_diff"],
            "num_matched_problems": invariance["num_matched_problems"],
        },
    }

    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved comparison figures to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures for MultiCoT results")

    # Single condition mode
    parser.add_argument("--importance", type=Path, help="Path to importance results JSON")

    # Comparison mode
    parser.add_argument("--importance-1", type=Path, help="Path to condition 1 importance results")
    parser.add_argument("--importance-2", type=Path, help="Path to condition 2 importance results")
    parser.add_argument("--name-1", default="Condition 1", help="Name for condition 1")
    parser.add_argument("--name-2", default="Condition 2", help="Name for condition 2")

    # Output
    parser.add_argument("--output", type=Path, required=True, help="Output directory for figures")

    args = parser.parse_args()

    # Validate arguments
    if args.importance and (args.importance_1 or args.importance_2):
        parser.error("Cannot use both --importance and --importance-1/--importance-2")

    if (args.importance_1 and not args.importance_2) or (args.importance_2 and not args.importance_1):
        parser.error("Must provide both --importance-1 and --importance-2 for comparison")

    if not args.importance and not args.importance_1:
        parser.error("Must provide either --importance or --importance-1/--importance-2")

    # Generate figures
    if args.importance:
        # Single condition mode
        results = load_results(args.importance)
        print(f"Loaded {len(results)} results from {args.importance}")
        generate_single_condition_figures(results, args.output, args.importance.stem)

    else:
        # Comparison mode
        results_1 = load_results(args.importance_1)
        results_2 = load_results(args.importance_2)
        print(f"Loaded {len(results_1)} results from {args.importance_1}")
        print(f"Loaded {len(results_2)} results from {args.importance_2}")

        # Also generate individual figures
        generate_single_condition_figures(results_1, args.output / "condition_1", args.name_1)
        generate_single_condition_figures(results_2, args.output / "condition_2", args.name_2)

        # Generate comparison figures
        generate_comparison_figures(
            results_1, results_2,
            args.output / "comparison",
            args.name_1, args.name_2,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
