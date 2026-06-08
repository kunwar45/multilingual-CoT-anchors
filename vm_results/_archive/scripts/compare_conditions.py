#!/usr/bin/env python3
"""
Compare metrics across language conditions.

Usage:
    python scripts/compare_conditions.py \
        --runs runs/pilot_en_native runs/pilot_es_native runs/pilot_es_en_thinking \
        --output figures/condition_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_summary(run_dir: Path) -> dict:
    """Load summary from a run's analysis directory."""
    summary_path = run_dir / "analysis" / "summary.json"
    if not summary_path.exists():
        # Try direct path
        summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json found in {run_dir}")

    with open(summary_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare conditions")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories")
    parser.add_argument("--output", required=True, help="Output file for comparison")
    args = parser.parse_args()

    print("Comparing conditions across runs")
    print("-" * 50)

    comparisons = {}

    for run_path in args.runs:
        run_dir = Path(run_path)
        run_name = run_dir.name

        try:
            summary = load_summary(run_dir)
            comparisons[run_name] = {
                "accuracy": summary["accuracy"],
                "total_rollouts": summary["total_rollouts"],
                "correct": summary["correct"],
                "mean_sentences": summary.get("mean_sentences", 0),
                "mean_concentration": summary.get("mean_concentration", 0),
            }
            print(f"  {run_name}: accuracy={summary['accuracy']:.2%}, "
                  f"concentration={summary.get('mean_concentration', 0):.3f}")
        except FileNotFoundError as e:
            print(f"  {run_name}: SKIPPED - {e}")

    # Compute pairwise differences
    runs = list(comparisons.keys())
    pairwise = {}

    for i, r1 in enumerate(runs):
        for r2 in runs[i + 1:]:
            acc_diff = comparisons[r1]["accuracy"] - comparisons[r2]["accuracy"]
            conc_diff = comparisons[r1]["mean_concentration"] - comparisons[r2]["mean_concentration"]
            pairwise[f"{r1}_vs_{r2}"] = {
                "accuracy_diff": acc_diff,
                "concentration_diff": conc_diff,
            }

    result = {
        "by_run": comparisons,
        "pairwise": pairwise,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved comparison: {output_path}")


if __name__ == "__main__":
    main()
