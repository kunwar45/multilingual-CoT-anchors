#!/usr/bin/env python3
"""
Prepare pilot dataset: select 20 MGSM problems and export to JSONL.

Usage:
    python scripts/prepare_pilot_data.py [--n 20] [--languages en,es,fr]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders import MGSMLoader, select_pilot_problems


def main():
    parser = argparse.ArgumentParser(description="Prepare pilot dataset")
    parser.add_argument("--n", type=int, default=20, help="Number of problems")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--languages",
        default="en,es,fr",
        help="Comma-separated language codes",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory",
    )
    args = parser.parse_args()

    languages = args.languages.split(",")
    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing pilot data: {args.n} problems in {languages}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    loader = MGSMLoader()

    # Select problems (same IDs across all languages)
    problem_ids = select_pilot_problems(loader, n=args.n, seed=args.seed)
    print(f"Selected problems: {problem_ids[:5]}... ({len(problem_ids)} total)")

    # Export for each language
    for lang in languages:
        output_path = output_dir / f"pilot_{lang}_{args.n}.jsonl"
        loader.export_subset(output_path, problem_ids, lang)
        print(f"  Exported {lang}: {output_path}")

    # Also save the problem ID list
    ids_path = output_dir / f"pilot_problem_ids_{args.n}.txt"
    with open(ids_path, "w") as f:
        for pid in problem_ids:
            f.write(pid + "\n")
    print(f"  Problem IDs: {ids_path}")

    print("\nDone! Files ready for rollout generation.")


if __name__ == "__main__":
    main()
