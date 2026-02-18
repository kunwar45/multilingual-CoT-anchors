#!/usr/bin/env python3
"""
Generate a job manifest from experiment configuration.

The manifest defines all tasks (rollouts) to be generated, enabling
parallel execution across multiple workers.

Usage:
    python scripts/generate_manifest.py --config configs/runs/r1_mgsm_2lang_c1.json \
        --output runs/r1_mgsm_2lang_c1

    python scripts/generate_manifest.py --config configs/runs/r_main_mgsm_5lang_200.json \
        --output runs/main_experiment --name my_run
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from data_loaders import MGSMLoader
from rollouts.manifest import generate_manifest


def main():
    parser = argparse.ArgumentParser(description="Generate job manifest")
    parser.add_argument("--config", type=Path, required=True, help="Path to config JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for manifest")
    parser.add_argument("--name", help="Custom run name (defaults to config name)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(config_path=args.config)
    print(f"Loaded config: {config.config_name}")
    print(f"  Languages: {config.experiment.languages}")
    print(f"  Conditions: {config.experiment.conditions}")
    print(f"  Num problems: {config.experiment.num_problems}")
    print(f"  Rollouts per: {config.generation.num_rollouts}")

    # Get problem IDs
    loader = MGSMLoader()
    if config.experiment.problem_ids:
        problem_ids = config.experiment.problem_ids
    else:
        problem_ids = loader.get_problem_ids(n=config.experiment.num_problems)

    print(f"  Problem IDs: {len(problem_ids)} ({problem_ids[0]} to {problem_ids[-1]})")

    # Calculate total tasks
    total_tasks = (
        len(problem_ids)
        * len(config.experiment.languages)
        * len(config.experiment.conditions)
        * config.generation.num_rollouts
    )
    print(f"  Total tasks: {total_tasks}")
    print("-" * 50)

    # Generate manifest
    run_name = args.name or config.config_name or "run"
    manifest = generate_manifest(
        run_name=run_name,
        config=config,
        problem_ids=problem_ids,
        output_dir=args.output,
    )

    # Summary
    manifest_path = args.output / f"{run_name}_manifest.json"
    print(f"\nManifest generated: {manifest_path}")
    print(f"  Run name: {manifest.run_name}")
    print(f"  Total tasks: {manifest.total_tasks}")
    print(f"  Pending: {manifest.pending_tasks}")

    # Show breakdown
    by_language = {}
    by_condition = {}
    for task in manifest.tasks:
        by_language[task.language] = by_language.get(task.language, 0) + 1
        by_condition[task.condition] = by_condition.get(task.condition, 0) + 1

    print("\nTasks by language:")
    for lang, count in sorted(by_language.items()):
        print(f"  {lang}: {count}")

    print("\nTasks by condition:")
    for cond, count in sorted(by_condition.items()):
        print(f"  {cond}: {count}")


if __name__ == "__main__":
    main()
