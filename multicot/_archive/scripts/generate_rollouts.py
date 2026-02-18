#!/usr/bin/env python3
"""
Generate chain-of-thought rollouts for multilingual analysis.

Usage:
    python scripts/generate_rollouts.py --input data/processed/pilot_en_20.jsonl \
        --output runs/pilot_en_native --condition native

    python scripts/generate_rollouts.py --input data/processed/pilot_es_20.jsonl \
        --output runs/pilot_es_en_thinking --condition english_thinking

    # Using config file
    python scripts/generate_rollouts.py --config configs/runs/r1_mgsm_2lang_c1.json \
        --input data/processed/pilot_es_20.jsonl --output runs/test
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders import MGSMLoader, MGSMProblem
from prompts import (
    CONDITION_NATIVE,
    CONDITION_ENGLISH_BASELINE,
    CONDITION_ENGLISH_THINKING,
    CONDITION_TRANSLATE_SOLVE,
)
from rollouts import RolloutGenerator, RolloutConfig, create_run_manifest
from config import load_config, ExperimentConfig


CONDITIONS = {
    "native": CONDITION_NATIVE,
    "english_baseline": CONDITION_ENGLISH_BASELINE,
    "english_thinking": CONDITION_ENGLISH_THINKING,
    "translate_solve": CONDITION_TRANSLATE_SOLVE,
}


def main():
    parser = argparse.ArgumentParser(description="Generate CoT rollouts")

    # Config file option
    parser.add_argument("--config", type=Path, help="Path to config JSON file")

    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output directory for run")
    parser.add_argument(
        "--condition",
        choices=list(CONDITIONS.keys()),
        default=None,
        help="Language condition (overrides config)",
    )
    parser.add_argument("--model", default=None, help="Model name (overrides config)")
    parser.add_argument("--num-rollouts", type=int, default=None, help="Rollouts per problem")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)

    # API options
    parser.add_argument("--use-api", action="store_true", help="Use API instead of local model")
    parser.add_argument("--api-provider", choices=["openrouter", "together", "fireworks"])
    parser.add_argument("--api-key", help="API key (or set via environment)")

    args = parser.parse_args()

    # Load config (merges config file with CLI args)
    if args.config:
        exp_config = load_config(config_path=args.config, cli_args=args)
        print(f"Loaded config from: {args.config}")
    else:
        exp_config = load_config(cli_args=args)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get condition from args or config
    condition_name = args.condition or (exp_config.experiment.conditions[0] if exp_config.experiment.conditions else "native")
    condition = CONDITIONS[condition_name]

    # Convert experiment config to rollout config
    config = exp_config.to_rollout_config()

    print(f"Generating rollouts")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_dir}")
    print(f"  Condition: {condition}")
    print(f"  Model: {config.model_name}")
    print(f"  Rollouts per problem: {config.num_rollouts}")
    print("-" * 50)

    # Load problems
    loader = MGSMLoader()
    problems = loader.load_from_jsonl(input_path)
    print(f"Loaded {len(problems)} problems")

    # Create run name from input file
    run_name = f"{input_path.stem}_{args.condition}"

    # Save manifest
    manifest_path = create_run_manifest(
        run_name=run_name,
        config=config,
        condition=condition,
        problem_ids=[p.problem_id for p in problems],
        output_dir=output_dir,
    )
    print(f"Saved manifest: {manifest_path}")

    # Generate
    generator = RolloutGenerator(config, output_dir)

    def progress_callback(current, total):
        pass  # tqdm handles this

    total = len(problems) * config.num_rollouts
    with tqdm(total=total, desc="Generating") as pbar:
        def update_progress(current, total):
            pbar.update(1)

        output_path = generator.generate_rollouts(
            problems=problems,
            condition=condition,
            run_name=run_name,
            progress_callback=update_progress,
        )

    print(f"\nRollouts saved to: {output_path}")

    # Quick summary
    rollouts = generator.load_rollouts(run_name)
    correct = sum(1 for r in rollouts if r.correct)
    parsed = sum(1 for r in rollouts if r.final_answer_text is not None)

    print(f"\nSummary:")
    print(f"  Total rollouts: {len(rollouts)}")
    print(f"  Parsed answers: {parsed} ({100*parsed/len(rollouts):.1f}%)")
    print(f"  Correct: {correct} ({100*correct/len(rollouts):.1f}%)")


if __name__ == "__main__":
    main()
