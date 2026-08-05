#!/usr/bin/env python3
"""
Compute full Stage 2 resampling importance for analyzed rollouts.

This script takes the output of analyze_rollouts.py (which computes Stage 1
sensitivity scores) and computes actual importance via resampling.

Usage:
    python scripts/compute_importance.py --analysis runs/pilot/analysis/sentence_analysis.json \
        --rollouts runs/pilot/pilot_en_20_native.jsonl \
        --output runs/pilot/importance

    # With config file
    python scripts/compute_importance.py --config configs/runs/r1_mgsm_2lang_c1.json \
        --analysis runs/r1/analysis/sentence_analysis.json \
        --rollouts runs/r1/rollouts.jsonl \
        --output runs/r1/importance
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import segment_cot, Sentence
from analysis.importance import (
    ResamplingConfig,
    compute_anchor_importance_full,
    save_full_importance_results,
    FullImportanceResult,
)
from rollouts import Rollout
from rollouts.parser import parse_final_answer, normalize_number
from config import load_config


def create_generate_fn(generator, config):
    """Create a generate function for importance computation."""

    def generate_fn(prefix: str, seed: int) -> tuple[str, str]:
        """Generate completion from prefix and parse answer."""
        text, _ = generator._generate_single(prefix, seed)
        answer = parse_final_answer(text)
        return text, answer

    return generate_fn


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None:
        return False

    pred_num = normalize_number(predicted)
    gt_num = normalize_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) < 1e-6
    else:
        return predicted.strip() == ground_truth.strip()


def main():
    parser = argparse.ArgumentParser(description="Compute Stage 2 importance via resampling")

    parser.add_argument("--config", type=Path, help="Path to config JSON file")
    parser.add_argument("--analysis", type=Path, required=True, help="Path to sentence_analysis.json from analyze_rollouts.py")
    parser.add_argument("--rollouts", type=Path, required=True, help="Path to rollouts JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for importance results")

    # Resampling parameters
    parser.add_argument("--num-samples", type=int, default=None, help="Number of resamples per masked sentence")
    parser.add_argument("--max-candidates", type=int, default=None, help="Maximum candidates per rollout")
    parser.add_argument("--min-sensitivity", type=float, default=None, help="Minimum sensitivity threshold")

    # Filtering
    parser.add_argument("--limit", type=int, help="Limit number of rollouts to process")
    parser.add_argument("--only-correct", action="store_true", help="Only process correct rollouts")

    args = parser.parse_args()

    # Load config
    if args.config:
        exp_config = load_config(config_path=args.config, cli_args=args)
        print(f"Loaded config from: {args.config}")
    else:
        exp_config = load_config(cli_args=args)

    # Build resampling config
    resampling_config = ResamplingConfig(
        num_samples=args.num_samples or exp_config.analysis.num_importance_samples,
        max_candidates=args.max_candidates or exp_config.analysis.top_k_candidates,
        min_sensitivity=args.min_sensitivity or exp_config.analysis.sensitivity_threshold,
        temperature=exp_config.generation.temperature,
        top_p=exp_config.generation.top_p,
        max_new_tokens=exp_config.generation.max_new_tokens,
    )

    print(f"Resampling config:")
    print(f"  Num samples: {resampling_config.num_samples}")
    print(f"  Max candidates: {resampling_config.max_candidates}")
    print(f"  Min sensitivity: {resampling_config.min_sensitivity}")
    print("-" * 50)

    # Load analysis
    with open(args.analysis, "r", encoding="utf-8") as f:
        analyses = json.load(f)
    print(f"Loaded {len(analyses)} analyses from {args.analysis}")

    # Load rollouts
    rollouts = []
    with open(args.rollouts, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rollouts.append(Rollout.from_dict(json.loads(line)))
    print(f"Loaded {len(rollouts)} rollouts from {args.rollouts}")

    # Build lookup maps
    rollout_map = {r.rollout_id: r for r in rollouts}
    analysis_map = {a["rollout_id"]: a for a in analyses}

    # Filter rollouts
    process_rollouts = []
    for analysis in analyses:
        rollout_id = analysis["rollout_id"]
        if rollout_id not in rollout_map:
            continue

        rollout = rollout_map[rollout_id]

        if args.only_correct and not rollout.correct:
            continue

        # Must have candidates
        if not analysis.get("candidate_indices"):
            continue

        process_rollouts.append((rollout, analysis))

    if args.limit:
        process_rollouts = process_rollouts[:args.limit]

    print(f"Processing {len(process_rollouts)} rollouts")
    print("-" * 50)

    # Initialize generator (for resampling)
    from rollouts import RolloutGenerator

    rollout_config = exp_config.to_rollout_config()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = RolloutGenerator(rollout_config, output_dir)

    # Create generate function
    generate_fn = create_generate_fn(generator, exp_config)

    # Process rollouts
    results = []
    for rollout, analysis in tqdm(process_rollouts, desc="Computing importance"):
        # Reconstruct sentences from analysis
        sentences = []
        for sent_data in analysis.get("sentences", []):
            # We need start_char and end_char which might not be in analysis
            # Re-segment to get proper Sentence objects
            pass

        # Re-segment to get proper Sentence objects with char positions
        sentences = segment_cot(
            rollout.cot_text,
            language=rollout.language,
            min_sentence_length=exp_config.analysis.min_sentence_length,
        )

        # Get candidate indices from analysis
        candidate_indices = analysis.get("candidate_indices", [])

        # Get sensitivity scores from analysis
        sensitivity_scores = [
            s.get("sensitivity", 0.0)
            for s in analysis.get("sentences", [])
        ]

        # Pad if needed (if sentence count differs)
        while len(sensitivity_scores) < len(sentences):
            sensitivity_scores.append(0.0)

        # Compute importance
        result = compute_anchor_importance_full(
            rollout_id=rollout.rollout_id,
            problem_id=rollout.problem_id,
            language=rollout.language,
            condition=rollout.condition,
            sentences=sentences,
            candidate_indices=candidate_indices,
            cot_prefix=rollout.cot_text,
            ground_truth=rollout.ground_truth,
            generate_fn=generate_fn,
            check_answer_fn=check_answer,
            config=resampling_config,
            sensitivity_scores=sensitivity_scores[:len(sentences)],
        )
        results.append(result)

    # Save results
    output_path = output_dir / "full_importance.json"
    save_full_importance_results(results, output_path)
    print(f"\nSaved {len(results)} importance results to {output_path}")

    # Summary statistics
    print("\nSummary:")
    total_candidates = sum(r.num_candidates for r in results)
    avg_importance = [
        r.mean_importance for r in results
        if r.mean_importance is not None
    ]
    max_importance = [
        r.max_importance for r in results
        if r.max_importance is not None
    ]

    print(f"  Total candidates analyzed: {total_candidates}")
    if avg_importance:
        print(f"  Mean importance: {sum(avg_importance) / len(avg_importance):.4f}")
    if max_importance:
        print(f"  Max importance: {max(max_importance):.4f}")

    # Top anchors
    top_results = sorted(
        [r for r in results if r.max_importance is not None],
        key=lambda r: r.max_importance,
        reverse=True,
    )[:5]

    if top_results:
        print("\nTop 5 rollouts by max importance:")
        for r in top_results:
            print(f"  {r.rollout_id}: max={r.max_importance:.4f}, mean={r.mean_importance:.4f}")


if __name__ == "__main__":
    main()
