#!/usr/bin/env python3
"""
Analyze rollouts: segment into sentences, compute sensitivity scores,
and optionally compute full importance for candidate anchors.

Usage:
    python scripts/analyze_rollouts.py --input runs/pilot_en_native/pilot_en_20_native.jsonl \
        --output runs/pilot_en_native/analysis

    # Using config file
    python scripts/analyze_rollouts.py --config configs/runs/r1_mgsm_2lang_c1.json \
        --input runs/pilot/rollouts.jsonl --output runs/pilot/analysis
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rollouts import Rollout
from analysis import (
    segment_cot,
    compute_sensitivity_scores,
    select_candidate_anchors,
    compute_sentence_stats,
    compute_accuracy,
    compute_anchor_concentration,
    compute_position_stats,
)
from config import load_config


def main():
    parser = argparse.ArgumentParser(description="Analyze rollouts")

    # Config file option
    parser.add_argument("--config", type=Path, help="Path to config JSON file")

    parser.add_argument("--input", required=True, help="Input rollouts JSONL file")
    parser.add_argument("--output", required=True, help="Output directory for analysis")
    parser.add_argument("--min-sentence-length", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="Top k candidate anchors")
    parser.add_argument("--threshold", type=float, default=None, help="Sensitivity threshold")
    args = parser.parse_args()

    # Load config
    if args.config:
        exp_config = load_config(config_path=args.config, cli_args=args)
        print(f"Loaded config from: {args.config}")
    else:
        exp_config = load_config(cli_args=args)

    # Get analysis parameters from config (CLI overrides)
    min_sentence_length = args.min_sentence_length or exp_config.analysis.min_sentence_length
    top_k = args.top_k or exp_config.analysis.top_k_candidates
    threshold = args.threshold or exp_config.analysis.sensitivity_threshold

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing rollouts")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_dir}")
    print(f"  Min sentence length: {min_sentence_length}")
    print(f"  Top-k candidates: {top_k}")
    print(f"  Sensitivity threshold: {threshold}")
    print("-" * 50)

    # Load rollouts
    rollouts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rollouts.append(Rollout.from_dict(json.loads(line)))

    print(f"Loaded {len(rollouts)} rollouts")

    # Compute accuracy
    rollout_dicts = [r.to_dict() for r in rollouts]
    accuracy_result = compute_accuracy(rollout_dicts)
    print(f"Overall accuracy: {accuracy_result.accuracy:.2%}")

    # Analyze each rollout
    all_analyses = []
    sentence_counts = []

    for rollout in tqdm(rollouts, desc="Analyzing"):
        # Segment into sentences
        sentences = segment_cot(
            rollout.cot_text,
            language=rollout.language,
            min_sentence_length=min_sentence_length,
        )

        sentence_counts.append(len(sentences))

        # Compute sensitivity scores (now with language support)
        sensitivity_scores = compute_sensitivity_scores(
            sentences, rollout.cot_text, language=rollout.language
        )

        # Select candidate anchors
        candidates = select_candidate_anchors(
            sentences,
            sensitivity_scores,
            top_k=top_k,
            threshold=threshold,
        )

        # Compute concentration metrics
        from analysis.importance import AnchorResult
        results = [
            AnchorResult(
                sentence_idx=s.idx,
                sentence_text=s.text,
                sensitivity_score=sensitivity_scores[s.idx],
                importance_score=None,
                is_candidate=s.idx in candidates,
            )
            for s in sentences
        ]

        concentration = compute_anchor_concentration(results, top_k=top_k)
        position_stats = compute_position_stats(results)

        analysis = {
            "rollout_id": rollout.rollout_id,
            "problem_id": rollout.problem_id,
            "language": rollout.language,
            "condition": rollout.condition,
            "correct": rollout.correct,
            "num_sentences": len(sentences),
            "sentences": [
                {
                    "idx": s.idx,
                    "text": s.text,
                    "sensitivity": sensitivity_scores[s.idx],
                    "is_candidate": s.idx in candidates,
                }
                for s in sentences
            ],
            "candidate_indices": candidates,
            "concentration": concentration,
            "position_stats": position_stats,
        }

        all_analyses.append(analysis)

    # Save analyses
    analysis_path = output_dir / "sentence_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    print(f"Saved analysis: {analysis_path}")

    # Summary statistics
    summary = {
        "total_rollouts": len(rollouts),
        "accuracy": accuracy_result.accuracy,
        "correct": accuracy_result.correct,
        "mean_sentences": sum(sentence_counts) / len(sentence_counts),
        "min_sentences": min(sentence_counts),
        "max_sentences": max(sentence_counts),
        "by_problem": accuracy_result.by_problem,
    }

    # Aggregate concentration across rollouts
    all_concentrations = [a["concentration"]["concentration_ratio"] for a in all_analyses]
    summary["mean_concentration"] = sum(all_concentrations) / len(all_concentrations)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\nSummary:")
    print(f"  Accuracy: {summary['accuracy']:.2%}")
    print(f"  Mean sentences per rollout: {summary['mean_sentences']:.1f}")
    print(f"  Mean concentration ratio: {summary['mean_concentration']:.3f}")


if __name__ == "__main__":
    main()
