#!/usr/bin/env bash
# ABOUTME: Runs the logprob_pivots pipeline end to end: smoke -> data -> generate -> accuracy -> pivots -> redo scaffold.
# ABOUTME: Each stage is also runnable alone via python -m scripts.logprob_pivots.<stage>; see CLAUDE.md pipeline table.
set -euo pipefail

# Simple end-to-end runner for the main pipeline.

echo "[1/6] Smoke test models"
python -m scripts.logprob_pivots.smoke_test_models

echo "[2/6] Make MGSM subset"
mkdir -p data
python -m scripts.logprob_pivots.build_mgsm_subset

echo "[3/6] Run generation"
mkdir -p output/logprob_pivots/runs
python -m scripts.logprob_pivots.generate_cot

echo "[4/6] Evaluate accuracy"
python -m scripts.logprob_pivots.eval_accuracy

echo "[5/6] Compute sentence-level pivot scores"
python -m scripts.logprob_pivots.compute_pivot_scores --only-reason

echo "[6/6] Run pivot-triggered redo scaffold (reason model, small subset)"
python -m scripts.logprob_pivots.run_redo_scaffold --model reason --n-branches 3 --limit 50

echo "Done. See output/logprob_pivots/runs/<run_id>/ for artifacts."


