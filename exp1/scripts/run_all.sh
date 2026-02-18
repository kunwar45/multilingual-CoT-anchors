#!/usr/bin/env bash
set -euo pipefail

# Simple end-to-end runner for the main pipeline.

echo "[1/6] Smoke test models"
python -m scripts.smoke_test_models

echo "[2/6] Make MGSM subset"
mkdir -p data
python -m scripts.make_dataset_subset

echo "[3/6] Run generation"
mkdir -p outputs/runs
python -m scripts.run_generation

echo "[4/6] Evaluate accuracy"
python -m scripts.eval_accuracy

echo "[5/6] Compute sentence-level pivot scores"
python -m scripts.compute_sentence_kl --only-reason

echo "[6/6] Run pivot-triggered redo scaffold (reason model, small subset)"
python -m scripts.redo_scaffold --model reason --n-branches 3 --limit 50

echo "Done. See outputs/runs/<run_id>/ for artifacts."


