#!/usr/bin/env bash
# ABOUTME: One-time LOGIN-NODE setup for Alliance (Compute Canada) clusters: venv + requirements + vLLM,
# ABOUTME: then prefetches the run's model, dataset, and GlotLID into $HF_HOME — compute nodes have NO internet.
#
# Usage (on a login node, from the repo root):
#   bash scripts/slurm/setup_environment_and_prefetch.sh configs/rollout_importance/<run>.yaml
#
# Re-run once per new run config (each may need a different model/dataset prefetched).
# HF_HOME defaults to $SCRATCH/hf_cache — model weights are far too big for $HOME quota.

set -euo pipefail

RUN_CONFIG="${1:?Usage: bash scripts/slurm/setup_environment_and_prefetch.sh configs/rollout_importance/<run>.yaml}"
[[ -f "$RUN_CONFIG" ]] || { echo "ERROR: run config not found: $RUN_CONFIG" >&2; exit 1; }

export HF_HOME="${HF_HOME:-${SCRATCH:?SCRATCH not set — are you on an Alliance cluster?}/hf_cache}"
mkdir -p "$HF_HOME" slurm_logs

# --- Python environment -----------------------------------------------------
module load python/3.11 2>/dev/null || module load python 2>/dev/null || true
if [[ ! -d venv ]]; then
    python -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
# Alliance wheelhouse first (fast, guaranteed ABI match), PyPI as fallback
pip install -r requirements.txt
pip install vllm

# --- Prefetch everything the compute node will need offline -----------------
MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['model'])")
DATASET=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['dataset'])")
LANGUAGES=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['languages'])")

echo "Prefetching model weights: $MODEL -> $HF_HOME"
hf download "$MODEL"

echo "Prefetching dataset ($DATASET, $LANGUAGES) and GlotLID"
python - "$DATASET" "$LANGUAGES" <<'EOF'
import sys
from src.rollout_importance.data_loaders import load_dataset_problems
from src.rollout_importance.answer_extraction import load_glotlid_model

dataset, languages = sys.argv[1], sys.argv[2].split(",")
problems = load_dataset_problems(dataset=dataset, languages=languages, num_problems=1)
assert problems, f"no problems loaded for {dataset} {languages}"
load_glotlid_model()
print("Prefetch OK: dataset + GlotLID cached")
EOF

echo
echo "Setup complete. Submit with:"
echo "  sbatch --account=def-<your-pi> scripts/slurm/generate_rollouts_job.sbatch $RUN_CONFIG"
