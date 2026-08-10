#!/usr/bin/env bash
# ABOUTME: SLURM job for rollout_importance stage 1 on Alliance (Compute Canada) clusters:
# ABOUTME: boots a vLLM server on the job's GPU(s), then runs generate_rollouts against it via the Vertex provider.
#
# Usage (from the repo root, AFTER scripts/slurm/setup_environment_and_prefetch.sh):
#   sbatch --account=def-<your-pi> scripts/slurm/generate_rollouts_job.sbatch configs/rollout_importance/<run>.yaml
#
# GPU sizing (override on the sbatch command line):
#   7B fp16  -> any 1 GPU:            --gpus-per-node=1
#   32B fp16 -> 2x A100-40GB (Narval): --gpus-per-node=a100:2  (tensor parallel picked up automatically)
#               or 1x H100-80GB where available
#
# The job is resumable: if it hits the time limit, resubmit the same command —
# generate_rollouts only fills in missing rollouts.

#SBATCH --job-name=generate-rollouts
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

RUN_CONFIG="${1:?Usage: sbatch ... generate_rollouts_job.sbatch configs/rollout_importance/<run>.yaml}"

# Compute nodes have no internet: everything was prefetched by setup_environment_and_prefetch.sh
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load python/3.11 2>/dev/null || module load python 2>/dev/null || true
source venv/bin/activate

MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['model'])")
PORT="${VLLM_PORT:-8000}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

echo "[job] serving $MODEL on $NUM_GPUS GPU(s), port $PORT"
vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$NUM_GPUS" \
    > "slurm_logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Wait for the server (32B weights off scratch can take several minutes to load)
for _ in $(seq 1 180); do
    curl -sf "http://localhost:${PORT}/health" > /dev/null && break
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "[job] vLLM died — see slurm_logs/vllm_${SLURM_JOB_ID}.log" >&2; exit 1; }
    sleep 10
done
curl -sf "http://localhost:${PORT}/health" > /dev/null || { echo "[job] vLLM never became healthy" >&2; exit 1; }
echo "[job] vLLM is up"

export VERTEX_API_BASE_URL="http://localhost:${PORT}/v1"
python -m src.rollout_importance.generate_rollouts \
    --config "$RUN_CONFIG" \
    --provider Vertex \
    --max_concurrent_requests 32

echo "[job] done — results under output/rollouts/; publish from a login node (compute nodes are offline)"
