#!/usr/bin/env bash
# ABOUTME: SLURM job for rollout_importance stage 1 on Alliance (Compute Canada) clusters:
# ABOUTME: boots a vLLM server on the job's GPU(s), then runs generate_rollouts against it via the Vertex provider.
#
# Usage (from the repo root, AFTER scripts/slurm/setup_environment_and_prefetch.sh):
#   sbatch --account=def-<your-pi> scripts/slurm/generate_rollouts_job.sbatch configs/rollout_importance/<run>.yaml
#
# Everything after the run config is forwarded verbatim to generate_rollouts, so a
# smoke run shrinks the config from the command line:
#   sbatch --account=def-<pi> --time=1:00:00 scripts/slurm/generate_rollouts_job.sbatch \
#       configs/rollout_importance/qwen25_7b_mgsm.yaml -np 2 -nr 5
#
# GPU sizing (override on the sbatch command line). fp16 weights need ~2 GB per 1B
# params, and the job FAILS FAST below if the allocation is too small:
#   7B fp16  -> ~14 GB, any 1 GPU:      --gpus-per-node=1
#   32B fp16 -> ~64 GB, needs 80 GB:    --gpus-per-node=h100:1   (Killarney)
#                                       --gpus-per-node=a100:2   (Narval, 2x40 GB, tensor parallel automatic)
# Do NOT leave 32B on a bare `--gpus-per-node=1` on a mixed cluster: an L40S (48 GB)
# or A100-40GB allocation cannot hold the weights.
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

RUN_CONFIG="${1:?Usage: sbatch ... generate_rollouts_job.sbatch configs/rollout_importance/<run>.yaml [extra generate_rollouts flags]}"
shift  # everything left in "$@" is forwarded to generate_rollouts (e.g. -np 2 -nr 5)

# Compute nodes have no internet: everything was prefetched by setup_environment_and_prefetch.sh
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load python/3.11 2>/dev/null || module load python 2>/dev/null || true
source venv/bin/activate

MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['model'])")
PORT="${VLLM_PORT:-8000}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

# --- GPU preflight: fail in seconds, not after a 10-minute weight load ------
# Killarney/Narval mix GPU sizes; a bare --gpus-per-node=1 can land 32B fp16 on a
# 48 GB L40S. Compare total allocated VRAM against ~2 GB per 1B params + overhead.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | awk '{sum += $1} END {print sum+0}')
REQUIRED_VRAM_MB=$(python - "$MODEL" <<'EOF'
import re, sys
# Parameter count from the model id (…-7B-…, …-32B-…); fp16 = 2 GB per 1B params,
# plus 10% headroom. Deliberately loose: this catches gross mismatches (32B on a
# 48GB L40S) without false-aborting a genuine 80GB card. Unknown ids skip the check.
match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", sys.argv[1].split("/")[-1])
print(int(float(match.group(1)) * 2 * 1024 * 1.1) if match else 0)
EOF
)
if [[ "$REQUIRED_VRAM_MB" -gt 0 && "$TOTAL_VRAM_MB" -lt "$REQUIRED_VRAM_MB" ]]; then
    echo "[job] ABORT: $MODEL needs ~${REQUIRED_VRAM_MB} MB of VRAM but this allocation has ${TOTAL_VRAM_MB} MB." >&2
    echo "[job] Resubmit with a bigger GPU, e.g. --gpus-per-node=h100:1 (or a100:2 on Narval)." >&2
    exit 1
fi

# --- Parallelism mode -------------------------------------------------------
# Tensor parallel splits ONE model across GPUs: it cuts per-request latency but pays
# all-reduce overhead on every layer, so throughput scales sublinearly. This job is
# throughput-bound, so when the weights fit on a SINGLE GPU we prefer data parallel
# (N independent replicas, requests load-balanced across them) for near-linear scaling.
PARALLEL_FLAG="--tensor-parallel-size"
if [[ "$NUM_GPUS" -gt 1 ]]; then
    PER_GPU_VRAM_MB=$((TOTAL_VRAM_MB / NUM_GPUS))
    if [[ "$REQUIRED_VRAM_MB" -gt 0 && "$PER_GPU_VRAM_MB" -ge "$REQUIRED_VRAM_MB" ]]; then
        # Older vLLM builds have no --data-parallel-size; fall back rather than crash.
        if vllm serve --help 2>/dev/null | grep -q -- "--data-parallel-size"; then
            PARALLEL_FLAG="--data-parallel-size"
        else
            echo "[job] vLLM has no --data-parallel-size; falling back to tensor parallel"
        fi
    fi
fi
echo "[job] serving $MODEL on $NUM_GPUS GPU(s) (${TOTAL_VRAM_MB} MB total VRAM), port $PORT"
echo "[job] parallelism: $PARALLEL_FLAG $NUM_GPUS"
vllm serve "$MODEL" \
    --port "$PORT" \
    "$PARALLEL_FLAG" "$NUM_GPUS" \
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
# "$@" holds any flags passed after the run config (e.g. -np 2 -nr 5); they come last
# so they override both the run config and the defaults set above.
python -m src.rollout_importance.generate_rollouts \
    --config "$RUN_CONFIG" \
    --provider Vertex \
    --max_concurrent_requests 128 \
    "$@"

echo "[job] done — results under output/rollouts/; publish from a login node (compute nodes are offline)"
