#!/usr/bin/env bash
# ABOUTME: Submits one generate_rollouts SLURM job per language so a run's languages execute in parallel across GPUs.
# ABOUTME: Each shard writes its own output/rollouts/.../<lang>/ tree, so shards never collide and can finish independently.
#
# Usage (login node, repo root, AFTER setup_environment_and_prefetch.sh):
#   bash scripts/slurm/submit_language_shards.sh --account def-<pi> \
#       configs/rollout_importance/qwen25_32b_mgsm.yaml [extra generate_rollouts flags]
#
# Why shard: generate_rollouts writes to output/rollouts/{dataset}/{model}/.../{lang}/,
# so per-language jobs touch disjoint paths — no locking, no coordination. Three
# languages on three GPUs finish in roughly a third of the wall clock of one job.
#
# Everything after the run config is forwarded to every shard (e.g. -np 2 -nr 5 for a
# smoke run). Each shard is independently resumable: resubmit the same command.

set -euo pipefail

ACCOUNT=""
GPUS="${SHARD_GPUS:-1}"
TIME_LIMIT="${SHARD_TIME:-23:00:00}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account) ACCOUNT="$2"; shift 2 ;;
        --gpus-per-node) GPUS="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        *) break ;;
    esac
done

RUN_CONFIG="${1:?Usage: bash scripts/slurm/submit_language_shards.sh --account def-<pi> configs/rollout_importance/<run>.yaml [extra flags]}"
shift
[[ -n "$ACCOUNT" ]] || { echo "ERROR: --account is required (e.g. --account def-<pi>)" >&2; exit 1; }
[[ -f "$RUN_CONFIG" ]] || { echo "ERROR: run config not found: $RUN_CONFIG" >&2; exit 1; }

LANGUAGES=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['languages'])")
mkdir -p slurm_logs

echo "Sharding $RUN_CONFIG across languages: $LANGUAGES"
for lang in ${LANGUAGES//,/ }; do
    echo "  submitting shard: $lang"
    sbatch --account="$ACCOUNT" \
           --gpus-per-node="$GPUS" \
           --time="$TIME_LIMIT" \
           --job-name="rollouts-${lang}" \
           scripts/slurm/generate_rollouts_job.sbatch \
           "$RUN_CONFIG" --languages "$lang" "$@"
done

echo
echo "Watch with: squeue -u \$USER"
