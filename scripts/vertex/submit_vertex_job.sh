#!/usr/bin/env bash
# ABOUTME: Submits a Vertex AI Custom Job from a generated run config in configs/vertex/runs/.
# ABOUTME: Third step of the Vertex flow; takes --project/--region/--display-name/--config.
# Submit a Vertex AI Custom Job.
# Usage:
#   scripts/vertex/submit_vertex_job.sh \
#     --project project-25ea6636-1c58-40fa-88b \
#     --region us-central1 \
#     --display-name kunwar-en-mgsm-qwen32b \
#     --config configs/vertex/runs/20260430_120000_kunwar-en-mgsm-qwen32b.yaml

set -euo pipefail

PROJECT=""
REGION="us-central1"
DISPLAY_NAME=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)   PROJECT="$2";      shift 2 ;;
        --region)    REGION="$2";       shift 2 ;;
        --display-name) DISPLAY_NAME="$2"; shift 2 ;;
        --config)    CONFIG="$2";       shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$PROJECT" || -z "$DISPLAY_NAME" || -z "$CONFIG" ]]; then
    echo "Usage: $0 --project PROJECT --display-name NAME --config YAML [--region REGION]" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 1
fi

echo "Submitting Vertex AI Custom Job"
echo "  Project:      $PROJECT"
echo "  Region:       $REGION"
echo "  Display name: $DISPLAY_NAME"
echo "  Config:       $CONFIG"
echo ""

gcloud ai custom-jobs create \
    --project="$PROJECT" \
    --region="$REGION" \
    --display-name="$DISPLAY_NAME" \
    --config="$CONFIG"
