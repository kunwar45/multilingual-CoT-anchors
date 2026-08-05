#!/usr/bin/env bash
# ABOUTME: Downloads a finished Vertex run's artifacts from its GCS URI to a local directory.
# ABOUTME: Final step of the Vertex flow; results should then be published to HF, not kept locally.
# Download Vertex run artifacts from GCS to a local directory.
# Usage:
#   scripts/vertex/download_vertex_results.sh \
#     --project project-25ea6636-1c58-40fa-88b \
#     --gcs-uri gs://genai-1010101/multilingual-cot-anchors/mgsm-en-repro/kunwar-en-mgsm-qwen32b/<run_id> \
#     --dest-dir ./vertex_downloads

set -euo pipefail

PROJECT=""
GCS_URI=""
DEST_DIR="./vertex_downloads"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)   PROJECT="$2";   shift 2 ;;
        --gcs-uri)   GCS_URI="$2";   shift 2 ;;
        --dest-dir)  DEST_DIR="$2";  shift 2 ;;
        --dry-run)   DRY_RUN=true;   shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$GCS_URI" ]]; then
    echo "Usage: $0 --gcs-uri GCS_URI [--project PROJECT] [--dest-dir DIR] [--dry-run]" >&2
    exit 1
fi

echo "Downloading from: $GCS_URI"
echo "Destination:      $DEST_DIR"
echo ""

if $DRY_RUN; then
    echo "[dry-run] Would run:"
    echo "  gcloud storage cp -r \"${GCS_URI}\" \"${DEST_DIR}/\""
    exit 0
fi

mkdir -p "$DEST_DIR"
gcloud storage cp -r "${GCS_URI}" "${DEST_DIR}/"
echo ""
echo "Done. Results in: $DEST_DIR"
