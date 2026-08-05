#!/usr/bin/env bash
# Build and push the Docker image to GCR.
# Usage:
#   scripts/vertex/build_and_push.sh [TAG]        # default tag: latest
#   scripts/vertex/build_and_push.sh v20260430    # versioned tag

set -euo pipefail

PROJECT="project-25ea6636-1c58-40fa-88b"
IMAGE_NAME="multilingual-cot"
REGISTRY="gcr.io"

TAG="${1:-latest}"
IMAGE_URI="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${TAG}"

echo "Building: ${IMAGE_URI}"
echo "(--platform linux/amd64 ensures x86 image for Vertex VMs even on Apple Silicon)"
docker build --platform linux/amd64 -t "${IMAGE_URI}" .

echo ""
echo "Pushing: ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "Done. Image at: ${IMAGE_URI}"
echo ""
echo "Generate a run config with:"
echo "  python scripts/vertex/create_vertex_run_config.py \\"
echo "    --template experiment_a100.yaml \\"
echo "    --run-name en-mgsm-qwen32b \\"
echo "    --owner kunwar \\"
echo "    --experiment-name mgsm-en-repro \\"
echo "    --image-uri ${IMAGE_URI}"
