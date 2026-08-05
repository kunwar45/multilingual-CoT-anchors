#!/usr/bin/env python3
# ABOUTME: Container entrypoint for Vertex AI jobs: runs the command after `--`, then uploads output/rollouts/ to GCS.
# ABOUTME: Referenced by the root Dockerfile ENTRYPOINT; not meant to be run on a laptop.
"""
Vertex AI job wrapper for multilingual-CoT-anchors experiments.

Runs the experiment command, then uploads output/rollouts/ output to GCS.

Usage (called by the Docker container entrypoint):
    python scripts/vertex/vertex_job_runner.py -- python -m src.rollout_importance.generate_rollouts \
        --dataset mgsm --languages en --provider Local --model Qwen/Qwen2.5-32B-Instruct ...
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage as gcs


GCS_BASE = os.environ.get("GCS_OUTPUT_URI", "gs://genai-1010101/multilingual-cot-anchors")
OUTPUT_DIRS = ["output/rollouts"]


def run_command(cmd: list[str]) -> int:
    print(f"[runner] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), f"Expected gs:// URI: {uri}"
    parts = uri[5:].split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def upload_to_gcs(local_dir: str, gcs_uri: str) -> None:
    local_path = Path(local_dir)
    if not local_path.exists() or not any(local_path.rglob("*")):
        print(f"[runner] Skipping upload — {local_dir} is empty or missing", flush=True)
        return

    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    print(f"[runner] Uploading {local_dir}/ → {gcs_uri}/", flush=True)
    files_uploaded = 0
    for file_path in sorted(local_path.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(local_path)
        blob_name = f"{prefix}/{rel}" if prefix else str(rel)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(file_path))
        files_uploaded += 1

    print(f"[runner] Uploaded {files_uploaded} files to gs://{bucket_name}/{prefix}/", flush=True)


def main() -> None:
    if "--" in sys.argv:
        sep = sys.argv.index("--")
        experiment_cmd = sys.argv[sep + 1:]
    else:
        experiment_cmd = sys.argv[1:]

    if not experiment_cmd:
        print("[runner] ERROR: no experiment command provided (pass it after --)", flush=True)
        sys.exit(1)

    job_id = os.environ.get("CLOUD_ML_JOB_ID", "local")
    display_name = os.environ.get("VERTEX_DISPLAY_NAME", "unnamed")
    experiment_name = os.environ.get("VERTEX_EXPERIMENT_NAME", "default")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{GCS_BASE}/{experiment_name}/{display_name}/{timestamp}_{job_id}"

    metadata = {
        "job_id": job_id,
        "display_name": display_name,
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "gcs_prefix": run_prefix,
        "command": experiment_cmd,
        "status": "running",
    }

    meta_dir = Path(".vertex_job")
    meta_dir.mkdir(exist_ok=True)
    meta_file = meta_dir / "metadata.json"
    meta_file.write_text(json.dumps(metadata, indent=2))

    start = time.time()
    exit_code = run_command(experiment_cmd)
    elapsed = time.time() - start

    metadata["status"] = "success" if exit_code == 0 else f"failed (exit {exit_code})"
    metadata["elapsed_seconds"] = round(elapsed, 1)
    meta_file.write_text(json.dumps(metadata, indent=2))

    print(f"\n[runner] Experiment finished in {elapsed:.0f}s (exit {exit_code})", flush=True)
    print(f"[runner] Uploading artifacts to {run_prefix}", flush=True)

    for d in OUTPUT_DIRS:
        upload_to_gcs(d, f"{run_prefix}/{d}")

    upload_to_gcs(".vertex_job", f"{run_prefix}/.vertex_job")

    print(f"[runner] Run artifacts at: {run_prefix}", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
