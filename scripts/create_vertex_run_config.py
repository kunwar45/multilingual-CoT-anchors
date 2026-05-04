#!/usr/bin/env python3
"""
Generate a per-run Vertex AI YAML config from a template.
Reads HF_TOKEN and OPENAI_API_KEY from local .env if present.

Usage:
    python scripts/create_vertex_run_config.py \
        --template experiment_a100.yaml \
        --run-name en-mgsm-qwen32b \
        --owner kunwar \
        --experiment-name mgsm-en-repro \
        --image-uri gcr.io/project-25ea6636-1c58-40fa-88b/multilingual-cot:latest
"""

import argparse
import getpass
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


TEMPLATE_DIR = Path("vertex_jobs")
RUNS_DIR = TEMPLATE_DIR / "runs"


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", s.lower()).strip("-")


def load_env_keys() -> dict[str, str]:
    """Load HF_TOKEN and OPENAI_API_KEY from .env, then override with actual env vars."""
    keys: dict[str, str] = {}
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in ("HF_TOKEN", "OPENAI_API_KEY"):
                keys[k] = v
    for k in ("HF_TOKEN", "OPENAI_API_KEY"):
        if k in os.environ:
            keys[k] = os.environ[k]
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, help="Template filename in vertex_jobs/ (e.g. experiment_a100.yaml)")
    parser.add_argument("--run-name", required=True, help="Short descriptive name for this run")
    parser.add_argument("--owner", default=None, help="Owner slug (default: VERTEX_RUN_OWNER env, then current user)")
    parser.add_argument("--experiment-name", required=True, help="Experiment group name (maps to GCS prefix)")
    parser.add_argument("--image-uri", required=True, help="Docker image URI to run")
    args = parser.parse_args()

    owner = args.owner or os.environ.get("VERTEX_RUN_OWNER") or getpass.getuser()
    owner = slugify(owner)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = slugify(args.run_name)
    display_name = f"{owner}-{run_name}"
    output_filename = f"{timestamp}_{display_name}.yaml"

    template_path = TEMPLATE_DIR / args.template
    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    template = template_path.read_text()
    env_keys = load_env_keys()

    replacements = {
        "{{IMAGE_URI}}": args.image_uri,
        "{{DISPLAY_NAME}}": display_name,
        "{{EXPERIMENT_NAME}}": args.experiment_name,
        "{{OWNER}}": owner,
        "{{RUN_NAME}}": run_name,
        "{{TIMESTAMP}}": timestamp,
        "{{HF_TOKEN}}": env_keys.get("HF_TOKEN", ""),
        "{{OPENAI_API_KEY}}": env_keys.get("OPENAI_API_KEY", ""),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUNS_DIR / output_filename
    output_path.write_text(template)

    print(f"Created: {output_path}")
    print(f"Display name: {display_name}")
    print(f"Experiment:   {args.experiment_name}")

    if not env_keys.get("HF_TOKEN"):
        print("\nWARNING: HF_TOKEN not found — fill in the generated YAML manually before submitting")
    if not env_keys.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found — fill in if using GPT-4o labeling (analyze_rollouts)")

    print(f"\nNext step:")
    print(f"  scripts/submit_vertex_job.sh \\")
    print(f"    --project project-25ea6636-1c58-40fa-88b \\")
    print(f"    --region us-central1 \\")
    print(f"    --display-name {display_name} \\")
    print(f"    --config {output_path}")


if __name__ == "__main__":
    main()
