#!/usr/bin/env python3
# ABOUTME: Generates a per-run Vertex job YAML from a configs/vertex/ template, INJECTING API keys from .env.
# ABOUTME: Output goes to configs/vertex/runs/ which is gitignored BECAUSE of those keys — never commit generated configs.
"""
Generate a per-run Vertex AI YAML config from a template.

The template carries machine/infra concerns; the experiment itself (dataset,
languages, model, sampling) comes from a configs/rollout_importance/ run config
injected as {{RUN_CONFIG}}. Reads HF_TOKEN and OPENAI_API_KEY from local .env.

Usage:
    python scripts/vertex/create_vertex_run_config.py \
        --template experiment_a100.yaml \
        --run-config configs/rollout_importance/qwen25_32b_mgsm.yaml \
        --owner kunwar \
        --experiment-name mgsm-qwen32b \
        --image-uri gcr.io/project-25ea6636-1c58-40fa-88b/multilingual-cot:latest
"""

import argparse
import getpass
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


TEMPLATE_DIR = Path("configs/vertex")
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
    parser.add_argument("--template", required=True, help="Template filename in configs/vertex/ (e.g. experiment_a100.yaml)")
    parser.add_argument("--run-config", required=True,
                        help="Track run config defining the experiment (e.g. configs/rollout_importance/qwen25_32b_mgsm.yaml)")
    parser.add_argument("--run-name", default=None,
                        help="Short descriptive name for this run (default: the run config's filename stem)")
    parser.add_argument("--owner", default=None, help="Owner slug (default: VERTEX_RUN_OWNER env, then current user)")
    parser.add_argument("--experiment-name", required=True, help="Experiment group name (maps to GCS prefix)")
    parser.add_argument("--image-uri", required=True, help="Docker image URI to run")
    args = parser.parse_args()

    run_config_path = Path(args.run_config)
    if not run_config_path.exists():
        print(f"ERROR: run config not found: {run_config_path}", file=sys.stderr)
        sys.exit(1)
    # The container resolves the path relative to /app (the repo root), so it must
    # be a repo-relative path that the Docker image actually contains.
    if not str(run_config_path).startswith("configs/"):
        print(f"ERROR: run config must be a repo-relative path under configs/, got: {run_config_path}", file=sys.stderr)
        sys.exit(1)

    owner = args.owner or os.environ.get("VERTEX_RUN_OWNER") or getpass.getuser()
    owner = slugify(owner)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = slugify(args.run_name or run_config_path.stem)
    display_name = f"{owner}-{run_name}"
    output_filename = f"{timestamp}_{display_name}.yaml"

    template_path = TEMPLATE_DIR / args.template
    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    template = template_path.read_text()
    if "{{RUN_CONFIG}}" not in template:
        print(f"ERROR: template {template_path} has no {{{{RUN_CONFIG}}}} placeholder", file=sys.stderr)
        sys.exit(1)
    env_keys = load_env_keys()

    replacements = {
        "{{IMAGE_URI}}": args.image_uri,
        "{{RUN_CONFIG}}": str(run_config_path),
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
    print(f"Run config:   {run_config_path}")

    if not env_keys.get("HF_TOKEN"):
        print("\nWARNING: HF_TOKEN not found — fill in the generated YAML manually before submitting")
    if not env_keys.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found — fill in if using GPT-4o labeling (compute_importance)")

    print(f"\nNext step:")
    print(f"  scripts/vertex/submit_vertex_job.sh \\")
    print(f"    --project project-25ea6636-1c58-40fa-88b \\")
    print(f"    --region us-central1 \\")
    print(f"    --display-name {display_name} \\")
    print(f"    --config {output_path}")


if __name__ == "__main__":
    main()
