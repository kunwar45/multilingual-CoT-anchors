"""Publish run artifacts (datasets, eval results, model files) to the Hugging Face Hub.

Implements the policy in CLAUDE.md ("Datasets, run artifacts and eval results go to
Hugging Face"): repo names are <YYYY-MM-DD>-<short-description> where the date is when
the data was GENERATED (not uploaded), and every upload carries a dataset card stating
the experiment, models, languages and provenance. Both rules are enforced here in code.

Thin CLI: scripts/publish_to_hf.py. Import as src.hf_publishing.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*$")

# Every card must fill these in; publish() refuses otherwise. `languages` is the
# field most easily lost and the one a future reader needs most — never omit it.
REQUIRED_CARD_FIELDS = (
    "experiment",      # which experiment produced this, in one sentence
    "date_generated",  # ISO date the data was produced (not uploaded)
    "track",           # logprob_pivots | rollout_importance
    "languages",       # languages covered
    "models",          # every model id involved
    "provenance",      # how to regenerate: the exact script and arguments
)

IGNORE_PATTERNS = ["__pycache__/**", "*.pyc", ".DS_Store", "._*", ".vertex_job/**"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def validate_repo_name(name: str) -> None:
    if not REPO_NAME_RE.match(name):
        raise ValueError(
            f"Repo name {name!r} violates the naming policy "
            "<YYYY-MM-DD>-<short-description> (lowercase, hyphens only), where the date "
            "is the date the data was GENERATED, "
            "e.g. 2026-04-30-rollout-importance-pilot-rollouts"
        )


def build_dataset_card(repo_name: str, fields: dict) -> str:
    missing = [f for f in REQUIRED_CARD_FIELDS if not fields.get(f)]
    if missing:
        raise ValueError(f"Dataset card is missing required fields: {missing}")
    lines = [
        f"# {repo_name}",
        "",
        fields["experiment"],
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    ordered = [k for k in REQUIRED_CARD_FIELDS if k != "experiment"]
    ordered += [k for k in fields if k not in REQUIRED_CARD_FIELDS and k != "experiment"]
    for key in ordered:
        if fields.get(key):
            lines.append(f"| `{key}` | {fields[key]} |")
    lines += [
        "",
        "Published from the multilingual-CoT-anchors repository; see its CLAUDE.md for "
        "the pipeline this artifact belongs to.",
    ]
    return "\n".join(lines) + "\n"


def publish(
    local_path: str | Path,
    repo_name: str,
    card_fields: dict,
    namespace: str | None = None,
    private: bool = False,
    repo_type: str = "dataset",
    path_in_repo: str = "",
) -> str:
    """Create (or update) the HF repo, upload the dataset card, upload the artifact.

    Returns the repo URL. Token comes from HF_TOKEN in .env (or a cached hf login);
    namespace defaults to HF_NAMESPACE in .env, else the token's own account.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Nothing to publish at {local_path}")
    validate_repo_name(repo_name)

    card_fields = dict(card_fields)
    card_fields.setdefault(
        "source_repo", f"multilingual-CoT-anchors @ commit `{git_sha()}`"
    )
    card = build_dataset_card(repo_name, card_fields)

    api = HfApi()
    namespace = namespace or os.getenv("HF_NAMESPACE") or api.whoami()["name"]
    repo_id = f"{namespace}/{repo_name}"

    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="dataset card",
    )
    if local_path.is_dir():
        api.upload_folder(
            folder_path=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"upload {local_path}",
        )
    else:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=str(Path(path_in_repo) / local_path.name) if path_in_repo else local_path.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"upload {local_path.name}",
        )
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}"
