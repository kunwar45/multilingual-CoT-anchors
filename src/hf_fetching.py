# ABOUTME: Fetches canonical project data from the Hugging Face multicot org: CANONICAL_DATASETS registry + ensure_* helpers.
# ABOUTME: Pipeline stages call these so a fresh clone runs with no local data; update CANONICAL_DATASETS whenever publishing new canonical data.
"""Fetch canonical project data from the Hugging Face Hub.

HF is the canonical data source for this project (CLAUDE.md "Datasets ... go to
Hugging Face"): local `data/` and `output/` are iteration caches that any machine can
rebuild by downloading from the multicot org. Pipeline stages call the ensure_* helpers
below so a fresh clone works with no local data at all.

Counterpart to src/hf_publishing.py (which uploads). When you publish a NEW canonical
artifact, update CANONICAL_DATASETS here — this registry is the code's single map of
where data lives.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

load_dotenv()

# The single registry of canonical data repos on the Hub. Public — no token needed.
CANONICAL_DATASETS = {
    # fixed MGSM input subset for the logprob_pivots track (mgsm_subset.csv)
    "mgsm_subset": "multicot/2026-01-01-mgsm-subset-en-es-fr-de",
    # the three logprob_pivots generation runs (run_<unix_ts>/ trees)
    "logprob_pivots_runs": "multicot/2026-01-03-logprob-pivots-qwen25-05b-mgsm-runs",
    # pilot rollout_importance trees ({dataset}/{model}/... layout)
    "rollouts": "multicot/2026-04-30-rollout-importance-pilot-rollouts",
    # frozen GCP VM results snapshot (same layout; stale code copies inside — never import)
    "vm_results": "multicot/2026-05-04-rollout-importance-vm-results",
}

_CARD_FILES = ["README.md", ".gitattributes"]


def ensure_mgsm_subset(dest: str | Path = "data/mgsm_subset.csv") -> Path:
    """Return the MGSM subset CSV, downloading it from the Hub if not present locally."""
    dest = Path(dest)
    if dest.exists():
        return dest
    print(f"[hf_fetching] {dest} missing — fetching from {CANONICAL_DATASETS['mgsm_subset']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        CANONICAL_DATASETS["mgsm_subset"],
        filename=dest.name,
        repo_type="dataset",
        local_dir=dest.parent,
    )
    return dest


def ensure_logprob_runs(runs_root: str | Path = "output/logprob_pivots/runs") -> Path:
    """Ensure at least one run_* directory exists locally, fetching all published runs
    from the Hub if none do."""
    runs_root = Path(runs_root)
    if runs_root.is_dir() and any(runs_root.glob("run_*")):
        return runs_root
    print(
        f"[hf_fetching] no run_* under {runs_root} — fetching from "
        f"{CANONICAL_DATASETS['logprob_pivots_runs']}"
    )
    snapshot_download(
        CANONICAL_DATASETS["logprob_pivots_runs"],
        repo_type="dataset",
        local_dir=runs_root,
        ignore_patterns=_CARD_FILES,
    )
    return runs_root


def latest_logprob_run_dir(runs_root: str | Path = "output/logprob_pivots/runs") -> str:
    """Path of the newest run_<unix_ts> directory, fetching published runs from the Hub
    when none exist locally. Shared by every logprob_pivots analysis stage."""
    runs_root = ensure_logprob_runs(runs_root)
    candidates = sorted(
        d for d in runs_root.glob("run_*") if d.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run_* directories under {runs_root!r} locally or on the Hub "
            f"({CANONICAL_DATASETS['logprob_pivots_runs']})"
        )
    return str(candidates[-1])


def ensure_rollouts(dataset: str, rollouts_base: str | Path = "output/rollouts") -> Path:
    """Ensure the rollout tree for `dataset` exists locally, fetching the published
    pilot rollouts from the Hub if it is missing. Returns the dataset's tree root.

    Only the canonical (published) rollouts can be fetched; a model/language absent from
    the published tree still has to be generated with generate_rollouts.
    """
    tree = Path(rollouts_base) / dataset
    if tree.is_dir() and any(tree.iterdir()):
        return tree
    print(
        f"[hf_fetching] {tree} missing — fetching from {CANONICAL_DATASETS['rollouts']}"
    )
    snapshot_download(
        CANONICAL_DATASETS["rollouts"],
        repo_type="dataset",
        local_dir=rollouts_base,
        allow_patterns=[f"{dataset}/**"],
    )
    if not tree.is_dir():
        print(f"[hf_fetching] the published rollouts contain no {dataset!r} tree")
    return tree
