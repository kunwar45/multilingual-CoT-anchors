# ABOUTME: v1 package exports for the archived rollout generation modules.
# ABOUTME: Superseded v1 (pre-2026-08-05) code kept for reference only — the live pipeline is src/rollout_importance/; nothing imports from here.
from .generator import RolloutGenerator, RolloutConfig, Rollout, create_run_manifest
from .parser import parse_final_answer, normalize_number, FINAL_MARKERS
from .manifest import (
    JobTask,
    JobManifest,
    ManifestManager,
    TaskStatus,
    generate_manifest,
)

__all__ = [
    "RolloutGenerator",
    "RolloutConfig",
    "Rollout",
    "create_run_manifest",
    "parse_final_answer",
    "normalize_number",
    "FINAL_MARKERS",
    "JobTask",
    "JobManifest",
    "ManifestManager",
    "TaskStatus",
    "generate_manifest",
]
