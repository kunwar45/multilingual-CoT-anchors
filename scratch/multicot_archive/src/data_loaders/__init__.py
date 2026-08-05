# ABOUTME: v1 package exports for the archived MGSM loader.
# ABOUTME: Superseded v1 (pre-2026-08-05) code kept for reference only — the live pipeline is src/rollout_importance/; nothing imports from here.
from .mgsm import MGSMLoader, MGSMProblem, MGSM_LANGUAGES, LANGUAGE_NAMES, select_pilot_problems

__all__ = ["MGSMLoader", "MGSMProblem", "MGSM_LANGUAGES", "LANGUAGE_NAMES", "select_pilot_problems"]
