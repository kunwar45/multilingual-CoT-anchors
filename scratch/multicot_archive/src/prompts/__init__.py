# ABOUTME: v1 package exports for the archived prompt templates.
# ABOUTME: Superseded v1 (pre-2026-08-05) code kept for reference only — the live pipeline is src/rollout_importance/; nothing imports from here.
from .templates import (
    LanguageCondition,
    build_cot_prompt,
    CONDITION_NATIVE,
    CONDITION_ENGLISH_BASELINE,
    CONDITION_ENGLISH_THINKING,
    CONDITION_TRANSLATE_SOLVE,
)

__all__ = [
    "LanguageCondition",
    "build_cot_prompt",
    "CONDITION_NATIVE",
    "CONDITION_ENGLISH_BASELINE",
    "CONDITION_ENGLISH_THINKING",
    "CONDITION_TRANSLATE_SOLVE",
]
