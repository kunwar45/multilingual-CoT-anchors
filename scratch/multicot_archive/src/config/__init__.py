# ABOUTME: v1 package marker for the archived configuration module.
# ABOUTME: Superseded v1 (pre-2026-08-05) code kept for reference only — the live pipeline is src/rollout_importance/; nothing imports from here.
"""
Configuration module for MultiCoT experiments.

Handles loading, validation, and merging of experiment configurations.
"""

from .loader import (
    ExperimentConfig,
    ModelConfig,
    GenerationConfig,
    APIConfig,
    ExperimentSettings,
    AnalysisConfig,
    load_config,
    load_defaults,
)

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "GenerationConfig",
    "APIConfig",
    "ExperimentSettings",
    "AnalysisConfig",
    "load_config",
    "load_defaults",
]
