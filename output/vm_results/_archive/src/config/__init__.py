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
