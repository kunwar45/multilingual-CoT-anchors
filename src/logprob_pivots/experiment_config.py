# ABOUTME: Frozen Config dataclass for the logprob_pivots track: dataset, model pair, languages, generation parameters.
# ABOUTME: Values come from configs/logprob_pivots/<run>.yaml via load_config(); the field defaults are only a fallback.
"""Run configuration for the logprob_pivots track.

`load_config()` is the entry point every stage uses — it reads a YAML run config
from configs/logprob_pivots/ so run parameters live in configs, not in code
(CLAUDE.md: "NEVER hardcode run params in scripts"). The dataclass defaults below
exist only so `Config()` still works in tests and ad-hoc scratch code; they are
kept in sync with DEFAULT_CONFIG_PATH by test_matches_default_yaml in the smoke test.
"""
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "configs/logprob_pivots/qwen25_05b_mgsm.yaml"


@dataclass(frozen=True)
class Config:
    seed: int = 42

    # dataset
    # Hugging Face dataset identifier for MGSM
    dataset_name: str = "juletxara/mgsm"
    languages: tuple[str, ...] = ("en", "es", "fr", "de")
    n_per_lang: int = 50

    # models (edit these if you change choices)
    base_model: str = "Qwen/Qwen2.5-0.5B"
    reason_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # generation
    max_new_tokens: int = 256
    temperature: float = 0.2

    # runtime
    # Default device preference for local runs on macOS with MPS support.
    # Options: "cuda", "mps", "cpu"
    device_preference: str = "mps"

    # redo scaffold (stage 5) — CLI flags override these
    n_branches: int = 3
    pivot_top_k: int = 1


# Maps the YAML's nested layout onto the flat dataclass fields. Every leaf the YAML
# is allowed to contain appears here exactly once, so an unknown or misspelled key
# raises instead of being silently ignored.
_YAML_TO_FIELD = {
    ("seed",): "seed",
    ("dataset", "name"): "dataset_name",
    ("dataset", "languages"): "languages",
    ("dataset", "n_per_lang"): "n_per_lang",
    ("models", "base"): "base_model",
    ("models", "reason"): "reason_model",
    ("generation", "max_new_tokens"): "max_new_tokens",
    ("generation", "temperature"): "temperature",
    ("runtime", "device_preference"): "device_preference",
    ("scaffold", "n_branches"): "n_branches",
    ("scaffold", "pivot_top_k"): "pivot_top_k",
}

# Keys that are documentation only — read by humans, not by any pipeline stage.
# `controls` points at configs/logprob_pivots/controls.yaml, whose ablations are a
# written spec with no implementation yet; see that file's header.
_IGNORED_YAML_KEYS = {("controls", "config_path")}


def _flatten(node, prefix=()):
    """Yield (path_tuple, leaf_value) for every leaf in a nested dict."""
    for key, value in node.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            yield from _flatten(value, path)
        else:
            yield path, value


def load_config(path: str | Path | None = None) -> Config:
    """Build a Config from a YAML run config (DEFAULT_CONFIG_PATH when path is None).

    Raises on unknown keys so a typo in the YAML fails loudly instead of silently
    falling back to a dataclass default.
    """
    config_path = Path(path) if path else Path(DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"logprob_pivots run config not found: {config_path} "
            "(run from the repository root, or pass --config)"
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    values: dict = {}
    unknown: list[str] = []
    for path_tuple, value in _flatten(raw):
        if path_tuple in _IGNORED_YAML_KEYS:
            continue
        field_name = _YAML_TO_FIELD.get(path_tuple)
        if field_name is None:
            unknown.append(".".join(path_tuple))
            continue
        values[field_name] = tuple(value) if field_name == "languages" else value

    if unknown:
        known = ", ".join(".".join(k) for k in _YAML_TO_FIELD)
        raise ValueError(
            f"unknown key(s) in {config_path}: {', '.join(sorted(unknown))}. "
            f"Recognized keys: {known}"
        )

    field_names = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in values.items() if k in field_names})
