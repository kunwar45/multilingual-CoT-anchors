"""
Configuration Loading and Validation

Provides ExperimentConfig dataclass that:
- Loads from JSON files
- Merges CLI args (CLI takes precedence)
- Validates configuration
- Converts to RolloutConfig for generator
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
import os


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device: str = "auto"
    dtype: str = "bfloat16"

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GenerationConfig:
    """Generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1024
    num_rollouts: int = 5
    base_seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class APIConfig:
    """API configuration for external providers."""
    use_api: bool = False
    provider: Optional[str] = None  # "openrouter", "together", "fireworks"
    base_url: Optional[str] = None
    key_env_var: str = "API_KEY"  # Environment variable name for API key

    @classmethod
    def from_dict(cls, d: dict) -> "APIConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable."""
        if self.provider:
            # Try provider-specific env var first
            provider_var = f"{self.provider.upper()}_API_KEY"
            key = os.environ.get(provider_var)
            if key:
                return key
        return os.environ.get(self.key_env_var)


@dataclass
class ExperimentSettings:
    """Experiment-level settings."""
    languages: list[str] = field(default_factory=lambda: ["en"])
    conditions: list[str] = field(default_factory=lambda: ["native"])
    num_problems: int = 20
    problem_ids: Optional[list[str]] = None  # Specific problem IDs to use

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentSettings":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AnalysisConfig:
    """Analysis parameters."""
    min_sentence_length: int = 10
    top_k_candidates: int = 5
    sensitivity_threshold: float = 0.3
    num_importance_samples: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Can be loaded from JSON, merged with CLI args, validated,
    and converted to RolloutConfig for the generator.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # Metadata
    config_name: Optional[str] = None
    config_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            generation=GenerationConfig.from_dict(d.get("generation", {})),
            api=APIConfig.from_dict(d.get("api", {})),
            experiment=ExperimentSettings.from_dict(d.get("experiment", {})),
            analysis=AnalysisConfig.from_dict(d.get("analysis", {})),
            config_name=d.get("config_name"),
            config_path=d.get("config_path"),
        )

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls.from_dict(data)
        config.config_path = str(path)
        config.config_name = path.stem
        return config

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model": asdict(self.model),
            "generation": asdict(self.generation),
            "api": asdict(self.api),
            "experiment": asdict(self.experiment),
            "analysis": asdict(self.analysis),
            "config_name": self.config_name,
            "config_path": self.config_path,
        }

    def to_json(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def merge_cli_args(self, args: Any) -> "ExperimentConfig":
        """
        Merge CLI arguments into config (CLI takes precedence).

        Args:
            args: argparse.Namespace or similar with CLI arguments

        Returns:
            New ExperimentConfig with merged values
        """
        # Create a copy of current config
        d = self.to_dict()

        # Model args
        if hasattr(args, "model") and args.model:
            d["model"]["name"] = args.model
        if hasattr(args, "device") and args.device:
            d["model"]["device"] = args.device
        if hasattr(args, "dtype") and args.dtype:
            d["model"]["dtype"] = args.dtype

        # Generation args
        if hasattr(args, "temperature") and args.temperature is not None:
            d["generation"]["temperature"] = args.temperature
        if hasattr(args, "top_p") and args.top_p is not None:
            d["generation"]["top_p"] = args.top_p
        if hasattr(args, "max_tokens") and args.max_tokens is not None:
            d["generation"]["max_new_tokens"] = args.max_tokens
        if hasattr(args, "num_rollouts") and args.num_rollouts is not None:
            d["generation"]["num_rollouts"] = args.num_rollouts
        if hasattr(args, "seed") and args.seed is not None:
            d["generation"]["base_seed"] = args.seed

        # API args
        if hasattr(args, "use_api") and args.use_api:
            d["api"]["use_api"] = args.use_api
        if hasattr(args, "api_provider") and args.api_provider:
            d["api"]["provider"] = args.api_provider
        if hasattr(args, "api_key") and args.api_key:
            # Store key in env var for security
            os.environ["API_KEY"] = args.api_key

        # Experiment args
        if hasattr(args, "languages") and args.languages:
            d["experiment"]["languages"] = args.languages
        if hasattr(args, "conditions") and args.conditions:
            d["experiment"]["conditions"] = args.conditions
        if hasattr(args, "num_problems") and args.num_problems is not None:
            d["experiment"]["num_problems"] = args.num_problems

        # Analysis args
        if hasattr(args, "min_sentence_length") and args.min_sentence_length is not None:
            d["analysis"]["min_sentence_length"] = args.min_sentence_length
        if hasattr(args, "top_k") and args.top_k is not None:
            d["analysis"]["top_k_candidates"] = args.top_k
        if hasattr(args, "threshold") and args.threshold is not None:
            d["analysis"]["sensitivity_threshold"] = args.threshold

        return ExperimentConfig.from_dict(d)

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Model validation
        if not self.model.name:
            errors.append("Model name is required")
        if self.model.dtype not in ["float16", "bfloat16", "float32"]:
            errors.append(f"Invalid dtype: {self.model.dtype}")

        # Generation validation
        if self.generation.temperature < 0 or self.generation.temperature > 2:
            errors.append(f"Temperature must be in [0, 2], got {self.generation.temperature}")
        if self.generation.top_p < 0 or self.generation.top_p > 1:
            errors.append(f"top_p must be in [0, 1], got {self.generation.top_p}")
        if self.generation.num_rollouts < 1:
            errors.append(f"num_rollouts must be >= 1, got {self.generation.num_rollouts}")
        if self.generation.max_new_tokens < 1:
            errors.append(f"max_new_tokens must be >= 1, got {self.generation.max_new_tokens}")

        # API validation
        if self.api.use_api:
            if not self.api.provider:
                errors.append("API provider is required when use_api is True")
            elif self.api.provider not in ["openrouter", "together", "fireworks"]:
                errors.append(f"Unknown API provider: {self.api.provider}")

        # Experiment validation
        valid_languages = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]
        for lang in self.experiment.languages:
            if lang not in valid_languages:
                errors.append(f"Unknown language: {lang}")

        valid_conditions = ["native", "english_baseline", "english_thinking", "translate_solve"]
        for cond in self.experiment.conditions:
            if cond not in valid_conditions:
                errors.append(f"Unknown condition: {cond}")

        if self.experiment.num_problems < 1:
            errors.append(f"num_problems must be >= 1, got {self.experiment.num_problems}")

        # Analysis validation
        if self.analysis.min_sentence_length < 1:
            errors.append(f"min_sentence_length must be >= 1")
        if self.analysis.sensitivity_threshold < 0 or self.analysis.sensitivity_threshold > 1:
            errors.append(f"sensitivity_threshold must be in [0, 1]")

        return errors

    def to_rollout_config(self) -> "RolloutConfig":
        """
        Convert to RolloutConfig for the generator.

        Returns:
            RolloutConfig instance
        """
        # Import here to avoid circular dependency
        from ..rollouts.generator import RolloutConfig

        return RolloutConfig(
            model_name=self.model.name,
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
            max_new_tokens=self.generation.max_new_tokens,
            num_rollouts=self.generation.num_rollouts,
            base_seed=self.generation.base_seed,
            device=self.model.device,
            dtype=self.model.dtype,
            use_api=self.api.use_api,
            api_provider=self.api.provider,
            api_key=self.api.get_api_key(),
            api_base_url=self.api.base_url,
        )


def load_defaults(defaults_path: Optional[Path] = None) -> ExperimentConfig:
    """
    Load default configuration.

    Args:
        defaults_path: Path to defaults JSON file (optional)

    Returns:
        ExperimentConfig with defaults
    """
    if defaults_path and defaults_path.exists():
        return ExperimentConfig.from_json(defaults_path)

    # Try to find default.json in configs directory
    module_dir = Path(__file__).parent.parent.parent
    default_path = module_dir / "configs" / "default.json"

    if default_path.exists():
        return ExperimentConfig.from_json(default_path)

    # Return built-in defaults
    return ExperimentConfig()


def load_config(
    config_path: Optional[Path] = None,
    cli_args: Optional[Any] = None,
    defaults_path: Optional[Path] = None,
) -> ExperimentConfig:
    """
    Load and merge configuration from multiple sources.

    Priority (highest to lowest):
    1. CLI arguments
    2. Config file (if provided)
    3. Defaults file
    4. Built-in defaults

    Args:
        config_path: Path to config JSON file
        cli_args: argparse.Namespace or similar with CLI arguments
        defaults_path: Path to defaults JSON file

    Returns:
        Merged and validated ExperimentConfig

    Raises:
        ValueError: If configuration is invalid
    """
    # Start with defaults
    config = load_defaults(defaults_path)

    # Override with config file if provided
    if config_path:
        file_config = ExperimentConfig.from_json(config_path)
        # Merge file config into defaults
        config = ExperimentConfig.from_dict({
            **config.to_dict(),
            **{k: v for k, v in file_config.to_dict().items() if v is not None},
        })
        config.config_path = str(config_path)
        config.config_name = config_path.stem

    # Override with CLI args if provided
    if cli_args:
        config = config.merge_cli_args(cli_args)

    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
