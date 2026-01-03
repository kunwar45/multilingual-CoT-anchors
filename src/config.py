from dataclasses import dataclass

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
