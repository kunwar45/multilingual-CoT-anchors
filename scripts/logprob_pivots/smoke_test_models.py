# ABOUTME: logprob_pivots STAGE 0 (smoke): verifies the Qwen2.5-0.5B base and instruct models both load and generate.
# ABOUTME: Run with venv/bin/python -m scripts.logprob_pivots.smoke_test_models before touching this track.
import os
import sys
from dataclasses import fields

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the project root (containing `src/`) is on sys.path when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logprob_pivots.experiment_config import Config, load_config


def check_defaults_match_yaml() -> None:
    """Fail loudly if the dataclass defaults have drifted from the run config.

    The YAML is the source of truth; the dataclass defaults exist only as a fallback
    for tests and scratch code. When the two disagree, `Config()` and `load_config()`
    silently produce different runs — exactly the trap this check exists to catch.
    """
    defaults, from_yaml = Config(), load_config()
    drifted = [
        (field.name, getattr(defaults, field.name), getattr(from_yaml, field.name))
        for field in fields(Config)
        if getattr(defaults, field.name) != getattr(from_yaml, field.name)
    ]
    if drifted:
        details = "; ".join(f"{n}: default={d!r} yaml={y!r}" for n, d, y in drifted)
        raise SystemExit(
            "Config dataclass defaults have drifted from "
            f"configs/logprob_pivots/qwen25_05b_mgsm.yaml — {details}"
        )
    print("Config defaults match the run config YAML.")


def pick_device(cfg: Config) -> torch.device:
    # Prefer CUDA on Colab / GPU machines, then MPS, then CPU.
    if cfg.device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    check_defaults_match_yaml()
    cfg = load_config()
    device = pick_device(cfg)

    print("Device:", device)
    for name in [cfg.base_model, cfg.reason_model]:
        print("\nLoading:", name)
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        # Use float16 on CUDA/MPS for speed and memory; float32 on CPU.
        dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            device_map=None,
        ).to(device)
        model.eval()

        prompt = "Solve: If 7+5=12, what is 17-9?"
        inputs = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.2,
            )
        print(tok.decode(out[0], skip_special_tokens=True)[-500:])

if __name__ == "__main__":
    main()