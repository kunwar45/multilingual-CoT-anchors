import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the project root (containing `src/`) is on sys.path when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config


def pick_device(cfg: Config) -> torch.device:
    # Prefer CUDA on Colab / GPU machines, then MPS, then CPU.
    if cfg.device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    cfg = Config()
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