#!/usr/bin/env python3
"""
Smoke test: verify model loads and generates output.

Usage:
    python scripts/smoke_model.py [--model MODEL_NAME]
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Smoke test for model loading")
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    args = parser.parse_args()

    print(f"Testing model: {args.model}")
    print(f"Device: {args.device}")
    print("-" * 50)

    # Check PyTorch
    print("Checking PyTorch...")
    import torch

    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    start = time.time()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Loaded in {time.time() - start:.1f}s")

    # Load model
    print("\nLoading model...")
    start = time.time()
    from transformers import AutoModelForCausalLM

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()
    print(f"  Loaded in {time.time() - start:.1f}s")
    print(f"  Model device: {next(model.parameters()).device}")

    # Test generation
    print("\nTesting generation...")
    prompt = "What is 2 + 2? Think step by step.\n\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gen_time = time.time() - start

    print(f"  Generated in {gen_time:.1f}s")
    print(f"\n  Prompt: {prompt.strip()}")
    print(f"  Output: {generated[len(prompt):].strip()[:200]}...")

    print("\n" + "=" * 50)
    print("SUCCESS: Model loads and generates output")
    print("=" * 50)


if __name__ == "__main__":
    main()
