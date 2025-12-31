
import os
import re
import sys
import json
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the project root (containing `src/`) is on sys.path when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.prompts import prompt_target_cot, prompt_en_cot

FINAL_RE = re.compile(r"FINAL:\s*([-+]?\d+(\.\d+)?)")

def pick_device(cfg: Config) -> torch.device:
    # Prefer CUDA on Colab / GPU machines, then MPS, then CPU.
    if cfg.device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def extract_final(text: str):
    m = FINAL_RE.search(text)
    return m.group(1) if m else None

def generate(model, tok, device, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
    return tok.decode(out[0], skip_special_tokens=True)

def load_model(name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    # Use float16 on CUDA/MPS for speed and memory; float32 on CPU.
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok

def main():
    cfg = Config()
    device = pick_device(cfg)

    df = pd.read_csv("data/mgsm_subset.csv")
    run_id = f"run_{int(time.time())}"
    out_dir = os.path.join("outputs", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading models...")
    base_model, base_tok = load_model(cfg.base_model, device)
    reason_model, reason_tok = load_model(cfg.reason_model, device)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lang = row["lang"]
        q = row["question"]
        gold = str(row["answer"]).strip()

        for cond in ["target_cot", "en_cot"]:
            if cond == "target_cot":
                prompt = prompt_target_cot(q, lang)
            else:
                prompt = prompt_en_cot(q)

            for model_tag, (model, tok) in [
                ("base", (base_model, base_tok)),
                ("reason", (reason_model, reason_tok)),
            ]:
                text = generate(model, tok, device, prompt, cfg.max_new_tokens, cfg.temperature)
                pred = extract_final(text)

                records.append({
                    "id": row["id"],
                    "lang": lang,
                    "cond": cond,
                    "model": model_tag,
                    "gold": gold,
                    "pred": pred,
                    "correct": (pred == gold),
                    "text": text,
                })

    out_path = os.path.join(out_dir, "generations.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Wrote:", out_path)

if __name__ == "__main__":
    main()