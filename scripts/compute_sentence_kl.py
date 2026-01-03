import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the project root (containing `src/`) is on sys.path when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.pivots import sentence_pivot_scores


def pick_device(cfg: Config) -> torch.device:
    # Prefer CUDA on GPU machines, then MPS, then CPU.
    if cfg.device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    # Use float16 on CUDA/MPS for speed and memory; float32 on CPU.
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok


def find_latest_run(runs_root: str = "outputs/runs") -> str:
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"No runs directory found at {runs_root!r}")
    candidates = [
        os.path.join(runs_root, d)
        for d in os.listdir(runs_root)
        if d.startswith("run_") and os.path.isdir(os.path.join(runs_root, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found under {runs_root!r}")
    return sorted(candidates)[-1]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute sentence-level pivot scores using logprob gaps between "
            "reasoning and base models."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a specific run directory containing generations.jsonl. "
        "Defaults to the latest run under outputs/runs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top-scoring sentences to keep per trace.",
    )
    parser.add_argument(
        "--only-reason",
        action="store_true",
        help="If set, restrict to generations from the 'reason' model only.",
    )
    args = parser.parse_args()

    cfg = Config()
    device = pick_device(cfg)

    run_dir = args.run_dir or find_latest_run()
    gen_path = os.path.join(run_dir, "generations.jsonl")
    if not os.path.isfile(gen_path):
        raise FileNotFoundError(f"generations.jsonl not found at {gen_path!r}")

    print("Using device:", device)
    print("Loading generations from:", gen_path)
    df = pd.read_json(gen_path, lines=True)
    print("Rows:", len(df))

    if args.only_reason:
        df = df[df["model"] == "reason"].reset_index(drop=True)
        print("Filtered to 'reason' model rows:", len(df))

    print("Loading models...")
    base_model, base_tok = load_model(cfg.base_model, device)
    reason_model, reason_tok = load_model(cfg.reason_model, device)

    # If vocab sizes match, we can safely use the base tokenizer for both.
    if base_tok.vocab_size == reason_tok.vocab_size:
        tok = base_tok
    else:
        # Fall back to using the reasoning tokenizer; alignment may be noisier.
        tok = reason_tok
    print("Tokenizer vocab size:", tok.vocab_size)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lang = row["lang"]
        text = row["text"]
        pivots = sentence_pivot_scores(
            text=text,
            lang=lang,
            tok=tok,
            base_model=base_model,
            reason_model=reason_model,
            device=device,
        )
        if pivots.empty:
            continue

        pivots = pivots.sort_values("pivot_score", ascending=False).head(args.top_k)
        for rank, p in enumerate(pivots.itertuples(index=False), start=1):
            records.append(
                {
                    "id": row["id"],
                    "lang": lang,
                    "cond": row["cond"],
                    "model": row["model"],
                    "rank": rank,
                    "sent_idx": p.sent_idx,
                    "pivot_score": p.pivot_score,
                    "pivot_p95": p.pivot_p95,
                    "char_start": p.char_start,
                    "char_end": p.char_end,
                    "n_tokens": p.n_tokens,
                    "sentence": p.sentence,
                }
            )

    if not records:
        print("No pivot records produced.")
        return

    out_df = pd.DataFrame(records)
    out_path = os.path.join(run_dir, "sentence_pivots.csv")
    out_df.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(out_df))


if __name__ == "__main__":
    main()


