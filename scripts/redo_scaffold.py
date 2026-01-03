import argparse
import os
import random
import re
import sys
from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure the project root (containing `src/`) is on sys.path when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.sentences import sentence_spans

FINAL_RE = re.compile(r"FINAL:\s*([-+]?\d+(\.\d+)?)")


def pick_device(cfg: Config) -> torch.device:
    # Prefer CUDA on GPU machines, then MPS, then CPU.
    if cfg.device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_final(text: str):
    m = FINAL_RE.search(text)
    return m.group(1) if m else None


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


def majority_vote(preds):
    preds = [p for p in preds if p is not None]
    if not preds:
        return None
    counts = Counter(preds)
    return counts.most_common(1)[0][0]


def generate_from_prefix(model, tok, device, prefix: str, max_new_tokens: int, temperature: float) -> str:
    # If the prefix is empty or whitespace, there is nothing meaningful to
    # condition on; avoid passing an empty sequence into generate, which can
    # trigger index errors in some models/generation utils.
    if not prefix or prefix.strip() == "":
        return prefix

    inputs = tok(prefix, return_tensors="pt")
    if inputs["input_ids"].shape[1] == 0:
        # Defensive: tokenizer produced no tokens.
        return prefix

    inputs = inputs.to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pivot-triggered redo scaffold vs baselines using existing generations and sentence pivots."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing generations.jsonl and sentence_pivots.csv. "
        "Defaults to latest run under outputs/runs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="reason",
        choices=["base", "reason"],
        help="Which model's generations to use as the baseline trace.",
    )
    parser.add_argument(
        "--n-branches",
        type=int,
        default=3,
        help="Number of redo branches to sample at the pivot / random sentence.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Use the top-k highest-scoring pivot sentences; k>1 currently "
        "means we pick the best (highest pivot_score) among them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of examples to process (for speed).",
    )
    args = parser.parse_args()

    cfg = Config()
    device = pick_device(cfg)

    run_dir = args.run_dir or find_latest_run()
    gen_path = os.path.join(run_dir, "generations.jsonl")
    piv_path = os.path.join(run_dir, "sentence_pivots.csv")

    if not os.path.isfile(gen_path):
        raise FileNotFoundError(f"generations.jsonl not found at {gen_path!r}")
    if not os.path.isfile(piv_path):
        raise FileNotFoundError(
            f"sentence_pivots.csv not found at {piv_path!r}. "
            "Run scripts.compute_sentence_kl first."
        )

    print("Using device:", device)
    print("Loading generations from:", gen_path)
    gen_df = pd.read_json(gen_path, lines=True)
    print("Rows:", len(gen_df))

    print("Loading sentence pivots from:", piv_path)
    piv_df = pd.read_csv(piv_path)

    # Filter to the chosen model's traces.
    gen_df = gen_df[gen_df["model"] == args.model].reset_index(drop=True)
    print(f"Filtered to model={args.model!r} rows:", len(gen_df))

    if args.limit is not None:
        gen_df = gen_df.head(args.limit).copy()
        print("Applying limit, rows:", len(gen_df))

    # Pick the top pivot per (id, lang, cond, model) by pivot_score.
    piv_group_cols = ["id", "lang", "cond", "model"]
    piv_top = (
        piv_df.sort_values(["pivot_score"], ascending=False)
        .groupby(piv_group_cols, as_index=False)
        .first()
    )

    merged = gen_df.merge(piv_top, on=["id", "lang", "cond", "model"], how="left", suffixes=("", "_pivot"))
    print("Merged generations with pivots; rows:", len(merged))

    # Load the generation model.
    model_name = cfg.reason_model if args.model == "reason" else cfg.base_model
    print("Loading generation model:", model_name)
    model, tok = load_model(model_name, device)

    results = []
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        gold = str(row["gold"]).strip()
        text_orig = row["text"]
        lang = row["lang"]

        # Baseline: existing prediction from the original run.
        pred_none = row["pred"]

        # Pivot-based prefix: use char_start from pivots if available.
        if not pd.isna(row.get("char_start")):
            cs = int(row["char_start"])
            prefix_pivot = text_orig[:cs]
        else:
            prefix_pivot = text_orig  # fallback: no pivot, effectively no-op

        # Random sentence prefix as baseline.
        spans = sentence_spans(text_orig, lang=lang)
        if spans:
            rs, _, _ = random.choice(spans)
            prefix_rand = text_orig[:rs]
        else:
            prefix_rand = text_orig

        # Helper to run N branches and majority vote.
        def redo_from_prefix(prefix: str):
            preds = []
            for _ in range(args.n_branches):
                new_text = generate_from_prefix(
                    model=model,
                    tok=tok,
                    device=device,
                    prefix=prefix,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                )
                preds.append(extract_final(new_text))
            return majority_vote(preds)

        pred_pivot = redo_from_prefix(prefix_pivot)
        pred_rand = redo_from_prefix(prefix_rand)

        results.append(
            {
                "id": row["id"],
                "lang": lang,
                "cond": row["cond"],
                "model": row["model"],
                "gold": gold,
                "pred_none": pred_none,
                "pred_pivot": pred_pivot,
                "pred_rand": pred_rand,
                "correct_none": (pred_none == gold),
                "correct_pivot": (pred_pivot == gold) if pred_pivot is not None else False,
                "correct_rand": (pred_rand == gold) if pred_rand is not None else False,
            }
        )

    if not results:
        print("No results produced.")
        return

    out_df = pd.DataFrame(results)
    out_path = os.path.join(run_dir, f"redo_scaffold_{args.model}.csv")
    out_df.to_csv(out_path, index=False)
    print("Wrote scaffold results to:", out_path)

    # Print quick summary.
    for col in ["correct_none", "correct_pivot", "correct_rand"]:
        acc = out_df[col].mean()
        print(f"{col}: {acc:.3f}")


if __name__ == "__main__":
    main()


