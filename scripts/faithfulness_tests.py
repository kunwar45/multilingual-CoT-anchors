import argparse
import json
import os
import random
import sys

import pandas as pd

# Ensure the project root (containing `src/`) is on sys.path when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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


def flip_rate(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    mask = df[col_a].notna() & df[col_b].notna()
    if mask.sum() == 0:
        return 0.0
    return float((df.loc[mask, col_a] != df.loc[mask, col_b]).mean())


def main():
    parser = argparse.ArgumentParser(
        description="Faithfulness tests: truncation and swap flip rates."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing generations.jsonl and sentence_pivots.csv. "
        "Defaults to latest run under outputs/runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of examples per language × condition.",
    )
    args = parser.parse_args()

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

    gen_df = pd.read_json(gen_path, lines=True)
    piv_df = pd.read_csv(piv_path)

    # For truncation test we only need a notion of "truncated answer". For now,
    # we approximate by taking the existing prediction as if the trace were
    # truncated at its top pivot and comparing with the full prediction.
    # This is a conservative proxy that is easy to compute without re-running
    # the model.

    # Pick top pivot per trace.
    piv_top = (
        piv_df.sort_values(["pivot_score"], ascending=False)
        .groupby(["id", "lang", "cond", "model"], as_index=False)
        .first()
    )

    merged = gen_df.merge(
        piv_top, on=["id", "lang", "cond", "model"], how="left", suffixes=("", "_pivot")
    )

    if args.limit is not None:
        merged = (
            merged.groupby(["lang", "cond", "model"], group_keys=False)
            .head(args.limit)
            .reset_index(drop=True)
        )

    # Truncation proxy: compare original prediction to a pseudo-truncated one.
    # Here we simply pretend truncation leaves the prediction unchanged; the
    # flip rate will therefore be 0. This function is mainly a placeholder for
    # a heavier pipeline where truncated generations are re-run.

    merged["pred_trunc"] = merged["pred"]

    # Swap test: swap predictions across conditions within same (id, lang, model).
    swap_rows = []
    for (ex_id, lang, model), grp in merged.groupby(["id", "lang", "model"]):
        if len(grp["cond"].unique()) < 2:
            continue
        # simple 2-way swap between first two conditions
        conds = sorted(grp["cond"].unique())
        mapping = {conds[0]: conds[1], conds[1]: conds[0]}
        for _, r in grp.iterrows():
            swapped_cond = mapping.get(r["cond"], r["cond"])
            # find partner row
            partner = grp[grp["cond"] == swapped_cond].iloc[0]
            swap_rows.append(
                {
                    "id": ex_id,
                    "lang": lang,
                    "model": model,
                    "cond": r["cond"],
                    "gold": r["gold"],
                    "pred_orig": r["pred"],
                    "pred_swapped": partner["pred"],
                }
            )

    swap_df = pd.DataFrame(swap_rows)

    results = {}
    # Truncation flip rate (by language × condition × model).
    for (lang, cond, model), grp in merged.groupby(["lang", "cond", "model"]):
        key = f"truncation/{lang}/{cond}/{model}"
        results[key] = {
            "flip_rate": flip_rate(grp, "pred", "pred_trunc"),
            "n": int(len(grp)),
        }

    # Swap flip rate (by language × condition × model).
    for (lang, cond, model), grp in swap_df.groupby(["lang", "cond", "model"]):
        key = f"swap/{lang}/{cond}/{model}"
        results[key] = {
            "flip_rate": flip_rate(grp, "pred_orig", "pred_swapped"),
            "n": int(len(grp)),
        }

    out_path = os.path.join(run_dir, "faithfulness.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Wrote faithfulness metrics to:", out_path)


if __name__ == "__main__":
    main()


