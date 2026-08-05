# ABOUTME: logprob_pivots STAGE 3 (accuracy): scores generations -> accuracy_by_lang_cond_model.csv in the run dir.
# ABOUTME: Defaults to the latest run under output/logprob_pivots/runs/; pass --run-dir for an older one.
import argparse
import os
import sys

import pandas as pd

# Ensure the project root (containing `src/`) is on sys.path when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.hf_fetching import latest_logprob_run_dir as find_latest_run


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy tables by language × condition × model."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a specific run directory containing generations.jsonl. "
        "Defaults to the latest run under output/logprob_pivots/runs.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run()
    gen_path = os.path.join(run_dir, "generations.jsonl")
    if not os.path.isfile(gen_path):
        raise FileNotFoundError(f"generations.jsonl not found at {gen_path!r}")

    df = pd.read_json(gen_path, lines=True)
    print("Loaded generations:", len(df), "rows from", gen_path)

    # Basic overall accuracy.
    overall = df["correct"].mean()
    print("\nOverall accuracy:", overall)

    # Accuracy by language × condition × model.
    acc_table = (
        df.groupby(["lang", "cond", "model"])["correct"]
        .mean()
        .reset_index()
        .pivot(index=["lang"], columns=["cond", "model"], values="correct")
    )
    print("\nAccuracy by language × condition × model:")
    print(acc_table)

    out_csv = os.path.join(run_dir, "accuracy_by_lang_cond_model.csv")
    acc_table.to_csv(out_csv)
    print("\nWrote accuracy table to:", out_csv)


if __name__ == "__main__":
    main()


