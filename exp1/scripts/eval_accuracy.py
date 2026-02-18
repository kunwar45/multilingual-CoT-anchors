import argparse
import os
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy tables by language × condition × model."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a specific run directory containing generations.jsonl. "
        "Defaults to the latest run under outputs/runs.",
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


