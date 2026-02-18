import os
import sys

import matplotlib.pyplot as plt
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
    run_dir = find_latest_run()
    print("Using run:", run_dir)

    # Paths
    acc_csv = os.path.join(run_dir, "accuracy_by_lang_cond_model.csv")
    pivots_csv = os.path.join(run_dir, "sentence_pivots.csv")
    scaffold_csv = os.path.join(run_dir, "redo_scaffold_reason.csv")

    fig_dir = os.path.join("reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Accuracy by language × condition (using reasoning model only).
    if os.path.isfile(acc_csv):
        acc = pd.read_csv(acc_csv, index_col=0)
        # Flatten MultiIndex columns if present.
        acc.columns = ["/".join(c for c in str(col).split()) for col in acc.columns]

        plt.figure(figsize=(8, 4))
        acc.plot(kind="bar")
        plt.title("Accuracy by language × condition × model")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=0)
        plt.tight_layout()
        out1 = os.path.join(fig_dir, "fig1_accuracy.png")
        plt.savefig(out1)
        print("Wrote", out1)
        plt.close()
    else:
        print("Missing accuracy CSV at", acc_csv)

    # Figure 2: Pivot score vs outcome (pivot quantiles vs failure probability).
    if os.path.isfile(pivots_csv):
        piv = pd.read_csv(pivots_csv)
        # Merge with correctness from generations.
        gen_path = os.path.join(run_dir, "generations.jsonl")
        gen_df = pd.read_json(gen_path, lines=True)
        merged = piv.merge(
            gen_df[["id", "lang", "cond", "model", "correct"]],
            on=["id", "lang", "cond", "model"],
            how="left",
        )
        merged["pivot_bin"] = pd.qcut(merged["pivot_score"], q=4, duplicates="drop")
        grp = (
            merged.groupby("pivot_bin")["correct"]
            .apply(lambda s: 1.0 - s.mean())
            .reset_index(name="failure_rate")
        )

        plt.figure(figsize=(6, 4))
        plt.plot(range(len(grp)), grp["failure_rate"], marker="o")
        plt.xticks(range(len(grp)), grp["pivot_bin"].astype(str), rotation=45, ha="right")
        plt.xlabel("Pivot score quantile")
        plt.ylabel("Failure rate")
        plt.title("Failure rate vs pivot score quantile")
        plt.tight_layout()
        out2 = os.path.join(fig_dir, "fig2_pivot_vs_failure.png")
        plt.savefig(out2)
        print("Wrote", out2)
        plt.close()
    else:
        print("Missing pivots CSV at", pivots_csv)

    # Figure 3: Mitigation wins – scaffold improvement over baselines by language.
    if os.path.isfile(scaffold_csv):
        sc = pd.read_csv(scaffold_csv)
        grp = (
            sc.groupby("lang")[["correct_none", "correct_pivot", "correct_rand"]]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(8, 4))
        x = range(len(grp))
        width = 0.25
        plt.bar([i - width for i in x], grp["correct_none"], width=width, label="none")
        plt.bar(x, grp["correct_pivot"], width=width, label="pivot")
        plt.bar([i + width for i in x], grp["correct_rand"], width=width, label="random")
        plt.xticks(x, grp["lang"])
        plt.ylabel("Accuracy")
        plt.title("Scaffold vs baselines by language")
        plt.legend()
        plt.tight_layout()
        out3 = os.path.join(fig_dir, "fig3_scaffold.png")
        plt.savefig(out3)
        print("Wrote", out3)
        plt.close()
    else:
        print("Missing scaffold CSV at", scaffold_csv)


if __name__ == "__main__":
    main()


