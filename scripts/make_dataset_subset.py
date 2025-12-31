import os
import random
import sys

import pandas as pd
from datasets import load_dataset

# Ensure the project root (containing `src/`) is on sys.path when running this script directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config


def main():
    cfg = Config()
    random.seed(cfg.seed)

    rows: list[dict] = []
    for lang in cfg.languages:
        ds = load_dataset(cfg.dataset_name, lang, split="test")
        idxs = list(range(len(ds)))
        random.shuffle(idxs)
        idxs = idxs[: cfg.n_per_lang]

        for i in idxs:
            ex = ds[i]
            rows.append(
                {
                    "lang": lang,
                    "id": f"{lang}_{i}",
                    "question": ex["question"],
                    "answer": ex["answer"],
                }
            )

    df = pd.DataFrame(rows).sort_values(["lang", "id"]).reset_index(drop=True)
    out_path = "data/mgsm_subset.csv"
    df.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(df))


if __name__ == "__main__":
    main()


