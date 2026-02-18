- New intersection: combine **base vs reasoning diffing**, **multilingual language-of-thought manipulation**, and a **pivot-triggered redo scaffold** in a single, controlled study.
- Pivot object is **mechanistic** (sentence-level activation/logit divergence) rather than phrase-based heuristics; pivot phrases are treated as labels for a regime, not the object itself.
- Explicit **tokenization/script controls** via sentence-boundary aggregation, within-language paraphrases, and (optional) back-translation reduce the risk that results are mere tokenizer artifacts.
- Sentence-level pivot scores **predict failures** (higher scores correlate with higher error rates) and support a conditional redo scaffold that outperforms random and heuristic redos under the same budget.
- The project exposes **language-dependent faithfulness** by measuring answer flip rates under truncation and cross-condition/ cross-language trace swaps.
- The codebase is **small but fully reproducible** (single `run_all.sh`, configs in YAML, one-run artifacts under `outputs/runs/`) and designed to scale to additional languages and models.


