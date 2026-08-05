# Research log — append-only, MOST RECENT FIRST

One entry per real result or major code change: hypothesis → method → result → next steps,
with absolute dates. Routine refactors, chores, and doc edits get no entry.

---

## 2026-08-05 — Repository restructured to src/scripts/scratch layout with explicit naming

**Method:** Reorganized the flat `exp1/` + `multicot/` layout into the
`src/ scripts/ scratch/ configs/ docs/ data/ output/` structure (modeled on the
teaching_claude_why_replication repo), then renamed every track and pipeline stage to be
self-describing. All moves via `git mv` (history preserved); verified by the 31-check
offline smoke test and `--help` runs of every driver.

**Rename map:**
- `exp1` → `logprob_pivots` (sentence pivot scores from the base↔instruct logprob gap, local models)
- `multicot` → `rollout_importance` (counterfactual chunk importance from resampled rollouts, API models)
- `sentences.py` → `sentence_segmentation.py`, `pivots.py` → `pivot_scores.py`
- `analyze_rollouts.py` → `compute_importance.py`, `plots.py` → `make_figures.py`, `utils.py` → `answer_utils.py`
- `run_generation.py` → `generate_cot.py`, `compute_sentence_kl.py` → `compute_pivot_scores.py`,
  `make_dataset_subset.py` → `build_mgsm_subset.py`, `run_all.sh` → `run_full_pipeline.sh`
- Vertex infra: `scripts/gpu` → `scripts/vertex`; rollout output → `output/rollouts/`

**Next steps:** rebuild + push the Docker image (entrypoint path changed) before the next
Vertex run; run faithfulness tests on real truncated generations (still a stub).

## 2026-08-05 and earlier — rollout_importance pipeline implemented end to end (backfilled)

**Hypothesis:** counterfactual chunk importance (remove chunk i, resample, measure
Δ accuracy / KL at chunk i+1, filtered by embedding dissimilarity cos < 0.8) can be
compared across languages to test whether reasoning pivots are language-invariant.

**Method:** async API rollout generation (Together/Fireworks/Vertex) over
MGSM/MMATH/MMMLU/PolyMath in en/fr/zh/ar; language-specific LaTeX-aware chunker;
GlotLID chunk-level language verification; 6 importance metrics + GPT-4o DAG labeling;
LaBSE DP cross-lingual chunk alignment; 5 figure types.

**Result:** all 5 pipeline steps implemented and smoke-tested (31/31 offline checks).
Partial data so far: en + fr MGSM rollouts for Llama-3.2-3B, one fr MMMLU problem,
mmath rollouts for Apriel-1.6-15b-Thinker (en) in git; larger trees from the GCP VM
frozen in `output/vm_results/`. Bugs fixed along the way: figures module was a stub
(now 5 plot types), OpenAI client made lazy (`_get_client()`), MMMLU added to
`--dataset` choices, MGSM-Arabic error made explicit (MGSM has no Arabic — use
en/fr/zh; use MMMLU for ar).

**Next steps:** full en/fr/zh MGSM sweep on one model, then cross-language importance
correlation on aligned chunks.

## 2026-01-01 → 2026-01-03 — logprob_pivots: three generation runs (backfilled)

**Hypothesis:** sentences where the instruct model diverges most from the base model
(pivot score = mean |Δ logprob| per sentence) mark reasoning pivots, and their
distribution transfers across languages.

**Method:** Qwen2.5-0.5B base vs Instruct on an MGSM subset (en/es/fr/de),
Target-CoT vs En-CoT conditions; accuracy by language × condition × model; pivot
scores; pivot-triggered redo scaffold (reason model, 3 branches).

**Result:** three runs under `output/logprob_pivots/runs/`:
run_1767244523 (2026-01-01), run_1767307353 (2026-01-01), run_1767410979 (2026-01-03).
Accuracy CSVs exist for all three; sentence pivot scores + redo-scaffold results only
for run_1767410979.

**Next steps:** faithfulness tests on real truncated generations
(`run_faithfulness_tests.py` is still a stub); figures over the latest run.
