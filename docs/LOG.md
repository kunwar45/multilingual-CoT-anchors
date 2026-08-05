# Research log — append-only, MOST RECENT FIRST

One entry per real result or major code change: hypothesis → method → result → next steps,
with absolute dates. Routine refactors, chores, and doc edits get no entry.

---

## 2026-08-05 — Local data deleted; code now fetches from HF as the canonical source

**Method:** with all artifacts verified on the Hub (file counts matched exactly and a
spot-checked file was byte-identical), deleted every local/git duplicate: the rollout
trees, the VM snapshot, the logprob run outputs, and the git-tracked (`git add -f`'d)
copies of all of them; `data/mgsm_subset.csv` was untracked (kept on disk as a staged
input). `output/` and `data/` are now fully gitignored. Pre-deletion copies remain in
git history.

Added `src/hf_fetching.py` (`CANONICAL_DATASETS` registry + `ensure_mgsm_subset` /
`ensure_logprob_runs` / `latest_logprob_run_dir` / `ensure_rollouts`) and wired it into
`generate_cot`, the five logprob analysis stages (replacing their duplicated
`find_latest_run`), and `compute_importance` / `align_chunks` / `make_figures`.

**Result:** verified from a fully empty `output/`: `eval_accuracy` auto-fetched the
published runs and rebuilt the accuracy table; `ensure_rollouts("mmmlu")` fetched the
published tree. Smoke test 31/31. A fresh clone now needs no local data at all.

**Next steps:** update `CANONICAL_DATASETS` whenever a new canonical artifact is
published.

## 2026-08-05 — All HF artifacts moved to the multicot org, made public (follow-up)

Correction to the entry below: the project now has an HF org, **https://huggingface.co/multicot** —
everything lives there, public, and the `kunwar45/` URLs below are obsolete:
- https://huggingface.co/datasets/multicot/2026-01-01-mgsm-subset-en-es-fr-de (new: the fixed MGSM input subset)
- https://huggingface.co/datasets/multicot/2026-01-03-logprob-pivots-qwen25-05b-mgsm-runs
- https://huggingface.co/datasets/multicot/2026-04-30-rollout-importance-pilot-rollouts
- https://huggingface.co/datasets/multicot/2026-05-04-rollout-importance-vm-results
- hf://buckets/multicot/rollout-scratch (mutable staging bucket for in-progress rollout
  syncs, layout `rollouts/<run-id>/<dataset>/<model>/...`)

Publishing defaults flipped to public + `HF_NAMESPACE=multicot`. Note: the `.env`
`HF_TOKEN` has org write access; the cached `hf auth login` token does not.

## 2026-08-05 — Hugging Face publishing policy adopted; existing artifacts published

**Method:** ported the reference repo's data policy: datasets, run artifacts and eval
results live on the HF Hub, not in git or only on local disk. `src/hf_publishing.py`
enforces the repo naming rule (`<YYYY-MM-DD>-<short-description>`, date generated) and
the required dataset-card fields (experiment, date_generated, track, languages, models,
provenance, source_repo+SHA) in code; `scripts/publish_to_hf.py` is the one entrypoint.
Repos are private by default.

**Result:** all pre-existing artifacts published (private, under `kunwar45/`):
- https://huggingface.co/datasets/kunwar45/2026-01-03-logprob-pivots-qwen25-05b-mgsm-runs
  (the three generation runs: accuracy tables, sentence pivot scores, redo-scaffold results)
- https://huggingface.co/datasets/kunwar45/2026-04-30-rollout-importance-pilot-rollouts
  (local pilot rollout trees for mgsm/mmath/mmmlu, 260 files)
- https://huggingface.co/datasets/kunwar45/2026-05-04-rollout-importance-vm-results
  (frozen GCP VM results snapshot, 320 files)

**Next steps:** publish each future run as it completes and record the URL in its LOG
entry; use `--repo-type model` when a trained model organism/adapter exists.

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
