<!-- ABOUTME: Append-only research log, most recent first: hypothesis, method, result, next steps per entry. -->
<!-- ABOUTME: Never rewrite a past entry — correct with a follow-up line; every published HF repo URL must be recorded here. -->
# Research log — append-only, MOST RECENT FIRST

One entry per real result or major code change: hypothesis → method → result → next steps,
with absolute dates. Routine refactors, chores, and doc edits get no entry.

---

## 2026-08-10 — generate_rollouts throughput: problem/chunk parallelism (measured 5.8x)

**Hypothesis:** the run was latency-bound on its own loop structure, not on the GPU.
`main()` iterated languages → problems → chunks serially, and only the innermost level
(rollouts for one chunk) ran concurrently. Peak in-flight requests therefore equalled
`--num_rollouts`, so `--max_concurrent_requests 32` never bound and a dedicated vLLM
server sat mostly idle. Dropping `num_rollouts` 40→10 earlier today made this worse: it
cut total work 4x but also shrank the only parallel dimension 4x.

**Method:** chunks are independent — a chunk's rollouts read only
`cumulative_chunks[chunk_idx - 1]`, computed up front from the base solution, and each
chunk writes its own `chunk_<i>/solutions.json`. So the chunk loop became
`asyncio.gather` over a `--max_concurrent_chunks` semaphore (default 4), and the problem
loop `asyncio.gather` over `--max_concurrent_problems` (default 8). Languages stay
sequential so partial results remain publishable and sharding by `--languages` stays
clean. `--max_concurrent_requests` is now the sole global throttle, so the new flags are
safe for rate-limited hosted providers; both are forced to 1 under `--provider Local`
(batched local generation blocks the event loop). Progress lines gained a `[lang] pid`
prefix since output now interleaves. Alongside: `scripts/slurm/submit_language_shards.sh`
(one job per language, disjoint output paths); the sbatch now prefers vLLM **data**
parallel over tensor parallel when the weights fit on one GPU, with a `--help` probe and
tensor-parallel fallback for older vLLM; job request cap 32 → 128; and `max_tokens`
16384 → 2048 in both MGSM run configs (MGSM traces are a few hundred tokens; the old
value only bought runaway tails).

**Result:** measured against a mock streaming server that counts in-flight requests
(6 problems x 4 rollouts, identical settings otherwise):

| mode | wall clock | peak concurrent requests | total requests | solutions.json |
|---|---|---|---|---|
| serial (old) | 32.3 s | 4 | 150 | 36 |
| parallel (new) | 5.6 s | 96 | 150 | 36 |

**5.8x faster on identical work** — same request count, same file count. Correctness
checked rather than assumed: the two output trees are structurally identical, and all 144
rollouts have byte-identical `prefix_without_chunk` / `chunk_removed`, confirming no race
on `cumulative_chunks`; prefix lengths stay monotonically increasing in chunk index.
Resumability intact (rerun issued 0 requests, logged 36 "already done"). Rollout smoke
test 28 passed / 0 failed / 4 skipped, unchanged.

**Next steps:** the mock has no GPU behind it, so 5.8x is the loop-structure ceiling, not
a vLLM throughput prediction — the real gain depends on where vLLM's batching curve
flattens for 32B. Measure `--max_concurrent_problems` on the 7B smoke run before assuming
8 is optimal. With the loop no longer the bottleneck, revisit `num_rollouts: 10` — larger
batches are also *more* GPU-efficient per request, so more rollouts may cost little extra
wall clock.

**Not changed (deliberate):** GlotLID verification stays on. Disabling it would cut fr/zh
regeneration cost, but language purity is a property of the experiment, not a knob —
`--no_verify_language` is available if a first pass wants it.

## 2026-08-10 — Config/code discrepancy audit before the first real rollout run

**Method:** audited every config against the code that consumes it, ahead of greenlighting
the first counterfactual rollout run on Killarney. Five discrepancies found and fixed:

1. **`generate_rollouts_job.sbatch` silently dropped extra CLI args.** It read only `$1`,
   so the smoke procedure documented in the 2026-08-06 entry below
   (`... <run>.yaml -np 2 -nr 5`) would have launched the **full 250×40 run**. It now
   `shift`s and forwards `"$@"` after the fixed flags, so overrides win over the config.
   *(Correction to the 2026-08-06 entry: that smoke command did not work as written.)*
2. **No GPU-size guard on a mixed-GPU cluster.** Killarney mixes H100-80GB and L40S-48GB;
   a bare `--gpus-per-node=1` could land 32B fp16 (~64GB) on a 48GB card and OOM ~10
   minutes into weight loading. Added an `nvidia-smi` preflight that infers the parameter
   count from the model id, requires 2GB/1B params + 10%, and aborts in seconds with a
   resubmit hint. Verified: rejects L40S-48GB and A100-40GB for 32B, accepts H100-80GB
   and 2×A100-40GB; unknown model ids skip the check.
3. **`configs/logprob_pivots/*.yaml` were dead files.** Nothing read them — all five
   stages called the frozen `Config()` dataclass, so editing the YAML changed nothing
   silently (contradicting CLAUDE.md "NEVER hardcode run params in scripts"). Added
   `load_config()` to `experiment_config.py`: reads the YAML, maps its nested layout onto
   the flat dataclass, and **raises on unknown keys**. All five stages now call it, and
   the three with an argparse gained `--config`. Two YAML keys had no code counterpart at
   all — `scaffold.n_branches` / `scaffold.pivot_top_k` are now real `Config` fields
   backing the `--n-branches` / `--top-k` defaults, and the entire `controls:` block
   (paraphrase / back-translation ablations) is **unimplemented**, now labelled as spec-only
   in `controls.yaml` instead of reading as live configuration.
   `smoke_test_models` gained `check_defaults_match_yaml()` so dataclass defaults can never
   drift from the YAML unnoticed.
4. **`configs/vertex/experiment_a100.yaml` requested a 40GB GPU for a 64GB model.**
   `a2-highgpu-1g` is the A100-**40GB**; the template's own header claimed 80GB. Changed to
   `a2-ultragpu-1g` + `NVIDIA_A100_80GB`.
5. **CLAUDE.md misplaced the GlotLID loader** — `load_glotlid_model()` lives in
   `answer_extraction.py`, not `language_verification.py` (which has `check_chunk_languages`).
   Corrected the module map.

**Result:** rollout smoke test 28 passed / 0 failed / 4 skipped (unchanged baseline);
`bash -n` clean on the sbatch; arg forwarding verified for both the with-args and no-args
cases; `load_config()` round-trips the run config and rejects a misspelled key; all three
`--config`-bearing stages `--help` cleanly; every config YAML still parses.

**Next steps:** no rollout data generated yet — the first real run is still pending. Submit
the 7B smoke run on Killarney (`-np 2 -nr 5`, now actually honored) before committing to the
32B run, and decide on `retry_on_wrong` (see below).

**Follow-up (same day):** both rollout run configs dropped from `num_rollouts: 40` to `10`
to make the first pass ~4x cheaper (~112k generations for 250 problems x 3 languages at
~15 chunks/problem). This is a pipeline-validation setting, not a publishable one — with
10 rollouts the per-chunk accuracy resolution is 0.1 before `compute_importance` discards
rollouts failing the cos < 0.8 dissimilarity filter, leaving only a few usable samples per
chunk. Raise toward 40-100 before drawing cross-language conclusions.

**Open question, not yet fixed:** `--retry_on_wrong` defaults to False, so a base solution
that gets the wrong answer is kept and rolled out against, inside a directory named
`correct_base_solution`. Either flip it in the run configs or confirm `compute_importance`
filters on `is_correct` before trusting stage-2 numbers.

## 2026-08-06 — SLURM path for stage 1 (Alliance / Compute Canada, free GPU hours)

**Method:** added `scripts/slurm/` as a second, zero-cost execution route for
generate_rollouts, reusing the same run configs as the Vertex flow.
`setup_environment_and_prefetch.sh` (login node, once per run config) builds the venv,
installs vLLM, and prefetches the run's model weights + dataset + GlotLID into
`$SCRATCH/hf_cache`, because **Alliance compute nodes have no internet**.
`generate_rollouts_job.sbatch` boots `vllm serve` on the job's GPU(s)
(`--tensor-parallel-size` follows `SLURM_GPUS_ON_NODE`; 32B fp16 needs 2×A100-40GB on
Narval), waits for health, then runs `generate_rollouts --config <run>.yaml --provider
Vertex` against localhost with `HF_HUB_OFFLINE=1` — the existing Vertex provider is just
an OpenAI-compatible base URL, so no new provider code was needed. Added
`--max_concurrent_requests` to generate_rollouts (default 6, unchanged behavior; the
SLURM job passes 32 to keep a dedicated vLLM server busy). Time-limit deaths are handled
by resubmitting — generation is resumable. `slurm_logs/` gitignored.

**Result:** both scripts pass `bash -n`; the YAML-extraction one-liners and the new
concurrency flag verified locally; rollout smoke test unchanged (28/0/4). Untested on a
real cluster yet — first submission should use the 7B config with `-np 2 -nr 5` CLI
overrides as a smoke run.

## 2026-08-06 — Vertex jobs now driven by track run configs (dataset × languages × model)

**Method:** a run varies three things — dataset, languages, model — so those now live
exclusively in `configs/rollout_importance/<run>.yaml` (one file per run:
`qwen25_32b_mgsm.yaml`, new `qwen25_7b_mgsm.yaml` for L4-class GPUs). The Vertex
templates (`experiment_a100.yaml`, `smoke_l4.yaml`) no longer hardcode generate_rollouts
args: they keep only machine/infra choices, force `--provider Local` (model under test
self-hosted on the job's GPU), and take the experiment via a new `{{RUN_CONFIG}}`
placeholder; `create_vertex_run_config.py` gained a required `--run-config` flag that
validates and injects the repo-relative path (run name defaults to the config's stem).
`smoke_l4.yaml` shrinks any run config to 2 problems × 5 rollouts via CLI overrides.
Behavior change vs the old A100 template: it ran `--languages en` with
`--no_verify_language` hardcoded; jobs now run whatever the run config says — for
`qwen25_32b_mgsm.yaml` that is en/fr/zh with GlotLID verification ON (a ~3× larger run
than the old en-only template; override on the config or CLI to shrink).

**Result:** generated job YAMLs for both templates parse and contain the expected
`--config <run>.yaml --provider Local` command; container arg resolution simulated
locally (config + CLI overrides) resolves correctly; missing/invalid `--run-config`
rejected with clear errors. Rollout smoke test unchanged (28/0/4). No job submitted yet.

## 2026-08-06 — generate_rollouts: YAML run configs + counterfactual-prefix bug fix

**Method:** `generate_rollouts.py` now takes `--config` (YAML keys = long option names,
applied as argparse defaults so explicit CLI flags override); canonical params live in
`configs/rollout_importance/qwen25_32b_mgsm.yaml` (mgsm, en/fr/zh, Qwen2.5-32B-Instruct,
250 problems × 40 rollouts, t=0.6, top_p=0.95, seed 44). `--model` no longer has a
hardcoded default (the stale `Qwen/Qwen3.5-9B`) — it must come from the config or `-m`.
While in there: **fixed a correctness bug in counterfactual prefix construction** — the
prefix was built with `full_prefix.replace(chunk_text, "")`, which deletes *every*
occurrence of a repeated chunk (e.g. a short "Wait." or duplicated equation line), not
just chunk i; it now uses the cumulative prefix through chunk i−1 directly. Rollouts
generated before 2026-08-06 are unaffected unless a base solution contained an exactly
repeated chunk. Also deduplicated the thrice-copied GlotLID retry loop into
`verify_language_with_retries()` (identical save/retry semantics) and aligned base-
solution API retries with `--max_retries`. Config-file naming rule added to CLAUDE.md:
configs are named for the run they specify, never `default.yaml` (logprob config renamed
to `configs/logprob_pivots/qwen25_05b_mgsm.yaml` accordingly).

**Result:** rollout smoke test 28 passed / 0 failed / 4 skipped (unchanged baseline);
config load + CLI-override + unknown-key rejection verified by importing the module with
patched argv.

**Next steps:** point the Vertex job templates at track configs instead of hardcoding
args in the container command; consider `--config` for the other rollout stages.

## 2026-08-05 — File-name clarity pass + ABOUTME headers on every file

**Method:** repo-wide audit of file names for clarity; renamed the stragglers with
`git mv` and fixed all imports/references: `src/logprob_pivots/config.py` →
`experiment_config.py`, `src/rollout_importance/answer_utils.py` →
`answer_extraction.py`, `src/rollout_importance/lang_verifier.py` →
`language_verification.py`, `scripts/vertex/build_and_push.sh` →
`build_and_push_docker_image.sh`, `scratch/___dummy.ipynb` →
`empty_scratch_notebook.ipynb`. Added a two-line `ABOUTME:` comment header to every
file in the repo (87 files, including the `multicot_archive`; exceptions: `.gitkeep`
must stay empty, JSON has no comments). Both policies are now codified in CLAUDE.md
("Names are self-describing" / "Every file starts with an ABOUTME header"). Older
entries below reference the pre-rename paths; they remain correct for their time.

**Result:** rollout smoke test 28 passed / 0 failed / 4 skipped (skips = missing local
data, expected since HF is canonical); `smoke_test_models` exits 0 with both models
generating; every tracked Python file byte-compiles; YAML configs and the notebook
still parse.

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
