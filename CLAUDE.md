# CLAUDE.md — repo guide for agents

**AI agents: do NOT write to this file unless specifically asked to — and even when asked,
encourage human review of the exact diff. This file only stays useful if it stays curated;
unsupervised agent edits turn it to slop.**

Orientation + operating rules for this repo. Read this before touching anything.

## What this project is

Research on **language-invariant reasoning pivots in LLMs**: do "redo / check / backtrack"
reasoning regimes transfer across languages, or are they language-specific artifacts?
Inspired by the Thought Anchors literature. Two experiment tracks share the repo, named
after their method:

1. **logprob_pivots** — local models (Qwen2.5-0.5B base vs Instruct) on MGSM en/es/fr/de.
   Sentence-level *pivot scores* via the logprob gap between base and instruct models
   (mean |Δ logprob| per sentence over the tokens in its span), plus a pivot-triggered
   redo scaffold and faithfulness tests.
2. **rollout_importance** — API models (Together/Fireworks/Vertex) on MGSM + MMATH +
   MMMLU + PolyMath, languages en/fr/zh/ar. Rollout-based *counterfactual importance*:
   remove chunk i, resample, measure Δ accuracy / KL at chunk i+1, filtered by embedding
   dissimilarity (cos < 0.8). Six importance metrics, GPT-4o DAG labeling, LaBSE
   cross-lingual chunk alignment, GlotLID language-switch detection.

Core comparison condition in both tracks: **Target-CoT vs En-CoT** (reason in the problem's
language vs reason in English). Language switch = the model drifting into English mid-trace.

**Dataset constraint that WILL bite you: MGSM has no Arabic.** Use `en,fr,zh` for MGSM;
use MMMLU (or MMATH) for `ar`. The loader raises a clear error if you forget.

## Where things go (keep this structure)

```
src/                    correctness-critical reusable code (import as src.*):
  logprob_pivots/         pivot-score library:
                            config.py                frozen Config dataclass (models, langs, gen params)
                            sentence_segmentation.py pysbd segmentation → char spans
                            pivot_scores.py          logprob-gap pivot scores per sentence
                            prompts.py               prompt_target_cot / prompt_en_cot
  rollout_importance/     rollout pipeline:
                            data_loaders.py          MGSM / MMATH / MMMLU / PolyMath loaders
                            chunker.py               language-specific chunking (en/fr/zh/ar), LaTeX-aware
                            prompts.py               base-solution / rollout / DAG-labeling prompts
                            answer_utils.py          answer extraction, checking, normalization
                            lang_verifier.py         GlotLID chunk-level language detection
                            generate_rollouts.py     STAGE 1: async API rollout generation
                            compute_importance.py    STAGE 2: 6 importance metrics + GPT-4o DAG labeling
                            align_chunks.py          STAGE 3: LaBSE DP cross-lingual chunk alignment
                            make_figures.py          STAGE 4: 5 plot types
                            smoke_test.py            31 offline checks (python -m src.rollout_importance.smoke_test)
scripts/                pipeline drivers; a script pipes src/ functions together, no real logic
  logprob_pivots/         one script per pipeline stage — see "The pipelines" below
  vertex/                 Vertex AI job infra: build_and_push.sh → create_vertex_run_config.py →
                          submit_vertex_job.sh → download_vertex_results.sh;
                          vertex_job_runner.py is the container entrypoint (runs the
                          command after `--`, uploads output/rollouts/ to GCS)
configs/                YAML configs, foldered by track. NEVER hardcode run params in scripts.
  logprob_pivots/         default.yaml, controls.yaml
  vertex/                 job templates (experiment_a100.yaml, smoke_l4.yaml);
                          runs/ holds generated per-run copies — gitignored because the
                          generator INJECTS API KEYS into them; never commit them
scratch/                one-off and AI-generated scripts. Default home for new experimental
                        code; NOTHING imports from it. multicot_archive/ = superseded v1 code.
docs/                   docs/LOG.md (append-only research log, MOST RECENT FIRST) +
                        reports and writeups (docs/reports/)
data/                   small input datasets kept in git (mgsm_subset.csv)
output/                 ALL run artifacts:
  rollouts/               rollout_importance trees:
                          {dataset}/{model}/temperature_{T}_top_p_{P}/{lang}/
                          {base_solution_type}_base_solution/problem_{id}/chunk_{i}/solutions.json
  logprob_pivots/runs/    logprob_pivots runs: run_<unix_ts>/ (generations.jsonl, accuracy
                          CSVs, sentence_pivots.csv, redo_scaffold_reason.csv)
  figures/                plot output (default of src/rollout_importance/make_figures.py)
  vm_results/             frozen snapshot of results pulled from the GCP VM (code copies
                          inside it are stale duplicates — never edit or import them)
local/                  machine-local notes (gitignored)
```

**Run everything from the repository root.** All defaults (`data/`, `output/`,
`configs/`) are cwd-relative to the root, and `python -m scripts.logprob_pivots.<x>` /
`python -m src.rollout_importance.<x>` puts the root on `sys.path` so `src.*` imports
work without hacks.

**Respect the structure when adding code:**

- `src/` holds reusable, verified code, split by track (`logprob_pivots/`,
  `rollout_importance/`). If a script grows logic worth reusing, the logic moves into
  `src/` and the script stays thin.
- `scripts/` holds pipelines we expect to rerun. New AI-written one-offs do NOT go here —
  default to `scratch/` until the code earns promotion.
- **Names are self-describing.** A module or script name states what it computes or
  produces (`compute_pivot_scores.py`, `generate_rollouts.py`) — never a bare `utils.py`,
  `plots.py`, `run.py`, or `exp2/`. Stage drivers start with a verb.
- **Integrate, don't tack on.** Extend the existing module rather than adding
  `foo_v2.py` / `foo_new.py` siblings. Superseded code goes to git history (or
  `scratch/*_archive/` if it must stay visible), not next to the live copy.
- Config or job template changes go in `configs/<track>/`; never bury run parameters
  in a script.

## Environment

- **pip + venv only — never conda.** The venv lives at `venv/`; run everything with
  `venv/bin/python` (or activate it). One `requirements.txt` at the root serves both tracks.
- Secrets live in one gitignored `.env` at the root; copy `.env.example` and fill in.
  Keys: `TOGETHER_API_KEY` / `FIREWORKS_API_KEY` (rollout generation), `OPENAI_API_KEY`
  (GPT-4o DAG labeling in compute_importance), `OPENROUTER_API_KEY`,
  `VERTEX_API_BASE_URL` + `VERTEX_API_KEY` (self-hosted Vertex endpoint provider).
  Never print, log, or commit a secret. New env vars go into `.env.example`
  (names only, never values).
- logprob_pivots runs locally on MPS by default (`Config.device_preference = "mps"`);
  rollout_importance generation is pure API calls — no GPU needed on this machine.

## The pipelines (each stage = one explicit script)

### rollout_importance (counterfactual chunk importance, API models)

| Stage | Command | Produces |
|---|---|---|
| 0. smoke | `venv/bin/python -m src.rollout_importance.smoke_test` | 31 offline checks — run after ANY change to this track |
| 1. generate | `venv/bin/python -m src.rollout_importance.generate_rollouts --dataset mgsm --languages en,fr,zh -m <model> -p Together` | `output/rollouts/<dataset>/<model>/.../chunk_i/solutions.json` |
| 2. importance | `venv/bin/python -m src.rollout_importance.compute_importance --dataset mgsm --languages en,fr,zh -m <model>` | 6 importance metrics + GPT-4o DAG labels per chunk |
| 3. align | `venv/bin/python -m src.rollout_importance.align_chunks --dataset mgsm --lang1 en --lang2 fr` | LaBSE cross-lingual chunk alignment |
| 4. figures | `venv/bin/python -m src.rollout_importance.make_figures --dataset mgsm --languages en,fr,zh --model <model>` | `output/figures/` (5 plot types) |

Override locations with `-o` / `--rollouts_base` / `--rollouts_dir` if needed.

### logprob_pivots (logprob-gap pivot scores, local models)

`bash scripts/logprob_pivots/run_full_pipeline.sh` runs stages 0–5 end to end; stage by stage:

| Stage | Command | Produces |
|---|---|---|
| 0. smoke | `venv/bin/python -m scripts.logprob_pivots.smoke_test_models` | verifies both models load + generate |
| 1. data | `venv/bin/python -m scripts.logprob_pivots.build_mgsm_subset` | `data/mgsm_subset.csv` |
| 2. generate | `venv/bin/python -m scripts.logprob_pivots.generate_cot` | `output/logprob_pivots/runs/run_<ts>/generations.jsonl` |
| 3. accuracy | `venv/bin/python -m scripts.logprob_pivots.eval_accuracy` | `accuracy_by_lang_cond_model.csv` |
| 4. pivots | `venv/bin/python -m scripts.logprob_pivots.compute_pivot_scores --only-reason` | `sentence_pivots.csv` |
| 5. intervention | `venv/bin/python -m scripts.logprob_pivots.run_redo_scaffold --model reason --n-branches 3 --limit 50` | `redo_scaffold_reason.csv` |
| 6. faithfulness | `venv/bin/python -m scripts.logprob_pivots.run_faithfulness_tests` | STILL A STUB on real truncated generations |
| 7. figures | `venv/bin/python -m scripts.logprob_pivots.make_figures` | figures for the run |

Stages 3–7 default to the **latest** run under `output/logprob_pivots/runs/`; pass
`--run-dir` to target an older one.

### Vertex AI jobs (heavy rollout_importance runs on GCP)

```bash
bash scripts/vertex/build_and_push.sh [TAG]         # docker build + push (uses Dockerfile at root)
python scripts/vertex/create_vertex_run_config.py --template experiment_a100.yaml ...
                                                    # → configs/vertex/runs/<ts>_<name>.yaml (keys injected — DO NOT COMMIT)
bash scripts/vertex/submit_vertex_job.sh --project <p> --display-name <n> --config configs/vertex/runs/<file>.yaml
bash scripts/vertex/download_vertex_results.sh --gcs-uri gs://.../<run_id> --dest-dir ./vertex_downloads
```

The container entrypoint is `scripts/vertex/vertex_job_runner.py` — it runs whatever comes
after `--` (e.g. `python -m src.rollout_importance.generate_rollouts ...`) and uploads
`output/rollouts/` to GCS afterwards.

## Gotchas (learned the hard way)

1. **`.gitignore` globally ignores `*.json`, `*.csv`, `*.png`, `*.pkl`.** Result files you
   want tracked must be `git add -f`'d (that is how the tracked mmath rollouts and
   logprob_pivots run CSVs got in). Conversely: never assume a result file is in git just
   because it is on disk.
2. **`configs/vertex/runs/` contains real API keys** injected by
   `create_vertex_run_config.py`. It is gitignored and dockerignored — keep it that way.
3. **MGSM has no Arabic** (see above). `en,fr,zh` for MGSM; `ar` via MMMLU.
4. **Run the smoke test with the venv Python, never system python3**:
   `venv/bin/python -m src.rollout_importance.smoke_test`. 31 checks, offline, ~1 min.
5. `output/vm_results/` contains stale *copies* of the rollout pipeline code (under its
   old `multicot` name) alongside the VM's results. Only `src/rollout_importance/` is
   live; never edit or import the copies.
6. The OpenAI client in `compute_importance.py` is lazily initialized via `_get_client()` —
   keep it lazy so offline analysis paths don't require `OPENAI_API_KEY`.
7. `generate_rollouts.py` and `compute_importance.py` parse args at module import — they
   can only be run as `python -m ...`, never imported by other code (the smoke test works
   around this deliberately).
8. logprob_pivots accuracy runs exist for run_1767244523 / run_1767307353 /
   run_1767410979, but pivot scores + redo scaffold only for the latest (run_1767410979).
9. Comments citing "root utils.py" / "root analyze_rollouts.py" refer to the upstream
   thought-anchors codebase this pipeline was adapted from — they are provenance, not
   references to files in this repo.

## When you finish a task

- Append a `docs/LOG.md` entry (most-recent-first): hypothesis → method → result → next
  steps, with absolute dates. LOG.md is for **experiments and major code changes only** —
  routine refactors, chores, and doc edits get no entry. Never rewrite a past entry;
  correct it with a follow-up line.
- Rerun the relevant smoke test (`python -m src.rollout_importance.smoke_test` for the
  rollout track; `python -m scripts.logprob_pivots.smoke_test_models` for the local track)
  before declaring done.
- New experiment results go under `output/`; only small, deliberately-kept summaries get
  `git add -f`'d.
- Writeups and findings go in `docs/reports/`.
