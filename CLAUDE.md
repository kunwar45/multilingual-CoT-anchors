<!-- ABOUTME: Repo guide for AI agents: project overview, directory contract, pipelines, HF publishing policy, gotchas. -->
<!-- ABOUTME: Curated by humans — agents must not edit it unless specifically asked to. -->
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
  hf_publishing.py        Hugging Face publishing: enforces the repo naming policy and
                          required dataset-card fields in code (see "Datasets ... go to
                          Hugging Face" below)
  hf_fetching.py          Hugging Face fetching: CANONICAL_DATASETS registry + ensure_*
                          helpers — pipeline stages auto-download missing inputs, so a
                          fresh clone runs with no local data
  logprob_pivots/         pivot-score library:
                            experiment_config.py     frozen Config dataclass (models, langs, gen params)
                            sentence_segmentation.py pysbd segmentation → char spans
                            pivot_scores.py          logprob-gap pivot scores per sentence
                            prompts.py               prompt_target_cot / prompt_en_cot
  rollout_importance/     rollout pipeline:
                            data_loaders.py          MGSM / MMATH / MMMLU / PolyMath loaders
                            chunker.py               language-specific chunking (en/fr/zh/ar), LaTeX-aware
                            prompts.py               base-solution / rollout / DAG-labeling prompts
                            answer_extraction.py     answer extraction, checking, normalization
                            language_verification.py GlotLID chunk-level language detection
                            generate_rollouts.py     STAGE 1: async API rollout generation
                            compute_importance.py    STAGE 2: 6 importance metrics + GPT-4o DAG labeling
                            align_chunks.py          STAGE 3: LaBSE DP cross-lingual chunk alignment
                            make_figures.py          STAGE 4: 5 plot types
                            smoke_test.py            31 offline checks (python -m src.rollout_importance.smoke_test)
scripts/                pipeline drivers; a script pipes src/ functions together, no real logic
  publish_to_hf.py        THE publishing entrypoint: pushes an artifact to the HF Hub with
                          an enforced dataset card (spans both tracks, hence top-level)
  logprob_pivots/         one script per pipeline stage — see "The pipelines" below
  vertex/                 Vertex AI job infra: build_and_push_docker_image.sh → create_vertex_run_config.py →
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
data/                   staged inputs (gitignored) — mgsm_subset.csv; rebuild with
                        build_mgsm_subset or pull from HF multicot/2026-01-01-mgsm-subset-en-es-fr-de
output/                 run artifacts (gitignored), LOCAL ITERATION ONLY — the archive is
                        the multicot org on HF; delete local copies once published:
  rollouts/               rollout_importance trees:
                          {dataset}/{model}/temperature_{T}_top_p_{P}/{lang}/
                          {base_solution_type}_base_solution/problem_{id}/chunk_{i}/solutions.json
  logprob_pivots/runs/    logprob_pivots runs: run_<unix_ts>/ (generations.jsonl, accuracy
                          CSVs, sentence_pivots.csv, redo_scaffold_reason.csv)
  figures/                plot output (default of src/rollout_importance/make_figures.py)
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
  `plots.py`, `run.py`, or `exp2/`. No abbreviations (`language_verification.py`, not
  `lang_verifier.py`). Stage drivers start with a verb.
- **Every file starts with an ABOUTME header.** The first two comment lines of every
  file (after the shebang, if any) each begin with `ABOUTME:` and together say what the
  file is and how it fits the repo — greppable via `grep -r "ABOUTME:"`. Python/shell/
  YAML use `# ABOUTME: ...`; Markdown uses `<!-- ABOUTME: ... -->`; notebooks use a
  leading markdown cell. Exceptions: `.gitkeep` (must stay empty) and JSON (no comment
  syntax). When you add a file, add its ABOUTME; when a file's purpose changes, update it.
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

## Datasets, run artifacts and eval results go to Hugging Face

**Any dataset, rollout tree, eval result, run artifact, or trained model produced by work
in this repository is published to the project org: https://huggingface.co/multicot.**
The repository holds code and configuration; `output/` exists only for fast local
iteration and plots. The canonical location for artifacts and results is HF. This applies
to rollout trees, generation runs, accuracy tables, pivot scores, alignment outputs,
judge/labeler outputs, embeddings, and any model organism / adapter we ever train
(`--repo-type model`).

Publish with the one entrypoint (token = `HF_TOKEN` in `.env` — it must have **write
access to the multicot org**; the cached `hf auth login` token does not. Namespace comes
from `HF_NAMESPACE=multicot` in `.env`. Repos are **public by default**; pass `--private`
deliberately):

```bash
venv/bin/python scripts/publish_to_hf.py \
  --path output/rollouts \
  --name <YYYY-MM-DD>-<short-description> \
  --track rollout_importance --date-generated <date> --languages en,fr,zh \
  --models "<every model id>" --experiment "<one sentence>" \
  --provenance "<exact command to regenerate>" \
  [--generation-config ...] [--schema ...] [--notes ...]
```

### Naming: the title carries the date and the subject

`<YYYY-MM-DD>-<short-experiment-description>`, lowercase with hyphens. The date is the
date the data was **generated**, not uploaded. `src/hf_publishing.py` rejects
non-conforming names — do not work around it.

### Required dataset-card fields (enforced in code)

| field | meaning |
| --- | --- |
| `experiment` | Which experiment produced this, in one sentence |
| `date_generated` | ISO date the data was produced |
| `track` | `logprob_pivots` or `rollout_importance` |
| `languages` | Languages covered — the field most easily lost and the one a future reader needs most |
| `models` | Every model id involved |
| `provenance` | How to regenerate: the exact script and arguments |
| `source_repo` | Added automatically: this repo + the git SHA at publish time |

### What stays in git / what does not

**In git:** code, configs, prompts, seeds, docs/reports, and a **pointer to every HF repo
in `docs/LOG.md`** — the link must never live only in someone's memory. **Not in git and
not kept locally once published:** inputs, rollout trees, run outputs, model weights,
caches. `output/` and `data/` are fully gitignored; results deleted locally after
publishing live on in HF (and pre-2026-08-05 copies in git history).

### Code treats HF as the canonical source (not just the archive)

`src/hf_fetching.py` holds `CANONICAL_DATASETS` — the code's single map of where data
lives — plus `ensure_*` helpers. `generate_cot` fetches the MGSM subset if `data/` is
empty; the logprob analysis stages fetch the published runs if `output/` is empty
(`latest_logprob_run_dir` replaces the per-script `find_latest_run`); the rollout
analysis stages fetch the published rollout trees. **When you publish a new canonical
artifact, update `CANONICAL_DATASETS` in the same change** — otherwise the code keeps
fetching the old data. Local copies fetched this way are disposable; delete them freely.

### The scratch bucket (in-progress data only)

`hf://buckets/multicot/rollout-scratch` is a **mutable bucket with no history** — the
directory structure is the only organization it has. Use it to sync in-progress rollout
trees off a VM/Vertex box before a run is frozen into a dated dataset repo; layout is
`rollouts/<run-id>/<dataset>/<model>/...` — always include the run id, never sync to the
bucket root. Sync with `hf sync` (env: `set -a; source .env; set +a`). A finished run
graduates to a dated dataset repo via `scripts/publish_to_hf.py`; the bucket copy is then
deletable. Never treat the bucket as the archive.

### Published so far (all public, under `multicot/`)

- `2026-01-01-mgsm-subset-en-es-fr-de` — the fixed MGSM input subset for logprob_pivots
- `2026-01-03-logprob-pivots-qwen25-05b-mgsm-runs` — the three logprob_pivots runs
- `2026-04-30-rollout-importance-pilot-rollouts` — local pilot rollout trees (mgsm/mmath/mmmlu)
- `2026-05-04-rollout-importance-vm-results` — frozen GCP VM results snapshot

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
| 1. data | `venv/bin/python -m scripts.logprob_pivots.build_mgsm_subset` (or `hf download multicot/2026-01-01-mgsm-subset-en-es-fr-de --repo-type dataset --local-dir data/`) | `data/mgsm_subset.csv` |
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
bash scripts/vertex/build_and_push_docker_image.sh [TAG]   # docker build + push (uses Dockerfile at root)
python scripts/vertex/create_vertex_run_config.py --template experiment_a100.yaml ...
                                                    # → configs/vertex/runs/<ts>_<name>.yaml (keys injected — DO NOT COMMIT)
bash scripts/vertex/submit_vertex_job.sh --project <p> --display-name <n> --config configs/vertex/runs/<file>.yaml
bash scripts/vertex/download_vertex_results.sh --gcs-uri gs://.../<run_id> --dest-dir ./vertex_downloads
```

The container entrypoint is `scripts/vertex/vertex_job_runner.py` — it runs whatever comes
after `--` (e.g. `python -m src.rollout_importance.generate_rollouts ...`) and uploads
`output/rollouts/` to GCS afterwards.

## Gotchas (learned the hard way)

1. **No results live in git or on local disk long-term.** `.gitignore` blocks `output/`,
   `data/`, and all `*.json`/`*.csv`/`*.png` — do NOT `git add -f` result files; publish
   them to HF instead. A result that exists only in `output/` is one `rm -rf` from gone.
   (Result files committed before 2026-08-05 remain in git history at their old paths.)
2. **`configs/vertex/runs/` contains real API keys** injected by
   `create_vertex_run_config.py`. It is gitignored and dockerignored — keep it that way.
3. **MGSM has no Arabic** (see above). `en,fr,zh` for MGSM; `ar` via MMMLU.
4. **Run the smoke test with the venv Python, never system python3**:
   `venv/bin/python -m src.rollout_importance.smoke_test`. 31 checks, offline, ~1 min.
5. The GCP VM results snapshot (HF `multicot/2026-05-04-rollout-importance-vm-results`)
   contains stale *copies* of the rollout pipeline code under its old `multicot` module
   name. Only `src/rollout_importance/` is live; never import or resurrect the copies.
6. The OpenAI client in `compute_importance.py` is lazily initialized via `_get_client()` —
   keep it lazy so offline analysis paths don't require `OPENAI_API_KEY`.
7. `generate_rollouts.py` and `compute_importance.py` parse args at module import — they
   can only be run as `python -m ...`, never imported by other code (the smoke test works
   around this deliberately).
8. logprob_pivots accuracy runs exist for run_1767244523 / run_1767307353 /
   run_1767410979 (all in HF `multicot/2026-01-03-logprob-pivots-qwen25-05b-mgsm-runs`;
   nothing local), but pivot scores + redo scaffold only for the latest (run_1767410979).
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
- **Publish new datasets/results to Hugging Face** via `scripts/publish_to_hf.py`
  (naming + card rules above) and record the repo URL in the `docs/LOG.md` entry.
  `output/` is iteration space, not the archive.
- Writeups and findings go in `docs/reports/`.
