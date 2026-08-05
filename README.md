<!-- ABOUTME: Public front page: what the language-invariant reasoning pivots project is and how the two tracks differ. -->
<!-- ABOUTME: Agents should read CLAUDE.md instead — it is the operational guide for this repo. -->
# multilingual-CoT-anchors

Do "redo / check / backtrack" reasoning pivots in LLM chains of thought transfer across
languages, or are they language-specific artifacts? Two experiment tracks, named after
their method (inspired by the Thought Anchors literature):

- **logprob_pivots** — local Qwen2.5-0.5B base vs Instruct on MGSM (en/es/fr/de);
  sentence-level pivot scores from the base↔instruct logprob gap, plus a pivot-triggered
  redo scaffold.
- **rollout_importance** — API models on MGSM/MMATH/MMMLU/PolyMath (en/fr/zh/ar);
  rollout-based counterfactual chunk importance (6 metrics), GPT-4o DAG labeling, LaBSE
  cross-lingual chunk alignment, GlotLID language-switch detection.

```text
.
├── src/                     # reusable code (import as src.*)
│   ├── logprob_pivots/      #   config, sentence_segmentation, pivot_scores, prompts
│   └── rollout_importance/  #   loaders, chunker, generate_rollouts → compute_importance
│                            #   → align_chunks → make_figures, + smoke_test
├── scripts/                 # pipeline drivers: thin CLIs over src/
│   ├── logprob_pivots/      #   one script per stage + run_full_pipeline.sh
│   └── vertex/              #   Vertex AI build/submit/download + container job runner
├── configs/                 # YAML: logprob_pivots/ + vertex/ job templates
│                            #   (vertex/runs/ is gitignored — generated copies carry injected keys)
├── scratch/                 # one-off and AI-generated scripts; nothing imports from it
├── data/                    # small input datasets kept in git (mgsm_subset.csv)
├── docs/                    # LOG.md (research log) + reports/ (writeups)
├── output/                  # all run artifacts: rollouts/, logprob_pivots/runs/, figures/, vm_results/
└── local/                   # machine-local notes (gitignored)
```

**Run everything from the repository root** with the venv Python:

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cp .env.example .env      # fill in API keys

# rollout_importance: generate → compute importance → align → figures
venv/bin/python -m src.rollout_importance.smoke_test
venv/bin/python -m src.rollout_importance.generate_rollouts  --dataset mgsm --languages en,fr,zh -m <model> -p Together
venv/bin/python -m src.rollout_importance.compute_importance --dataset mgsm --languages en,fr,zh -m <model>
venv/bin/python -m src.rollout_importance.align_chunks       --dataset mgsm --lang1 en --lang2 fr
venv/bin/python -m src.rollout_importance.make_figures       --dataset mgsm --languages en,fr,zh --model <model>

# logprob_pivots: build subset → generate CoT → accuracy → pivot scores → redo scaffold
bash scripts/logprob_pivots/run_full_pipeline.sh
```

Note: MGSM has no Arabic — use `en,fr,zh` for MGSM and MMMLU for `ar`.

## Conventions

Read [`CLAUDE.md`](CLAUDE.md) — the agent operating guide and repository-wide
conventions — before running an experiment or committing. The rule that bites soonest:

> **Datasets, rollout trees, eval results and model artifacts go to the
> [multicot org on Hugging Face](https://huggingface.co/multicot), not into git.**
> HF repos are named `<YYYY-MM-DD>-<short-experiment-description>` using the
> date the data was *generated*. Every dataset card states the experiment, the models,
> and **the languages covered**. Publish with `scripts/publish_to_hf.py`, which enforces
> both rules, and record the repo URL in `docs/LOG.md`.

Code, configs, prompts, seeds, analysis and reports stay in git; bulk data does not.
`output/` is local iteration space only. Results are logged chronologically in
[`docs/LOG.md`](docs/LOG.md).
