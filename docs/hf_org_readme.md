<!-- ABOUTME: Draft README for the multicot Hugging Face org page (org cards are edited via the HF web UI). -->
<!-- ABOUTME: Keep in sync by hand when the dataset list or publishing policy changes. -->
# multicot — language-invariant reasoning pivots in LLMs

Do "redo / check / backtrack" reasoning pivots in chains of thought transfer across
languages, or are they language-specific artifacts? Two experiment tracks (inspired by
the Thought Anchors literature): the primary track **rollout_importance**
(counterfactual chunk importance from resampled rollouts, API models on
MGSM/MMATH/MMMLU/PolyMath in en/fr/zh/ar) and the secondary **logprob_pivots**
(sentence pivot scores from the base↔instruct logprob gap, local Qwen2.5-0.5B on MGSM
en/es/fr/de).

## What each repo holds

- **Dataset repos, named `<YYYY-MM-DD>-<description>`** (date the data was *generated*):
  frozen run artifacts — rollout trees, generation runs, accuracy tables, pivot scores.
  Every card states the experiment, models, **languages**, and the exact command to
  regenerate it.
- **`rollout-scratch` (bucket)**: mutable staging for in-progress rollout syncs from
  VMs/Vertex jobs, laid out `rollouts/<run-id>/<dataset>/<model>/...`. Not an archive —
  finished runs graduate to a dated dataset repo.
- **Model repos**: trained model organisms / adapters, when they exist.

Source code and configs live in the `multilingual-CoT-anchors` git repository (see its
`CLAUDE.md` for conventions and `docs/LOG.md` for the chronological research log).

Maintainer: Kunwar Nir (kunwar45 / lasr-g3s26@arcadiaimpact.org).
