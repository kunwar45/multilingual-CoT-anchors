<!-- ABOUTME: Research overview: the question (do reasoning pivots transfer across languages?) and how the rollout_importance pipeline answers it. -->
<!-- ABOUTME: Explains the primary experiment track end to end; CLAUDE.md remains the operational guide for running the repo. -->
# Project overview — language-invariant reasoning pivots

## What we are trying to show

When an LLM reasons through a math problem, certain sentences act as *pivots* —
moments where it checks, backtracks, or redoes work — and these moments
disproportionately determine whether it gets the answer right (the "Thought
Anchors" idea). We ask: **are these pivots a property of the reasoning itself,
or an artifact of the language it is written in?** If a model solves the same
problem in English, French, and Chinese, do the important chunks land at the
same conceptual steps? And does letting the model reason in English (**En-CoT**)
versus the problem's own language (**Target-CoT**) change where they land — or
whether the model drifts into English mid-trace when the reasoning gets hard?

If importance transfers across languages, that is evidence for
language-invariant reasoning structure. If it does not, "reasoning" behavior is
more surface-bound than the anchors literature assumes.

## How we measure it: rollout-based counterfactual importance

The primary method (`src/rollout_importance/`) treats importance causally,
black-box, on API models over MGSM / MMATH / MMMLU / PolyMath in en/fr/zh/ar:

1. **Chunk** the chain of thought with a language-aware, LaTeX-aware chunker.
2. **Ablate and resample**: remove chunk *i*, resample the continuation many
   times, and measure how much the outcome shifts — Δ accuracy and KL at chunk
   *i+1* — keeping only resamples that actually said something different
   (embedding cosine < 0.8). Six importance metrics in total.
3. **Label structure**: GPT-4o labels the chunk-dependency DAG (what each chunk
   builds on).
4. **Compare across languages**: LaBSE dynamic-programming alignment matches
   chunks between languages, so we can ask whether the important chunk in
   French is the same conceptual step as the important chunk in Chinese; GlotLID
   flags language switches mid-trace.

## What runs what

| Stage | Module | Produces |
|---|---|---|
| 0 | `src.rollout_importance.smoke_test` | 31 offline checks — run after any change to the track |
| 1 | `src.rollout_importance.generate_rollouts` | async API rollout trees under `output/rollouts/` |
| 2 | `src.rollout_importance.compute_importance` | 6 importance metrics + GPT-4o DAG labels per chunk |
| 3 | `src.rollout_importance.align_chunks` | LaBSE cross-lingual chunk alignment |
| 4 | `src.rollout_importance.make_figures` | 5 plot types under `output/figures/` |

Supporting modules in `src/rollout_importance/`: `data_loaders` (MGSM / MMATH /
MMMLU / PolyMath), `chunker`, `prompts`, `answer_extraction`,
`language_verification` (GlotLID). Heavy runs go through the Vertex AI infra in
`scripts/vertex/` + `configs/vertex/`. All frozen artifacts live on the
[multicot Hugging Face org](https://huggingface.co/multicot) —
`scripts/publish_to_hf.py` publishes with enforced naming and dataset cards, and
`src/hf_fetching.py` auto-downloads canonical inputs so a fresh clone runs with
no local data.

Dataset constraint worth stating up front: **MGSM has no Arabic** — use
`en,fr,zh` for MGSM; Arabic comes from MMMLU (or MMATH).

(A secondary, exploratory track — `logprob_pivots`, sentence-level pivot scores
from a base↔instruct logprob gap on small local models — also lives in the repo;
see CLAUDE.md.)
