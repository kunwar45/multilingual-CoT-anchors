# Language-Invariant Reasoning Anchors (draft MATS writeup)

## 1. Question and motivation

- **Question.** Do “redo/check/backtrack” reasoning regimes transfer across
  languages, and when they fail, is it due to superficial tokenization/script
  differences or deeper representational mismatch?
- **Motivation.** Multilingual CoT behavior is brittle: the language you think
  in affects both accuracy and faithfulness-like probes. At the same time, new
  work (e.g., Thought Anchors) shows that sentence-level causal structure and
  backtracking regimes can be measured. This project sits at the intersection:
  multilingual CoT × reasoning vs base diffing × pivot-triggered redo.

## 2. Setup

- **Dataset.** MGSM ([juletxara/mgsm on Hugging Face](https://huggingface.co/datasets/juletxara/mgsm)):
  human-translated GSM8K problems in multiple languages.
- **Languages.** English + three same-script languages (es, fr, de).
- **Models.** One base-ish Qwen model and one reasoning-tuned Qwen/Instruct
  variant with the same tokenizer.
- **Conditions.**
  - Target-CoT: think and answer in the target language.
  - En-CoT: think in English, answer in the target language.
  - (Optional) No-CoT: direct answer with minimal explanation.

## 3. Method

- **Pivot object.** Sentence-level pivot score based on logprob gaps between
  reasoning and base models; all scoring is done at sentence boundaries using
  `pysbd` segmentation to reduce tokenizer artifacts.
- **Tokenization/script controls.**
  - Sentence-boundary aggregation (not token-level spikes).
  - Within-language paraphrases of reasoning traces; check that pivot
    locations/scores persist.
  - Optional back-translation control.
- **Scaffold.** A simple pivot-triggered redo mechanism:
  - generate a trace;
  - compute sentence pivot scores;
  - if max pivot score exceeds a threshold, redo from the preceding sentence
    boundary with a small branch of alternative continuations;
  - aggregate branches via majority vote on the final numeric answer.
- **Baselines.**
  - No redo.
  - Random redo (same budget, random boundary).
  - Heuristic redo (length / keyword-based).

## 4. Results (to be filled after experiments)

- **Figure 1.** Accuracy by language × condition (Target-CoT vs En-CoT) for
  base vs reasoning models.
- **Figure 2.** Failure rate as a function of pivot score quantile.
- **Figure 3.** Accuracy improvement from pivot-triggered scaffold compared to
  random and heuristic baselines.

## 5. Limitations and confounds

- Small dataset slice (200 problems) and relatively small models.
- Quality of MGSM translations and paraphrases may confound language effects.
- Pivot score is a logprob-gap proxy rather than a full causal estimate
  (counterfactual resampling is only run on a small subset).

## 6. Next steps

- Scale to additional scripts (e.g., Arabic, Japanese) with explicit
  transliteration controls.
- Connect pivot directions to mechanistic features (e.g., linear probes on
  activations, steering vectors).
- Explore alternative scaffolds (e.g., using base model as verifier for the
  reasoning model’s trace).


