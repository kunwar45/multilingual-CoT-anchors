# Proxy task and success metrics

## Proxy task

**Question.** Given a non-English MGSM problem, does choosing the reasoning
language (Target-CoT vs En-CoT) change:

- **accuracy** on the math task, and
- **faithfulness** of the reasoning trace (in the sense that perturbing the
  reasoning changes the answer)?

**Core proxy task.**

For each problem and language \(L \in \{\text{es}, \text{fr}, \text{de}\}\),
we compare:

- **Target-CoT**: prompt and reasoning in \(L\)
- **En-CoT**: prompt in \(L\), reasoning in English, answer in \(L\)

and evaluate:

1. Accuracy per language × condition (Target-CoT vs En-CoT) for both base and
   reasoning-tuned models.
2. Faithfulness proxies based on answer sensitivity to perturbations of the
   reasoning trace (truncation and swaps).

## Primary metrics

1. **Task accuracy**

   - Exact match / numeric match between predicted answer and MGSM gold answer.
   - Reported by language × condition × model, saved as
     `accuracy_by_lang_cond_model.csv`.

2. **Faithfulness under perturbation**

   - **Truncation sensitivity.** For each trace and a chosen sentence boundary
     (typically a high-scoring pivot sentence), truncate the reasoning at that
     boundary, regenerate the final answer if needed, and measure the rate at
     which the answer changes.

   - **Trace swap sensitivity.** Swap reasoning traces across conditions or
     languages (e.g., use an En-CoT trace with a Target-CoT question) and
     measure the rate at which the answer changes.

   For both perturbations we use the **answer flip rate**

   \[
   \mathrm{FlipRate}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left[a_i \neq a_i'\right]
   \]

   where \(a_i\) is the original answer and \(a_i'\) is the answer after
   perturbation.


