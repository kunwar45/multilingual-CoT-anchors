# Error taxonomy (draft)

This document summarizes a small, structured taxonomy of failure modes
observed in the MGSM multilingual CoT experiments.

## Buckets

1. **Translation / phrasing ambiguity**
   - The target-language problem statement is ambiguous, mistranslated, or
     uses phrasing that suggests a different operation than the English
     canonical version.

2. **Early arithmetic / logic slip**
   - An arithmetic or logical mistake appears in one of the first 1–2
     reasoning steps and is never corrected.

3. **Mid-trace reasoning drift**
   - The model starts with a correct plan but drifts to an incorrect subgoal
     or misapplies an operation in the middle of the chain.

4. **Final-step answer flip**
   - The reasoning is essentially correct, but the final numeric answer is
     mis-copied, mis-formatted, or otherwise inconsistent with the preceding
     steps.

5. **Refusal / safety interference**
   - The model refuses to answer or produces meta-text about safety or policy
     instead of solving the math problem.

6. **Other / unclear**
   - Failures that do not cleanly fit into the above categories.

## Labelling protocol

- Sample ~50 failures where:
  - En-CoT succeeds / Target-CoT fails (or vice versa), and/or
  - the pivot-triggered scaffold helps vs does not help.
- Manually assign each example to exactly one primary bucket above.
- Store labels in `runs/<run_id>/labeled_failures.csv` with columns:
  - `id`, `lang`, `cond`, `model`, `gold`, `pred`, `bucket`, `notes`.


