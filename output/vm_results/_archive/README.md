# Multilingual Chain-of-Thought Analysis (MultiCoT)

Extension of Thought Anchors for multilingual reasoning analysis on MGSM.

## Research Question

Are model "reasoning reflexes" (planning, checking, backtracking) stable across languages, or do they become more effective when thinking in English?

## Language Conditions

Each experiment uses a (P_L, T_L, A_L) tuple:
- **P_L**: Problem language
- **T_L**: Thinking (reasoning) language
- **A_L**: Answer language

| Condition | (P_L, T_L, A_L) | Description |
|-----------|-----------------|-------------|
| `native` | (L, L, L) | Problem, thinking, answer all in language L |
| `english_baseline` | (en, en, en) | Everything in English |
| `english_thinking` | (L, en, L) | Problem in L, think in English, answer in L |
| `translate_solve` | (L→en, en, en) | Translate problem, solve in English |

## Quick Start

### 1. Setup

```bash
cd multicot
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

### 2. Verify Model

```bash
python scripts/smoke_model.py
```

### 3. Prepare Pilot Data

```bash
python scripts/prepare_pilot_data.py --n 20 --languages en,es,fr
```

### 4. Generate Rollouts

```bash
# English native condition
python scripts/generate_rollouts.py \
    --input data/processed/pilot_en_20.jsonl \
    --output runs/pilot_en_native \
    --condition native

# Spanish with English thinking
python scripts/generate_rollouts.py \
    --input data/processed/pilot_es_20.jsonl \
    --output runs/pilot_es_en_thinking \
    --condition english_thinking
```

### 5. Analyze

```bash
python scripts/analyze_rollouts.py \
    --input runs/pilot_en_native/pilot_en_20_native.jsonl \
    --output runs/pilot_en_native/analysis
```

### 6. Compare Conditions

```bash
python scripts/compare_conditions.py \
    --runs runs/pilot_en_native runs/pilot_es_native runs/pilot_es_en_thinking \
    --output figures/comparison.json
```

## Project Structure

```
multicot/
├── configs/          # Experiment configurations
│   ├── default.json  # Full experiment settings
│   └── pilot.json    # Pilot study (20 problems, English only)
├── data/
│   ├── raw/          # Downloaded datasets
│   └── processed/    # Prepared problem sets (JSONL)
├── runs/             # Experiment outputs
├── src/
│   ├── datasets/     # MGSM loader
│   ├── prompts/      # Language condition templates
│   ├── rollouts/     # Generation + answer parsing
│   └── analysis/     # Segmentation + importance
├── figures/          # Output plots and comparisons
└── scripts/          # CLI entry points
```

## Key Modules

### `src/datasets/mgsm.py`
- `MGSMLoader`: Load MGSM with stable problem IDs across languages
- `select_pilot_problems()`: Sample problems for experiments

### `src/prompts/templates.py`
- `LanguageCondition`: Define (P_L, T_L, A_L) conditions
- `build_cot_prompt()`: Generate prompts with language control
- Predefined conditions: `CONDITION_NATIVE`, `CONDITION_ENGLISH_THINKING`, etc.

### `src/rollouts/generator.py`
- `RolloutGenerator`: Resume-safe JSONL generation
- `RolloutConfig`: Model and generation settings
- `create_run_manifest()`: Record experiment parameters

### `src/rollouts/parser.py`
- `parse_final_answer()`: Extract answers from "Final:" marker
- `normalize_number()`: Handle comma separators, etc.

### `src/analysis/segmentation.py`
- `segment_cot()`: Split CoT into sentences (spacy or regex fallback)
- `Sentence`: Dataclass with char/token positions

### `src/analysis/importance.py`
- Two-stage importance: sensitivity → candidate anchors → full resampling
- `compute_sensitivity_scores()`: Cheap heuristic pass
- `compute_anchor_importance()`: Full resampling for candidates

### `src/analysis/metrics.py`
- `compute_accuracy()`: Overall and per-problem accuracy
- `compute_anchor_concentration()`: Gini coefficient, top-k ratio
- `compute_position_stats()`: Where anchors occur in trace

## API Usage

For API-based generation (OpenRouter, Together, Fireworks):

```bash
export OPENROUTER_API_KEY=your_key
python scripts/generate_rollouts.py \
    --input data/processed/pilot_en_20.jsonl \
    --output runs/api_test \
    --use-api --api-provider openrouter \
    --model deepseek/deepseek-r1
```

## Output Format

### Rollouts (`*.jsonl`)
```json
{
  "rollout_id": "abc123",
  "problem_id": "mgsm_042",
  "language": "es",
  "condition": "english_thinking",
  "prompt": "...",
  "cot_text": "Let me solve this step by step...",
  "final_answer_text": "42",
  "correct": true,
  "ground_truth": "42",
  "seed": 12345,
  "model_name": "deepseek-ai/...",
  "timestamp": "2025-01-30T..."
}
```

### Analysis (`sentence_analysis.json`)
```json
{
  "rollout_id": "abc123",
  "num_sentences": 15,
  "sentences": [
    {"idx": 0, "text": "First, let's understand...", "sensitivity": 0.8, "is_candidate": true}
  ],
  "concentration": {"concentration_ratio": 0.65, "gini_coefficient": 0.42},
  "position_stats": {"weighted_mean_position": 3.2}
}
```

## Caching

All outputs are cached:
- Rollouts: JSONL files with `rollout_id` for resume
- Analysis: JSON files per run

Regenerate by deleting the output file.
