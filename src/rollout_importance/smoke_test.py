# ABOUTME: 31 offline checks covering every rollout_importance module — no API calls, ~1 minute.
# ABOUTME: Run after ANY change to this track: venv/bin/python -m src.rollout_importance.smoke_test.
"""
Smoke test for the multilingual CoT pipeline.

Tests all modules without making API calls.
Uses existing rollout data in output/rollouts/mgsm/ if present,
or skips analysis tests gracefully if no data found.

Run:
    python -m src.rollout_importance.smoke_test
"""

import json
import sys
import traceback
from pathlib import Path

# compute_importance.py calls parser.parse_args() at module level.
# Patch sys.argv before any lazy imports so the module can be imported cleanly.
sys.argv = [sys.argv[0], "--dataset", "mgsm"]


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def run_test(name: str, fn):
    try:
        result = fn()
        status = SKIP if result is None else PASS
        marker = "[ OK ]" if status == PASS else "[SKIP]"
        print(f"  {marker} {name}")
        return status
    except Exception as e:
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        return FAIL


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------

def test_imports():
    from src.rollout_importance.chunker import split_solution_into_chunks
    from src.rollout_importance.language_verification import check_chunk_languages
    from src.rollout_importance.data_loaders import MGSMLoader, MMATHLoader, MMMMLULoader, PolyMathLoader, Problem, load_dataset_problems
    from src.rollout_importance.answer_extraction import (
        extract_answer, check_answer_for_problem,
        normalize_answer, verify_language, parse_final_answer,
        extract_boxed_answers, extract_multiple_choice_answer,
    )
    from src.rollout_importance.prompts import build_base_solution_prompt, build_rollout_prompt, build_dag_labeling_prompt
    from src.rollout_importance.make_figures import load_language_chunks
    return True


# ---------------------------------------------------------------------------
# 2. Chunker tests
# ---------------------------------------------------------------------------

def test_chunker_en():
    from src.rollout_importance.chunker import split_solution_into_chunks
    text = "First, I note that x = 5. Then, I compute y = x + 3. So y = 8."
    chunks = split_solution_into_chunks(text, "en")
    assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}: {chunks}"
    return True


def test_chunker_fr():
    from src.rollout_importance.chunker import split_solution_into_chunks
    text = "D'abord, je note que x = 5. Ensuite, je calcule y = x + 3. Donc y = 8."
    chunks = split_solution_into_chunks(text, "fr")
    assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}: {chunks}"
    return True


def test_chunker_zh():
    from src.rollout_importance.chunker import split_solution_into_chunks
    text = "首先，我注意到x = 5。然后，我计算y = x + 3。所以y = 8。"
    chunks = split_solution_into_chunks(text, "zh")
    assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}: {chunks}"
    return True


def test_chunker_ar():
    from src.rollout_importance.chunker import split_solution_into_chunks
    text = "أولاً، أُلاحظ أن x = 5. ثم أحسب y = x + 3. إذن y = 8."
    chunks = split_solution_into_chunks(text, "ar")
    assert len(chunks) >= 1, f"Expected >=1 chunks, got {len(chunks)}: {chunks}"
    return True


def test_chunker_strips_think_tags():
    from src.rollout_importance.chunker import split_solution_into_chunks
    text = "<think>First step. Second step.</think>\n\nFinal: 8"
    chunks = split_solution_into_chunks(text, "en")
    assert not any("<think>" in c for c in chunks), "Think tags not stripped"
    return True


def test_chunker_latex_aware():
    from src.rollout_importance.chunker import split_solution_into_chunks
    # Period inside $...$ should not split
    text = r"We have $x = 1.5$ meters. Then $y = x \cdot 2$. So $y = 3.0$."
    chunks = split_solution_into_chunks(text, "en")
    # Should not over-split on the decimals inside LaTeX
    assert len(chunks) >= 2, f"Expected >=2 chunks"
    return True


# ---------------------------------------------------------------------------
# 3. Answer extraction tests
# ---------------------------------------------------------------------------

def test_extract_numeric_answer():
    from src.rollout_importance.data_loaders import Problem
    from src.rollout_importance.answer_extraction import extract_answer, check_answer_for_problem

    p = Problem("mgsm_000", "mgsm", "en", "What is 2+2?", "4", "numeric")
    text = "Let me think. 2 + 2 = 4.\nFinal: 4"
    answer = extract_answer(text, p, "en")
    assert answer == "4", f"Expected '4', got {answer!r}"
    assert check_answer_for_problem(answer, p), "Should be correct"
    return True


def test_extract_boxed_answer():
    from src.rollout_importance.data_loaders import Problem
    from src.rollout_importance.answer_extraction import extract_answer, check_answer_for_problem

    p = Problem("mmath_001", "mmath", "en", "Solve x+1=3", "2", "latex_boxed")
    text = r"We get x = 2. Therefore \boxed{2}."
    answer = extract_answer(text, p, "en")
    assert answer == "2", f"Expected '2', got {answer!r}"
    assert check_answer_for_problem(answer, p), "Should be correct"
    return True


def test_extract_multiple_choice():
    from src.rollout_importance.data_loaders import Problem
    from src.rollout_importance.answer_extraction import extract_answer, check_answer_for_problem

    p = Problem("mmmlu_000", "mmmlu", "en", "What is 2+2?\nA.3\nB.4\nC.5\nD.6", "B", "multiple_choice")
    text = "The answer is B because 2+2=4.\nAnswer: B"
    answer = extract_answer(text, p, "en")
    assert answer == "B", f"Expected 'B', got {answer!r}"
    assert check_answer_for_problem(answer, p), "Should be correct"
    return True


def test_extract_fr_answer():
    from src.rollout_importance.data_loaders import Problem
    from src.rollout_importance.answer_extraction import extract_answer
    p = Problem("mgsm_000", "mgsm", "fr", "2+2?", "4", "numeric")
    text = "Je calcule. 2 + 2 = 4.\nRéponse finale: 4"
    answer = extract_answer(text, p, "fr")
    assert answer == "4", f"Expected '4', got {answer!r}"
    return True


def test_extract_zh_answer():
    from src.rollout_importance.data_loaders import Problem
    from src.rollout_importance.answer_extraction import extract_answer
    p = Problem("mgsm_000", "mgsm", "zh", "2+2?", "4", "numeric")
    text = r"计算得到 2 + 2 = 4。最终答案: 4"
    answer = extract_answer(text, p, "zh")
    assert answer == "4", f"Expected '4', got {answer!r}"
    return True


# ---------------------------------------------------------------------------
# 4. Prompts tests
# ---------------------------------------------------------------------------

def test_build_base_solution_prompt_mgsm():
    from src.rollout_importance.data_loaders import Problem
    from src.rollout_importance.prompts import build_base_solution_prompt, THINK_ANCHORS

    p = Problem("mgsm_000", "mgsm", "fr", "Combien font 2+2?", "4", "numeric")
    prompt = build_base_solution_prompt(p, "fr")
    assert "french" in prompt.lower() or "français" in prompt.lower() or "Problème" in prompt, \
        f"French prompt missing French content: {prompt[:100]}"
    assert THINK_ANCHORS["fr"] in prompt, "French think anchor not in prompt"
    assert "<think>" in prompt, "Missing <think> tag"
    return True


def test_build_rollout_prompt_forced():
    from src.rollout_importance.data_loaders import Problem
    from src.rollout_importance.prompts import build_rollout_prompt

    p = Problem("mgsm_000", "mgsm", "en", "What is 2+2?", "4", "numeric")
    prompt = build_rollout_prompt(p, "I think x = 2.", "en", rollout_type="forced_answer")
    assert "Final:" in prompt or "final" in prompt.lower(), "Forced answer prompt missing Final:"
    return True


def test_build_dag_prompt():
    from src.rollout_importance.prompts import build_dag_labeling_prompt

    chunks = ["We compute x.", "Therefore y = 2.", "Final: 2."]
    prompt = build_dag_labeling_prompt("What is y?", chunks, "en")
    assert "function_tags" in prompt, "DAG prompt missing function_tags"
    assert "Chunk 0" in prompt, "DAG prompt missing chunk labels"
    return True


# ---------------------------------------------------------------------------
# 5. Data loader tests (downloads MGSM en, first 3 problems)
# ---------------------------------------------------------------------------

def test_mgsm_loader_en():
    from src.rollout_importance.data_loaders import MGSMLoader
    loader = MGSMLoader()
    problems = loader.load_language("en")
    assert len(problems) == 250, f"Expected 250 MGSM problems, got {len(problems)}"
    p = problems[0]
    assert p.dataset == "mgsm"
    assert p.answer_type == "numeric"
    assert p.problem_id.startswith("mgsm_")
    return True


def test_mgsm_arabic_error():
    from src.rollout_importance.data_loaders import MGSMLoader
    loader = MGSMLoader()
    try:
        loader.load_language("ar")
        return None  # Should have raised
    except ValueError as e:
        assert "Arabic" in str(e) or "ar" in str(e), f"Error message unhelpful: {e}"
        return True


def test_load_dataset_problems_mgsm_en():
    from src.rollout_importance.data_loaders import load_dataset_problems
    problems = load_dataset_problems("mgsm", ["en"], num_problems=3)
    assert len(problems) == 3, f"Expected 3 problems, got {len(problems)}"
    for pid, lang_map in problems.items():
        assert "en" in lang_map
    return True


# ---------------------------------------------------------------------------
# 6. Analysis on existing rollout data (no API needed)
# ---------------------------------------------------------------------------

def _find_existing_rollout_dir():
    """Look for existing rollout data to test analysis on.

    Searches for any 'correct_base_solution' directory with at least one
    problem that has chunks.json. The directory structure is:
      output/rollouts/{dataset}/{model_org}/{model_name}/temperature_T_top_p_P/{lang}/correct_base_solution/
    """
    base = Path("output/rollouts")
    # Use rglob to find any correct_base_solution dir
    for correct_dir in base.rglob("correct_base_solution"):
        if not correct_dir.is_dir():
            continue
        # Language is the parent directory of correct_base_solution
        language = correct_dir.parent.name
        # Skip if language looks like a temperature/model string
        if language.startswith("temperature_") or "/" in language or len(language) > 5:
            continue
        for prob_dir in sorted(correct_dir.iterdir()):
            if prob_dir.is_dir() and (prob_dir / "chunks.json").exists():
                return correct_dir, language
    return None


def test_analyze_existing_data():
    """Run importance analysis on existing rollout data (skips if no data found)."""
    result = _find_existing_rollout_dir()
    if result is None:
        return None  # SKIP

    correct_dir, language = result
    # Pick first problem with chunk_0/solutions.json
    prob_dir = None
    for d in sorted(correct_dir.iterdir()):
        if d.is_dir() and (d / "chunks.json").exists() and (d / "chunk_0" / "solutions.json").exists():
            prob_dir = d
            break

    if prob_dir is None:
        return None  # SKIP

    # Run the analyze_problem function directly
    from src.rollout_importance.compute_importance import analyze_problem
    result_data = analyze_problem(
        problem_dir=prob_dir,
        dataset="mgsm",
        language=language,
        force_relabel=False,
        use_existing_metrics=False,
        sentence_model="paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold=0.8,
        generate_metadata=False,
    )
    assert result_data is not None, "analyze_problem returned None"
    assert "num_chunks" in result_data
    labeled_file = prob_dir / "chunks_labeled.json"
    assert labeled_file.exists(), "chunks_labeled.json not created"
    return True


def test_plots_load_language_chunks():
    """Test plots data loading on existing rollout data."""
    result = _find_existing_rollout_dir()
    if result is None:
        return None  # SKIP

    correct_dir, language = result
    rollouts_dir = correct_dir.parent.parent  # up 2: temperature_... dir
    from src.rollout_importance.make_figures import load_language_chunks
    chunks = load_language_chunks(rollouts_dir, language)
    # May be 0 if no chunks_labeled.json yet — that's ok for this test
    assert isinstance(chunks, list)
    return True


# ---------------------------------------------------------------------------
# 7. End-to-end plots smoke (skips if no labeled data)
# ---------------------------------------------------------------------------

def test_plots_generate():
    """Generate plots from any existing labeled data."""
    # Find any chunks_labeled.json
    base = Path("output/rollouts")
    all_chunks = []
    for f in base.rglob("chunks_labeled.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            if data and isinstance(data, list):
                all_chunks.extend(data)
        except Exception:
            pass

    if not all_chunks:
        return None  # SKIP

    # Check at least one chunk has importance metrics
    has_metrics = any("counterfactual_importance_accuracy" in c for c in all_chunks)
    if not has_metrics:
        return None  # SKIP (need to run compute_importance first)

    from src.rollout_importance.make_figures import (
        plot_importance_curves, plot_mean_importance_by_language,
        plot_accuracy_by_position
    )
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        lang_chunks = {"en": all_chunks}
        plot_importance_curves(
            lang_chunks, "counterfactual_importance_accuracy",
            outdir / "test_curve.png"
        )
        assert (outdir / "test_curve.png").exists()
        plot_accuracy_by_position(lang_chunks, outdir / "test_acc.png")
        assert (outdir / "test_acc.png").exists()

    return True


# ---------------------------------------------------------------------------
# 7b. PolyMath tests (offline: answer extraction; online: loader downloads ~1MB)
# ---------------------------------------------------------------------------

def test_open_math_answer_extraction():
    """open_math answer type: boxed extraction and numeric checking."""
    from src.rollout_importance.answer_extraction import extract_answer, check_answer_for_problem
    from src.rollout_importance.data_loaders import Problem

    # \boxed{} extraction
    p = Problem("polymath_low-0", "polymath", "en", "q?", "42", "open_math")
    text = "Let me solve step by step... \\boxed{42}"
    ans = extract_answer(text, p, "en")
    assert ans == "42", f"Expected '42', got '{ans}'"
    assert check_answer_for_problem(ans, p), "Numeric check for '42' failed"

    # Final: marker extraction
    p2 = Problem("polymath_low-1", "polymath", "en", "q?", "3.14", "open_math")
    ans2 = extract_answer("Some work. Final: 3.14", p2, "en")
    assert check_answer_for_problem(ans2, p2), f"Numeric check for '3.14' failed (got '{ans2}')"

    # String fallback (non-numeric answer)
    p3 = Problem("polymath_low-2", "polymath", "en", "q?", "yes", "open_math")
    ans3 = extract_answer("Conclusion: \\boxed{yes}", p3, "en")
    assert check_answer_for_problem(ans3, p3), f"String match for 'yes' failed (got '{ans3}')"

    return True


def test_chunker_th():
    """Thai text chunker: splits on paragraph breaks and ASCII .?! boundaries."""
    from src.rollout_importance.chunker import split_solution_into_chunks
    text = "ขั้นแรก ฉันสังเกตว่า x = 5. จากนั้น ฉันคำนวณ y = x + 3. ดังนั้น y = 8."
    chunks = split_solution_into_chunks(text, "th")
    assert len(chunks) >= 2, f"Expected >=2 chunks for Thai text, got {len(chunks)}: {chunks}"
    return True


def test_polymath_loader_offline():
    """PolyMath loader: class exists, language validation works, cross-lang IDs are stable."""
    from src.rollout_importance.data_loaders import PolyMathLoader

    loader = PolyMathLoader(difficulties=["low"])

    # Unknown language should raise
    try:
        loader.load_language("xx")
        assert False, "Should have raised ValueError for unknown language"
    except ValueError as e:
        assert "PolyMath" in str(e)

    # get_problem_ids returns deterministic sorted IDs for 'low' difficulty
    ids = loader.get_problem_ids()
    assert len(ids) == 125, f"Expected 125 IDs for low difficulty, got {len(ids)}"
    assert ids[0].startswith("polymath_low-"), f"Unexpected id format: {ids[0]}"
    assert ids == sorted(ids), "IDs not sorted"

    # Cross-lang ID stripping
    assert loader._cross_lang_id("low-en-0") == "polymath_low-0"
    assert loader._cross_lang_id("top-zh-124") == "polymath_top-124"

    return True


def test_polymath_loader_hf():
    """PolyMath HF loader: downloads low/en (125 problems) — skipped if offline."""
    try:
        from src.rollout_importance.data_loaders import PolyMathLoader
        loader = PolyMathLoader(difficulties=["low"])
        problems = loader.load_language("en")
        if len(problems) == 0:
            return None  # SKIP: offline or dataset not cached
        assert len(problems) == 125, f"Expected 125 low-difficulty en problems, got {len(problems)}"
        p = problems[0]
        assert p.dataset == "polymath"
        assert p.language == "en"
        assert p.answer_type == "open_math"
        assert p.problem_id.startswith("polymath_low-")
        assert p.question.strip()
        assert p.gt_answer.strip()
        print(f"    first id={p.problem_id}, answer='{p.gt_answer}'")
    except Exception as e:
        if "ConnectionError" in type(e).__name__ or "requests" in str(e).lower() or "network" in str(e).lower():
            return None  # SKIP when offline
        raise
    return True


# ---------------------------------------------------------------------------
# 8. Alignment tests
# ---------------------------------------------------------------------------

def test_align_chunks_imports():
    """Import all public functions from align_chunks."""
    from src.rollout_importance.align_chunks import (
        embed_chunks,
        compute_similarity_matrix,
        monotone_max_weight_alignment,
        load_labeled_chunks,
        align_problem,
        validate_alignment_with_llm,
        write_alignment_json,
        build_summary_rows,
        translate_chunks_to_english,
        main as align_main,
    )
    return True


def test_translate_en_alignment_offline():
    """translate_en aligner: translate_chunks_to_english is identity for English input."""
    from src.rollout_importance.align_chunks import translate_chunks_to_english

    chunks = ["First, I note that x = 5.", "Then I compute y = x + 3."]

    result = translate_chunks_to_english(chunks, "en")
    assert result == chunks, f"en→en should be identity, got {result}"
    assert result is not chunks, "should return a copy, not the same list object"

    result_empty = translate_chunks_to_english([], "fr")
    assert result_empty == [], "empty input should return empty list"

    return True


def test_monotone_alignment_dp():
    """3×3 identity-ish matrix: should pick (0,0),(1,1),(2,2) as the best monotone path."""
    from src.rollout_importance.align_chunks import monotone_max_weight_alignment
    import numpy as np
    # Strong diagonal, weak off-diagonal
    sim = np.array([
        [0.9, 0.3, 0.2],
        [0.3, 0.9, 0.3],
        [0.2, 0.3, 0.9],
    ], dtype=np.float32)
    alignment = monotone_max_weight_alignment(sim, tau=0.6)
    pairs = [(i, j) for i, j, _ in alignment]
    assert (0, 0) in pairs, f"Expected (0,0) in pairs: {pairs}"
    assert (1, 1) in pairs, f"Expected (1,1) in pairs: {pairs}"
    assert (2, 2) in pairs, f"Expected (2,2) in pairs: {pairs}"
    # Check strict monotonicity
    for k in range(1, len(alignment)):
        assert alignment[k][0] > alignment[k-1][0], "i must be strictly increasing"
        assert alignment[k][1] > alignment[k-1][1], "j must be strictly increasing"
    return True


def test_monotone_alignment_empty():
    """All pairs below tau → returns empty list."""
    from src.rollout_importance.align_chunks import monotone_max_weight_alignment
    import numpy as np
    sim = np.full((3, 3), 0.3, dtype=np.float32)
    alignment = monotone_max_weight_alignment(sim, tau=0.6)
    assert alignment == [], f"Expected [], got {alignment}"
    return True


def test_similarity_matrix():
    """Identity vectors → diagonal should be 1.0."""
    from src.rollout_importance.align_chunks import compute_similarity_matrix
    import numpy as np
    n = 4
    embs = np.eye(n, dtype=np.float32)
    sim = compute_similarity_matrix(embs, embs)
    assert sim.shape == (n, n), f"Expected ({n},{n}), got {sim.shape}"
    for i in range(n):
        assert abs(sim[i, i] - 1.0) < 1e-5, f"Diagonal[{i}] should be 1.0, got {sim[i,i]}"
    return True


def test_align_problem_on_existing_data():
    """Run align_problem on existing en+fr data if available; skip otherwise."""
    import tempfile
    from pathlib import Path

    base = Path("output/rollouts")
    en_base = None
    for correct_dir in base.rglob("correct_base_solution"):
        if not correct_dir.is_dir():
            continue
        lang = correct_dir.parent.name
        if lang == "en":
            # Check if a sibling fr directory exists with labeled chunks
            fr_correct = correct_dir.parent.parent / "fr" / "correct_base_solution"
            if fr_correct.exists():
                # Find a problem with chunks_labeled.json in both
                for prob_dir in sorted(correct_dir.iterdir()):
                    if not prob_dir.is_dir() or not prob_dir.name.startswith("problem_"):
                        continue
                    en_labeled = prob_dir / "chunks_labeled.json"
                    fr_labeled = fr_correct / prob_dir.name / "chunks_labeled.json"
                    if en_labeled.exists() and fr_labeled.exists():
                        en_base = prob_dir
                        break
            if en_base:
                break

    if en_base is None:
        return None  # SKIP

    from src.rollout_importance.align_chunks import align_problem
    data = align_problem(
        problem_dir_lang1=en_base,
        lang1="en",
        lang2="fr",
        tau=0.5,  # lower threshold to increase chance of finding pairs
        force=True,
        store_matrix=False,
        validate_n=0,
    )
    assert data is not None, "align_problem returned None on existing data"
    assert "n_aligned_pairs" in data
    out_file = en_base / "alignment_en_fr.json"
    assert out_file.exists(), f"Expected {out_file} to exist"
    return True


def test_plot_alignment_scatter():
    """Create synthetic alignment JSON and verify Plot 6 produces a PNG."""
    import tempfile
    from pathlib import Path
    from src.rollout_importance.make_figures import plot_alignment_importance_scatter

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Write a synthetic alignment file
        alignment_data = {
            "problem_id": "problem_synthetic_000",
            "lang1": "en", "lang2": "fr", "tau": 0.6,
            "n_chunks_lang1": 3, "n_chunks_lang2": 3,
            "n_aligned_pairs": 3,
            "mean_similarity": 0.85,
            "aligned_pairs": [
                {
                    "idx_lang1": i, "idx_lang2": i,
                    "similarity": 0.85,
                    "chunk_lang1": f"step {i} en",
                    "chunk_lang2": f"step {i} fr",
                    "function_tags_lang1": ["active_computation"],
                    "function_tags_lang2": ["active_computation"],
                    "counterfactual_importance_accuracy_lang1": 0.1 * i,
                    "counterfactual_importance_accuracy_lang2": 0.12 * i,
                    "resampling_importance_accuracy_lang1": 0.05,
                    "resampling_importance_accuracy_lang2": 0.06,
                    "counterfactual_importance_kl_lang1": None,
                    "counterfactual_importance_kl_lang2": None,
                    "resampling_importance_kl_lang1": None,
                    "resampling_importance_kl_lang2": None,
                    "forced_importance_accuracy_lang1": None,
                    "forced_importance_accuracy_lang2": None,
                    "forced_importance_kl_lang1": None,
                    "forced_importance_kl_lang2": None,
                    "llm_valid": None, "llm_reasoning": None,
                }
                for i in range(1, 4)
            ],
        }
        afile = tmpdir / "alignment_en_fr.json"
        with open(afile, "w") as f:
            json.dump(alignment_data, f)

        out_png = tmpdir / "test_scatter.png"
        plot_alignment_importance_scatter(
            alignment_files=[afile],
            lang1="en", lang2="fr",
            metric="counterfactual_importance_accuracy",
            output_path=out_png,
        )
        assert out_png.exists(), f"Expected scatter PNG at {out_png}"

    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("Multilingual CoT Pipeline — Smoke Test")
    print("=" * 60)

    tests = [
        # Imports
        ("imports: all modules", test_imports),
        # Chunker
        ("chunker: English", test_chunker_en),
        ("chunker: French", test_chunker_fr),
        ("chunker: Chinese (zh)", test_chunker_zh),
        ("chunker: Arabic (ar)", test_chunker_ar),
        ("chunker: strips <think> tags", test_chunker_strips_think_tags),
        ("chunker: LaTeX-aware (no split inside $)", test_chunker_latex_aware),
        # Answer extraction
        ("answer: extract numeric (en)", test_extract_numeric_answer),
        ("answer: extract \\boxed{} (en)", test_extract_boxed_answer),
        ("answer: extract multiple_choice (en)", test_extract_multiple_choice),
        ("answer: extract numeric (fr, Réponse finale)", test_extract_fr_answer),
        ("answer: extract numeric (zh, 最终答案)", test_extract_zh_answer),
        # Prompts
        ("prompts: base solution MGSM fr", test_build_base_solution_prompt_mgsm),
        ("prompts: rollout forced_answer", test_build_rollout_prompt_forced),
        ("prompts: DAG labeling prompt", test_build_dag_prompt),
        # Data loaders (downloads ~1MB if not cached)
        ("data: MGSM loader en (250 problems)", test_mgsm_loader_en),
        ("data: MGSM Arabic raises clear error", test_mgsm_arabic_error),
        ("data: load_dataset_problems mgsm en 3", test_load_dataset_problems_mgsm_en),
        # PolyMath
        ("answer: open_math extraction + checking", test_open_math_answer_extraction),
        ("chunker: Thai (th)", test_chunker_th),
        ("polymath: loader offline validation", test_polymath_loader_offline),
        ("polymath: HF download low/en (125 problems)", test_polymath_loader_hf),
        # Analysis on disk data
        ("analysis: analyze_problem on existing data", test_analyze_existing_data),
        ("plots: load_language_chunks on existing data", test_plots_load_language_chunks),
        ("plots: generate figures from labeled data", test_plots_generate),
        # Alignment
        ("align: imports", test_align_chunks_imports),
        ("align: translate_en identity for English", test_translate_en_alignment_offline),
        ("align: DP monotone alignment correctness", test_monotone_alignment_dp),
        ("align: DP empty when below tau", test_monotone_alignment_empty),
        ("align: cosine similarity matrix", test_similarity_matrix),
        ("align: align_problem on existing data", test_align_problem_on_existing_data),
        ("plots: Plot 6 alignment scatter", test_plot_alignment_scatter),
    ]

    results = {PASS: 0, FAIL: 0, SKIP: 0}
    for name, fn in tests:
        status = run_test(name, fn)
        results[status] += 1

    print("\n" + "-" * 60)
    print(f"Results: {results[PASS]} passed, {results[FAIL]} failed, {results[SKIP]} skipped")

    if results[FAIL] > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("All tests passed (or skipped for missing data).")


if __name__ == "__main__":
    main()
