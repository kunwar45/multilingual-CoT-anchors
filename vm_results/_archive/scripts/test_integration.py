#!/usr/bin/env python3
"""
Integration test: verify all modules load and work together.

This does NOT run model inference - just tests the pipeline logic.

Usage:
    python scripts/test_integration.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_mgsm_loader():
    """Test MGSM dataset loading (requires HuggingFace datasets)."""
    print("Testing MGSM loader...")

    try:
        import datasets as hf_datasets
    except ImportError:
        print("  SKIPPED (HuggingFace datasets not installed)")
        return "skipped"

    from data_loaders import MGSMLoader, MGSMProblem

    loader = MGSMLoader()

    # Get problem IDs
    ids = loader.get_problem_ids(n=5)
    assert len(ids) == 5
    assert ids[0] == "mgsm_000"
    print(f"  Problem IDs: {ids}")

    # Load English problems
    problems = loader.load_language("en")
    assert len(problems) == 250
    assert isinstance(problems[0], MGSMProblem)
    print(f"  Loaded {len(problems)} English problems")

    # Check stable IDs
    p = loader.load_problem("mgsm_042", "en")
    assert p.problem_id == "mgsm_042"
    print(f"  Problem 42: {p.question[:50]}...")

    print("  PASSED")


def test_prompts():
    """Test prompt building."""
    print("Testing prompts...")

    from prompts import (
        build_cot_prompt,
        CONDITION_NATIVE,
        CONDITION_ENGLISH_THINKING,
    )

    # Native condition
    prompt = build_cot_prompt(
        question="What is 2 + 2?",
        problem_lang="en",
        condition=CONDITION_NATIVE,
    )
    assert "step by step" in prompt.lower()
    assert "Final:" in prompt
    print(f"  Native prompt: {prompt[:100]}...")

    # English thinking with Spanish problem
    prompt = build_cot_prompt(
        question="¿Cuánto es 2 + 2?",
        problem_lang="es",
        condition=CONDITION_ENGLISH_THINKING,
    )
    assert "English" in prompt
    assert "¿Cuánto" in prompt
    print(f"  English thinking prompt: {prompt[:100]}...")

    print("  PASSED")


def test_parser():
    """Test answer parsing (no dependencies)."""
    print("Testing answer parser...")

    # Import directly from parser module to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "parser",
        Path(__file__).parent.parent / "src" / "rollouts" / "parser.py"
    )
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)

    parse_final_answer = parser_module.parse_final_answer
    normalize_number = parser_module.normalize_number

    # Standard format
    assert parse_final_answer("The answer is Final: 42") == "42"
    assert parse_final_answer("Blah blah\nFinal: 123.45\nMore text") == "123.45"

    # With commas
    assert parse_final_answer("Final: 1,234") == "1234"

    # Boxed fallback
    assert parse_final_answer("The answer is \\boxed{99}") == "99"

    # "The answer is" pattern
    assert parse_final_answer("Therefore, the answer is 7.") == "7"

    # Normalize
    assert normalize_number("1,234.56") == 1234.56
    assert normalize_number("42") == 42.0

    print("  PASSED")


def test_segmentation():
    """Test sentence segmentation."""
    print("Testing segmentation...")

    from analysis import segment_cot

    text = "First, let's understand the problem. We need to add 2 and 2. The answer is 4."
    sentences = segment_cot(text, language="en")

    assert len(sentences) >= 2
    print(f"  Segmented into {len(sentences)} sentences")

    for s in sentences:
        print(f"    [{s.idx}] {s.text[:50]}...")

    print("  PASSED")


def test_sensitivity():
    """Test sensitivity score computation."""
    print("Testing sensitivity scores...")

    from analysis import segment_cot, compute_sensitivity_scores, select_candidate_anchors

    text = """First, let's plan our approach to this problem.
We need to calculate the total cost.
The price is $10 per item.
We have 5 items.
Therefore, the total is 10 * 5 = 50.
Let me verify: 10 + 10 + 10 + 10 + 10 = 50. Correct!
Final: 50"""

    sentences = segment_cot(text, language="en", min_sentence_length=5)
    scores = compute_sensitivity_scores(sentences, text)

    assert len(scores) == len(sentences)
    print(f"  Computed {len(scores)} sensitivity scores")

    for s, score in zip(sentences, scores):
        print(f"    [{s.idx}] {score:.3f}: {s.text[:40]}...")

    # Select candidates
    candidates = select_candidate_anchors(sentences, scores, top_k=3)
    print(f"  Selected {len(candidates)} candidates: {candidates}")

    print("  PASSED")


def test_metrics():
    """Test metric computation."""
    print("Testing metrics...")

    from analysis import compute_accuracy, compute_anchor_concentration
    from analysis.importance import AnchorResult

    # Accuracy
    rollouts = [
        {"problem_id": "p1", "correct": True},
        {"problem_id": "p1", "correct": False},
        {"problem_id": "p2", "correct": True},
        {"problem_id": "p2", "correct": True},
    ]
    acc = compute_accuracy(rollouts)
    assert acc.accuracy == 0.75
    print(f"  Accuracy: {acc.accuracy}")

    # Concentration
    results = [
        AnchorResult(0, "s0", 0.9, None, True),
        AnchorResult(1, "s1", 0.1, None, False),
        AnchorResult(2, "s2", 0.1, None, False),
        AnchorResult(3, "s3", 0.1, None, False),
    ]
    conc = compute_anchor_concentration(results, top_k=1)
    assert conc["concentration_ratio"] > 0.5
    print(f"  Concentration: {conc}")

    print("  PASSED")


def test_rollout_config():
    """Test rollout configuration (requires torch)."""
    print("Testing rollout config...")

    try:
        import torch
    except ImportError:
        print("  SKIPPED (torch not installed)")
        return "skipped"

    from rollouts import RolloutConfig, Rollout

    config = RolloutConfig(
        model_name="test-model",
        temperature=0.7,
        num_rollouts=5,
    )

    assert config.model_name == "test-model"
    assert config.num_rollouts == 5

    # Test serialization
    d = config.to_dict()
    config2 = RolloutConfig.from_dict(d)
    assert config2.model_name == config.model_name
    print(f"  Config roundtrip OK")

    # Test Rollout
    rollout = Rollout(
        problem_id="mgsm_001",
        language="en",
        condition="native",
        rollout_idx=0,
        prompt="test",
        cot_text="Let me think...",
        final_answer_text="42",
        correct=True,
        ground_truth="42",
        seed=123,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=1024,
        model_name="test",
        timestamp="2025-01-01",
        generation_time_ms=100,
    )
    assert rollout.rollout_id  # Should be auto-generated
    print(f"  Rollout ID: {rollout.rollout_id}")

    print("  PASSED")


def test_config_loading():
    """Test config loading and validation."""
    print("Testing config loading...")

    from config import load_config, ExperimentConfig, load_defaults

    # Test load_defaults
    config = load_defaults()
    assert config.model.name is not None
    print(f"  Default model: {config.model.name}")

    # Test from_dict
    d = {
        "model": {"name": "test-model"},
        "experiment": {"languages": ["en", "es"], "conditions": ["native"]},
    }
    config = ExperimentConfig.from_dict(d)
    assert config.model.name == "test-model"
    assert config.experiment.languages == ["en", "es"]
    print(f"  from_dict: OK")

    # Test validation
    errors = config.validate()
    assert len(errors) == 0, f"Validation errors: {errors}"
    print(f"  Validation: OK")

    # Test to_dict roundtrip
    d2 = config.to_dict()
    config2 = ExperimentConfig.from_dict(d2)
    assert config2.model.name == config.model.name
    print(f"  Roundtrip: OK")

    print("  PASSED")


def test_multilingual_parser():
    """Test multilingual answer parsing."""
    print("Testing multilingual answer parsing...")

    # Import directly from file to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "parser",
        Path(__file__).parent.parent / "src" / "rollouts" / "parser.py"
    )
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)

    parse_final_answer = parser_module.parse_final_answer
    FINAL_MARKERS = parser_module.FINAL_MARKERS

    # English
    assert parse_final_answer("Final: 42", "en") == "42"

    # Spanish
    assert parse_final_answer("Respuesta final: 42", "es") == "42"
    assert parse_final_answer("La respuesta es: 123", "es") == "123"

    # Chinese (with full-width colon)
    assert parse_final_answer("最终答案：42", "zh") == "42"
    assert parse_final_answer("答案: 123", "zh") == "123"

    # Russian
    assert parse_final_answer("Ответ: 42", "ru") == "42"

    # French
    assert parse_final_answer("Réponse finale: 42", "fr") == "42"

    # Verify all MGSM languages have markers
    expected_languages = {"en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"}
    assert set(FINAL_MARKERS.keys()) == expected_languages
    print(f"  All {len(expected_languages)} languages have markers")

    print("  PASSED")


def test_multilingual_sensitivity():
    """Test multilingual sensitivity heuristics."""
    print("Testing multilingual sensitivity heuristics...")

    from analysis import segment_cot, compute_sensitivity_scores
    from analysis.importance import PLANNING_PHRASES, REASONING_PHRASES

    # English text
    text_en = "First, let's plan. Therefore, the answer is 5."
    sentences_en = segment_cot(text_en, language="en", min_sentence_length=5)
    scores_en = compute_sensitivity_scores(sentences_en, text_en, language="en")
    assert len(scores_en) == len(sentences_en)
    print(f"  English: {len(scores_en)} scores")

    # Spanish text
    text_es = "Primero, vamos a planificar. Por lo tanto, la respuesta es 5."
    sentences_es = segment_cot(text_es, language="es", min_sentence_length=5)
    scores_es = compute_sensitivity_scores(sentences_es, text_es, language="es")
    assert len(scores_es) == len(sentences_es)
    print(f"  Spanish: {len(scores_es)} scores")

    # Verify phrase dictionaries
    expected_languages = {"en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"}
    assert set(PLANNING_PHRASES.keys()) == expected_languages
    assert set(REASONING_PHRASES.keys()) == expected_languages
    print(f"  All languages have phrase dictionaries")

    print("  PASSED")


def test_cross_condition():
    """Test cross-condition analysis."""
    print("Testing cross-condition analysis...")

    from analysis.importance import AnchorResult, FullImportanceResult
    from analysis.cross_condition import (
        compare_conditions,
        position_normalize_importance,
        bootstrap_confidence_interval,
        paired_ttest_importance,
    )

    # Create mock results
    results_1 = [
        FullImportanceResult(
            rollout_id="r1",
            problem_id="p1",
            language="en",
            condition="native",
            ground_truth="42",
            baseline_accuracy=0.8,
            anchor_results=[
                AnchorResult(0, "s0", 0.9, 0.3, True, 0.5, 0.8),
                AnchorResult(1, "s1", 0.5, 0.1, False),
            ],
        ),
    ]
    results_2 = [
        FullImportanceResult(
            rollout_id="r2",
            problem_id="p1",
            language="en",
            condition="english_thinking",
            ground_truth="42",
            baseline_accuracy=0.7,
            anchor_results=[
                AnchorResult(0, "s0", 0.8, 0.2, True, 0.4, 0.7),
                AnchorResult(1, "s1", 0.6, 0.1, False),
            ],
        ),
    ]

    # Test compare_conditions
    comparison = compare_conditions(results_1, results_2, "c1", "c2")
    # Use approximate comparison for floating point
    assert abs(comparison.accuracy_diff - 0.1) < 0.01, f"Expected ~0.1, got {comparison.accuracy_diff}"
    assert comparison.n_matched_problems == 1
    print(f"  compare_conditions: OK")

    # Test position_normalize_importance
    curve = position_normalize_importance(results_1, num_bins=5)
    assert len(curve) == 5
    print(f"  position_normalize: OK")

    # Test bootstrap_confidence_interval
    ci = bootstrap_confidence_interval([0.1, 0.2, 0.3, 0.4, 0.5], n_bootstrap=100, seed=42)
    assert ci["ci_lower"] < ci["estimate"] < ci["ci_upper"]
    print(f"  bootstrap_ci: OK")

    # Test paired_ttest
    ttest = paired_ttest_importance([0.8, 0.7, 0.9], [0.7, 0.6, 0.8])
    assert "pvalue" in ttest
    print(f"  paired_ttest: OK")

    print("  PASSED")


def test_manifest():
    """Test manifest-based job generation."""
    print("Testing manifest...")

    # Import directly from file to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "manifest",
        Path(__file__).parent.parent / "src" / "rollouts" / "manifest.py"
    )
    manifest_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest_module)

    JobTask = manifest_module.JobTask
    JobManifest = manifest_module.JobManifest
    TaskStatus = manifest_module.TaskStatus

    # Test JobTask
    task = JobTask.create(
        problem_id="mgsm_001",
        language="en",
        condition="native",
        rollout_idx=0,
        base_seed=42,
    )
    assert task.task_id is not None
    assert task.status == TaskStatus.PENDING
    print(f"  JobTask: {task.task_id}")

    # Test roundtrip
    d = task.to_dict()
    task2 = JobTask.from_dict(d)
    assert task2.task_id == task.task_id
    print(f"  JobTask roundtrip: OK")

    # Test JobManifest
    manifest = JobManifest(
        run_name="test_run",
        config_name="test",
        created_at="2025-01-01",
        tasks=[task],
    )
    assert manifest.total_tasks == 1
    assert manifest.pending_tasks == 1
    print(f"  JobManifest: OK")

    print("  PASSED")


def main():
    print("=" * 60)
    print("MultiCoT Integration Tests")
    print("=" * 60)

    tests = [
        test_mgsm_loader,
        test_prompts,
        test_parser,
        test_segmentation,
        test_sensitivity,
        test_metrics,
        test_rollout_config,
        test_config_loading,
        test_multilingual_parser,
        test_multilingual_sensitivity,
        test_cross_condition,
        test_manifest,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            result = test()
            if result == "skipped":
                skipped += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {skipped} skipped, {failed} failed")
    if skipped > 0:
        print("  (Install dependencies with: pip install -r ../requirements.txt)")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
