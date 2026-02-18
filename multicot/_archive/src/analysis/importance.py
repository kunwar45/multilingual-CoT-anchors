"""
Two-Stage Importance Computation

Stage 1: Cheap sensitivity pass to identify candidate anchor sentences
Stage 2: Full resampling on candidates to compute true importance

This approach is feasible on limited compute by spending resampling budget
only on promising candidates.

Supports multilingual sensitivity heuristics for all 11 MGSM languages.
"""

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Callable
import logging

from .segmentation import Sentence

logger = logging.getLogger(__name__)


# Multilingual phrase dictionaries for sensitivity heuristics
PLANNING_PHRASES = {
    "en": ["first", "let's", "we need", "approach", "strategy", "plan", "start by", "begin with"],
    "es": ["primero", "vamos a", "necesitamos", "enfoque", "estrategia", "plan", "empezar por", "comenzar con"],
    "fr": ["d'abord", "premièrement", "nous devons", "approche", "stratégie", "plan", "commençons par"],
    "de": ["zuerst", "lass uns", "wir müssen", "ansatz", "strategie", "plan", "beginnen wir"],
    "ru": ["сначала", "давайте", "нам нужно", "подход", "стратегия", "план", "начнём с"],
    "zh": ["首先", "让我们", "我们需要", "方法", "策略", "计划", "开始", "第一步"],
    "ja": ["まず", "最初に", "必要", "方法", "戦略", "計画", "始めましょう"],
    "th": ["ก่อนอื่น", "เริ่มต้น", "จำเป็นต้อง", "วิธีการ", "กลยุทธ์", "แผน"],
    "sw": ["kwanza", "tuanze", "tunahitaji", "mbinu", "mkakati", "mpango"],
    "bn": ["প্রথমে", "শুরু করি", "আমাদের দরকার", "পদ্ধতি", "কৌশল", "পরিকল্পনা"],
    "te": ["మొదట", "మొదలు", "మనకు అవసరం", "విధానం", "వ్యూహం", "ప్రణాళిక"],
}

REASONING_PHRASES = {
    "en": ["therefore", "thus", "so", "because", "since", "hence", "this means", "we get"],
    "es": ["por lo tanto", "así que", "entonces", "porque", "ya que", "esto significa", "obtenemos"],
    "fr": ["donc", "ainsi", "alors", "parce que", "puisque", "cela signifie", "on obtient"],
    "de": ["deshalb", "also", "daher", "weil", "da", "das bedeutet", "wir bekommen"],
    "ru": ["поэтому", "следовательно", "так что", "потому что", "значит", "получаем"],
    "zh": ["所以", "因此", "因为", "这意味着", "得到", "由此", "可得"],
    "ja": ["したがって", "だから", "なので", "ゆえに", "これは", "得られる"],
    "th": ["ดังนั้น", "เพราะฉะนั้น", "เพราะ", "หมายความว่า", "ได้"],
    "sw": ["kwa hiyo", "hivyo", "kwa sababu", "hii inamaanisha", "tunapata"],
    "bn": ["তাই", "অতএব", "কারণ", "এর মানে", "আমরা পাই"],
    "te": ["కాబట్టి", "అందువల్ల", "ఎందుకంటే", "అంటే", "మనకు వస్తుంది"],
}

CHECKING_PHRASES = {
    "en": ["check", "verify", "confirm", "wait", "actually", "but", "let me see", "hold on", "hmm"],
    "es": ["verificar", "comprobar", "confirmar", "espera", "en realidad", "pero", "déjame ver", "un momento"],
    "fr": ["vérifier", "confirmer", "attends", "en fait", "mais", "laisse-moi voir", "un instant"],
    "de": ["überprüfen", "verifizieren", "bestätigen", "warte", "eigentlich", "aber", "mal sehen", "moment"],
    "ru": ["проверить", "убедиться", "подтвердить", "подожди", "на самом деле", "но", "дай посмотрю", "хм"],
    "zh": ["检查", "验证", "确认", "等等", "实际上", "但是", "让我看看", "嗯"],
    "ja": ["確認", "検証", "待って", "実際", "しかし", "ちょっと待って", "うーん"],
    "th": ["ตรวจสอบ", "ยืนยัน", "รอ", "จริงๆ", "แต่", "ให้ฉันดู"],
    "sw": ["angalia", "thibitisha", "subiri", "kweli", "lakini", "ngoja"],
    "bn": ["যাচাই", "নিশ্চিত", "অপেক্ষা", "আসলে", "কিন্তু", "দেখি"],
    "te": ["తనిఖీ", "ధృవీకరించు", "ఆగు", "నిజానికి", "కానీ", "చూద్దాం"],
}

CONCLUSION_PHRASES = {
    "en": ["final answer", "in conclusion", "to summarize", "the answer is", "result is"],
    "es": ["respuesta final", "en conclusión", "para resumir", "la respuesta es", "el resultado es"],
    "fr": ["réponse finale", "en conclusion", "pour résumer", "la réponse est", "le résultat est"],
    "de": ["endgültige antwort", "zusammenfassend", "die antwort ist", "das ergebnis ist"],
    "ru": ["окончательный ответ", "в заключение", "ответ", "результат"],
    "zh": ["最终答案", "总结", "答案是", "结果是", "因此答案"],
    "ja": ["最終的な答え", "結論", "答えは", "結果は"],
    "th": ["คำตอบสุดท้าย", "สรุป", "คำตอบคือ", "ผลลัพธ์คือ"],
    "sw": ["jibu la mwisho", "kwa muhtasari", "jibu ni", "matokeo ni"],
    "bn": ["চূড়ান্ত উত্তর", "সংক্ষেপে", "উত্তর হল", "ফলাফল হল"],
    "te": ["చివరి సమాధానం", "సారాంశంలో", "సమాధానం", "ఫలితం"],
}


@dataclass
class AnchorResult:
    """Result of anchor importance computation for a single sentence."""

    sentence_idx: int
    sentence_text: str
    sensitivity_score: float  # Stage 1: cheap proxy
    importance_score: Optional[float]  # Stage 2: resampling-based (may be None)
    is_candidate: bool  # Whether selected for stage 2
    masked_accuracy: Optional[float] = None  # Accuracy when this sentence is masked
    original_accuracy: Optional[float] = None  # Baseline accuracy

    def to_dict(self) -> dict:
        return asdict(self)


def compute_sensitivity_scores(
    sentences: list[Sentence],
    full_text: str,
    model_fn: Optional[Callable[[str], float]] = None,
    language: str = "en",
) -> list[float]:
    """
    Stage 1: Compute cheap sensitivity scores for all sentences.

    Uses heuristics to estimate which sentences might be important:
    - Position in trace (early planning, late summary)
    - Contains key phrases (therefore, thus, so, because)
    - Numeric density (calculations)
    - Length (longer = more content)

    Supports multilingual phrase matching for all 11 MGSM languages.

    If model_fn is provided, uses single-pass perplexity change.

    Args:
        sentences: List of Sentence objects
        full_text: Original full text
        model_fn: Optional function that returns perplexity for text
        language: Language code for phrase matching (en, es, fr, de, ru, zh, ja, th, sw, bn, te)

    Returns:
        List of sensitivity scores (higher = more likely anchor)
    """
    if not sentences:
        return []

    # Get language-specific phrases (fallback to English if not available)
    planning_phrases = PLANNING_PHRASES.get(language, PLANNING_PHRASES["en"])
    reasoning_phrases = REASONING_PHRASES.get(language, REASONING_PHRASES["en"])
    checking_phrases = CHECKING_PHRASES.get(language, CHECKING_PHRASES["en"])
    conclusion_phrases = CONCLUSION_PHRASES.get(language, CONCLUSION_PHRASES["en"])

    scores = []
    num_sentences = len(sentences)

    for sent in sentences:
        score = 0.0

        # Position features (U-shaped: early and late are often important)
        relative_pos = sent.idx / max(1, num_sentences - 1)
        # Early sentences (planning)
        if relative_pos < 0.2:
            score += 0.3
        # Late sentences (consolidation)
        if relative_pos > 0.8:
            score += 0.2

        # Key phrase features (language-aware)
        text_lower = sent.text.lower()

        for phrase in planning_phrases:
            if phrase.lower() in text_lower:
                score += 0.2
                break

        for phrase in reasoning_phrases:
            if phrase.lower() in text_lower:
                score += 0.15
                break

        for phrase in checking_phrases:
            if phrase.lower() in text_lower:
                score += 0.25  # Backtracking is often important
                break

        for phrase in conclusion_phrases:
            if phrase.lower() in text_lower:
                score += 0.2  # Conclusions are important
                break

        # Numeric density (calculations matter)
        numbers = re.findall(r"\d+(?:\.\d+)?", sent.text)
        if len(numbers) >= 2:
            score += 0.1 * min(len(numbers), 5)

        # Length feature (normalized)
        avg_len = sum(len(s.text) for s in sentences) / len(sentences)
        if len(sent.text) > avg_len * 1.5:
            score += 0.1
        elif len(sent.text) < avg_len * 0.5:
            score -= 0.1  # Short sentences less likely to be anchors

        scores.append(max(0, score))

    # Normalize to [0, 1]
    max_score = max(scores) if scores else 1
    if max_score > 0:
        scores = [s / max_score for s in scores]

    return scores


def select_candidate_anchors(
    sentences: list[Sentence],
    sensitivity_scores: list[float],
    top_k: int = 5,
    threshold: float = 0.3,
) -> list[int]:
    """
    Select candidate anchor sentences for stage 2.

    Args:
        sentences: List of Sentence objects
        sensitivity_scores: Scores from stage 1
        top_k: Maximum number of candidates
        threshold: Minimum sensitivity score to be considered

    Returns:
        List of sentence indices selected as candidates
    """
    # Filter by threshold
    candidates = [
        (idx, score)
        for idx, score in enumerate(sensitivity_scores)
        if score >= threshold
    ]

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Take top k
    return [idx for idx, _ in candidates[:top_k]]


def compute_anchor_importance(
    sentences: list[Sentence],
    candidate_indices: list[int],
    full_text: str,
    generate_fn: Callable[[str], list[tuple[str, bool]]],
    num_samples: int = 10,
) -> list[AnchorResult]:
    """
    Stage 2: Compute importance via resampling for candidate sentences.

    For each candidate, mask the sentence and regenerate multiple times.
    Importance = accuracy_drop when sentence is masked.

    Args:
        sentences: List of Sentence objects
        candidate_indices: Indices of candidate sentences
        full_text: Original full text
        generate_fn: Function that generates samples and returns (answer, correct) tuples
        num_samples: Number of resamples per masking

    Returns:
        List of AnchorResult for all sentences
    """
    # First, compute baseline accuracy
    baseline_samples = generate_fn(full_text)
    baseline_accuracy = sum(1 for _, correct in baseline_samples if correct) / len(baseline_samples)

    results = []
    sensitivity_scores = compute_sensitivity_scores(sentences, full_text)

    for sent in sentences:
        is_candidate = sent.idx in candidate_indices

        if is_candidate:
            # Mask this sentence and regenerate
            masked_text = _mask_sentence(full_text, sent)
            masked_samples = generate_fn(masked_text)
            masked_accuracy = sum(1 for _, correct in masked_samples if correct) / len(masked_samples)

            # Importance = accuracy drop
            importance = baseline_accuracy - masked_accuracy
        else:
            masked_accuracy = None
            importance = None

        results.append(
            AnchorResult(
                sentence_idx=sent.idx,
                sentence_text=sent.text,
                sensitivity_score=sensitivity_scores[sent.idx],
                importance_score=importance,
                is_candidate=is_candidate,
                masked_accuracy=masked_accuracy,
                original_accuracy=baseline_accuracy,
            )
        )

    return results


def _mask_sentence(text: str, sentence: Sentence) -> str:
    """
    Remove a sentence from text.

    Args:
        text: Original text
        sentence: Sentence to remove

    Returns:
        Text with sentence removed
    """
    before = text[: sentence.start_char]
    after = text[sentence.end_char :]

    # Clean up whitespace
    result = before.rstrip() + " " + after.lstrip()
    return result.strip()


def save_importance_results(
    results: list[AnchorResult],
    output_path: Path,
) -> None:
    """Save importance results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)


def load_importance_results(path: Path) -> list[AnchorResult]:
    """Load importance results from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [AnchorResult(**d) for d in data]


# =============================================================================
# Full Stage 2 Resampling Implementation
# =============================================================================


@dataclass
class ResamplingConfig:
    """Configuration for full resampling importance computation."""
    num_samples: int = 10  # K samples per masked sentence
    max_candidates: int = 5  # Maximum candidates per rollout
    min_sensitivity: float = 0.3  # Minimum sensitivity to be a candidate
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1024


@dataclass
class FullImportanceResult:
    """
    Full importance computation result for a single rollout.

    Contains both Stage 1 (sensitivity) and Stage 2 (resampling) results.
    """
    rollout_id: str
    problem_id: str
    language: str
    condition: str
    ground_truth: str
    baseline_accuracy: float  # Accuracy without any masking
    anchor_results: list[AnchorResult]

    # Summary statistics
    num_sentences: int = 0
    num_candidates: int = 0
    max_importance: Optional[float] = None
    mean_importance: Optional[float] = None
    top_anchor_idx: Optional[int] = None

    def __post_init__(self):
        self._compute_summary()

    def _compute_summary(self):
        """Compute summary statistics."""
        self.num_sentences = len(self.anchor_results)
        candidates = [r for r in self.anchor_results if r.is_candidate and r.importance_score is not None]
        self.num_candidates = len(candidates)

        if candidates:
            importances = [r.importance_score for r in candidates]
            self.max_importance = max(importances)
            self.mean_importance = sum(importances) / len(importances)
            self.top_anchor_idx = max(candidates, key=lambda r: r.importance_score).sentence_idx

    def to_dict(self) -> dict:
        return {
            "rollout_id": self.rollout_id,
            "problem_id": self.problem_id,
            "language": self.language,
            "condition": self.condition,
            "ground_truth": self.ground_truth,
            "baseline_accuracy": self.baseline_accuracy,
            "anchor_results": [r.to_dict() for r in self.anchor_results],
            "num_sentences": self.num_sentences,
            "num_candidates": self.num_candidates,
            "max_importance": self.max_importance,
            "mean_importance": self.mean_importance,
            "top_anchor_idx": self.top_anchor_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FullImportanceResult":
        anchor_results = [AnchorResult(**r) for r in d.pop("anchor_results", [])]
        return cls(anchor_results=anchor_results, **d)


def compute_anchor_importance_full(
    rollout_id: str,
    problem_id: str,
    language: str,
    condition: str,
    sentences: list[Sentence],
    candidate_indices: list[int],
    cot_prefix: str,
    ground_truth: str,
    generate_fn: Callable[[str, int], tuple[str, Optional[str]]],
    check_answer_fn: Callable[[Optional[str], str], bool],
    config: ResamplingConfig,
    sensitivity_scores: Optional[list[float]] = None,
) -> FullImportanceResult:
    """
    Full Stage 2: Compute importance via resampling for candidate sentences.

    For each candidate sentence:
    1. Mask sentence from CoT prefix
    2. Regenerate K completions
    3. Compute accuracy drop vs baseline

    Args:
        rollout_id: ID of the rollout
        problem_id: Problem identifier
        language: Language code
        condition: Condition name
        sentences: List of Sentence objects
        candidate_indices: Indices of candidate sentences (from Stage 1)
        cot_prefix: The chain-of-thought text prefix (before final answer)
        ground_truth: Ground truth answer
        generate_fn: Function (prefix, seed) -> (generated_text, parsed_answer)
        check_answer_fn: Function (predicted, ground_truth) -> is_correct
        config: ResamplingConfig
        sensitivity_scores: Pre-computed sensitivity scores (optional)

    Returns:
        FullImportanceResult with all anchor results
    """
    # Compute sensitivity scores if not provided
    if sensitivity_scores is None:
        sensitivity_scores = compute_sensitivity_scores(sentences, cot_prefix, language=language)

    # Compute baseline accuracy (generate K samples from full prefix)
    baseline_correct = 0
    for k in range(config.num_samples):
        seed = hash(f"{rollout_id}:baseline:{k}") % (2**31)
        _, answer = generate_fn(cot_prefix, seed)
        if check_answer_fn(answer, ground_truth):
            baseline_correct += 1

    baseline_accuracy = baseline_correct / config.num_samples

    # Compute importance for each sentence
    results = []

    for sent in sentences:
        is_candidate = sent.idx in candidate_indices

        if is_candidate:
            # Mask this sentence and regenerate K samples
            masked_text = _mask_sentence(cot_prefix, sent)

            masked_correct = 0
            for k in range(config.num_samples):
                seed = hash(f"{rollout_id}:masked:{sent.idx}:{k}") % (2**31)
                _, answer = generate_fn(masked_text, seed)
                if check_answer_fn(answer, ground_truth):
                    masked_correct += 1

            masked_accuracy = masked_correct / config.num_samples

            # Importance = accuracy drop when masked
            importance = baseline_accuracy - masked_accuracy
        else:
            masked_accuracy = None
            importance = None

        results.append(
            AnchorResult(
                sentence_idx=sent.idx,
                sentence_text=sent.text,
                sensitivity_score=sensitivity_scores[sent.idx],
                importance_score=importance,
                is_candidate=is_candidate,
                masked_accuracy=masked_accuracy,
                original_accuracy=baseline_accuracy,
            )
        )

    return FullImportanceResult(
        rollout_id=rollout_id,
        problem_id=problem_id,
        language=language,
        condition=condition,
        ground_truth=ground_truth,
        baseline_accuracy=baseline_accuracy,
        anchor_results=results,
    )


def compute_importance_batch(
    rollouts: list[dict],
    sentences_by_rollout: dict[str, list[Sentence]],
    candidates_by_rollout: dict[str, list[int]],
    generate_fn: Callable[[str, int], tuple[str, Optional[str]]],
    check_answer_fn: Callable[[Optional[str], str], bool],
    config: ResamplingConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[FullImportanceResult]:
    """
    Compute importance for a batch of rollouts.

    Args:
        rollouts: List of rollout dicts with id, problem_id, language, condition, cot_text, ground_truth
        sentences_by_rollout: Dict mapping rollout_id to Sentence list
        candidates_by_rollout: Dict mapping rollout_id to candidate indices
        generate_fn: Generation function
        check_answer_fn: Answer checking function
        config: ResamplingConfig
        progress_callback: Optional callback(current, total)

    Returns:
        List of FullImportanceResult
    """
    results = []
    total = len(rollouts)

    for i, rollout in enumerate(rollouts):
        rollout_id = rollout["rollout_id"]
        sentences = sentences_by_rollout.get(rollout_id, [])
        candidates = candidates_by_rollout.get(rollout_id, [])

        if not sentences:
            logger.warning(f"No sentences for rollout {rollout_id}")
            continue

        result = compute_anchor_importance_full(
            rollout_id=rollout_id,
            problem_id=rollout["problem_id"],
            language=rollout["language"],
            condition=rollout["condition"],
            sentences=sentences,
            candidate_indices=candidates,
            cot_prefix=rollout.get("cot_text", ""),
            ground_truth=rollout["ground_truth"],
            generate_fn=generate_fn,
            check_answer_fn=check_answer_fn,
            config=config,
        )
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total)

    return results


def save_full_importance_results(
    results: list[FullImportanceResult],
    output_path: Path,
) -> None:
    """Save full importance results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)


def load_full_importance_results(path: Path) -> list[FullImportanceResult]:
    """Load full importance results from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [FullImportanceResult.from_dict(d) for d in data]
