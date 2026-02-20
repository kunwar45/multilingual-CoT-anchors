"""
Core Utilities for Multilingual CoT Experiments

Includes:
- GlotLID language verification
- Unified answer extraction and checking (numeric + LaTeX)
- LaTeX normalization utilities (copied from root utils.py)

Chunking is handled by multicot.chunker.
"""

import re
from typing import List, Optional, Tuple

from multicot.data_loaders import Problem
from multicot.chunker import split_solution_into_chunks  # re-exported for callers


# ---------------------------------------------------------------------------
# LaTeX utilities (copied from root utils.py)
# ---------------------------------------------------------------------------

def extract_boxed_answers(text: str) -> List[str]:
    """Extract answers enclosed in \\boxed{} with nested brace handling."""
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not boxed_starts:
        return [""]

    answers = []
    for start_idx in boxed_starts:
        idx = start_idx + 7
        brace_count = 1
        answer = ""
        while idx < len(text) and brace_count > 0:
            char = text[idx]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            if brace_count > 0:
                answer += char
            idx += 1
        if answer:
            answers.append(answer)

    return answers if answers else [""]


def normalize_latex(latex_str: str) -> str:
    """Normalize LaTeX string for comparison."""
    normalized = latex_str.strip().lower()
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("\\%", "")
    normalized = normalized.replace("{,}", "")
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")
    normalized = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", normalized)
    normalized = re.sub(r"{([^{}]+)}", r"\1", normalized)
    normalized = normalized.replace("\\pi", "pi")
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")
    return normalized


def prepare_latex_for_sympy(latex_str):
    """Prepare a LaTeX string for SymPy parsing."""
    if not isinstance(latex_str, str):
        return str(latex_str)
    latex_str = re.sub(r"\\boxed\{(.*?)\}", r"\1", latex_str)
    replacements = {
        r"\\dfrac": r"\\frac",
        r"\\tfrac": r"\\frac",
        r"\\cdot": r"*",
        r"\\times": r"*",
        r"\\div": r"/",
        r"\\left": r"",
        r"\\right": r"",
        r"\\textbf": r"",
        r"\\text": r"",
        r"\\mathrm": r"",
        r"\\!": r"",
        r",": r"",
    }
    for old, new in replacements.items():
        latex_str = re.sub(old, new, latex_str)
    return latex_str


def get_latex_equivalent(answer0, answer1):
    """Check if two LaTeX expressions are mathematically equivalent using SymPy."""
    try:
        from sympy.parsing.latex import parse_latex
        answer0 = prepare_latex_for_sympy(answer0)
        answer1 = prepare_latex_for_sympy(answer1)
        expr1 = parse_latex(answer0)
        expr2 = parse_latex(answer1)
        return expr1.equals(expr2)
    except Exception:
        return False


def normalize_answer(answer: str, use_sympy: bool = False) -> str:
    """Get the final normalized version of an answer."""
    normalized = normalize_latex(answer)
    if use_sympy:
        try:
            sympy_ready = prepare_latex_for_sympy(answer)
            if sympy_ready != normalized and len(sympy_ready) > 0:
                return sympy_ready
        except Exception:
            pass
    return normalized


def check_answer_latex(answer: str, gt_answer: str) -> bool:
    """Check if a generated LaTeX answer matches ground truth."""
    normalized_answer = normalize_latex(answer)
    normalized_gt_answer = normalize_latex(gt_answer)
    if normalized_answer == normalized_gt_answer:
        return True
    try:
        return get_latex_equivalent(answer, gt_answer)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Numeric answer utilities (from multicot parser.py)
# ---------------------------------------------------------------------------

# Language-specific "Final:" markers
FINAL_MARKERS = {
    "en": ["Final:", "final:", "FINAL:"],
    "fr": [
        "Final:", "final:", "FINAL:",
        "Réponse finale:", "réponse finale:",
        "La réponse est:", "la réponse est:",
        "Réponse:", "réponse:",
    ],
    "zh": [
        "Final:", "final:", "FINAL:",
        "Final：", "final：",
        "最终答案:", "最终答案：",
        "答案:", "答案：",
        "最后答案:", "最后答案：",
        "结果:", "结果：",
    ],
    "ar": [
        "Final:", "final:", "FINAL:",
        "الإجابة النهائية:", "الإجابة:",
        "الجواب النهائي:", "الجواب:",
        "النتيجة:",
    ],
}

ALL_FINAL_MARKERS = set()
for markers in FINAL_MARKERS.values():
    ALL_FINAL_MARKERS.update(markers)


def parse_final_answer(text: str, language: str = "en") -> Optional[str]:
    """
    Extract the final numeric answer from a chain-of-thought output.
    Supports language-specific answer markers.
    """
    markers = FINAL_MARKERS.get(language, list(ALL_FINAL_MARKERS))

    for marker in markers:
        escaped_marker = re.escape(marker)
        # Pattern: marker followed by a number
        pattern1 = escaped_marker + r"\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
        match = re.search(pattern1, text)
        if match:
            return match.group(1).replace(",", "")

        # Pattern: marker followed by anything on the same line
        pattern2 = escaped_marker + r"\s*(.+?)(?:\n|$)"
        match = re.search(pattern2, text)
        if match:
            answer_text = match.group(1).strip()
            number_match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer_text)
            if number_match:
                return number_match.group(0).replace(",", "")
            if answer_text:
                return answer_text

    # Fallback: case-insensitive "Final:" pattern
    pattern1 = r"Final[:：]\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    pattern2 = r"Final[:：]\s*(.+?)(?:\n|$)"
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip()
        number_match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer_text)
        if number_match:
            return number_match.group(0).replace(",", "")
        if answer_text:
            return answer_text

    # Fallback: boxed answer
    pattern3 = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern3, text)
    if match:
        answer = match.group(1).strip()
        number_match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
        if number_match:
            return number_match.group(0).replace(",", "")
        return answer

    # Language-specific "the answer is" patterns
    answer_patterns = {
        "en": r"[Tt]he (?:final )?answer is[:\s]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        "fr": r"[Ll]a réponse (?:finale )?est[:\s]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
    }
    if language in answer_patterns:
        match = re.search(answer_patterns[language], text)
        if match:
            return match.group(1).replace(",", "")

    return None


def normalize_number(s: str) -> Optional[float]:
    """Normalize a string to a float."""
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Unified answer extraction and checking
# ---------------------------------------------------------------------------

def extract_multiple_choice_answer(text: str) -> str:
    """
    Extract the chosen letter (A/B/C/D) from a multiple-choice solution.

    Tries, in order:
    1. Language-specific "Answer: X" marker patterns
    2. \\boxed{X} where X is a letter
    3. Last line that is a bare letter
    4. Last standalone A/B/C/D in the final 200 chars
    """
    # Multilingual "answer label: X" patterns (markdown bold stripped via \**)
    answer_patterns = [
        r"[Aa]nswer\s*[:：]\s*\**([ABCD])\**",
        r"[Rr][eé]ponse\s*[:：]\s*\**([ABCD])\**",
        r"答案\s*[:：]\s*\**([ABCD])\**",
        r"الإجابة\s*[:：]\s*\**([ABCD])\**",
        r"[Aa]ntwort\s*[:：]\s*\**([ABCD])\**",
        r"[Rr]espuesta\s*[:：]\s*\**([ABCD])\**",
        r"उत्तर\s*[:：]\s*\**([ABCD])\**",
        r"উত্তর\s*[:：]\s*\**([ABCD])\**",
        r"[Jj]awaban\s*[:：]\s*\**([ABCD])\**",
        r"[Rr]isposta\s*[:：]\s*\**([ABCD])\**",
        r"答え\s*[:：]\s*\**([ABCD])\**",
        r"정답\s*[:：]\s*\**([ABCD])\**",
        r"[Rr]esposta\s*[:：]\s*\**([ABCD])\**",
        r"[Jj]ibu\s*[:：]\s*\**([ABCD])\**",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    # \boxed{A} style
    boxed = re.search(r"\\boxed\{([ABCD])\}", text)
    if boxed:
        return boxed.group(1).upper()

    # Last line that is just a bare letter (possibly bold)
    for line in reversed(text.strip().split("\n")):
        stripped = line.strip().strip("*").strip()
        if stripped in ("A", "B", "C", "D"):
            return stripped

    # Last standalone letter in the final 200 chars
    tail_matches = list(re.finditer(r"\b([ABCD])\b", text[-200:]))
    if tail_matches:
        return tail_matches[-1].group(1).upper()

    return ""


def extract_answer(text: str, problem: Problem, language: str) -> str:
    """
    Extract answer from generated text based on dataset type.

    MMATH: extract from \\boxed{}
    MGSM: parse Final: <number> with multilingual markers
    MMMLU: extract multiple-choice letter (A/B/C/D)
    """
    if problem.answer_type == "latex_boxed":
        answers = extract_boxed_answers(text)
        return answers[0] if answers and answers[0] else ""
    elif problem.answer_type == "multiple_choice":
        return extract_multiple_choice_answer(text)
    else:  # numeric
        answer = parse_final_answer(text, language)
        if answer is None:
            # Fallback: try boxed
            answers = extract_boxed_answers(text)
            if answers and answers[0]:
                number_match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answers[0])
                if number_match:
                    return number_match.group(0).replace(",", "")
        return answer if answer is not None else ""


def check_answer_for_problem(predicted: str, problem: Problem) -> bool:
    """
    Check if predicted answer is correct for a given problem.

    numeric: float comparison with tolerance 1e-6
    latex_boxed: normalize_latex + sympy equivalence
    multiple_choice: exact letter match (case-insensitive)
    """
    if not predicted:
        return False

    if problem.answer_type == "multiple_choice":
        return predicted.strip().upper() == problem.gt_answer.strip().upper()
    elif problem.answer_type == "numeric":
        pred_num = normalize_number(predicted)
        gt_num = normalize_number(problem.gt_answer)
        if pred_num is not None and gt_num is not None:
            return abs(pred_num - gt_num) < 1e-6
        return predicted.strip() == problem.gt_answer.strip()
    else:  # latex_boxed
        return check_answer_latex(predicted, problem.gt_answer)


# ---------------------------------------------------------------------------
# GlotLID language verification
# ---------------------------------------------------------------------------

_glotlid_model = None

# Map GlotLID ISO 639-3 codes to our 2-letter codes
GLOTLID_TO_LANG = {
    "eng": "en", "eng_Latn": "en",
    "fra": "fr", "fra_Latn": "fr",
    "zho": "zh", "zho_Hans": "zh", "zho_Hant": "zh",
    "cmn": "zh", "cmn_Hans": "zh", "cmn_Hant": "zh",
    "arb": "ar", "arb_Arab": "ar",
    "ara": "ar", "ara_Arab": "ar",
    # MMMLU languages
    "deu": "de", "deu_Latn": "de",
    "spa": "es", "spa_Latn": "es",
    "hin": "hi", "hin_Deva": "hi",
    "ben": "bn", "ben_Beng": "bn",
    "ind": "id", "ind_Latn": "id",
    "ita": "it", "ita_Latn": "it",
    "jpn": "ja", "jpn_Jpan": "ja",
    "kor": "ko", "kor_Hang": "ko",
    "por": "pt", "por_Latn": "pt",
    "swh": "sw", "swh_Latn": "sw",
    "swa": "sw", "swa_Latn": "sw",
    "yor": "yo", "yor_Latn": "yo",
}


def _apply_numpy2_compat():
    """Work around NumPy 2.x: libraries (e.g. fasttext) using np.array(..., copy=False) can raise
    'Unable to avoid copy while creating an array'. Use np.asarray when copy=False."""
    import numpy as np
    if getattr(np, "_glotlid_numpy2_patched", False):
        return
    _orig = np.array

    def _array(obj, dtype=None, copy=None, order="K", subok=False, ndmin=0):
        if copy is False:
            return np.asarray(obj, dtype=dtype, order=order)
        return _orig(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    np.array = _array
    np._glotlid_numpy2_patched = True


def load_glotlid_model():
    """Load the GlotLID model (downloads from HuggingFace if not cached)."""
    global _glotlid_model
    if _glotlid_model is not None:
        return _glotlid_model

    _apply_numpy2_compat()
    import fasttext
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="cis-lmu/glotlid",
        filename="model.bin",
    )
    _glotlid_model = fasttext.load_model(model_path)
    return _glotlid_model


def verify_language(text: str, expected_lang: str) -> Tuple[bool, str, float]:
    """
    Verify that text is in the expected language using GlotLID.

    Args:
        text: Text to verify
        expected_lang: Expected 2-letter language code

    Returns:
        (is_correct, detected_lang_code, confidence)
    """
    model = load_glotlid_model()

    # Clean text for detection: remove LaTeX, keep natural language
    clean_text = re.sub(r"\$[^$]+\$", "", text)
    clean_text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", clean_text)
    clean_text = re.sub(r"[0-9+\-*/=(){}\\]+", " ", clean_text)
    clean_text = " ".join(clean_text.split())  # normalize whitespace

    if len(clean_text) < 10:
        # Too short to reliably detect language
        return True, expected_lang, 0.0

    # Replace newlines for fasttext
    clean_text = clean_text.replace("\n", " ")

    predictions = model.predict(clean_text, k=3)
    labels, scores = predictions

    # Parse the top prediction
    top_label = labels[0].replace("__label__", "")
    top_score = float(scores[0])

    # Map to our language code
    detected_lang = GLOTLID_TO_LANG.get(top_label, top_label[:2])

    is_correct = detected_lang == expected_lang
    return is_correct, detected_lang, top_score
