"""
Answer Parsing Utilities

Extracts and normalizes final answers from chain-of-thought outputs.
Handles the "Final:" marker format and various numeric representations.
Supports multilingual answer markers for all 11 MGSM languages.
"""

import re
from typing import Optional


# Language-specific "Final:" markers
# Includes variations with different colons (full-width for CJK)
FINAL_MARKERS = {
    "en": ["Final:", "final:", "FINAL:"],
    "es": [
        "Final:", "final:", "FINAL:",
        "Respuesta final:", "respuesta final:",
        "La respuesta es:", "la respuesta es:",
        "Respuesta:", "respuesta:",
    ],
    "fr": [
        "Final:", "final:", "FINAL:",
        "Réponse finale:", "réponse finale:",
        "La réponse est:", "la réponse est:",
        "Réponse:", "réponse:",
    ],
    "de": [
        "Final:", "final:", "FINAL:",
        "Endgültige Antwort:", "endgültige Antwort:",
        "Die Antwort ist:", "die Antwort ist:",
        "Antwort:", "antwort:",
    ],
    "ru": [
        "Final:", "final:", "FINAL:",
        "Итоговый ответ:", "итоговый ответ:",
        "Ответ:", "ответ:",
        "Окончательный ответ:", "окончательный ответ:",
    ],
    "zh": [
        "Final:", "final:", "FINAL:",
        # Full-width colon variants
        "Final：", "final：",
        "最终答案:", "最终答案：",
        "答案:", "答案：",
        "最后答案:", "最后答案：",
        "结果:", "结果：",
    ],
    "ja": [
        "Final:", "final:", "FINAL:",
        # Full-width colon variants
        "Final：", "final：",
        "最終答え:", "最終答え：",
        "答え:", "答え：",
        "回答:", "回答：",
        "最終的な答え:", "最終的な答え：",
    ],
    "th": [
        "Final:", "final:", "FINAL:",
        "คำตอบสุดท้าย:",
        "คำตอบ:",
        "ผลลัพธ์:",
    ],
    "sw": [
        "Final:", "final:", "FINAL:",
        "Jibu la mwisho:",
        "Jibu:",
        "Matokeo:",
    ],
    "bn": [
        "Final:", "final:", "FINAL:",
        "চূড়ান্ত উত্তর:",
        "উত্তর:",
        "ফলাফল:",
    ],
    "te": [
        "Final:", "final:", "FINAL:",
        "చివరి సమాధానం:",
        "సమాధానం:",
        "ఫలితం:",
    ],
}

# Build a combined list of all markers for fallback
ALL_FINAL_MARKERS = set()
for markers in FINAL_MARKERS.values():
    ALL_FINAL_MARKERS.update(markers)


def parse_final_answer(text: str, language: str = "en") -> Optional[str]:
    """
    Extract the final answer from a chain-of-thought output.

    Supports language-specific answer markers for all 11 MGSM languages.

    Looks for patterns like:
    - "Final: 42" (English)
    - "Respuesta final: 42" (Spanish)
    - "最终答案：42" (Chinese, with full-width colon)
    - And many more language-specific variants

    Args:
        text: The full generated text
        language: Language code (en, es, fr, de, ru, zh, ja, th, sw, bn, te)

    Returns:
        The extracted answer as a string, or None if not found
    """
    # Get language-specific markers, falling back to all markers
    markers = FINAL_MARKERS.get(language, list(ALL_FINAL_MARKERS))

    # Try language-specific markers first
    for marker in markers:
        # Escape special regex characters in marker
        escaped_marker = re.escape(marker)

        # Pattern: marker followed by a number (with optional comma separators)
        pattern1 = escaped_marker + r"\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
        match = re.search(pattern1, text)
        if match:
            answer = match.group(1).replace(",", "")
            return answer

        # Pattern: marker followed by anything on the same line
        pattern2 = escaped_marker + r"\s*(.+?)(?:\n|$)"
        match = re.search(pattern2, text)
        if match:
            answer_text = match.group(1).strip()
            # Try to extract a number from whatever follows
            number_match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer_text)
            if number_match:
                return number_match.group(0).replace(",", "")
            # Return non-empty answer text
            if answer_text:
                return answer_text

    # Fallback: Try standard "Final:" pattern (case-insensitive)
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

    # Pattern 3: Look for boxed answer (LaTeX style) as fallback
    pattern3 = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern3, text)
    if match:
        answer = match.group(1).strip()
        # Try to extract number
        number_match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
        if number_match:
            return number_match.group(0).replace(",", "")
        return answer

    # Pattern 4: "The answer is" pattern (English only)
    if language == "en":
        pattern4 = r"[Tt]he (?:final )?answer is[:\s]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)"
        match = re.search(pattern4, text)
        if match:
            return match.group(1).replace(",", "")

    # Language-specific "the answer is" patterns
    answer_patterns = {
        "es": r"[Ll]a respuesta (?:final )?es[:\s]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        "fr": r"[Ll]a réponse (?:finale )?est[:\s]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        "de": r"[Dd]ie (?:endgültige )?Antwort ist[:\s]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        "ru": r"[Оо]твет[:\s]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
    }

    if language in answer_patterns:
        match = re.search(answer_patterns[language], text)
        if match:
            return match.group(1).replace(",", "")

    return None


def normalize_number(s: str) -> Optional[float]:
    """
    Normalize a string to a float.

    Handles:
    - Comma separators (1,234 -> 1234)
    - Whitespace
    - Sign prefixes

    Args:
        s: String representation of a number

    Returns:
        Float value, or None if parsing fails
    """
    if s is None:
        return None

    # Remove commas and whitespace
    s = s.replace(",", "").strip()

    try:
        return float(s)
    except ValueError:
        return None


def check_answer(predicted: Optional[str], ground_truth: str) -> Optional[bool]:
    """
    Check if a predicted answer matches the ground truth.

    Args:
        predicted: Predicted answer string (may be None)
        ground_truth: Ground truth answer string

    Returns:
        True if correct, False if incorrect, None if predicted is None
    """
    if predicted is None:
        return None

    pred_num = normalize_number(predicted)
    gt_num = normalize_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        # Numeric comparison with small tolerance for floating point
        return abs(pred_num - gt_num) < 1e-6
    else:
        # String comparison as fallback
        return predicted.strip() == ground_truth.strip()


def extract_all_numbers(text: str) -> list[str]:
    """
    Extract all numbers from text.

    Useful for analysis of reasoning steps.

    Args:
        text: Input text

    Returns:
        List of number strings found
    """
    pattern = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
    matches = re.findall(pattern, text)
    return [m.replace(",", "") for m in matches]
