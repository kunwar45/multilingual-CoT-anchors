# ABOUTME: Language-specific chunking of CoT traces into sentence-level chunks (en/fr/zh/ar), LaTeX-aware.
# ABOUTME: Dispatches on language code; chunks are the unit of rollout generation, importance metrics, and alignment.
"""
Language-specific chunking for chain-of-thought reasoning traces.

Splits solution text into sentence-level chunks used for rollout
generation and importance analysis. Dispatches to a per-language
splitter based on the language code.

Supported languages:
  en / fr  – Latin-script with . ? ! sentence boundaries
  zh / ja  – Chinese/Japanese punctuation (。！？) plus ASCII equivalents
  ar       – Arabic text with ؟ plus ASCII .?!
  th       – Thai text; splits on paragraph breaks and ASCII .?! (LaTeX contexts)
  (other)  – falls back to en/fr splitter (de, es, hi, bn, id, it, ko, ms,
              pt, ru, sw, te, vi, yo)
"""

from __future__ import annotations

import re
from typing import List


# ---------------------------------------------------------------------------
# LaTeX-awareness helper
# ---------------------------------------------------------------------------

def _is_inside_latex(text: str, pos: int) -> bool:
    """Return True if *pos* is inside a LaTeX math delimiter ($…$ or \\(…\\))."""
    # $…$ inline math: odd number of unescaped $ before pos
    dollar_count = text[:pos].count("$") - text[:pos].count("\\$")
    if dollar_count % 2 == 1:
        return True

    # \(…\) inline math
    open_paren = len(re.findall(r"\\\(", text[:pos]))
    close_paren = len(re.findall(r"\\\)", text[:pos]))
    if open_paren > close_paren:
        return True

    return False


# ---------------------------------------------------------------------------
# Language-specific splitters
# ---------------------------------------------------------------------------

def _split_en_fr(solution_text: str) -> List[str]:
    """Split English/French text into chunks on . ? ! or double-newline."""
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    chunks: List[str] = []
    current_chunk = ""

    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if (next_char == " " or next_char == "\n") and not _is_inside_latex(solution_text, i):
                is_sentence_end = True

        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Merge chunks shorter than 10 characters into neighbors
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    return chunks


def _split_zh(solution_text: str) -> List[str]:
    """Split Chinese text on 。！？ (immediately) and .?! (when followed by space/newline)."""
    sentence_endings = ["。", "！", "？", ".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    chunks: List[str] = []
    current_chunk = ""

    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        is_sentence_end = False
        if solution_text[i] in sentence_endings and not _is_inside_latex(solution_text, i):
            if solution_text[i] in ["。", "！", "？"]:
                is_sentence_end = True
            elif i < len(solution_text) - 1:
                next_char = solution_text[i + 1]
                if next_char == " " or next_char == "\n":
                    is_sentence_end = True

        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Merge chunks shorter than 6 characters (Chinese characters are denser)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 6:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    return chunks


def _split_ar(solution_text: str) -> List[str]:
    """Split Arabic text on ؟ (immediately) and .?! (when followed by space/newline)."""
    sentence_endings = [".", "?", "!", "؟"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    chunks: List[str] = []
    current_chunk = ""

    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        is_sentence_end = False
        if solution_text[i] in sentence_endings and not _is_inside_latex(solution_text, i):
            if solution_text[i] == "؟":
                is_sentence_end = True
            elif i < len(solution_text) - 1:
                next_char = solution_text[i + 1]
                if next_char == " " or next_char == "\n":
                    is_sentence_end = True

        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Merge chunks shorter than 8 characters
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 8:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    return chunks


def _split_th(solution_text: str) -> List[str]:
    """Split Thai text on paragraph breaks and ASCII .?! (for LaTeX-mixed content).

    Thai doesn't use Western sentence-ending punctuation consistently, so we
    split primarily on paragraph boundaries and newline-terminated ASCII
    punctuation (which models use when writing LaTeX math in Thai CoT).
    """
    sentence_endings = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    chunks: List[str] = []
    current_chunk = ""

    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        is_sentence_end = False
        if solution_text[i] in sentence_endings and not _is_inside_latex(solution_text, i):
            if i < len(solution_text) - 1:
                next_char = solution_text[i + 1]
                if next_char == " " or next_char == "\n":
                    is_sentence_end = True

        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Merge chunks shorter than 8 characters
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 8:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_solution_into_chunks(solution_text: str, language: str = "en") -> List[str]:
    """
    Split a solution into sentence-level chunks for rollout generation.

    Strips <think>…</think> tags first, then dispatches to a
    language-specific splitter.

    Args:
        solution_text: Full solution (may contain <think> tags).
        language:      BCP-47-style code: "en", "fr", "zh", "ar".
                       Unknown codes fall back to the en/fr splitter.

    Returns:
        List of non-empty chunk strings.
    """
    # Strip think tags
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    if language in ("en", "fr"):
        return _split_en_fr(solution_text)
    elif language in ("zh", "ja"):
        return _split_zh(solution_text)
    elif language == "ar":
        return _split_ar(solution_text)
    elif language == "th":
        return _split_th(solution_text)
    else:
        # Latin-script and other languages (de, es, hi, bn, id, it, ko, ms,
        # pt, ru, sw, te, vi, yo) — all use . ? ! sentence boundaries
        return _split_en_fr(solution_text)
