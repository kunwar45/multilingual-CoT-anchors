"""
Sentence segmentation utilities.

We use `pysbd` to segment reasoning traces into sentence spans, and
return character-level spans so that downstream code can align tokens
and activations with sentences in a tokenizer-agnostic way.
"""

from __future__ import annotations

from typing import List, Tuple

import pysbd

_SEGMENTERS: dict[str, pysbd.Segmenter] = {}


def _get_segmenter(lang: str) -> pysbd.Segmenter:
    """
    Return a cached pysbd segmenter for the given language code.

    We currently care about same-script languages (en, es, fr, de) for MGSM.
    For unknown languages we fall back to English rules, which are usually
    reasonable for Latin script.
    """
    lang = (lang or "en").lower()
    mapping = {
        "en": "en",
        "english": "en",
        "es": "es",
        "spa": "es",
        "spanish": "es",
        "fr": "fr",
        "fra": "fr",
        "french": "fr",
        "de": "de",
        "ger": "de",
        "german": "de",
    }
    code = mapping.get(lang, "en")
    if code not in _SEGMENTERS:
        _SEGMENTERS[code] = pysbd.Segmenter(language=code, clean=False)
    return _SEGMENTERS[code]


def sentence_spans(text: str, lang: str = "en") -> List[Tuple[int, int, str]]:
    """
    Segment `text` into sentences and return a list of (start, end, sentence).

    - `start` / `end` are character indices into `text` (Python slice
      semantics: `text[start:end] == sentence`).
    - `sentence` is the raw sentence string as returned by pysbd.
    """
    if not text:
        return []

    segmenter = _get_segmenter(lang)
    sents = segmenter.segment(text)

    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for s in sents:
        idx = text.find(s, cursor)
        if idx == -1:
            # Fallback: skip if we cannot align this sentence cleanly.
            continue
        start = idx
        end = idx + len(s)
        spans.append((start, end, s))
        cursor = end
    return spans


