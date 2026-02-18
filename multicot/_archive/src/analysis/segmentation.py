"""
Sentence Segmentation for Chain-of-Thought

Segments reasoning traces into comparable "steps" (sentences).
Handles multiple languages with language-aware sentence splitting.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Sentence:
    """A single sentence/step in a chain-of-thought trace."""

    idx: int  # 0-indexed position in the trace
    text: str  # The sentence text
    start_char: int  # Start character position in original text
    end_char: int  # End character position in original text
    start_token: Optional[int] = None  # Start token position (if tokenized)
    end_token: Optional[int] = None  # End token position (if tokenized)

    @property
    def char_range(self) -> tuple[int, int]:
        return (self.start_char, self.end_char)


def segment_cot(
    text: str,
    language: str = "en",
    min_sentence_length: int = 10,
) -> list[Sentence]:
    """
    Segment a chain-of-thought trace into sentences.

    Uses language-aware sentence boundary detection.

    Args:
        text: The full chain-of-thought text
        language: Language code for sentence splitting
        min_sentence_length: Minimum characters for a valid sentence

    Returns:
        List of Sentence objects
    """
    # Try to use spacy for better multilingual support if available
    try:
        return _segment_with_spacy(text, language, min_sentence_length)
    except ImportError:
        pass

    # Fallback to regex-based splitting
    return _segment_with_regex(text, language, min_sentence_length)


def _segment_with_spacy(
    text: str,
    language: str,
    min_sentence_length: int,
) -> list[Sentence]:
    """Segment using spacy's sentence tokenizer."""
    import spacy

    # Map language codes to spacy models
    model_map = {
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_core_news_sm",
        "ru": "ru_core_news_sm",
    }

    model_name = model_map.get(language, "xx_sent_ud_sm")

    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Model not installed, try multilingual
        try:
            nlp = spacy.load("xx_sent_ud_sm")
        except OSError:
            raise ImportError(f"No spacy model available for {language}")

    # Disable everything except sentence segmentation for speed
    nlp.disable_pipes([p for p in nlp.pipe_names if p != "senter" and p != "parser"])

    doc = nlp(text)

    sentences = []
    for idx, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        if len(sent_text) >= min_sentence_length:
            sentences.append(
                Sentence(
                    idx=len(sentences),
                    text=sent_text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                )
            )

    return sentences


def _segment_with_regex(
    text: str,
    language: str,
    min_sentence_length: int,
) -> list[Sentence]:
    """
    Segment using regex-based sentence splitting.

    Handles common sentence boundaries across languages.
    """
    # Sentence-ending punctuation by language
    if language in ["zh", "ja"]:
        # Chinese/Japanese use different punctuation
        split_pattern = r"(?<=[。！？\.\!\?])\s*"
    elif language == "th":
        # Thai uses spaces more flexibly, split on explicit markers
        split_pattern = r"(?<=[\.!?])\s+"
    else:
        # Western languages
        split_pattern = r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u017F])|(?<=[.!?])\s*\n"

    # First, normalize newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # Split by paragraph first, then by sentence
    paragraphs = text.split("\n\n")

    sentences = []
    current_pos = 0

    for para in paragraphs:
        if not para.strip():
            current_pos += len(para) + 2  # Account for \n\n
            continue

        # Find paragraph position in original text
        para_start = text.find(para, current_pos)
        if para_start == -1:
            para_start = current_pos

        # Split paragraph into sentences
        sent_texts = re.split(split_pattern, para)

        sent_pos = para_start
        for sent_text in sent_texts:
            sent_text = sent_text.strip()
            if len(sent_text) >= min_sentence_length:
                # Find actual position
                actual_start = text.find(sent_text, sent_pos)
                if actual_start == -1:
                    actual_start = sent_pos

                sentences.append(
                    Sentence(
                        idx=len(sentences),
                        text=sent_text,
                        start_char=actual_start,
                        end_char=actual_start + len(sent_text),
                    )
                )
                sent_pos = actual_start + len(sent_text)

        current_pos = para_start + len(para)

    # Renumber indices to be contiguous
    for i, sent in enumerate(sentences):
        sent.idx = i

    return sentences


def add_token_positions(
    sentences: list[Sentence],
    text: str,
    tokenizer,
) -> list[Sentence]:
    """
    Add token positions to sentences using a tokenizer.

    Args:
        sentences: List of Sentence objects
        text: Original text
        tokenizer: HuggingFace tokenizer

    Returns:
        Sentences with start_token and end_token filled in
    """
    # Tokenize full text with offset mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    offsets = encoding["offset_mapping"]

    for sent in sentences:
        # Find tokens that overlap with this sentence
        start_token = None
        end_token = None

        for token_idx, (start, end) in enumerate(offsets):
            # Token overlaps with sentence
            if end > sent.start_char and start < sent.end_char:
                if start_token is None:
                    start_token = token_idx
                end_token = token_idx + 1

        sent.start_token = start_token
        sent.end_token = end_token

    return sentences


def compute_sentence_stats(sentences: list[Sentence]) -> dict:
    """
    Compute statistics about sentence segmentation.

    Args:
        sentences: List of Sentence objects

    Returns:
        Dict with statistics
    """
    if not sentences:
        return {
            "num_sentences": 0,
            "mean_length": 0,
            "median_length": 0,
            "min_length": 0,
            "max_length": 0,
        }

    lengths = [len(s.text) for s in sentences]

    return {
        "num_sentences": len(sentences),
        "mean_length": sum(lengths) / len(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
