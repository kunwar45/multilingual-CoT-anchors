"""Chunk-level language verification for multilingual CoT generation."""
from multicot.chunker import split_solution_into_chunks
from multicot.utils import verify_language


def check_chunk_languages(text: str, expected_lang: str) -> dict:
    """
    Split text into chunks and verify each chunk's language using GlotLID.

    Returns:
        {
            "has_switch": bool,
            "first_switch_chunk_idx": int | None,
            "switch_count": int,
            "chunks_checked": int,
            "switches": [
                {"chunk_idx": int, "text_preview": str, "detected_lang": str, "confidence": float}
            ],
        }
    """
    empty = {"has_switch": False, "first_switch_chunk_idx": None,
             "switch_count": 0, "chunks_checked": 0, "switches": []}

    if expected_lang == "en":
        return empty

    chunks = split_solution_into_chunks(text, expected_lang)
    if not chunks:
        return empty

    switches = []
    first_switch_idx = None

    for i, chunk in enumerate(chunks):
        # verify_language strips math internally; returns True if < 10 clean chars
        is_correct, detected, confidence = verify_language(chunk, expected_lang)
        if not is_correct:
            if first_switch_idx is None:
                first_switch_idx = i
            switches.append({
                "chunk_idx": i,
                "text_preview": chunk[:80].replace("\n", " "),
                "detected_lang": detected,
                "confidence": round(confidence, 4),
            })

    return {
        "has_switch": len(switches) > 0,
        "first_switch_chunk_idx": first_switch_idx,
        "switch_count": len(switches),
        "chunks_checked": len(chunks),
        "switches": switches,
    }
