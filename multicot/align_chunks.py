"""
Cross-Language Step Alignment

Aligns semantically equivalent reasoning chunks between two language traces
using LaBSE embeddings and monotone maximum-weight DP alignment.

Usage:
    python -m multicot.align_chunks --dataset mgsm --lang1 en --lang2 fr [--tau 0.6]
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# CLI Arguments (module-level, mirrors analyze_rollouts.py pattern)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Cross-language chunk alignment via LaBSE")
parser.add_argument("--dataset", type=str, required=True, choices=["mgsm", "mmath", "mmmlu", "polymath"])
parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--lang1", type=str, default="en")
parser.add_argument("--lang2", type=str, default="fr")
parser.add_argument("--tau", type=float, default=0.6,
                    help="Cosine similarity threshold for alignment")
parser.add_argument("--validate_n", type=int, default=0,
                    help="Number of aligned pairs to validate with GPT-4o (0=skip)")
parser.add_argument("--force", action="store_true", default=False,
                    help="Overwrite existing alignment files")
parser.add_argument("--no_store_matrix", action="store_true", default=False,
                    help="Omit full similarity matrix from JSON (saves space)")
parser.add_argument("--labse_model", type=str, default="LaBSE",
                    help="SentenceTransformer model name for cross-lingual embeddings")
parser.add_argument("--aligner", type=str, default="labse",
                    choices=["labse", "translate_en"],
                    help="Alignment method: 'labse' embeds original text with LaBSE; "
                         "'translate_en' translates both traces to English with GPT-4o first")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Lazy globals
# ---------------------------------------------------------------------------

_labse_model = None
_openai_client = None

IMPORTANCE_METRICS = [
    "resampling_importance_accuracy",
    "resampling_importance_kl",
    "counterfactual_importance_accuracy",
    "counterfactual_importance_kl",
    "forced_importance_accuracy",
    "forced_importance_kl",
]


def _get_labse() -> SentenceTransformer:
    global _labse_model
    if _labse_model is None:
        print(f"Loading LaBSE model: {args.labse_model}")
        _labse_model = SentenceTransformer(args.labse_model)
    return _labse_model


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Embed chunks with LaBSE; returns L2-normalized array of shape (N, d)."""
    if not chunks:
        return np.zeros((0, 768), dtype=np.float32)
    model = _get_labse()
    embs = model.encode(chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return embs.astype(np.float32)


def translate_chunks_to_english(chunks: List[str], language: str) -> List[str]:
    """
    Translate a list of reasoning chunks to English using GPT-4o.

    Mathematical notation and LaTeX are preserved unchanged.
    Returns a copy of the input list if language is already 'en' or chunks is empty.
    Falls back to the original text (with a warning) on API failure or count mismatch.
    """
    if language == "en" or not chunks:
        return list(chunks)

    from multicot.data_loaders import LANGUAGE_NAMES
    lang_name = LANGUAGE_NAMES.get(language, language)

    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(chunks))
    prompt = (
        f"Translate each of the following mathematical reasoning steps from {lang_name} to English. "
        f"Preserve all mathematical notation, LaTeX formulas, and variable names exactly as written. "
        f'Reply with JSON in this format: {{"translations": ["<step 1 in English>", "<step 2 in English>", ...]}}\n\n'
        f"{numbered}"
    )

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        translated = result.get("translations", [])
        if len(translated) != len(chunks):
            print(f"  [warn] translate_en: got {len(translated)} translations for "
                  f"{len(chunks)} chunks — falling back to originals")
            return list(chunks)
        return [str(t) for t in translated]
    except Exception as e:
        print(f"  [warn] translate_en: translation failed ({e}) — falling back to originals")
        return list(chunks)


def compute_similarity_matrix(embs1: np.ndarray, embs2: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix, shape (m, n). Assumes L2-normalized inputs."""
    if embs1.shape[0] == 0 or embs2.shape[0] == 0:
        return np.zeros((embs1.shape[0], embs2.shape[0]), dtype=np.float32)
    return (embs1 @ embs2.T).astype(np.float32)


def monotone_max_weight_alignment(
    sim_matrix: np.ndarray, tau: float = 0.6
) -> List[Tuple[int, int, float]]:
    """
    Find max-weight monotone alignment between two chunk sequences.

    Returns a list of (i, j, sim) tuples with strictly increasing i and j,
    where sim >= tau. Uses O(m*n) DP with prefix-max sweep.
    """
    if sim_matrix.size == 0:
        return []

    m, n = sim_matrix.shape
    NEG_INF = -1e9

    # dp[i][j] = best total weight of alignment ending at (i, j)
    dp = np.full((m, n), NEG_INF, dtype=np.float64)
    prev = [[None] * n for _ in range(m)]

    # best[j] = max dp[i'][j'] over all i' processed so far, for j' <= j
    # We maintain this as a running prefix max after each row.
    # For row i, prefix_max[j] = max(best[0..j-1])
    best = np.full(n, NEG_INF, dtype=np.float64)

    for i in range(m):
        # prefix_max[j] = max(best[0], ..., best[j-1])
        prefix_max = np.full(n, NEG_INF, dtype=np.float64)
        for j in range(1, n):
            prefix_max[j] = max(prefix_max[j - 1], best[j - 1])

        for j in range(n):
            sim = float(sim_matrix[i, j])
            if sim < tau:
                continue
            base = max(0.0, prefix_max[j])
            dp[i, j] = sim + base

            # Record predecessor
            if prefix_max[j] > 0 and j > 0:
                # Find the best (i', j') with i' < i, j' < j
                best_val = NEG_INF
                best_ij = None
                for i2 in range(i):
                    for j2 in range(j):
                        if dp[i2, j2] > best_val:
                            best_val = dp[i2, j2]
                            best_ij = (i2, j2)
                prev[i][j] = best_ij

        # Update best[] with current row dp values
        for j in range(n):
            if dp[i, j] > best[j]:
                best[j] = dp[i, j]

    # Find the endpoint with maximum weight
    max_val = NEG_INF
    end_i, end_j = -1, -1
    for i in range(m):
        for j in range(n):
            if dp[i, j] > max_val:
                max_val = dp[i, j]
                end_i, end_j = i, j

    if end_i == -1 or max_val <= NEG_INF / 2:
        return []

    # Traceback
    path = []
    ci, cj = end_i, end_j
    while ci != -1 and cj != -1:
        sim = float(sim_matrix[ci, cj])
        if sim >= tau:
            path.append((ci, cj, sim))
        p = prev[ci][cj]
        if p is None:
            break
        ci, cj = p

    path.reverse()
    return path


def load_labeled_chunks(problem_dir_lang: Path) -> Optional[List[dict]]:
    """Load chunks_labeled.json from a problem directory. Returns None if missing."""
    f = problem_dir_lang / "chunks_labeled.json"
    if not f.exists():
        return None
    try:
        with open(f, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as e:
        print(f"  [warn] Could not load {f}: {e}")
        return None


def validate_alignment_with_llm(
    problem_text: str,
    aligned_pairs: List[Dict],
    lang1: str,
    lang2: str,
    n_samples: int,
) -> List[Dict]:
    """
    Use GPT-4o to validate a sample of aligned pairs.

    Adds llm_valid (bool) and llm_reasoning (str) fields to sampled pairs.
    """
    if not aligned_pairs or n_samples <= 0:
        return aligned_pairs

    import random
    indices = random.sample(range(len(aligned_pairs)), min(n_samples, len(aligned_pairs)))

    client = _get_client()
    for idx in indices:
        pair = aligned_pairs[idx]
        c1 = pair.get("chunk_lang1", "")
        c2 = pair.get("chunk_lang2", "")

        prompt = (
            f"You are evaluating cross-language reasoning alignment.\n\n"
            f"Problem: {problem_text}\n\n"
            f"The following two reasoning steps from different language traces "
            f"({lang1} and {lang2}) have been automatically aligned as semantically equivalent.\n\n"
            f"Step in {lang1}:\n{c1}\n\n"
            f"Step in {lang2}:\n{c2}\n\n"
            f"Do these two steps represent the same reasoning operation? "
            f"Reply with JSON: {{\"valid\": true/false, \"reasoning\": \"<brief explanation>\"}}"
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            pair["llm_valid"] = result.get("valid", None)
            pair["llm_reasoning"] = result.get("reasoning", None)
        except Exception as e:
            print(f"  [warn] GPT-4o validation failed: {e}")
            pair["llm_valid"] = None
            pair["llm_reasoning"] = None

    return aligned_pairs


def write_alignment_json(
    problem_dir_lang1: Path,
    lang1: str,
    lang2: str,
    tau: float,
    chunks1: List[dict],
    chunks2: List[dict],
    sim_matrix: np.ndarray,
    alignment: List[Tuple[int, int, float]],
    store_matrix: bool = True,
    problem_text: str = "",
    validate_n: int = 0,
    translated1: Optional[List[str]] = None,
    translated2: Optional[List[str]] = None,
    aligner: str = "labse",
) -> Dict:
    """Build aligned_pairs, optionally validate, write JSON, and return data dict."""
    aligned_pairs = []
    for i, j, sim in alignment:
        c1 = chunks1[i]
        c2 = chunks2[j]
        pair: Dict = {
            "idx_lang1": i,
            "idx_lang2": j,
            "similarity": round(float(sim), 4),
            "chunk_lang1": c1.get("chunk", ""),
            "chunk_lang2": c2.get("chunk", ""),
            "function_tags_lang1": c1.get("function_tags", []),
            "function_tags_lang2": c2.get("function_tags", []),
            "llm_valid": None,
            "llm_reasoning": None,
        }
        # Include English translations when using translate_en aligner
        if translated1 is not None:
            pair["translated_lang1"] = translated1[i]
        if translated2 is not None:
            pair["translated_lang2"] = translated2[j]
        for m in IMPORTANCE_METRICS:
            pair[f"{m}_lang1"] = c1.get(m, None)
            pair[f"{m}_lang2"] = c2.get(m, None)
        aligned_pairs.append(pair)

    if validate_n > 0 and aligned_pairs:
        aligned_pairs = validate_alignment_with_llm(
            problem_text, aligned_pairs, lang1, lang2, validate_n
        )

    mean_sim = float(np.mean([p["similarity"] for p in aligned_pairs])) if aligned_pairs else 0.0

    data: Dict = {
        "problem_id": problem_dir_lang1.name,
        "lang1": lang1,
        "lang2": lang2,
        "tau": tau,
        "aligner": aligner,
        "n_chunks_lang1": len(chunks1),
        "n_chunks_lang2": len(chunks2),
        "n_aligned_pairs": len(aligned_pairs),
        "mean_similarity": round(mean_sim, 4),
        "aligned_pairs": aligned_pairs,
    }

    if store_matrix:
        data["similarity_matrix"] = sim_matrix.tolist()

    suffix = "" if aligner == "labse" else f"_{aligner}"
    out_path = problem_dir_lang1 / f"alignment_{lang1}_{lang2}{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


def build_summary_rows(alignment_data: Dict, problem_id: str) -> List[Dict]:
    """Build one CSV row per aligned pair for the summary CSV."""
    rows = []
    lang1 = alignment_data["lang1"]
    lang2 = alignment_data["lang2"]
    for pair in alignment_data.get("aligned_pairs", []):
        row: Dict = {
            "problem_id": problem_id,
            "lang1": lang1,
            "lang2": lang2,
            "idx_lang1": pair["idx_lang1"],
            "idx_lang2": pair["idx_lang2"],
            "similarity": pair["similarity"],
            "function_tags_lang1": "|".join(pair.get("function_tags_lang1") or []),
            "function_tags_lang2": "|".join(pair.get("function_tags_lang2") or []),
            "llm_valid": pair.get("llm_valid"),
        }
        for m in IMPORTANCE_METRICS:
            row[f"{m}_lang1"] = pair.get(f"{m}_lang1")
            row[f"{m}_lang2"] = pair.get(f"{m}_lang2")
        rows.append(row)
    return rows


def align_problem(
    problem_dir_lang1: Path,
    lang1: str,
    lang2: str,
    tau: float,
    force: bool,
    store_matrix: bool = True,
    validate_n: int = 0,
    aligner: str = "labse",
) -> Optional[Dict]:
    """
    Orchestrate embed → similarity matrix → DP alignment → write JSON.

    aligner="labse"       — embed original text with LaBSE (default)
    aligner="translate_en" — translate both traces to English with GPT-4o, then embed

    Returns the alignment data dict, or None on failure/skip.
    """
    suffix = "" if aligner == "labse" else f"_{aligner}"
    out_path = problem_dir_lang1 / f"alignment_{lang1}_{lang2}{suffix}.json"
    if out_path.exists() and not force:
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Resolve lang2 problem directory (sibling language, same problem_id)
    lang2_problem_dir = (
        problem_dir_lang1.parent.parent.parent / lang2 / "correct_base_solution" / problem_dir_lang1.name
    )

    chunks1 = load_labeled_chunks(problem_dir_lang1)
    chunks2 = load_labeled_chunks(lang2_problem_dir)

    if chunks1 is None or chunks2 is None:
        return None
    if len(chunks1) == 0 or len(chunks2) == 0:
        return None

    texts1 = [c.get("chunk", "") for c in chunks1]
    texts2 = [c.get("chunk", "") for c in chunks2]

    if aligner == "translate_en":
        translated1 = translate_chunks_to_english(texts1, lang1)
        translated2 = translate_chunks_to_english(texts2, lang2)
        embs1 = embed_chunks(translated1)
        embs2 = embed_chunks(translated2)
    else:  # labse
        translated1 = translated2 = None
        embs1 = embed_chunks(texts1)
        embs2 = embed_chunks(texts2)

    sim_matrix = compute_similarity_matrix(embs1, embs2)
    alignment = monotone_max_weight_alignment(sim_matrix, tau=tau)

    # Load problem text for optional GPT-4o validation
    problem_text = ""
    problem_file = problem_dir_lang1 / "problem.json"
    if problem_file.exists():
        try:
            with open(problem_file, "r", encoding="utf-8") as f:
                pdata = json.load(f)
            problem_text = pdata.get("question", pdata.get("problem", ""))
        except Exception:
            pass

    return write_alignment_json(
        problem_dir_lang1=problem_dir_lang1,
        lang1=lang1,
        lang2=lang2,
        tau=tau,
        chunks1=chunks1,
        chunks2=chunks2,
        sim_matrix=sim_matrix,
        alignment=alignment,
        store_matrix=store_matrix,
        problem_text=problem_text,
        validate_n=validate_n,
        translated1=translated1,
        translated2=translated2,
        aligner=aligner,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    lang1 = args.lang1
    lang2 = args.lang2
    tau = args.tau
    aligner = args.aligner
    store_matrix = not args.no_store_matrix

    rollouts_dir = (
        Path("multicot") / args.dataset / args.model
        / f"temperature_{args.temperature}_top_p_{args.top_p}"
    )

    lang1_base = rollouts_dir / lang1 / "correct_base_solution"
    if not lang1_base.exists():
        print(f"Lang1 directory not found: {lang1_base}")
        return

    problem_dirs = sorted([
        d for d in lang1_base.iterdir()
        if d.is_dir() and d.name.startswith("problem_")
    ])

    if not problem_dirs:
        print(f"No problem directories found in {lang1_base}")
        return

    print(f"\nAligning {lang1} → {lang2} for {args.dataset} "
          f"({len(problem_dirs)} problems, tau={tau}, aligner={aligner})")

    all_summary_rows = []
    n_aligned = 0

    for problem_dir in problem_dirs:
        data = align_problem(
            problem_dir_lang1=problem_dir,
            lang1=lang1,
            lang2=lang2,
            tau=tau,
            force=args.force,
            store_matrix=store_matrix,
            validate_n=args.validate_n,
            aligner=aligner,
        )
        if data is None:
            print(f"  [skip] {problem_dir.name}")
            continue

        rows = build_summary_rows(data, problem_dir.name)
        all_summary_rows.extend(rows)
        n_aligned += data.get("n_aligned_pairs", 0)
        print(f"  [ok] {problem_dir.name}: {data['n_aligned_pairs']} pairs aligned "
              f"(mean sim={data['mean_similarity']:.3f})")

    # Write summary CSV — filename encodes the aligner so both results can coexist
    if all_summary_rows:
        suffix = "" if aligner == "labse" else f"_{aligner}"
        csv_path = rollouts_dir / f"alignment_summary_{lang1}_{lang2}{suffix}.csv"
        fieldnames = list(all_summary_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_summary_rows)
        print(f"\nSummary CSV: {csv_path}")

    print(f"\nDone. Total aligned pairs: {n_aligned}")


if __name__ == "__main__":
    main()
