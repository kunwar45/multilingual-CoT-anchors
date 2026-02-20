"""
Multilingual Rollout Analysis

Mirrors the root analyze_rollouts.py pipeline with multilingual support.
Calculates all 6 importance metrics, performs DAG labeling with GPT-4o,
and uses multilingual sentence embeddings for counterfactual importance.
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from functools import partial

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

from multicot.data_loaders import load_dataset_problems, Problem
from multicot.utils import (
    normalize_answer,
    normalize_latex,
    extract_answer,
    check_answer_for_problem,
)
from multicot.prompts import build_dag_labeling_prompt

# Load environment variables
load_dotenv()

# Use HF token from .env (HF_TOKEN) for dataset downloads
if os.getenv("HF_TOKEN"):
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

# Set tokenizers parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global cache for embedding model
_embedding_model_cache = {}

IMPORTANCE_METRICS = [
    "resampling_importance_accuracy",
    "resampling_importance_kl",
    "counterfactual_importance_accuracy",
    "counterfactual_importance_kl",
    "forced_importance_accuracy",
    "forced_importance_kl",
]

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Analyze multilingual rollout data")
parser.add_argument("--dataset", type=str, required=True, choices=["mgsm", "mmath"],
                    help="Dataset to analyze")
parser.add_argument("--languages", type=str, default="en,fr,zh,ar",
                    help="Comma-separated language codes")
parser.add_argument("--rollouts_dir", type=str, default=None,
                    help="Override rollouts directory (default: auto-detected from dataset/model)")
parser.add_argument("--model", type=str, default="deepseek-r1-distill-qwen-14b",
                    help="Model name (for directory structure)")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--base_solution_type", type=str, default="correct",
                    choices=["correct", "incorrect"])
parser.add_argument("-r", "--rollout_type", type=str, default="default",
                    choices=["default", "forced_answer"],
                    help="Rollout type (must match generate_rollouts.py)")
parser.add_argument("--output_suffix", type=str, default=None,
                    help="Output suffix (must match generate_rollouts.py)")
parser.add_argument("--forced_answer_dir", type=str, default=None,
                    help="Directory containing forced answer rollouts")
parser.add_argument("-f", "--force_relabel", action="store_true", default=False,
                    help="Force relabeling of chunks with GPT-4o")
parser.add_argument("--generate_metadata", action="store_true", default=False,
                    help="Generate chunk summaries and problem nicknames (default off)")
parser.add_argument("--force_metadata", action="store_true", default=False,
                    help="Force regeneration of metadata")
parser.add_argument("--sentence_model", type=str,
                    default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="Sentence transformer model for embeddings")
parser.add_argument("--similarity_threshold", type=float, default=0.8,
                    help="Similarity threshold for counterfactual split")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for embedding model")
parser.add_argument("--use_similar_chunks", action="store_true", default=True)
parser.add_argument("--use_existing_metrics", action="store_true", default=False,
                    help="Use existing metrics without recalculating")
parser.add_argument("--importance_metric", type=str,
                    default="counterfactual_importance_accuracy",
                    choices=IMPORTANCE_METRICS)
parser.add_argument("--use_prob_true", action="store_true", default=False,
                    help="Use P(true) instead of answer distribution for KL")
parser.add_argument("--laplace_alpha", type=float, default=1.0,
                    help="Smoothing parameter for KL divergence")
parser.add_argument("--use_global_vocab", action="store_true", default=False,
                    help="Use global vocabulary for KL smoothing")
parser.add_argument("--top_chunks", type=int, default=500,
                    help="Number of top chunks for counterfactual importance")
parser.add_argument("--absolute", action="store_true", default=False,
                    help="Use absolute value for importance")
parser.add_argument("--max_problems", type=int, default=None,
                    help="Maximum number of problems to analyze")

args = parser.parse_args()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# ImportanceArgs (same as root)
# ---------------------------------------------------------------------------

class ImportanceArgs:
    """Arguments for importance calculation functions."""

    def __init__(
        self,
        use_absolute=False,
        forced_answer_dir=None,
        similarity_threshold=0.8,
        use_similar_chunks=True,
        use_abs_importance=False,
        top_chunks=100,
        use_prob_true=True,
        global_vocab=None,
        laplace_alpha=1e-9,
    ):
        self.use_absolute = use_absolute
        self.forced_answer_dir = forced_answer_dir
        self.similarity_threshold = similarity_threshold
        self.use_similar_chunks = use_similar_chunks
        self.use_abs_importance = use_abs_importance
        self.top_chunks = top_chunks
        self.use_prob_true = use_prob_true
        self.global_vocab = global_vocab
        self.laplace_alpha = laplace_alpha


# ---------------------------------------------------------------------------
# KL Divergence (copied from root)
# ---------------------------------------------------------------------------

def calculate_kl_divergence(
    chunk_sols1,
    chunk_sols2,
    use_prob_true=False,
    global_vocab=None,
    alpha=1e-9,
):
    """Calculate KL divergence between answer distributions of two chunks."""
    if use_prob_true:
        correct1 = sum(1 for sol in chunk_sols1 if sol.get("is_correct", False) is True)
        total1 = sum(1 for sol in chunk_sols1 if sol.get("is_correct", None) is not None)
        correct2 = sum(1 for sol in chunk_sols2 if sol.get("is_correct", False) is True)
        total2 = sum(1 for sol in chunk_sols2 if sol.get("is_correct", None) is not None)

        if total1 == 0 or total2 == 0:
            return 0.0

        p = (correct1 + alpha) / (total1 + 2 * alpha)
        q = (correct2 + alpha) / (total2 + 2 * alpha)

        kl_div = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        return max(0.0, kl_div)
    else:
        answer_counts1 = defaultdict(int)
        answer_counts2 = defaultdict(int)

        for sol in chunk_sols1:
            answer = normalize_answer(sol.get("answer", ""))
            if answer:
                answer_counts1[answer] += 1

        for sol in chunk_sols2:
            answer = normalize_answer(sol.get("answer", ""))
            if answer:
                answer_counts2[answer] += 1

        if not answer_counts1 or not answer_counts2:
            return 0.0

        if global_vocab is not None:
            V = len(global_vocab)
        else:
            all_answers = set(answer_counts1.keys()) | set(answer_counts2.keys())
            V = len(all_answers)

        total1 = sum(answer_counts1.values())
        total2 = sum(answer_counts2.values())

        if total1 == 0 or total2 == 0:
            return 0.0

        smoothed_total1 = total1 + alpha * V
        smoothed_total2 = total2 + alpha * V

        observed_answers = set(answer_counts1.keys()) | set(answer_counts2.keys())
        kl_div = 0.0
        for answer in observed_answers:
            count1 = answer_counts1[answer]
            count2 = answer_counts2[answer]
            p = (count1 + alpha) / smoothed_total1
            q = (count2 + alpha) / smoothed_total2
            p_unsmoothed = count1 / total1
            kl_div += p_unsmoothed * math.log(p / q)

        return max(0.0, kl_div)


# ---------------------------------------------------------------------------
# Importance Calculation Functions (copied from root)
# ---------------------------------------------------------------------------

def calculate_counterfactual_importance_accuracy(
    chunk_idx, chunk_info, chunk_embedding_cache, chunk_accuracies, importance_args,
):
    """Calculate counterfactual importance based on accuracy differences."""
    if chunk_idx not in chunk_info:
        return 0.0, 0.0, 0.0

    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0, 0.0, 0.0

    next_chunk_idx = min(next_chunks)
    current_solutions = chunk_info[chunk_idx]
    next_solutions = chunk_info.get(next_chunk_idx, [])

    dissimilar_solutions = []
    different_trajectories_count = 0
    chunk_pairs = []

    for sol in current_solutions:
        removed = sol.get("chunk_removed", "")
        resampled = sol.get("chunk_resampled", "")

        if (
            isinstance(removed, str)
            and isinstance(resampled, str)
            and removed in chunk_embedding_cache
            and resampled in chunk_embedding_cache
        ):
            removed_emb = chunk_embedding_cache[removed]
            resampled_emb = chunk_embedding_cache[resampled]

            similarity = np.dot(removed_emb, resampled_emb) / (
                np.linalg.norm(removed_emb) * np.linalg.norm(resampled_emb)
            )

            if similarity < importance_args.similarity_threshold:
                dissimilar_solutions.append(sol)
                different_trajectories_count += 1

            chunk_pairs.append((removed, resampled, sol))

    different_trajectories_fraction = (
        different_trajectories_count / len(current_solutions) if current_solutions else 0.0
    )

    resampled_chunks_str = [pair[1] for pair in chunk_pairs]
    unique_resampled = set(resampled_chunks_str)
    overdeterminedness = (
        1.0 - (len(unique_resampled) / len(resampled_chunks_str)) if resampled_chunks_str else 0.0
    )

    if not dissimilar_solutions or not next_solutions:
        return 0.0, different_trajectories_fraction, overdeterminedness

    dissimilar_correct = sum(
        1 for sol in dissimilar_solutions
        if sol.get("is_correct", False) is True and sol.get("answer", "") != ""
    )
    dissimilar_total = len([
        sol for sol in dissimilar_solutions
        if sol.get("is_correct", None) is not None and sol.get("answer", "") != ""
    ])
    dissimilar_accuracy = dissimilar_correct / dissimilar_total if dissimilar_total > 0 else 0.0

    next_correct = sum(
        1 for sol in next_solutions
        if sol.get("is_correct", False) is True and sol.get("answer", "") != ""
    )
    next_total = len([
        sol for sol in next_solutions
        if sol.get("is_correct", None) is not None and sol.get("answer", "") != ""
    ])
    next_accuracy = next_correct / next_total if next_total > 0 else 0.0

    diff = dissimilar_accuracy - next_accuracy
    diff = abs(diff) if importance_args.use_abs_importance else diff
    return diff, different_trajectories_fraction, overdeterminedness


def calculate_counterfactual_importance_kl(
    chunk_idx, chunk_info, chunk_embedding_cache, chunk_accuracies, chunk_answers, importance_args,
):
    """Calculate counterfactual importance based on KL divergence."""
    if chunk_idx not in chunk_info:
        return 0.0

    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0

    next_chunk_idx = min(next_chunks)
    current_solutions = chunk_info[chunk_idx]
    next_solutions = chunk_info.get(next_chunk_idx, [])

    dissimilar_solutions = []
    similar_solutions = []

    for sol in current_solutions:
        removed = sol.get("chunk_removed", "")
        resampled = sol.get("chunk_resampled", "")

        if (
            isinstance(removed, str)
            and isinstance(resampled, str)
            and removed in chunk_embedding_cache
            and resampled in chunk_embedding_cache
        ):
            removed_emb = chunk_embedding_cache[removed]
            resampled_emb = chunk_embedding_cache[resampled]

            similarity = np.dot(removed_emb, resampled_emb) / (
                np.linalg.norm(removed_emb) * np.linalg.norm(resampled_emb)
            )

            if similarity < importance_args.similarity_threshold:
                dissimilar_solutions.append(sol)
            else:
                similar_solutions.append(sol)

    if not dissimilar_solutions or not next_solutions:
        return 0.0

    # Build answer distributions
    answer_correctness = {}
    for sol in current_solutions + next_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            answer_correctness[answer] = sol.get("is_correct", False)

    def _to_sol_list(answers_dict):
        sols = []
        for answer, count in answers_dict.items():
            for _ in range(count):
                sols.append({"answer": answer, "is_correct": answer_correctness.get(answer, False)})
        return sols

    similar_answers = defaultdict(int)
    for sol in similar_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            similar_answers[answer] += 1

    dissimilar_answers = defaultdict(int)
    for sol in dissimilar_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            dissimilar_answers[answer] += 1

    next_answers = defaultdict(int)
    for sol in next_solutions:
        answer = normalize_answer(sol.get("answer", ""))
        if answer:
            next_answers[answer] += 1

    dissimilar_sols = _to_sol_list(dissimilar_answers)
    similar_sols = _to_sol_list(similar_answers)
    next_sols = _to_sol_list(next_answers)

    reference = next_sols + similar_sols if importance_args.use_similar_chunks else next_sols

    return calculate_kl_divergence(
        dissimilar_sols,
        reference,
        use_prob_true=importance_args.use_prob_true,
        global_vocab=importance_args.global_vocab,
        alpha=importance_args.laplace_alpha,
    )


def calculate_resampling_importance_accuracy(chunk_idx, chunk_accuracies, importance_args=None):
    """Calculate resampling importance based on accuracy differences."""
    if chunk_idx not in chunk_accuracies:
        return 0.0

    current_accuracy = chunk_accuracies[chunk_idx]
    next_accuracies = [acc for idx, acc in chunk_accuracies.items() if idx == chunk_idx + 1]

    if not next_accuracies:
        return 0.0

    next_avg_accuracy = sum(next_accuracies) / len(next_accuracies)
    diff = next_avg_accuracy - current_accuracy

    if importance_args and importance_args.use_abs_importance:
        return abs(diff)
    return diff


def calculate_resampling_importance_kl(chunk_idx, chunk_info, problem_dir, importance_args=None):
    """Calculate resampling importance based on KL divergence."""
    if chunk_idx not in chunk_info:
        return 0.0

    next_chunks = [idx for idx in chunk_info.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0

    next_chunk_idx = min(next_chunks)

    chunk_dir = problem_dir / f"chunk_{chunk_idx}"
    next_chunk_dir = problem_dir / f"chunk_{next_chunk_idx}"

    chunk_sols1 = []
    chunk_sols2 = []

    solutions_file = chunk_dir / "solutions.json"
    if solutions_file.exists():
        try:
            with open(solutions_file, "r", encoding="utf-8") as f:
                chunk_sols1 = json.load(f)
        except Exception:
            pass

    next_solutions_file = next_chunk_dir / "solutions.json"
    if next_solutions_file.exists():
        try:
            with open(next_solutions_file, "r", encoding="utf-8") as f:
                chunk_sols2 = json.load(f)
        except Exception:
            pass

    if not chunk_sols1 or not chunk_sols2:
        return 0.0

    return calculate_kl_divergence(
        chunk_sols1, chunk_sols2,
        use_prob_true=importance_args.use_prob_true,
        global_vocab=importance_args.global_vocab,
        alpha=importance_args.laplace_alpha,
    )


def calculate_forced_importance_accuracy(chunk_idx, forced_answer_accuracies, importance_args=None):
    """Calculate forced importance based on accuracy differences."""
    if chunk_idx not in forced_answer_accuracies:
        return 0.0

    next_chunks = [idx for idx in forced_answer_accuracies.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0

    next_chunk_idx = min(next_chunks)
    current = forced_answer_accuracies[chunk_idx]
    next_val = forced_answer_accuracies[next_chunk_idx]

    diff = next_val - current
    if importance_args and importance_args.use_abs_importance:
        return abs(diff)
    return diff


def calculate_forced_importance_kl(
    chunk_idx, forced_answer_accuracies, problem_dir, forced_answer_dir, importance_args=None,
):
    """Calculate forced importance based on KL divergence."""
    if chunk_idx not in forced_answer_accuracies:
        return 0.0

    if not forced_answer_dir:
        return 0.0

    forced_problem_dir = forced_answer_dir / problem_dir.name
    if not forced_problem_dir.exists():
        return 0.0

    current_chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"
    current_solutions = []
    if current_chunk_dir.exists():
        solutions_file = current_chunk_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, "r", encoding="utf-8") as f:
                    current_solutions = json.load(f)
            except Exception:
                pass

    next_chunks = [idx for idx in forced_answer_accuracies.keys() if idx > chunk_idx]
    if not next_chunks:
        return 0.0

    next_chunk_idx = min(next_chunks)
    next_chunk_dir = forced_problem_dir / f"chunk_{next_chunk_idx}"
    next_solutions = []
    if next_chunk_dir.exists():
        solutions_file = next_chunk_dir / "solutions.json"
        if solutions_file.exists():
            try:
                with open(solutions_file, "r", encoding="utf-8") as f:
                    next_solutions = json.load(f)
            except Exception:
                pass

    if not current_solutions or not next_solutions:
        return 0.0

    return calculate_kl_divergence(
        current_solutions, next_solutions,
        use_prob_true=importance_args.use_prob_true,
        global_vocab=importance_args.global_vocab,
        alpha=importance_args.laplace_alpha,
    )


# ---------------------------------------------------------------------------
# Chunk importance orchestrator (same as root)
# ---------------------------------------------------------------------------

def process_chunk_importance(
    chunk_idx,
    chunk_info,
    chunk_embedding_cache,
    chunk_accuracies,
    importance_args,
    problem_dir=None,
    forced_answer_accuracies=None,
    chunk_answers=None,
):
    """Process importance metrics for a single chunk."""
    metrics = {}

    # Counterfactual
    cf_acc, dtf, od = calculate_counterfactual_importance_accuracy(
        chunk_idx, chunk_info, chunk_embedding_cache, chunk_accuracies, importance_args,
    )
    cf_kl = calculate_counterfactual_importance_kl(
        chunk_idx, chunk_info, chunk_embedding_cache, chunk_accuracies, chunk_answers, importance_args,
    )
    metrics.update({
        "counterfactual_importance_accuracy": cf_acc,
        "counterfactual_importance_kl": cf_kl,
        "different_trajectories_fraction": dtf,
        "overdeterminedness": od,
    })

    # Resampling
    rs_acc = calculate_resampling_importance_accuracy(chunk_idx, chunk_accuracies, importance_args)
    rs_kl = (
        calculate_resampling_importance_kl(chunk_idx, chunk_info, problem_dir, importance_args)
        if problem_dir else 0.0
    )
    metrics.update({
        "resampling_importance_accuracy": rs_acc,
        "resampling_importance_kl": rs_kl,
    })

    # Forced
    if forced_answer_accuracies and importance_args.forced_answer_dir:
        forced_acc = calculate_forced_importance_accuracy(
            chunk_idx, forced_answer_accuracies, importance_args,
        )
        forced_kl = (
            calculate_forced_importance_kl(
                chunk_idx, forced_answer_accuracies, problem_dir,
                importance_args.forced_answer_dir, importance_args,
            )
            if problem_dir else 0.0
        )
        metrics.update({
            "forced_importance_accuracy": forced_acc,
            "forced_importance_kl": forced_kl,
        })

    return chunk_idx, metrics


# ---------------------------------------------------------------------------
# GPT-4o Labeling
# ---------------------------------------------------------------------------

def label_chunks_with_dag(problem_text: str, chunks: List[str], language: str) -> Dict:
    """Label all chunks using GPT-4o with the DAG prompt."""
    formatted_prompt = build_dag_labeling_prompt(problem_text, chunks, language)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error labeling chunks: {e}")
        return {}


def generate_chunk_summary(chunk_text: str) -> str:
    """Generate a 2-4 word summary of a chunk using GPT-4o-mini."""
    prompt = f"""Please provide a 2-4 word maximum summary of what specifically happens in this text chunk. Focus on the concrete action or calculation.

    Text chunk:
    {chunk_text}

    Summary (2-4 words max):
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        summary = response.choices[0].message.content.strip()
        words = summary.split()
        if len(words) > 5:
            summary = " ".join(words[:5])
        return summary.replace('"', "")
    except Exception as e:
        print(f"Error generating chunk summary: {e}")
        return "unknown action"


def generate_problem_nickname(problem_text: str) -> str:
    """Generate a 2-4 word nickname for a problem using GPT-4o-mini."""
    prompt = f"""Please provide a 2-4 word maximum nickname for this math problem that captures its essence.

    Problem:
    {problem_text}

    Nickname (2-4 words max):
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        nickname = response.choices[0].message.content.strip()
        words = nickname.split()
        if len(words) > 5:
            nickname = " ".join(words[:5])
        return nickname.replace('"', "")
    except Exception as e:
        print(f"Error generating problem nickname: {e}")
        return "math problem"


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_problem(
    problem_dir: Path,
    dataset: str,
    language: str,
    force_relabel: bool = False,
    forced_answer_dir: Optional[Path] = None,
    use_existing_metrics: bool = False,
    sentence_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold: float = 0.8,
    generate_metadata: bool = False,
    force_metadata: bool = False,
) -> Optional[Dict]:
    """Analyze a single problem's rollouts and compute importance metrics."""
    base_solution_file = problem_dir / "base_solution.json"
    chunks_file = problem_dir / "chunks.json"
    problem_file = problem_dir / "problem.json"

    if not (base_solution_file.exists() and chunks_file.exists() and problem_file.exists()):
        print(f"Problem {problem_dir.name}: Missing required files")
        return None

    # Load problem
    with open(problem_file, "r", encoding="utf-8") as f:
        problem_data = json.load(f)

    # Generate nickname if metadata is enabled
    if generate_metadata and (force_metadata or "nickname" not in problem_data):
        question = problem_data.get("question", problem_data.get("problem", ""))
        problem_data["nickname"] = generate_problem_nickname(question)
        with open(problem_file, "w", encoding="utf-8") as f:
            json.dump(problem_data, f, indent=2, ensure_ascii=False)

    # Load chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = chunks_data["chunks"]

    # Filter short chunks
    valid_chunks = []
    valid_chunk_indices = []
    for i, chunk in enumerate(chunks):
        if len(chunk) >= 4:
            valid_chunks.append(chunk)
            valid_chunk_indices.append(i)

    if len(valid_chunks) < len(chunks):
        print(f"Problem {problem_dir.name}: Filtered {len(chunks) - len(valid_chunks)} short chunks")
        chunks = valid_chunks

    # Check chunk folders exist
    chunk_folders = [problem_dir / f"chunk_{i}" for i in valid_chunk_indices]
    existing_folders = [f for f in chunk_folders if f.exists()]
    if len(existing_folders) < 0.1 * len(chunks):
        print(f"Problem {problem_dir.name}: Only {len(existing_folders)}/{len(chunks)} chunk folders")
        return None

    # Pre-calculate accuracies, answers, info
    chunk_accuracies = {}
    chunk_answers = {}
    chunk_info = {}

    # Load forced answer accuracies
    forced_answer_accuracies = {}
    if forced_answer_dir:
        forced_problem_dir = forced_answer_dir / problem_dir.name
        if forced_problem_dir.exists():
            for chunk_idx in valid_chunk_indices:
                chunk_dir = forced_problem_dir / f"chunk_{chunk_idx}"
                if chunk_dir.exists():
                    solutions_file = chunk_dir / "solutions.json"
                    if solutions_file.exists():
                        try:
                            with open(solutions_file, "r", encoding="utf-8") as f:
                                solutions = json.load(f)
                            correct = sum(1 for s in solutions if s.get("is_correct") is True and s.get("answer", "") != "")
                            total = sum(1 for s in solutions if s.get("is_correct") is not None and s.get("answer", "") != "")
                            forced_answer_accuracies[chunk_idx] = correct / total if total > 0 else 0.0
                        except Exception:
                            forced_answer_accuracies[chunk_idx] = 0.0

    for chunk_idx in valid_chunk_indices:
        solutions_file = problem_dir / f"chunk_{chunk_idx}" / "solutions.json"
        if solutions_file.exists():
            with open(solutions_file, "r", encoding="utf-8") as f:
                solutions = json.load(f)

            correct = sum(1 for s in solutions if s.get("is_correct") is True and s.get("answer", "") != "")
            total = sum(1 for s in solutions if s.get("is_correct") is not None and s.get("answer", "") != "")
            chunk_accuracies[chunk_idx] = correct / total if total > 0 else 0.0

            chunk_answers[chunk_idx] = defaultdict(int)
            chunk_info[chunk_idx] = []
            for sol in solutions:
                if sol.get("answer", "") not in ("", "None"):
                    chunk_answers[chunk_idx][normalize_answer(sol.get("answer", ""))] += 1
                    chunk_info[chunk_idx].append({
                        "chunk_removed": sol.get("chunk_removed", ""),
                        "chunk_resampled": sol.get("chunk_resampled", ""),
                        "full_cot": sol.get("full_cot", ""),
                        "is_correct": sol.get("is_correct", False),
                        "answer": normalize_answer(sol.get("answer", "")),
                    })

    # Initialize embedding model
    global _embedding_model_cache
    if sentence_model not in _embedding_model_cache:
        _embedding_model_cache[sentence_model] = SentenceTransformer(sentence_model)
    embedding_model = _embedding_model_cache[sentence_model]

    # Compute embeddings for all unique chunks
    all_chunks_to_embed = set()
    for solutions in chunk_info.values():
        for sol in solutions:
            if isinstance(sol.get("chunk_removed", ""), str):
                all_chunks_to_embed.add(sol["chunk_removed"])
            if isinstance(sol.get("chunk_resampled", ""), str):
                all_chunks_to_embed.add(sol["chunk_resampled"])

    chunk_embedding_cache = {}
    all_chunks_list = list(all_chunks_to_embed)
    batch_size = args.batch_size

    print(f"Problem {problem_dir.name}: Computing embeddings for {len(all_chunks_list)} unique chunks")
    for i in range(0, len(all_chunks_list), batch_size):
        batch = all_chunks_list[i : i + batch_size]
        try:
            batch_embeddings = embedding_model.encode(batch, batch_size=batch_size, show_progress_bar=False)
            for chunk, embedding in zip(batch, batch_embeddings):
                chunk_embedding_cache[chunk] = embedding
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mini_size = max(1, batch_size // 2)
                for j in range(0, len(batch), mini_size):
                    mini = batch[j : j + mini_size]
                    mini_embs = embedding_model.encode(mini, batch_size=mini_size, show_progress_bar=False)
                    for c, e in zip(mini, mini_embs):
                        chunk_embedding_cache[c] = e
            else:
                raise

    # Load or create labeled chunks
    labeled_chunks_file = problem_dir / "chunks_labeled.json"
    labeled_chunks = [{"chunk_idx": i} for i in valid_chunk_indices]

    if labeled_chunks_file.exists() and not force_relabel:
        with open(labeled_chunks_file, "r", encoding="utf-8") as f:
            labeled_chunks = json.load(f)
        labeled_chunks = [c for c in labeled_chunks if c.get("chunk_idx") in valid_chunk_indices]

        # Generate summaries if metadata enabled
        if generate_metadata:
            for chunk in labeled_chunks:
                if force_metadata or "summary" not in chunk:
                    chunk_text = chunk.get("chunk", "")
                    if chunk_text:
                        chunk["summary"] = generate_chunk_summary(chunk_text)
    else:
        # Label with GPT-4o DAG prompt
        question = problem_data.get("question", problem_data.get("problem", ""))
        print(f"Problem {problem_dir.name}: Labeling {len(chunks)} chunks with GPT-4o")

        dag_result = label_chunks_with_dag(question, chunks, language)

        labeled_chunks = []
        for i, chunk_idx in enumerate(valid_chunk_indices):
            chunk_data = {
                "chunk": chunks[i],
                "chunk_idx": chunk_idx,
            }
            chunk_key = str(i)
            if chunk_key in dag_result:
                chunk_data["function_tags"] = dag_result[chunk_key].get("function_tags", ["unknown"])
                chunk_data["depends_on"] = dag_result[chunk_key].get("depends_on", [])
            else:
                chunk_data["function_tags"] = ["unknown"]
                chunk_data["depends_on"] = []

            if generate_metadata:
                chunk_data["summary"] = generate_chunk_summary(chunks[i])

            labeled_chunks.append(chunk_data)

    # Calculate importance metrics
    if not use_existing_metrics:
        print(f"Problem {problem_dir.name}: Calculating importance for {len(labeled_chunks)} chunks")

        importance_args = ImportanceArgs(
            use_absolute=args.absolute,
            forced_answer_dir=Path(args.forced_answer_dir) if args.forced_answer_dir else None,
            similarity_threshold=similarity_threshold,
            use_similar_chunks=args.use_similar_chunks,
            use_abs_importance=args.absolute,
            top_chunks=args.top_chunks,
            use_prob_true=args.use_prob_true,
            laplace_alpha=args.laplace_alpha,
        )

        for chunk in labeled_chunks:
            cidx = chunk["chunk_idx"]
            _, metrics = process_chunk_importance(
                cidx,
                chunk_info=chunk_info,
                chunk_embedding_cache=chunk_embedding_cache,
                chunk_accuracies=chunk_accuracies,
                importance_args=importance_args,
                problem_dir=problem_dir,
                forced_answer_accuracies=forced_answer_accuracies,
                chunk_answers=chunk_answers,
            )
            chunk["accuracy"] = chunk_accuracies.get(cidx, 0.0)
            chunk.update(metrics)

            # Reorder keys
            key_order = [
                "chunk", "chunk_idx", "function_tags", "depends_on", "accuracy",
                "resampling_importance_accuracy", "resampling_importance_kl",
                "counterfactual_importance_accuracy", "counterfactual_importance_kl",
                "forced_importance_accuracy", "forced_importance_kl",
                "different_trajectories_fraction", "overdeterminedness", "summary",
            ]
            ordered = {}
            for key in key_order:
                if key in chunk:
                    ordered[key] = chunk[key]
            for key, value in chunk.items():
                if key not in ordered:
                    ordered[key] = value
            chunk.clear()
            chunk.update(ordered)

    # Save labeled chunks
    with open(labeled_chunks_file, "w", encoding="utf-8") as f:
        json.dump(labeled_chunks, f, indent=2, ensure_ascii=False)

    print(f"Problem {problem_dir.name}: Saved {len(labeled_chunks)} labeled chunks")
    return {"problem_dir": str(problem_dir), "num_chunks": len(labeled_chunks)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    languages = [l.strip() for l in args.languages.split(",")]

    for language in languages:
        # Build rollouts directory path
        if args.rollouts_dir:
            rollouts_dir = Path(args.rollouts_dir)
        else:
            suffix_parts = [f"{args.base_solution_type}_base_solution"]
            if args.rollout_type == "forced_answer":
                suffix_parts.append(f"_{args.rollout_type}")
            if args.output_suffix:
                suffix_parts.append(f"_{args.output_suffix}")
            rollouts_dir = (
                Path("multicot") / args.dataset / args.model
                / f"temperature_{args.temperature}_top_p_{args.top_p}"
                / language
                / "".join(suffix_parts)
            )

        if not rollouts_dir.exists():
            print(f"[{language}] Rollouts directory not found: {rollouts_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing language: {language} from {rollouts_dir}")
        print(f"{'='*60}")

        # Find all problem directories
        problem_dirs = sorted([
            d for d in rollouts_dir.iterdir()
            if d.is_dir() and d.name.startswith("problem_")
        ])

        if args.max_problems:
            problem_dirs = problem_dirs[:args.max_problems]

        print(f"Found {len(problem_dirs)} problems")

        forced_answer_dir = Path(args.forced_answer_dir) if args.forced_answer_dir else None

        for problem_dir in tqdm(problem_dirs, desc=f"[{language}] Analyzing"):
            analyze_problem(
                problem_dir=problem_dir,
                dataset=args.dataset,
                language=language,
                force_relabel=args.force_relabel,
                forced_answer_dir=forced_answer_dir,
                use_existing_metrics=args.use_existing_metrics,
                sentence_model=args.sentence_model,
                similarity_threshold=args.similarity_threshold,
                generate_metadata=args.generate_metadata,
                force_metadata=args.force_metadata,
            )


if __name__ == "__main__":
    main()
