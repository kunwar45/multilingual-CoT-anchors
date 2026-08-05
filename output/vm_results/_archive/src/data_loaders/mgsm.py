"""
MGSM (Multilingual Grade School Math) Dataset Loader

Provides stable problem_ids across languages for controlled experiments.
MGSM contains 250 problems translated into 10 languages.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json

from datasets import load_dataset


# MGSM language codes
MGSM_LANGUAGES = [
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "ru",  # Russian
    "zh",  # Chinese
    "ja",  # Japanese
    "th",  # Thai
    "sw",  # Swahili
    "bn",  # Bengali
    "te",  # Telugu
]

# Human-readable names
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "th": "Thai",
    "sw": "Swahili",
    "bn": "Bengali",
    "te": "Telugu",
}


@dataclass
class MGSMProblem:
    """A single MGSM problem with stable ID across languages."""

    problem_id: str  # Stable ID: "mgsm_{idx}" where idx is 0-249
    language: str  # Language code (e.g., "en", "es", "fr")
    question: str  # The problem text in the target language
    answer: str  # Numeric answer (same across all languages)
    answer_number: float  # Parsed numeric value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MGSMProblem":
        return cls(**d)


class MGSMLoader:
    """
    Loader for MGSM dataset with stable problem IDs.

    MGSM problems are indexed 0-249, and we create stable IDs as "mgsm_{idx}".
    This ensures the same problem can be compared across languages.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the MGSM loader.

        Args:
            cache_dir: Optional directory for caching the dataset
        """
        self.cache_dir = cache_dir
        self._cache: dict[str, list[MGSMProblem]] = {}

    def load_language(self, language: str) -> list[MGSMProblem]:
        """
        Load all MGSM problems for a specific language.

        Args:
            language: Language code (e.g., "en", "es", "fr")

        Returns:
            List of MGSMProblem objects with stable problem_ids
        """
        if language not in MGSM_LANGUAGES:
            raise ValueError(
                f"Unknown language: {language}. "
                f"Available: {MGSM_LANGUAGES}"
            )

        if language in self._cache:
            return self._cache[language]

        # Load from HuggingFace
        dataset = load_dataset(
            "juletxara/mgsm",
            language,
            split="test",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        problems = []
        for idx, row in enumerate(dataset):
            # Parse the numeric answer
            answer_str = row["answer_number"]
            if isinstance(answer_str, str):
                # Handle comma-separated numbers
                answer_str = answer_str.replace(",", "")
                answer_number = float(answer_str)
            else:
                answer_number = float(answer_str)

            problem = MGSMProblem(
                problem_id=f"mgsm_{idx:03d}",
                language=language,
                question=row["question"],
                answer=str(row["answer_number"]),
                answer_number=answer_number,
            )
            problems.append(problem)

        self._cache[language] = problems
        return problems

    def load_problem(self, problem_id: str, language: str) -> MGSMProblem:
        """
        Load a specific problem by ID and language.

        Args:
            problem_id: Problem ID (e.g., "mgsm_042")
            language: Language code

        Returns:
            The MGSMProblem object
        """
        problems = self.load_language(language)

        # Extract index from problem_id
        idx = int(problem_id.split("_")[1])
        if idx < 0 or idx >= len(problems):
            raise ValueError(f"Invalid problem_id: {problem_id}")

        return problems[idx]

    def load_parallel(
        self,
        problem_ids: list[str],
        languages: list[str]
    ) -> dict[str, dict[str, MGSMProblem]]:
        """
        Load problems in parallel across multiple languages.

        Args:
            problem_ids: List of problem IDs to load
            languages: List of language codes

        Returns:
            Nested dict: {problem_id: {language: MGSMProblem}}
        """
        result = {}
        for problem_id in problem_ids:
            result[problem_id] = {}
            for lang in languages:
                result[problem_id][lang] = self.load_problem(problem_id, lang)
        return result

    def get_problem_ids(self, n: Optional[int] = None) -> list[str]:
        """
        Get list of all problem IDs (or first n).

        Args:
            n: Optional limit on number of IDs to return

        Returns:
            List of problem IDs
        """
        total = 250  # MGSM has 250 problems
        if n is not None:
            total = min(n, total)
        return [f"mgsm_{i:03d}" for i in range(total)]

    def export_subset(
        self,
        output_path: Path,
        problem_ids: list[str],
        language: str,
    ) -> None:
        """
        Export a subset of problems to JSONL for reproducibility.

        Args:
            output_path: Path to write JSONL file
            problem_ids: List of problem IDs to export
            language: Language code
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for problem_id in problem_ids:
                problem = self.load_problem(problem_id, language)
                f.write(json.dumps(problem.to_dict(), ensure_ascii=False) + "\n")

    def load_from_jsonl(self, path: Path) -> list[MGSMProblem]:
        """
        Load problems from a JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of MGSMProblem objects
        """
        problems = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problems.append(MGSMProblem.from_dict(json.loads(line)))
        return problems


def select_pilot_problems(
    loader: MGSMLoader,
    n: int = 20,
    seed: int = 42,
) -> list[str]:
    """
    Select a random subset of problem IDs for pilot experiments.

    Args:
        loader: MGSMLoader instance
        n: Number of problems to select
        seed: Random seed for reproducibility

    Returns:
        List of problem IDs
    """
    import random

    all_ids = loader.get_problem_ids()
    rng = random.Random(seed)
    return rng.sample(all_ids, min(n, len(all_ids)))
