"""
Unified Dataset Interface for Multilingual CoT Experiments

Provides loaders for MGSM and MMATH datasets with stable cross-language
problem IDs and a unified Problem dataclass.
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def _hf_token():
    """Hugging Face token from HF_TOKEN or HUGGING_FACE_HUB_TOKEN (e.g. from .env)."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or True


def _load_dataset_from_hub(path: str, name: Optional[str], split: str, cache_dir: Optional[str], token, **kwargs):
    """Load dataset from Hugging Face; if cache is missing the config, download from Hub."""
    try:
        return load_dataset(
            path,
            name,
            split=split,
            cache_dir=cache_dir,
            token=token,
            **kwargs,
        )
    except ValueError as e:
        if "Couldn't find cache" in str(e) and "Available configs" in str(e):
            # Cache exists for other configs but not this one — download from Hub
            return load_dataset(
                path,
                name,
                split=split,
                cache_dir=cache_dir,
                token=token,
                download_mode="force_redownload",
                **kwargs,
            )
        raise


SUPPORTED_LANGUAGES = ["en", "fr", "zh", "ar", "de", "es", "hi", "bn", "id", "it", "ja", "ko", "pt", "sw", "yo"]

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "zh": "Chinese",
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "hi": "Hindi",
    "bn": "Bengali",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "sw": "Swahili",
    "yo": "Yoruba",
}


@dataclass
class Problem:
    """A single problem with stable ID across languages."""

    problem_id: str       # "mgsm_042" or mmath gid
    dataset: str          # "mgsm" or "mmath"
    language: str         # "en", "fr", "zh", "ar"
    question: str
    gt_answer: str        # numeric string (MGSM) or LaTeX content (MMATH)
    answer_type: str      # "numeric" or "latex_boxed"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Problem":
        return cls(**d)


class MGSMLoader:
    """
    Loader for MGSM dataset with stable problem IDs.

    MGSM problems are indexed 0-249, and we create stable IDs as "mgsm_{idx:03d}".
    This ensures the same problem can be compared across languages.
    """

    # MGSM languages that overlap with our supported set
    MGSM_LANGUAGES = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self._cache: dict[str, list[Problem]] = {}

    def load_language(self, language: str) -> list[Problem]:
        """Load all MGSM problems for a specific language."""
        if language not in self.MGSM_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not available in MGSM. "
                f"Available: {self.MGSM_LANGUAGES}"
            )

        if language in self._cache:
            return self._cache[language]

        dataset = _load_dataset_from_hub(
            "juletxara/mgsm",
            language,
            split="test",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            token=_hf_token(),
        )

        problems = []
        for idx, row in enumerate(dataset):
            answer_str = row["answer_number"]
            if isinstance(answer_str, str):
                answer_str = answer_str.replace(",", "")

            problem = Problem(
                problem_id=f"mgsm_{idx:03d}",
                dataset="mgsm",
                language=language,
                question=row["question"],
                gt_answer=str(answer_str),
                answer_type="numeric",
            )
            problems.append(problem)

        self._cache[language] = problems
        return problems

    def load_parallel(
        self,
        languages: list[str],
        problem_ids: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Problem]]:
        """
        Load problems across multiple languages.

        Returns:
            Nested dict: {problem_id: {language: Problem}}
        """
        # Load all languages
        all_problems = {}
        for lang in languages:
            for p in self.load_language(lang):
                all_problems.setdefault(p.problem_id, {})[lang] = p

        # Filter to requested problem_ids if given
        if problem_ids is not None:
            all_problems = {
                pid: langs for pid, langs in all_problems.items()
                if pid in problem_ids
            }

        return all_problems

    def get_problem_ids(self, n: Optional[int] = None) -> list[str]:
        """Get list of all problem IDs (or first n)."""
        total = 250  # MGSM has 250 problems
        if n is not None:
            total = min(n, total)
        return [f"mgsm_{i:03d}" for i in range(total)]


class MMATHLoader:
    """
    Loader for MMATH dataset from local JSON files.

    Cross-language matching via 'gid' field in each problem.
    Expects files named like: {language}.json or mmath_{language}.json
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._cache: dict[str, list[Problem]] = {}

    def _find_language_file(self, language: str) -> Path:
        """Find the JSON file for a given language."""
        candidates = [
            self.data_dir / f"{language}.json",
            self.data_dir / f"mmath_{language}.json",
            self.data_dir / f"MMATH_{language}.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No MMATH data file found for language '{language}' in {self.data_dir}. "
            f"Tried: {[c.name for c in candidates]}"
        )

    def load_language(self, language: str) -> list[Problem]:
        """Load all MMATH problems for a specific language."""
        if language in self._cache:
            return self._cache[language]

        path = self._find_language_file(language)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            items = list(data.values())
        else:
            items = data

        problems = []
        for item in items:
            gid = str(item.get("gid", item.get("id", item.get("problem_id", ""))))
            question = item.get("question", item.get("problem", ""))
            answer = item.get("answer", item.get("gt_answer", ""))

            problem = Problem(
                problem_id=f"mmath_{gid}",
                dataset="mmath",
                language=language,
                question=question,
                gt_answer=str(answer),
                answer_type="latex_boxed",
            )
            problems.append(problem)

        self._cache[language] = problems
        return problems

    def load_parallel(
        self,
        languages: list[str],
        problem_ids: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Problem]]:
        """
        Load problems across multiple languages.

        Returns:
            Nested dict: {problem_id: {language: Problem}}
        """
        all_problems = {}
        for lang in languages:
            for p in self.load_language(lang):
                all_problems.setdefault(p.problem_id, {})[lang] = p

        if problem_ids is not None:
            all_problems = {
                pid: langs for pid, langs in all_problems.items()
                if pid in problem_ids
            }

        return all_problems


class MMMMLULoader:
    """
    Loader for the MMMLU dataset (openai/MMMLU on HuggingFace).

    14,042 multiple-choice questions from MMLU, professionally translated into
    14 languages. Cross-language alignment is by row index (all configs share
    the same fixed ordering of MMLU problems).

    Languages: ar, bn, de, es, fr, hi, id, it, ja, ko, pt, sw, yo, zh
    English is the source language of MMLU and has no translated config;
    use --dataset mgsm or mmath for English experiments.
    """

    # Map our 2-letter codes to the HuggingFace config names
    LANG_TO_CONFIG = {
        "ar": "AR_XY",
        "bn": "BN_BD",
        "de": "DE_DE",
        "es": "ES_LA",
        "fr": "FR_FR",
        "hi": "HI_IN",
        "id": "ID_ID",
        "it": "IT_IT",
        "ja": "JA_JP",
        "ko": "KO_KR",
        "pt": "PT_BR",
        "sw": "SW_KE",
        "yo": "YO_NG",
        "zh": "ZH_CN",
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self._cache: dict[str, list[Problem]] = {}

    def load_language(self, language: str) -> list[Problem]:
        """Load all MMMLU problems for a specific language."""
        if language not in self.LANG_TO_CONFIG:
            raise ValueError(
                f"Language '{language}' not available in MMMLU. "
                f"Available: {sorted(self.LANG_TO_CONFIG.keys())}"
            )

        if language in self._cache:
            return self._cache[language]

        config = self.LANG_TO_CONFIG[language]
        dataset = _load_dataset_from_hub(
            "openai/MMMLU",
            config,
            split="test",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            token=_hf_token(),
        )

        problems = []
        for idx, row in enumerate(dataset):
            # Bake the four options into the question text — standard MMLU format
            question = (
                f"{row['Question']}\n\n"
                f"A. {row['A']}\n"
                f"B. {row['B']}\n"
                f"C. {row['C']}\n"
                f"D. {row['D']}"
            )
            problem = Problem(
                problem_id=f"mmmlu_{idx:05d}",
                dataset="mmmlu",
                language=language,
                question=question,
                gt_answer=str(row["Answer"]).strip().upper(),
                answer_type="multiple_choice",
            )
            problems.append(problem)

        self._cache[language] = problems
        return problems

    def load_parallel(
        self,
        languages: list[str],
        problem_ids: Optional[list[str]] = None,
    ) -> dict[str, dict[str, Problem]]:
        """
        Load problems across multiple languages.

        Returns:
            Nested dict: {problem_id: {language: Problem}}
        """
        all_problems = {}
        for lang in languages:
            for p in self.load_language(lang):
                all_problems.setdefault(p.problem_id, {})[lang] = p

        if problem_ids is not None:
            all_problems = {
                pid: langs for pid, langs in all_problems.items()
                if pid in problem_ids
            }

        return all_problems


def load_dataset_problems(
    dataset: str,
    languages: list[str],
    num_problems: Optional[int] = None,
    data_dir: Optional[Path] = None,
    problem_ids: Optional[list[str]] = None,
) -> dict[str, dict[str, Problem]]:
    """
    Unified entry point for loading problems.

    Returns only problems available in ALL requested languages.

    Args:
        dataset: "mgsm", "mmath", or "mmmlu"
        languages: List of language codes
        num_problems: Optional limit on number of problems
        data_dir: Path to MMATH data directory (required for mmath)
        problem_ids: Optional list of specific problem IDs to load

    Returns:
        Nested dict: {problem_id: {language: Problem}}
    """
    if dataset == "mgsm":
        loader = MGSMLoader()
    elif dataset == "mmath":
        if data_dir is None:
            raise ValueError("--data_dir is required for MMATH dataset")
        loader = MMATHLoader(data_dir=Path(data_dir))
    elif dataset == "mmmlu":
        loader = MMMMLULoader()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mgsm', 'mmath', or 'mmmlu'.")

    parallel = loader.load_parallel(languages, problem_ids=problem_ids)

    # Keep only problems available in ALL requested languages
    complete = {
        pid: langs for pid, langs in parallel.items()
        if all(lang in langs for lang in languages)
    }

    # Sort by problem_id for deterministic ordering
    sorted_ids = sorted(complete.keys())

    # Limit number of problems
    if num_problems is not None:
        sorted_ids = sorted_ids[:num_problems]

    return {pid: complete[pid] for pid in sorted_ids}
