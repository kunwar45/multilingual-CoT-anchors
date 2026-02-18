"""
Rollout Generator with Resume-Safe JSONL Writing

Generates chain-of-thought rollouts with:
- Reproducible seeds
- Resume capability (skips already-generated rollouts)
- JSONL output format
- Configurable decoding parameters
"""

import json
import hashlib
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Callable
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..data_loaders.mgsm import MGSMProblem
from ..prompts.templates import LanguageCondition, build_cot_prompt, build_system_prompt

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    """Configuration for rollout generation."""

    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1024
    num_rollouts: int = 5
    base_seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"

    # API settings (if using API instead of local)
    use_api: bool = False
    api_provider: Optional[str] = None  # "openrouter", "together", "fireworks"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RolloutConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Rollout:
    """A single chain-of-thought rollout."""

    problem_id: str
    language: str
    condition: str  # Name of the LanguageCondition
    rollout_idx: int
    prompt: str
    cot_text: str  # Full generated text
    final_answer_text: Optional[str]  # Extracted answer (may be None if parsing fails)
    correct: Optional[bool]  # Whether answer matches ground truth
    ground_truth: str

    # Metadata
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    model_name: str
    timestamp: str
    generation_time_ms: int

    # Unique ID for deduplication
    rollout_id: str = field(default="")

    def __post_init__(self):
        if not self.rollout_id:
            # Create deterministic ID from key fields
            key = f"{self.problem_id}:{self.language}:{self.condition}:{self.rollout_idx}:{self.seed}"
            self.rollout_id = hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Rollout":
        return cls(**d)


class RolloutGenerator:
    """
    Generates chain-of-thought rollouts with resume capability.

    Writes to JSONL format with one rollout per line.
    Skips rollouts that already exist in the output file.
    """

    def __init__(
        self,
        config: RolloutConfig,
        output_dir: Path,
    ):
        """
        Initialize the rollout generator.

        Args:
            config: RolloutConfig with model and generation settings
            output_dir: Directory to write rollout JSONL files
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self._existing_ids: set[str] = set()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        if self.model is not None:
            return

        if self.config.use_api:
            logger.info(f"Using API provider: {self.config.api_provider}")
            return

        logger.info(f"Loading model: {self.config.model_name}")

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        # Determine device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=device if device == "auto" else None,
            trust_remote_code=True,
        )

        if device != "auto" and device != "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        logger.info(f"Model loaded on {device}")

    def _get_output_path(self, run_name: str) -> Path:
        """Get the output JSONL path for a run."""
        return self.output_dir / f"{run_name}.jsonl"

    def _load_existing_ids(self, output_path: Path) -> set[str]:
        """Load IDs of already-generated rollouts for resume capability."""
        existing_ids = set()
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rollout = json.loads(line)
                            existing_ids.add(rollout.get("rollout_id", ""))
                        except json.JSONDecodeError:
                            continue
        return existing_ids

    def _append_rollout(self, output_path: Path, rollout: Rollout) -> None:
        """Append a rollout to the JSONL file."""
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rollout.to_dict(), ensure_ascii=False) + "\n")

    def _generate_single(
        self,
        prompt: str,
        seed: int,
    ) -> tuple[str, int]:
        """
        Generate a single completion.

        Args:
            prompt: The input prompt
            seed: Random seed

        Returns:
            Tuple of (generated_text, generation_time_ms)
        """
        import time

        if self.config.use_api:
            return self._generate_api(prompt, seed)

        self._load_model()

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        start_time = time.time()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (exclude prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        generation_time_ms = int((time.time() - start_time) * 1000)

        return generated_text, generation_time_ms

    def _generate_api(self, prompt: str, seed: int) -> tuple[str, int]:
        """Generate using an API provider."""
        import time
        import httpx

        start_time = time.time()

        if self.config.api_provider == "openrouter":
            url = self.config.api_base_url or "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_new_tokens,
                "seed": seed,
            }

            response = httpx.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]

        elif self.config.api_provider == "together":
            url = self.config.api_base_url or "https://api.together.xyz/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_new_tokens,
                "seed": seed,
            }

            response = httpx.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]

        else:
            raise ValueError(f"Unknown API provider: {self.config.api_provider}")

        generation_time_ms = int((time.time() - start_time) * 1000)
        return generated_text, generation_time_ms

    def generate_rollouts(
        self,
        problems: list[MGSMProblem],
        condition: LanguageCondition,
        run_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Generate rollouts for a list of problems.

        Args:
            problems: List of MGSMProblem objects
            condition: LanguageCondition defining (P_L, T_L, A_L)
            run_name: Name for this run (used in output filename)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Path to the output JSONL file
        """
        from .parser import parse_final_answer

        output_path = self._get_output_path(run_name)
        existing_ids = self._load_existing_ids(output_path)

        total = len(problems) * self.config.num_rollouts
        current = 0

        for problem in problems:
            prompt = build_cot_prompt(
                question=problem.question,
                problem_lang=problem.language,
                condition=condition,
            )

            for rollout_idx in range(self.config.num_rollouts):
                # Compute seed deterministically
                seed = self.config.base_seed + hash(f"{problem.problem_id}:{rollout_idx}") % (2**31)

                # Create rollout to check if it exists
                rollout = Rollout(
                    problem_id=problem.problem_id,
                    language=problem.language,
                    condition=condition.name,
                    rollout_idx=rollout_idx,
                    prompt=prompt,
                    cot_text="",  # Will be filled
                    final_answer_text=None,
                    correct=None,
                    ground_truth=problem.answer,
                    seed=seed,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_new_tokens=self.config.max_new_tokens,
                    model_name=self.config.model_name,
                    timestamp="",
                    generation_time_ms=0,
                )

                # Skip if already exists
                if rollout.rollout_id in existing_ids:
                    logger.debug(f"Skipping existing rollout: {rollout.rollout_id}")
                    current += 1
                    if progress_callback:
                        progress_callback(current, total)
                    continue

                # Generate
                cot_text, generation_time_ms = self._generate_single(prompt, seed)

                # Parse answer (with language-aware parsing)
                final_answer = parse_final_answer(cot_text, language=problem.language)
                correct = None
                if final_answer is not None:
                    try:
                        correct = float(final_answer) == problem.answer_number
                    except ValueError:
                        correct = final_answer.strip() == problem.answer.strip()

                # Update rollout with results
                rollout.cot_text = cot_text
                rollout.final_answer_text = final_answer
                rollout.correct = correct
                rollout.timestamp = datetime.now().isoformat()
                rollout.generation_time_ms = generation_time_ms

                # Write to file
                self._append_rollout(output_path, rollout)

                current += 1
                if progress_callback:
                    progress_callback(current, total)

        return output_path

    def load_rollouts(self, run_name: str) -> list[Rollout]:
        """Load rollouts from a run file."""
        output_path = self._get_output_path(run_name)
        rollouts = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rollouts.append(Rollout.from_dict(json.loads(line)))
        return rollouts


def create_run_manifest(
    run_name: str,
    config: RolloutConfig,
    condition: LanguageCondition,
    problem_ids: list[str],
    output_dir: Path,
) -> Path:
    """
    Create a manifest file for a run.

    The manifest records all parameters for reproducibility.

    Args:
        run_name: Name of the run
        config: RolloutConfig
        condition: LanguageCondition
        problem_ids: List of problem IDs included
        output_dir: Output directory

    Returns:
        Path to the manifest file
    """
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "config": config.to_dict(),
        "condition": {
            "name": condition.name,
            "problem_lang": condition.problem_lang,
            "thinking_lang": condition.thinking_lang,
            "answer_lang": condition.answer_lang,
            "description": condition.description,
        },
        "problem_ids": problem_ids,
        "num_problems": len(problem_ids),
        "total_rollouts": len(problem_ids) * config.num_rollouts,
    }

    manifest_path = output_dir / f"{run_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path
