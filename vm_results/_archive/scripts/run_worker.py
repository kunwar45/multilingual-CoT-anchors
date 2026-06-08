#!/usr/bin/env python3
"""
Worker for parallel rollout generation.

Claims tasks from a manifest and generates rollouts. Multiple workers
can run in parallel, each claiming and completing tasks atomically.

Usage:
    # Run a single worker
    python scripts/run_worker.py --manifest runs/r1/r1_manifest.json

    # Run with specific worker ID
    python scripts/run_worker.py --manifest runs/r1/r1_manifest.json --worker-id worker_0

    # Run specific number of tasks then exit
    python scripts/run_worker.py --manifest runs/r1/r1_manifest.json --max-tasks 100

    # Dry run (claim but don't execute)
    python scripts/run_worker.py --manifest runs/r1/r1_manifest.json --dry-run
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders import MGSMLoader
from prompts import (
    CONDITION_NATIVE,
    CONDITION_ENGLISH_BASELINE,
    CONDITION_ENGLISH_THINKING,
    CONDITION_TRANSLATE_SOLVE,
)
from rollouts import RolloutGenerator, RolloutConfig, Rollout
from rollouts.manifest import ManifestManager, TaskStatus
from rollouts.parser import parse_final_answer
from config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


CONDITIONS = {
    "native": CONDITION_NATIVE,
    "english_baseline": CONDITION_ENGLISH_BASELINE,
    "english_thinking": CONDITION_ENGLISH_THINKING,
    "translate_solve": CONDITION_TRANSLATE_SOLVE,
}


def process_task(
    task,
    config: ExperimentConfig,
    loader: MGSMLoader,
    generator: RolloutGenerator,
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Process a single task.

    Args:
        task: JobTask to process
        config: Experiment configuration
        loader: MGSM data loader
        generator: Rollout generator
        output_dir: Directory for rollout output
        dry_run: If True, skip actual generation

    Returns:
        Tuple of (success, output_path or error_message)
    """
    from prompts.templates import build_cot_prompt
    from datetime import datetime

    try:
        # Load problem
        problem = loader.load_problem(task.problem_id, task.language)
        if problem is None:
            return False, f"Problem not found: {task.problem_id} ({task.language})"

        # Get condition
        condition = CONDITIONS.get(task.condition)
        if condition is None:
            return False, f"Unknown condition: {task.condition}"

        if dry_run:
            logger.info(f"[DRY RUN] Would generate rollout for {task.problem_id} ({task.language}, {task.condition})")
            return True, "dry_run"

        # Build prompt
        prompt = build_cot_prompt(
            question=problem.question,
            problem_lang=problem.language,
            condition=condition,
        )

        # Generate
        import time
        import torch

        torch.manual_seed(task.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(task.seed)

        start_time = time.time()

        # Use the generator's internal method
        cot_text, generation_time_ms = generator._generate_single(prompt, task.seed)

        # Parse answer
        final_answer = parse_final_answer(cot_text, language=task.language)
        correct = None
        if final_answer is not None:
            try:
                correct = float(final_answer) == problem.answer_number
            except ValueError:
                correct = final_answer.strip() == problem.answer.strip()

        # Create rollout object
        rollout = Rollout(
            problem_id=task.problem_id,
            language=task.language,
            condition=task.condition,
            rollout_idx=task.rollout_idx,
            prompt=prompt,
            cot_text=cot_text,
            final_answer_text=final_answer,
            correct=correct,
            ground_truth=problem.answer,
            seed=task.seed,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            max_new_tokens=config.generation.max_new_tokens,
            model_name=config.model.name,
            timestamp=datetime.now().isoformat(),
            generation_time_ms=generation_time_ms,
        )

        # Save rollout
        output_path = output_dir / f"{task.language}_{task.condition}.jsonl"
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rollout.to_dict(), ensure_ascii=False) + "\n")

        return True, str(output_path)

    except Exception as e:
        logger.exception(f"Error processing task {task.task_id}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Worker for parallel rollout generation")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSON file")
    parser.add_argument("--worker-id", help="Worker identifier (default: auto-generated)")
    parser.add_argument("--max-tasks", type=int, help="Maximum tasks to process before exiting")
    parser.add_argument("--dry-run", action="store_true", help="Claim tasks but don't execute")
    parser.add_argument("--reset-stale", action="store_true", help="Reset stale in-progress tasks")
    parser.add_argument("--stale-hours", type=float, default=2.0, help="Hours before task is considered stale")
    args = parser.parse_args()

    # Generate worker ID
    worker_id = args.worker_id or f"worker_{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting worker: {worker_id}")

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manager = ManifestManager(manifest_path)
    manifest = manager.get_manifest()

    logger.info(f"Loaded manifest: {manifest.run_name}")
    logger.info(f"  Total tasks: {manifest.total_tasks}")
    logger.info(f"  Pending: {manifest.pending_tasks}")
    logger.info(f"  In progress: {manifest.in_progress_tasks}")
    logger.info(f"  Completed: {manifest.completed_tasks}")

    # Reset stale tasks if requested
    if args.reset_stale:
        reset_count = manager.reset_stale_tasks(max_age_hours=args.stale_hours)
        logger.info(f"Reset {reset_count} stale tasks")

    # Load configuration from manifest
    config = ExperimentConfig.from_dict(manifest.config_snapshot)
    output_dir = manifest_path.parent / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader and generator
    loader = MGSMLoader()

    rollout_config = config.to_rollout_config()
    generator = RolloutGenerator(rollout_config, output_dir)

    # Process tasks
    tasks_completed = 0
    tasks_failed = 0

    while True:
        # Check max tasks limit
        if args.max_tasks and tasks_completed + tasks_failed >= args.max_tasks:
            logger.info(f"Reached max tasks limit ({args.max_tasks})")
            break

        # Claim next task
        task = manager.claim_next_task(worker_id)
        if task is None:
            logger.info("No more pending tasks")
            break

        logger.info(f"Processing task {task.task_id}: {task.problem_id} ({task.language}, {task.condition}, rollout {task.rollout_idx})")

        # Process task
        success, result = process_task(
            task=task,
            config=config,
            loader=loader,
            generator=generator,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )

        # Mark task complete
        manager.complete_task(
            task_id=task.task_id,
            success=success,
            output_path=result if success else None,
            error_message=result if not success else None,
        )

        if success:
            tasks_completed += 1
        else:
            tasks_failed += 1
            logger.error(f"Task {task.task_id} failed: {result}")

        # Log progress
        progress = manager.get_progress()
        logger.info(
            f"Progress: {progress['completed']}/{progress['total']} "
            f"({progress['percent_complete']:.1f}%) "
            f"[worker completed: {tasks_completed}, failed: {tasks_failed}]"
        )

    # Final summary
    logger.info("-" * 50)
    logger.info(f"Worker {worker_id} finished")
    logger.info(f"  Tasks completed: {tasks_completed}")
    logger.info(f"  Tasks failed: {tasks_failed}")

    final_progress = manager.get_progress()
    logger.info(f"Overall progress: {final_progress['completed']}/{final_progress['total']} ({final_progress['percent_complete']:.1f}%)")


if __name__ == "__main__":
    main()
