"""
Manifest-based Job Management

Provides atomic task claiming for parallel rollout generation across
multiple workers. Uses file locking for coordination.

Key classes:
- JobTask: Single rollout generation task
- JobManifest: Collection of tasks with metadata
- ManifestManager: Thread-safe task claiming and completion
"""

import json
import hashlib
import fcntl
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Iterator
import logging

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a job task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobTask:
    """
    A single rollout generation task.

    Each task represents one rollout for one problem in one condition.
    """
    task_id: str
    problem_id: str
    language: str
    condition: str
    seed: int
    rollout_idx: int
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)

    @classmethod
    def create(
        cls,
        problem_id: str,
        language: str,
        condition: str,
        rollout_idx: int,
        base_seed: int,
    ) -> "JobTask":
        """Create a new task with deterministic ID and seed."""
        # Deterministic task ID
        key = f"{problem_id}:{language}:{condition}:{rollout_idx}"
        task_id = hashlib.md5(key.encode()).hexdigest()[:12]

        # Deterministic seed
        seed = base_seed + hash(f"{problem_id}:{rollout_idx}") % (2**31)

        return cls(
            task_id=task_id,
            problem_id=problem_id,
            language=language,
            condition=condition,
            seed=seed,
            rollout_idx=rollout_idx,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JobTask":
        return cls(**d)


@dataclass
class JobManifest:
    """
    Collection of job tasks with metadata.

    Tracks overall progress and configuration for a run.
    """
    run_name: str
    config_name: str
    created_at: str
    tasks: list[JobTask]

    # Statistics (computed)
    total_tasks: int = 0
    pending_tasks: int = 0
    in_progress_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    # Configuration snapshot
    config_snapshot: dict = field(default_factory=dict)

    def __post_init__(self):
        self._update_stats()

    def _update_stats(self):
        """Update task statistics."""
        self.total_tasks = len(self.tasks)
        self.pending_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)
        self.in_progress_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.IN_PROGRESS)
        self.completed_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        self.failed_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)

    def get_task(self, task_id: str) -> Optional[JobTask]:
        """Get task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_pending_tasks(self) -> list[JobTask]:
        """Get all pending tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_tasks_by_status(self, status: TaskStatus) -> list[JobTask]:
        """Get tasks with a specific status."""
        return [t for t in self.tasks if t.status == status]

    def to_dict(self) -> dict:
        self._update_stats()
        return {
            "run_name": self.run_name,
            "config_name": self.config_name,
            "created_at": self.created_at,
            "tasks": [t.to_dict() for t in self.tasks],
            "total_tasks": self.total_tasks,
            "pending_tasks": self.pending_tasks,
            "in_progress_tasks": self.in_progress_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "config_snapshot": self.config_snapshot,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JobManifest":
        tasks = [JobTask.from_dict(t) for t in d.get("tasks", [])]
        return cls(
            run_name=d["run_name"],
            config_name=d["config_name"],
            created_at=d["created_at"],
            tasks=tasks,
            config_snapshot=d.get("config_snapshot", {}),
        )


class ManifestManager:
    """
    Thread-safe manifest manager with file locking.

    Provides atomic task claiming and completion for parallel workers.
    """

    def __init__(self, manifest_path: Path):
        """
        Initialize the manifest manager.

        Args:
            manifest_path: Path to the manifest JSON file
        """
        self.manifest_path = Path(manifest_path)
        self.lock_path = self.manifest_path.with_suffix(".lock")
        self._manifest: Optional[JobManifest] = None

    def _acquire_lock(self, blocking: bool = True) -> bool:
        """Acquire file lock."""
        self._lock_file = open(self.lock_path, "w")
        try:
            if blocking:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
            else:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            self._lock_file.close()
            return False

    def _release_lock(self):
        """Release file lock."""
        if hasattr(self, "_lock_file") and self._lock_file:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()

    def _load_manifest(self) -> JobManifest:
        """Load manifest from file."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JobManifest.from_dict(data)

    def _save_manifest(self, manifest: JobManifest):
        """Save manifest to file."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

    def get_manifest(self) -> JobManifest:
        """Get the current manifest (read-only, no locking)."""
        return self._load_manifest()

    def claim_next_task(self, worker_id: str) -> Optional[JobTask]:
        """
        Atomically claim the next pending task.

        Args:
            worker_id: Identifier for the claiming worker

        Returns:
            The claimed task, or None if no tasks available
        """
        self._acquire_lock()
        try:
            manifest = self._load_manifest()

            # Find first pending task
            for task in manifest.tasks:
                if task.status == TaskStatus.PENDING:
                    # Claim it
                    task.status = TaskStatus.IN_PROGRESS
                    task.worker_id = worker_id
                    task.started_at = datetime.now().isoformat()

                    self._save_manifest(manifest)
                    logger.info(f"Worker {worker_id} claimed task {task.task_id}")
                    return task

            return None
        finally:
            self._release_lock()

    def claim_specific_task(self, task_id: str, worker_id: str) -> Optional[JobTask]:
        """
        Atomically claim a specific task by ID.

        Args:
            task_id: ID of the task to claim
            worker_id: Identifier for the claiming worker

        Returns:
            The claimed task, or None if not available
        """
        self._acquire_lock()
        try:
            manifest = self._load_manifest()
            task = manifest.get_task(task_id)

            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.IN_PROGRESS
                task.worker_id = worker_id
                task.started_at = datetime.now().isoformat()

                self._save_manifest(manifest)
                logger.info(f"Worker {worker_id} claimed task {task_id}")
                return task

            return None
        finally:
            self._release_lock()

    def complete_task(
        self,
        task_id: str,
        success: bool,
        output_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Mark a task as completed or failed.

        Args:
            task_id: ID of the task
            success: Whether the task succeeded
            output_path: Path to output file (if successful)
            error_message: Error message (if failed)

        Returns:
            True if task was updated, False if not found
        """
        self._acquire_lock()
        try:
            manifest = self._load_manifest()
            task = manifest.get_task(task_id)

            if task and task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.completed_at = datetime.now().isoformat()
                task.output_path = output_path
                task.error_message = error_message

                self._save_manifest(manifest)
                logger.info(f"Task {task_id} marked as {'completed' if success else 'failed'}")
                return True

            return False
        finally:
            self._release_lock()

    def reset_stale_tasks(self, max_age_hours: float = 2.0) -> int:
        """
        Reset tasks that have been in_progress for too long.

        Args:
            max_age_hours: Maximum hours a task can be in_progress

        Returns:
            Number of tasks reset
        """
        from datetime import timedelta

        self._acquire_lock()
        try:
            manifest = self._load_manifest()
            reset_count = 0
            now = datetime.now()

            for task in manifest.tasks:
                if task.status == TaskStatus.IN_PROGRESS and task.started_at:
                    started = datetime.fromisoformat(task.started_at)
                    age = now - started
                    if age > timedelta(hours=max_age_hours):
                        task.status = TaskStatus.PENDING
                        task.worker_id = None
                        task.started_at = None
                        reset_count += 1
                        logger.warning(f"Reset stale task {task.task_id}")

            if reset_count > 0:
                self._save_manifest(manifest)

            return reset_count
        finally:
            self._release_lock()

    def get_progress(self) -> dict:
        """Get progress statistics."""
        manifest = self.get_manifest()
        return {
            "total": manifest.total_tasks,
            "pending": manifest.pending_tasks,
            "in_progress": manifest.in_progress_tasks,
            "completed": manifest.completed_tasks,
            "failed": manifest.failed_tasks,
            "percent_complete": (
                100 * manifest.completed_tasks / manifest.total_tasks
                if manifest.total_tasks > 0 else 0
            ),
        }


def generate_manifest(
    run_name: str,
    config: "ExperimentConfig",
    problem_ids: list[str],
    output_dir: Path,
) -> JobManifest:
    """
    Generate a job manifest from experiment configuration.

    Args:
        run_name: Name for this run
        config: ExperimentConfig with experiment settings
        problem_ids: List of problem IDs to include
        output_dir: Directory to save the manifest

    Returns:
        Generated JobManifest
    """
    tasks = []

    for problem_id in problem_ids:
        for language in config.experiment.languages:
            for condition in config.experiment.conditions:
                for rollout_idx in range(config.generation.num_rollouts):
                    task = JobTask.create(
                        problem_id=problem_id,
                        language=language,
                        condition=condition,
                        rollout_idx=rollout_idx,
                        base_seed=config.generation.base_seed,
                    )
                    tasks.append(task)

    manifest = JobManifest(
        run_name=run_name,
        config_name=config.config_name or "unknown",
        created_at=datetime.now().isoformat(),
        tasks=tasks,
        config_snapshot=config.to_dict(),
    )

    # Save manifest
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{run_name}_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Generated manifest with {len(tasks)} tasks: {manifest_path}")
    return manifest
