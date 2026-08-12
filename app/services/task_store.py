"""In-memory async task store (the lightweight stand-in for Celery).

Thread/async safe for the <10-user target. Swap for Celery+Redis by
reimplementing the same four methods.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.schemas import Job, SourceError, TaskStatus

MAX_TASKS = 200
TASK_TTL_HOURS = 12


class Task:
    __slots__ = (
        "id", "status", "progress", "created_at", "finished_at",
        "results", "errors", "error", "meta",
    )

    def __init__(self, task_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.id = task_id
        self.status: TaskStatus = TaskStatus.pending
        self.progress: str = "queued"
        self.created_at = datetime.now(timezone.utc)
        self.finished_at: Optional[datetime] = None
        self.results: List[Job] = []
        self.errors: List[SourceError] = []
        self.error: Optional[str] = None
        self.meta: Dict[str, Any] = meta or {}


class TaskStore:
    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._active_by_cv: Dict[str, str] = {}  # cv_id -> task_id still running
        self._lock = asyncio.Lock()

    async def create(self, meta: Optional[Dict[str, Any]] = None) -> Task:
        task_id = uuid.uuid4().hex
        task = Task(task_id, meta)
        async with self._lock:
            self._tasks[task_id] = task
            self._evict_locked()
        return task

    async def get_active_for_cv(self, cv_id: str) -> Optional[Task]:
        """Returns the still-running search for this CV, if any.

        Prevents duplicate-click / rapid-rerun bugs from firing several
        overlapping searches against the same 5 sites within seconds of each
        other — exactly the pattern that gets an IP rate-limited or blocked.
        """
        async with self._lock:
            task_id = self._active_by_cv.get(cv_id)
            if task_id is None:
                return None
            task = self._tasks.get(task_id)
            if task is None or task.status in (TaskStatus.completed, TaskStatus.failed):
                self._active_by_cv.pop(cv_id, None)
                return None
            return task

    async def mark_active(self, cv_id: str, task_id: str) -> None:
        async with self._lock:
            self._active_by_cv[cv_id] = task_id

    async def get(self, task_id: str) -> Optional[Task]:
        async with self._lock:
            return self._tasks.get(task_id)

    async def update(self, task_id: str, **fields: Any) -> Optional[Task]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            for key, value in fields.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            if fields.get("status") in (TaskStatus.completed, TaskStatus.failed):
                task.finished_at = datetime.now(timezone.utc)
                cv_id = task.meta.get("cv_id")
                if cv_id and self._active_by_cv.get(cv_id) == task_id:
                    self._active_by_cv.pop(cv_id, None)
            return task

    async def list_ids(self) -> List[str]:
        async with self._lock:
            return list(self._tasks.keys())

    def _evict_locked(self) -> None:
        """Drop finished tasks older than the TTL, then cap total size."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=TASK_TTL_HOURS)
        stale = [
            tid for tid, t in self._tasks.items()
            if t.finished_at and t.finished_at < cutoff
        ]
        for tid in stale:
            self._tasks.pop(tid, None)
        if len(self._tasks) > MAX_TASKS:
            for tid in sorted(self._tasks, key=lambda t: self._tasks[t].created_at)[
                : len(self._tasks) - MAX_TASKS
            ]:
                self._tasks.pop(tid, None)


task_store = TaskStore()
