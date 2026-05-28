"""Durable task state for complex task execution."""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, get_args

TASKS_DIR = Path.home() / ".corecoder" / "tasks"

TaskStatus = Literal[
    "draft",
    "awaiting_approval",
    "running",
    "paused",
    "blocked",
    "completed",
    "failed",
    "cancelled",
]
StepStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
    "blocked",
    "skipped",
]


DEFAULT_BUDGETS = {
    "max_rounds": 50,
    "max_tool_calls": 200,
    "max_minutes": 60,
    "subagent_foreground_seconds": 60,
    "auto_continue_max_runs": 3,
}

ALLOWED_TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    "draft": {"awaiting_approval", "cancelled"},
    "awaiting_approval": {"running", "cancelled"},
    "running": {"paused", "blocked", "completed", "failed", "cancelled"},
    "paused": {"running", "cancelled"},
    "blocked": {"awaiting_approval", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


_SAFE_TASK_RE = re.compile(r"[^A-Za-z0-9._-]+")
_UNSET = object()


class TaskStatusError(ValueError):
    """Raised when task or step status is not supported."""


@dataclass
class TaskStep:
    id: str
    title: str
    status: StepStatus = "pending"
    depends_on: list[str] = field(default_factory=list)
    acceptance: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "depends_on": self.depends_on,
            "acceptance": self.acceptance,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskStep":
        return cls(
            id=data["id"],
            title=data["title"],
            status=_validate_step_status(data.get("status", "pending")),
            depends_on=list(data.get("depends_on", [])),
            acceptance=data.get("acceptance", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class TaskState:
    id: str
    session_id: str
    user_goal: str
    status: TaskStatus
    created_at: str
    updated_at: str
    model: str
    cwd: str
    current_step: str | None = None
    steps: list[TaskStep] = field(default_factory=list)
    budgets: dict[str, int] = field(default_factory=lambda: DEFAULT_BUDGETS.copy())
    files_changed: list[str] = field(default_factory=list)
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_goal": self.user_goal,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "cwd": self.cwd,
            "current_step": self.current_step,
            "steps": [step.to_dict() for step in self.steps],
            "budgets": self.budgets,
            "files_changed": self.files_changed,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            user_goal=data["user_goal"],
            status=_validate_task_status(data.get("status", "draft")),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            model=data["model"],
            cwd=data["cwd"],
            current_step=data.get("current_step"),
            steps=[TaskStep.from_dict(step) for step in data.get("steps", [])],
            budgets=dict(data.get("budgets", DEFAULT_BUDGETS)),
            files_changed=list(data.get("files_changed", [])),
            last_error=data.get("last_error"),
        )


class TaskStore:
    """Create, load, and update task state under ~/.corecoder/tasks."""

    def __init__(self, root: Path | None = None):
        self.root = root or TASKS_DIR

    def create_task(
        self,
        *,
        session_id: str,
        user_goal: str,
        model: str,
        cwd: str,
        task_id: str | None = None,
        steps: list[TaskStep] | None = None,
        budgets: dict[str, int] | None = None,
    ) -> TaskState:
        now = _now()
        task = TaskState(
            id=_normalize_task_id(task_id, user_goal),
            session_id=session_id,
            user_goal=user_goal,
            status="draft",
            created_at=now,
            updated_at=now,
            model=model,
            cwd=cwd,
            steps=steps or [],
            budgets={**DEFAULT_BUDGETS, **(budgets or {})},
        )
        self.save(task)
        return task

    def load(self, task_id: str) -> TaskState | None:
        path = self.task_path(task_id)
        if not path.exists():
            return None
        return TaskState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_tasks(self) -> list[TaskState]:
        if not self.root.exists():
            return []
        paths = sorted(
            self.root.glob("*/task.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        tasks = []
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                tasks.append(TaskState.from_dict(data))
            except (OSError, KeyError, TypeError, json.JSONDecodeError, TaskStatusError):
                continue
        return tasks

    def save(self, task: TaskState) -> None:
        path = self.task_path(task.id)
        _write_json_atomic(path, task.to_dict())

    def update_status(
        self,
        task_id: str,
        status: str,
        *,
        current_step: str | None | object = _UNSET,
        last_error: str | None | object = _UNSET,
    ) -> TaskState:
        task = self._require_task(task_id)
        next_status = _validate_task_status(status)
        _validate_task_transition(task.status, next_status)
        task.status = next_status
        task.updated_at = _now()
        if current_step is not _UNSET:
            task.current_step = current_step
        if last_error is not _UNSET:
            task.last_error = last_error
        self.save(task)
        return task

    def update_step(
        self,
        task_id: str,
        step_id: str,
        status: str,
        *,
        notes: str | None = None,
    ) -> TaskState:
        task = self._require_task(task_id)
        for step in task.steps:
            if step.id == step_id:
                step.status = _validate_step_status(status)
                if notes is not None:
                    step.notes = notes
                task.updated_at = _now()
                self.save(task)
                return task
        raise KeyError(f"Unknown task step: {step_id}")

    def task_dir(self, task_id: str) -> Path:
        return task_dir(task_id, self.root)

    def task_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "task.json"

    def _require_task(self, task_id: str) -> TaskState:
        task = self.load(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {_normalize_task_id(task_id)}")
        return task


def task_dir(task_id: str, root: Path | None = None) -> Path:
    safe_id = _normalize_task_id(task_id)
    base = (root or TASKS_DIR).resolve()
    path = (base / safe_id).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Invalid task id")
    return path


def _normalize_task_id(task_id: str | None, user_goal: str | None = None) -> str:
    if task_id:
        raw = task_id
    else:
        raw = f"{time.strftime('%Y%m%d_%H%M%S')}_{_slug(user_goal or 'task')}_{uuid.uuid4().hex[:8]}"
    name = raw.strip().replace("\\", "/").split("/")[-1]
    safe = _SAFE_TASK_RE.sub("-", name).strip(".-_")
    return safe or f"task_{uuid.uuid4().hex[:8]}"


def _validate_task_status(status: str) -> TaskStatus:
    if status not in get_args(TaskStatus):
        raise TaskStatusError(f"Invalid task status: {status}")
    return cast(TaskStatus, status)


def _validate_task_transition(current: TaskStatus, next_status: TaskStatus) -> None:
    if current == next_status:
        return
    allowed = ALLOWED_TASK_TRANSITIONS[current]
    if next_status not in allowed:
        raise TaskStatusError(
            f"Invalid task status transition: {current} -> {next_status}"
        )


def _validate_step_status(status: str) -> StepStatus:
    if status not in get_args(StepStatus):
        raise TaskStatusError(f"Invalid step status: {status}")
    return cast(StepStatus, status)



def _slug(value: str) -> str:
    return _SAFE_TASK_RE.sub("-", value.lower()).strip(".-_")[:40] or "task"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
