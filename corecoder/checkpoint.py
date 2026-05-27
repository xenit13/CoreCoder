"""Checkpoint snapshots for durable task recovery."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tasks import TASKS_DIR, TaskState, _write_json_atomic, task_dir


@dataclass
class Checkpoint:
    version: int
    task_id: str
    session_id: str
    saved_at: str
    task: TaskState
    messages: list[dict[str, Any]] = field(default_factory=list)
    cwd: str = ""
    tool_counters: dict[str, int] = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    background_subagents: list[dict[str, Any]] = field(default_factory=list)
    last_event_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "saved_at": self.saved_at,
            "task": self.task.to_dict(),
            "messages": self.messages,
            "cwd": self.cwd,
            "tool_counters": self.tool_counters,
            "changed_files": self.changed_files,
            "background_subagents": self.background_subagents,
            "last_event_offset": self.last_event_offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            version=data.get("version", 1),
            task_id=data["task_id"],
            session_id=data["session_id"],
            saved_at=data["saved_at"],
            task=TaskState.from_dict(data["task"]),
            messages=list(data.get("messages", [])),
            cwd=data.get("cwd", ""),
            tool_counters=dict(data.get("tool_counters", {})),
            changed_files=list(data.get("changed_files", [])),
            background_subagents=list(data.get("background_subagents", [])),
            last_event_offset=data.get("last_event_offset", 0),
        )


class CheckpointStore:
    """Save and load the latest checkpoint for a task."""

    def __init__(self, root: Path | None = None):
        self.root = root or TASKS_DIR

    def save(
        self,
        *,
        task: TaskState,
        messages: list[dict[str, Any]],
        tool_counters: dict[str, int] | None = None,
        changed_files: list[str] | None = None,
        background_subagents: list[dict[str, Any]] | None = None,
        last_event_offset: int = 0,
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            version=1,
            task_id=task.id,
            session_id=task.session_id,
            saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            task=task,
            messages=messages,
            cwd=task.cwd,
            tool_counters=tool_counters or {"rounds": 0, "tool_calls": 0},
            changed_files=changed_files or [],
            background_subagents=background_subagents or [],
            last_event_offset=last_event_offset,
        )
        _write_json_atomic(self.path(task.id), checkpoint.to_dict())
        return checkpoint

    def load(self, task_id: str) -> Checkpoint | None:
        path = self.path(task_id)
        if not path.exists():
            return None
        import json

        return Checkpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def path(self, task_id: str) -> Path:
        return task_dir(task_id, self.root) / "checkpoint.json"
