"""Plan file persistence for durable tasks."""

import time
import uuid
from pathlib import Path

from .tasks import TASKS_DIR, TaskState, task_dir


def plan_file_path(task_id: str, root: Path | None = None) -> Path:
    return task_dir(task_id, root or TASKS_DIR) / "plan.md"


def write_plan_file(task: TaskState, root: Path | None = None) -> Path:
    path = plan_file_path(task.id, root)
    lines = [
        "# Approved Plan",
        "",
        f"Task: {task.id}",
        f"Created: {task.created_at}",
        f"Updated: {task.updated_at}",
        f"Model: {task.model}",
        f"CWD: {task.cwd}",
        "",
        "## Original Request",
        "",
        task.user_goal,
        "",
        "## Steps",
        "",
    ]
    for index, step in enumerate(task.steps, start=1):
        lines.append(f"{index}. {step.id} - {step.title}")
        if step.acceptance:
            lines.append(f"   Acceptance: {step.acceptance}")
        if step.depends_on:
            lines.append(f"   Depends on: {', '.join(step.depends_on)}")
        lines.append("")
    _write_text_atomic(path, "\n".join(lines).rstrip() + "\n")
    return path


def read_plan_file(task_id: str, root: Path | None = None) -> str | None:
    path = plan_file_path(task_id, root)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}_{uuid.uuid4().hex}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
