"""Task progress updates for foreground runner execution."""

from collections.abc import Callable

from .base import Tool
from ..events import EventLog
from ..tasks import TaskState, TaskStore


class TaskProgressTool(Tool):
    name = "update_task_progress"
    description = (
        "Update the current approved task's step status. Use this when starting, "
        "completing, blocking, failing, or skipping a plan step."
    )
    parameters = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "description": "Plan step id, e.g. S1."},
            "status": {
                "type": "string",
                "enum": ["in_progress", "completed", "blocked", "failed", "skipped"],
                "description": "New status for the step.",
            },
            "notes": {
                "type": "string",
                "description": "Short reason, summary, or blocker details.",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Evidence for completed work, such as test commands.",
            },
        },
        "required": ["step_id", "status"],
    }

    def __init__(
        self,
        *,
        task_id: str,
        store: TaskStore,
        events: EventLog,
        checkpoint_callback: Callable[[], None] | None = None,
        progress_callback: Callable[[], None] | None = None,
    ):
        self.task_id = task_id
        self.store = store
        self.events = events
        self.checkpoint_callback = checkpoint_callback
        self.progress_callback = progress_callback

    def execute(
        self,
        step_id: str,
        status: str,
        notes: str = "",
        evidence: list[str] | None = None,
    ) -> str:
        task = self.store.load(self.task_id)
        if task is None:
            return f"Error: task not found: {self.task_id}"
        step = next((item for item in task.steps if item.id == step_id), None)
        if step is None:
            return f"Error: Unknown task step: {step_id}"
        if status not in {"in_progress", "completed", "blocked", "failed", "skipped"}:
            return f"Error: Invalid step status: {status}"
        evidence = evidence or []
        if status == "completed" and not notes.strip() and not evidence:
            return "Error: completed progress requires notes or evidence"
        if status in {"in_progress", "completed"}:
            incomplete = _incomplete_dependencies(task, step.depends_on)
            if incomplete:
                return f"Error: step {step_id} depends on incomplete steps: {', '.join(incomplete)}"

        if status == "in_progress":
            active = next(
                (
                    item
                    for item in task.steps
                    if item.id != step_id and item.status == "in_progress"
                ),
                None,
            )
            if active is not None:
                return f"Error: another step is already in_progress: {active.id}"

        updated_notes = _format_notes(notes, evidence)
        if status == "in_progress":
            task = self.store.update_status(task.id, task.status, current_step=step_id)
            task = self.store.update_step(task.id, step_id, status, notes=updated_notes)
            self.events.append("step_started", task_id=task.id, step_id=step_id)
        elif status == "completed":
            task = self.store.update_step(task.id, step_id, status, notes=updated_notes)
            self.events.append("step_completed", task_id=task.id, step_id=step_id)
        elif status == "skipped":
            task = self.store.update_step(task.id, step_id, status, notes=updated_notes)
            self.events.append("step_skipped", task_id=task.id, step_id=step_id)
        elif status == "blocked":
            task = self.store.update_step(task.id, step_id, status, notes=updated_notes)
            task = self.store.update_status(
                task.id,
                "blocked",
                current_step=step_id,
                last_error=notes or f"Step blocked: {step_id}",
            )
            self.events.append(
                "step_blocked",
                task_id=task.id,
                step_id=step_id,
                reason=notes,
            )
        elif status == "failed":
            task = self.store.update_step(task.id, step_id, status, notes=updated_notes)
            task = self.store.update_status(
                task.id,
                "failed",
                current_step=step_id,
                last_error=notes or f"Step failed: {step_id}",
            )
            self.events.append(
                "step_failed",
                task_id=task.id,
                step_id=step_id,
                reason=notes,
            )

        if self.progress_callback:
            self.progress_callback()
        if self.checkpoint_callback:
            self.checkpoint_callback()
        return f"updated {step_id} to {status}"


def _incomplete_dependencies(task: TaskState, depends_on: list[str]) -> list[str]:
    by_id = {step.id: step for step in task.steps}
    return [
        step_id
        for step_id in depends_on
        if step_id not in by_id or by_id[step_id].status not in {"completed", "skipped"}
    ]


def _format_notes(notes: str, evidence: list[str]) -> str:
    lines = [notes.strip()] if notes.strip() else []
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in evidence)
    return "\n".join(lines)
