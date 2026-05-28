"""Foreground task runner for approved durable tasks."""

import time
from typing import Any

from .checkpoint import Checkpoint, CheckpointStore
from .events import EventLog
from .plan_files import plan_file_path
from .tasks import TASKS_DIR, TaskState, TaskStatusError, TaskStore
from .tools.progress import TaskProgressTool


class TaskBudgetExceeded(RuntimeError):
    """Raised when a foreground task exceeds its configured budget."""


class TaskPaused(RuntimeError):
    """Raised when a foreground task is cooperatively paused."""


class TaskCancelled(RuntimeError):
    """Raised when a foreground task is cooperatively cancelled."""


class TaskRunner:
    """Run approved tasks through the normal foreground Agent loop."""

    def __init__(
        self,
        *,
        agent,
        store: TaskStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        progress_reminder_rounds: int = 5,
    ):
        self.agent = agent
        self.store = store or TaskStore(TASKS_DIR)
        self.checkpoints = checkpoint_store or CheckpointStore(self.store.root)
        self.progress_reminder_rounds = progress_reminder_rounds

    def run(self, task_id: str, user_input: str, on_token=None, on_tool=None) -> str:
        task = self._prepare_running_task(task_id)
        events = EventLog(task.id, self.store.root)
        events.append("task_started", task_id=task.id)
        counters = {"rounds": 0, "tool_calls": 0}
        started_at = time.monotonic()
        last_progress_round = 0

        def record_progress() -> None:
            nonlocal last_progress_round
            last_progress_round = counters["rounds"] + 1

        def on_round(info: dict[str, Any]) -> None:
            counters["rounds"] = int(info.get("round", counters["rounds"] + 1))
            counters["tool_calls"] += int(info.get("tool_calls", 0))
            self._save_checkpoint(task.id, counters, events)
            self._raise_if_controlled(task.id, counters, events)
            self._enforce_budgets(task.id, counters, started_at)
            if (
                not info.get("done")
                and self.progress_reminder_rounds > 0
                and counters["rounds"] - last_progress_round >= self.progress_reminder_rounds
            ):
                self._append_progress_reminder(task.id)

        progress_tool = TaskProgressTool(
            task_id=task.id,
            store=self.store,
            events=events,
            checkpoint_callback=lambda: self._save_checkpoint(task.id, counters, events),
            progress_callback=record_progress,
        )
        original_tools = self._install_progress_tool(progress_tool)
        try:
            response = self.agent.chat(
                user_input,
                on_token=on_token,
                on_tool=on_tool,
                on_round=on_round,
            )
            if response == "(reached maximum tool-call rounds)":
                raise TaskBudgetExceeded("round budget exceeded")
            return self._finalize_after_agent_return(task.id, response, counters, events)
        except TaskBudgetExceeded as exc:
            return self._block_task(task.id, str(exc), counters, events)
        except TaskPaused as exc:
            return f"Task paused: {exc}"
        except TaskCancelled as exc:
            return f"Task cancelled: {exc}"
        except KeyboardInterrupt:
            self._pause_task(task.id, counters, events, reason="Interrupted")
            raise
        except Exception as exc:
            self._fail_task(task.id, str(exc), counters, events)
            raise
        finally:
            self._restore_tools(original_tools)

    def restore_from_checkpoint(self, task_id: str) -> Checkpoint | None:
        checkpoint = self.checkpoints.load(task_id)
        if checkpoint is not None:
            self.agent.messages = list(checkpoint.messages)
        return checkpoint

    def restore_latest_for_session(self, session_id: str) -> Checkpoint | None:
        for task in self.store.list_tasks():
            if task.session_id != session_id:
                continue
            if task.status not in {"running", "paused", "blocked"}:
                continue
            checkpoint = self.restore_from_checkpoint(task.id)
            if checkpoint is not None:
                self._append_progress_reminder(task.id)
                return checkpoint
        return None

    def pause(self, task_id: str, *, reason: str = "User requested pause") -> TaskState:
        events = EventLog(task_id, self.store.root)
        return self._pause_task(
            task_id,
            {"rounds": 0, "tool_calls": 0},
            events,
            reason=reason,
        )

    def cancel(self, task_id: str, *, reason: str = "User requested cancel") -> TaskState:
        events = EventLog(task_id, self.store.root)
        return self._cancel_task(
            task_id,
            {"rounds": 0, "tool_calls": 0},
            events,
            reason=reason,
        )

    def _prepare_running_task(self, task_id: str) -> TaskState:
        task = self.store.load(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        if task.status == "paused":
            return self.store.update_status(task.id, "running", last_error=None)
        if task.status != "running":
            raise TaskStatusError(f"Task must be running or paused: {task.status}")
        return task

    def _finalize_after_agent_return(
        self,
        task_id: str,
        response: str,
        counters: dict[str, int],
        events: EventLog,
    ) -> str:
        task = self.store.load(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        if task.status in {"failed", "cancelled"}:
            self._save_checkpoint(task.id, counters, events)
            return response
        if task.status == "paused":
            self._save_checkpoint(task.id, counters, events)
            return f"Task paused: {task.last_error or 'Task paused'}"
        failed_step = next((step for step in task.steps if step.status == "failed"), None)
        if failed_step:
            self._fail_task(
                task.id,
                failed_step.notes or f"Step failed: {failed_step.id}",
                counters,
                events,
            )
            return response
        blocked_step = next((step for step in task.steps if step.status == "blocked"), None)
        if task.status == "blocked" or blocked_step:
            reason = task.last_error or (blocked_step.notes if blocked_step else "Task blocked")
            if task.status != "blocked":
                self._block_task(task.id, reason, counters, events)
            else:
                self._save_checkpoint(task.id, counters, events)
            return response
        if task.steps and all(step.status in {"completed", "skipped"} for step in task.steps):
            task = self.store.update_status(
                task.id,
                "completed",
                current_step=None,
                last_error=None,
            )
            events.append("task_completed", task_id=task.id)
            self._save_checkpoint(task.id, counters, events)
            return response
        return self._block_task(
            task.id,
            "Agent returned before completing all plan steps",
            counters,
            events,
        )

    def _block_task(
        self,
        task_id: str,
        reason: str,
        counters: dict[str, int],
        events: EventLog,
    ) -> str:
        task = self.store.update_status(task_id, "blocked", last_error=reason)
        events.append("task_blocked", task_id=task.id, reason=reason)
        self._save_checkpoint(task.id, counters, events)
        return f"Task blocked: {reason}"

    def _pause_task(
        self,
        task_id: str,
        counters: dict[str, int],
        events: EventLog,
        *,
        reason: str,
    ) -> TaskState:
        task = self.store.load(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        if task.status == "paused":
            self._save_checkpoint(task.id, counters, events)
            return task
        if task.status != "running":
            raise TaskStatusError(f"Task must be running to pause: {task.status}")
        task = self.store.update_status(task_id, "paused", last_error=reason)
        events.append("task_paused", task_id=task.id, reason=reason)
        self._save_checkpoint(task.id, counters, events)
        return task

    def _cancel_task(
        self,
        task_id: str,
        counters: dict[str, int],
        events: EventLog,
        *,
        reason: str,
    ) -> TaskState:
        task = self.store.load(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        if task.status == "cancelled":
            self._save_checkpoint(task.id, counters, events)
            return task
        if task.status in {"completed", "failed"}:
            raise TaskStatusError(f"Terminal task cannot be cancelled: {task.status}")
        task = self.store.update_status(task_id, "cancelled", last_error=reason)
        events.append("task_cancelled", task_id=task.id, reason=reason)
        self._save_checkpoint(task.id, counters, events)
        return task

    def _fail_task(
        self,
        task_id: str,
        reason: str,
        counters: dict[str, int],
        events: EventLog,
    ) -> None:
        task = self.store.update_status(task_id, "failed", last_error=reason)
        events.append("task_failed", task_id=task.id, reason=reason)
        self._save_checkpoint(task.id, counters, events)

    def _raise_if_controlled(
        self,
        task_id: str,
        counters: dict[str, int],
        events: EventLog,
    ) -> None:
        task = self.store.load(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        if task.status == "paused":
            self._save_checkpoint(task.id, counters, events)
            raise TaskPaused(task.last_error or "Task paused")
        if task.status == "cancelled":
            self._save_checkpoint(task.id, counters, events)
            raise TaskCancelled(task.last_error or "Task cancelled")

    def _enforce_budgets(
        self,
        task_id: str,
        counters: dict[str, int],
        started_at: float,
    ) -> None:
        task = self.store.load(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        max_rounds = task.budgets.get("max_rounds")
        if max_rounds is not None and counters["rounds"] > max_rounds:
            raise TaskBudgetExceeded("round budget exceeded")
        max_tool_calls = task.budgets.get("max_tool_calls")
        if max_tool_calls is not None and counters["tool_calls"] > max_tool_calls:
            raise TaskBudgetExceeded("tool call budget exceeded")
        max_minutes = task.budgets.get("max_minutes")
        if max_minutes is not None and time.monotonic() - started_at > max_minutes * 60:
            raise TaskBudgetExceeded("wall time budget exceeded")

    def _save_checkpoint(
        self,
        task_id: str,
        counters: dict[str, int],
        events: EventLog,
    ) -> Checkpoint:
        task = self.store.load(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        checkpoint = self.checkpoints.save(
            task=task,
            messages=list(getattr(self.agent, "messages", [])),
            tool_counters=dict(counters),
            changed_files=_changed_files(),
            last_event_offset=len(events.read()),
        )
        events.append("checkpoint_saved", task_id=task.id)
        return checkpoint

    def _append_progress_reminder(self, task_id: str) -> None:
        if not hasattr(self.agent, "messages"):
            return
        task = self.store.load(task_id)
        if task is None:
            return
        lines = [
            "<system-reminder>",
            "Current task progress:",
            *_format_progress_lines(task),
            "",
            "Use update_task_progress when starting, completing, blocking, or failing a step.",
            f"Plan file: {plan_file_path(task.id, self.store.root)}",
            "</system-reminder>",
        ]
        self.agent.messages.append({"role": "user", "content": "\n".join(lines)})

    def _install_progress_tool(self, progress_tool: TaskProgressTool):
        if not hasattr(self.agent, "tools"):
            return None
        original_tools = list(self.agent.tools)
        self.agent.tools = [*original_tools, progress_tool]
        return original_tools

    def _restore_tools(self, original_tools) -> None:
        if original_tools is not None and hasattr(self.agent, "tools"):
            self.agent.tools = original_tools


def _format_progress_lines(task: TaskState) -> list[str]:
    if not task.steps:
        return ["- (no plan steps)"]
    return [f"- {step.id} [{step.status}] {step.title}" for step in task.steps]


def _changed_files() -> list[str]:
    try:
        from .tools.edit import _changed_files as files
    except Exception:
        return []
    return sorted(files)
