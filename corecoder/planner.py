"""Plan-mode orchestration for durable tasks."""

import json
import re
from pathlib import Path
from typing import Any

from .agent import Agent
from .events import EventLog
from .tasks import TASKS_DIR, TaskState, TaskStep, TaskStore
from .tools.glob_tool import GlobTool
from .tools.grep import GrepTool
from .tools.read import ReadFileTool


class PlanError(ValueError):
    """Raised when a plan cannot be generated or parsed."""


class Planner:
    """Manage session-level plan mode and plan approval state."""

    def __init__(
        self,
        *,
        session_id: str = "default",
        store: TaskStore | None = None,
        root: Path | None = None,
    ):
        self.session_id = session_id
        self.store = store or TaskStore(root or TASKS_DIR)
        self.plan_mode_enabled = False
        self.active_task_id: str | None = None

    def enable_plan_mode(self) -> None:
        self.plan_mode_enabled = True

    def disable_plan_mode(self) -> None:
        self.plan_mode_enabled = False

    def generate_plan(
        self,
        *,
        llm,
        user_goal: str,
        model: str,
        cwd: str,
    ) -> TaskState:
        """Generate a structured plan and persist it for approval."""
        planning_agent = Agent(
            llm=llm,
            tools=_read_only_plan_tools(),
            max_rounds=8,
            system=_plan_system_prompt(),
        )
        plan_content = planning_agent.chat(user_goal)
        steps = _parse_plan_steps(plan_content)
        task = self.store.create_task(
            session_id=self.session_id,
            user_goal=user_goal,
            model=model,
            cwd=cwd,
            steps=steps,
        )
        task = self.store.update_status(task.id, "awaiting_approval")
        self.active_task_id = task.id
        EventLog(task.id, self.store.root).append(
            "plan_generated",
            task_id=task.id,
            steps=[step.id for step in task.steps],
        )
        return task

    def approve_plan(self, task_id: str | None = None) -> TaskState:
        task = self._active_or(task_id)
        task = self.store.update_status(task.id, "running")
        self.active_task_id = task.id
        self.disable_plan_mode()
        EventLog(task.id, self.store.root).append("plan_approved", task_id=task.id)
        return task

    def reject_plan(self, task_id: str | None = None, *, reason: str = "") -> TaskState:
        task = self._active_or(task_id)
        task = self.store.update_status(
            task.id,
            "cancelled",
            last_error=reason or "Plan rejected",
        )
        self.active_task_id = task.id
        self.disable_plan_mode()
        EventLog(task.id, self.store.root).append(
            "plan_rejected",
            task_id=task.id,
            reason=task.last_error,
        )
        return task

    def active_task(self) -> TaskState | None:
        if self.active_task_id:
            task = self.store.load(self.active_task_id)
            if task is not None:
                return task
        tasks = self.store.list_tasks()
        if not tasks:
            return None
        self.active_task_id = tasks[0].id
        return tasks[0]

    def _active_or(self, task_id: str | None) -> TaskState:
        if task_id:
            task = self.store.load(task_id)
        else:
            task = self.active_task()
        if task is None:
            raise PlanError("No active plan task")
        return task


def _read_only_plan_tools():
    return [GlobTool(), GrepTool(), ReadFileTool()]


def _plan_system_prompt() -> str:
    return """\
You are CoreCoder planning mode. Do not edit files, run shell commands, change
configuration, install dependencies, commit changes, or claim implementation work
is complete. You may only use the provided read-only tools: glob, grep, and
read_file.

Explore just enough of the codebase to understand the requested change, reuse
existing patterns where possible, then produce your final response as valid JSON
only. The JSON must be an object with a non-empty "steps" array. Each step must
include "title" and "acceptance" fields, and may include "id" and
"depends_on". Use short, concrete steps that can be verified independently.
"""


def _parse_plan_steps(content: str) -> list[TaskStep]:
    data = _parse_json_object(content)
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanError("Plan response must include a non-empty steps array")

    steps = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise PlanError("Each plan step must be an object")
        title = _required_text(raw_step, "title")
        acceptance = _required_text(raw_step, "acceptance")
        step_id = str(raw_step.get("id") or f"S{index}").strip() or f"S{index}"
        depends_on = raw_step.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise PlanError("depends_on must be a list when present")
        steps.append(
            TaskStep(
                id=step_id,
                title=title,
                acceptance=acceptance,
                depends_on=[str(item) for item in depends_on],
            )
        )
    return steps


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanError("Plan response must be valid JSON") from exc
    if not isinstance(data, dict):
        raise PlanError("Plan response must be a JSON object")
    return data


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"Plan step must include {key}")
    return value.strip()
