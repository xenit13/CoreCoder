from dataclasses import dataclass
from io import StringIO

from rich.console import Console

from corecoder import cli
from corecoder.cli import _restore_task_for_resumed_session


@dataclass
class CheckpointStub:
    task_id: str


class PlannerStub:
    active_task_id = None


class RunnerStub:
    def __init__(self):
        self.restored_sessions = []

    def restore_latest_for_session(self, session_id):
        self.restored_sessions.append(session_id)
        return CheckpointStub(task_id="task-from-checkpoint")


def test_resume_restore_sets_active_task_from_checkpoint():
    planner = PlannerStub()
    runner = RunnerStub()

    checkpoint = _restore_task_for_resumed_session(planner, runner, "session-a")

    assert checkpoint.task_id == "task-from-checkpoint"
    assert planner.active_task_id == "task-from-checkpoint"
    assert runner.restored_sessions == ["session-a"]


def test_show_help_lists_all_interactive_commands(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=120),
    )

    cli._show_help()

    help_text = output.getvalue()
    for command in (
        "/plan_mode",
        "/task",
        "/tasks",
        "/logs [agent_id]",
        "/reset",
        "/tokens",
        "/diff",
        "/sessions",
    ):
        assert command in help_text
