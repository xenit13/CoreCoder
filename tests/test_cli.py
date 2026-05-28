from dataclasses import dataclass

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
