from pathlib import Path

from corecoder import checkpoint as checkpoint_module
from corecoder import events as events_module
from corecoder import tasks as tasks_module
from corecoder.events import EventLog
from corecoder.tasks import TaskStep, TaskStore
from corecoder.tools.progress import TaskProgressTool


def _patch_task_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / ".corecoder" / "tasks"
    monkeypatch.setattr(tasks_module, "TASKS_DIR", root)
    monkeypatch.setattr(events_module, "TASKS_DIR", root)
    monkeypatch.setattr(checkpoint_module, "TASKS_DIR", root)
    return root


def _running_task(store: TaskStore):
    task = store.create_task(
        session_id="session-a",
        user_goal="Implement approved plan",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="progress-task",
        steps=[
            TaskStep(id="S1", title="Add tests", acceptance="Tests fail first."),
            TaskStep(
                id="S2",
                title="Implement code",
                acceptance="Tests pass.",
                depends_on=["S1"],
            ),
        ],
    )
    store.update_status(task.id, "awaiting_approval")
    return store.update_status(task.id, "running")


def _tool(task_id: str, store: TaskStore, root: Path, checkpoints: list[str]):
    return TaskProgressTool(
        task_id=task_id,
        store=store,
        events=EventLog(task_id, root),
        checkpoint_callback=lambda: checkpoints.append("saved"),
    )


def test_progress_tool_updates_step_events_current_step_and_checkpoint(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)
    checkpoints = []
    tool = _tool(task.id, store, root, checkpoints)

    started = tool.execute(step_id="S1", status="in_progress", notes="Starting tests")
    completed = tool.execute(
        step_id="S1",
        status="completed",
        notes="Tests cover the behavior.",
        evidence=["uv run pytest tests/test_progress_tool.py"],
    )

    reloaded = store.load(task.id)
    assert "updated S1 to in_progress" in started
    assert "updated S1 to completed" in completed
    assert reloaded.current_step == "S1"
    assert reloaded.steps[0].status == "completed"
    assert "Tests cover the behavior." in reloaded.steps[0].notes
    assert "uv run pytest tests/test_progress_tool.py" in reloaded.steps[0].notes
    assert checkpoints == ["saved", "saved"]
    events = [event["type"] for event in EventLog(task.id, root).read()]
    assert events == ["step_started", "step_completed"]


def test_progress_tool_rejects_invalid_completion(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)
    tool = _tool(task.id, store, root, [])

    assert "Unknown task step" in tool.execute(step_id="missing", status="completed")
    assert "requires notes or evidence" in tool.execute(step_id="S1", status="completed")
    assert "depends on incomplete steps" in tool.execute(
        step_id="S2",
        status="in_progress",
        notes="Starting dependent work",
    )


def test_progress_tool_rejects_second_in_progress_step(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = store.create_task(
        session_id="session-a",
        user_goal="Implement approved plan",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="parallel-progress-task",
        steps=[
            TaskStep(id="S1", title="First step", acceptance="Done."),
            TaskStep(id="S2", title="Second step", acceptance="Done."),
        ],
    )
    store.update_status(task.id, "awaiting_approval")
    task = store.update_status(task.id, "running")
    tool = _tool(task.id, store, root, [])

    assert "updated S1 to in_progress" in tool.execute(
        step_id="S1",
        status="in_progress",
        notes="Starting first step",
    )
    assert "another step is already in_progress: S1" in tool.execute(
        step_id="S2",
        status="in_progress",
        notes="Starting second step",
    )


def test_progress_tool_blocked_and_failed_update_task_status(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)
    tool = _tool(task.id, store, root, [])

    blocked = tool.execute(step_id="S1", status="blocked", notes="Need user decision")

    reloaded = store.load(task.id)
    assert "updated S1 to blocked" in blocked
    assert reloaded.status == "blocked"
    assert reloaded.last_error == "Need user decision"
    assert reloaded.steps[0].status == "blocked"
    assert EventLog(task.id, root).read()[-1]["type"] == "step_blocked"
