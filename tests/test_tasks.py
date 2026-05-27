from pathlib import Path

from corecoder import checkpoint as checkpoint_module
from corecoder import events as events_module
from corecoder import tasks as tasks_module
from corecoder.checkpoint import CheckpointStore
from corecoder.events import EventLog
from corecoder.tasks import TaskStatusError, TaskStep, TaskStore


def _patch_task_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / ".corecoder" / "tasks"
    monkeypatch.setattr(tasks_module, "TASKS_DIR", root)
    monkeypatch.setattr(events_module, "TASKS_DIR", root)
    monkeypatch.setattr(checkpoint_module, "TASKS_DIR", root)
    return root


def test_task_can_be_created_updated_and_reloaded(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore()

    task = store.create_task(
        session_id="default",
        user_goal="Add durable task storage",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="../Durable Task!",
        steps=[
            TaskStep(id="S1", title="Create tests", acceptance="Task tests fail first."),
            TaskStep(id="S2", title="Implement storage", acceptance="Task state reloads."),
        ],
    )

    assert task.id == "Durable-Task"
    assert task.status == "draft"
    assert task.current_step is None
    assert task.steps[0].status == "pending"
    assert (root / task.id / "task.json").exists()
    assert (root / task.id).resolve().is_relative_to(root.resolve())

    store.update_step(task.id, "S1", "completed", notes="covered by tests")
    store.update_status(task.id, "awaiting_approval")
    store.update_status(task.id, "running", current_step="S2")

    reloaded = TaskStore().load(task.id)

    assert reloaded is not None
    assert reloaded.status == "running"
    assert reloaded.current_step == "S2"
    assert reloaded.steps[0].status == "completed"
    assert reloaded.steps[0].notes == "covered by tests"
    assert reloaded.steps[1].title == "Implement storage"


def test_task_store_rejects_invalid_status(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore()
    task = store.create_task(
        session_id="default",
        user_goal="Add durable task storage",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="safe-task",
    )

    try:
        store.update_status(task.id, "unknown")
    except TaskStatusError as exc:
        assert "Invalid task status" in str(exc)
    else:
        raise AssertionError("expected invalid task status to fail")


def test_task_store_rejects_invalid_step_status(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore()
    task = store.create_task(
        session_id="default",
        user_goal="Add durable task storage",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="safe-task",
        steps=[TaskStep(id="S1", title="Create tests", acceptance="Step validation is covered.")],
    )

    try:
        store.update_step(task.id, "S1", "unknown")
    except TaskStatusError as exc:
        assert "Invalid step status" in str(exc)
    else:
        raise AssertionError("expected invalid step status to fail")


def test_event_log_appends_and_reads_jsonl(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore()
    task = store.create_task(
        session_id="default",
        user_goal="Add durable task storage",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="eventful-task",
    )
    log = EventLog(task.id)

    created = log.append("task_created", task_id=task.id)
    log.append("step_started", task_id=task.id, step_id="S1")

    events = EventLog(task.id).read()

    assert created["type"] == "task_created"
    assert "time" in created
    assert [event["type"] for event in events] == ["task_created", "step_started"]
    assert events[1]["step_id"] == "S1"


def test_checkpoint_can_be_saved_and_reloaded(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore()
    task = store.create_task(
        session_id="default",
        user_goal="Add durable task storage",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="checkpoint-task",
        steps=[TaskStep(id="S1", title="Create checkpoint", acceptance="Reload succeeds.")],
    )
    store.update_status(task.id, "awaiting_approval")
    store.update_status(task.id, "running", current_step="S1")
    task = store.load(task.id)
    assert task is not None

    checkpoint_store = CheckpointStore()
    checkpoint_store.save(
        task=task,
        messages=[{"role": "user", "content": "continue"}],
        tool_counters={"rounds": 1, "tool_calls": 2},
        changed_files=["corecoder/tasks.py"],
        background_subagents=[{"id": "agent-1", "status": "running", "log_path": "output.log"}],
        last_event_offset=128,
    )

    loaded = CheckpointStore().load(task.id)

    assert (root / task.id / "checkpoint.json").exists()
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.task.status == "running"
    assert loaded.task.current_step == "S1"
    assert loaded.messages == [{"role": "user", "content": "continue"}]
    assert loaded.tool_counters == {"rounds": 1, "tool_calls": 2}
    assert loaded.changed_files == ["corecoder/tasks.py"]
    assert loaded.background_subagents[0]["id"] == "agent-1"
    assert loaded.last_event_offset == 128
