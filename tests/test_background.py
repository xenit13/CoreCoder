import threading
import time
from pathlib import Path

from corecoder import checkpoint as checkpoint_module
from corecoder import events as events_module
from corecoder import runner as runner_module
from corecoder import tasks as tasks_module
from corecoder.agent import Agent
from corecoder.background import BackgroundRegistry
from corecoder.cli import _format_task_summary
from corecoder.checkpoint import CheckpointStore
from corecoder.llm import LLMResponse, ToolCall
from corecoder.runner import TaskRunner
from corecoder.tasks import TaskStep, TaskStore
from corecoder.tools.agent import AgentTool


class FakeLLM:
    def __init__(self, subagent_done: threading.Event | None = None):
        self.calls = []
        self.subagent_done = subagent_done
        self.model = "gpt-5"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append({"messages": messages, "tools": tools})
        last_user = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "",
        )
        if last_user == "slow background research":
            time.sleep(0.05)
            if self.subagent_done is not None:
                self.subagent_done.set()
            return LLMResponse(content="sub-agent result")
        if len(self.calls) == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-agent",
                        name="agent",
                        arguments={"task": "slow background research"},
                    ),
                ]
            )
        return LLMResponse(content="main agent continues")


def _patch_task_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / ".corecoder" / "tasks"
    monkeypatch.setattr(tasks_module, "TASKS_DIR", root)
    monkeypatch.setattr(events_module, "TASKS_DIR", root)
    monkeypatch.setattr(checkpoint_module, "TASKS_DIR", root)
    monkeypatch.setattr(runner_module, "TASKS_DIR", root)
    return root


def _running_task(store: TaskStore, *, budgets=None):
    task = store.create_task(
        session_id="session-a",
        user_goal="Use a background sub-agent",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="background-task",
        steps=[
            TaskStep(id="S1", title="Explore in background", acceptance="Result is persisted."),
            TaskStep(id="S2", title="Continue after result", acceptance="Notification is injected."),
        ],
        budgets=budgets,
    )
    store.update_status(task.id, "awaiting_approval")
    return store.update_status(task.id, "running")


def _wait_for(predicate, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_background_registry_returns_handle_persists_result_and_reconciles_once(tmp_path):
    registry = BackgroundRegistry("task-a", tmp_path)
    done = threading.Event()

    def run_subagent():
        time.sleep(0.05)
        done.set()
        return "background result"

    result = registry.run_with_foreground_budget(
        description="Explore code",
        prompt="slow background research",
        foreground_seconds=0.001,
        run=run_subagent,
        format_foreground_result=lambda value: f"[Sub-agent completed]\n{value}",
    )

    assert "[Sub-agent backgrounded]" in result
    assert "agent-" in result
    assert registry.list_handles()[0]["status"] == "backgrounded"

    assert done.wait(1.0)
    handle = _wait_for(
        lambda: next(
            (item for item in registry.list_handles() if item["status"] == "completed"),
            None,
        )
    )
    assert handle["reconciled"] is False
    assert (tmp_path / "task-a" / handle["transcript_path"]).exists()
    assert (tmp_path / "task-a" / handle["result_path"]).read_text(encoding="utf-8")

    notifications = registry.reconcile_completed()

    assert len(notifications) == 1
    assert notifications[0].startswith("<task-notification>")
    assert "Background sub-agent completed" in notifications[0]
    assert "background result" in notifications[0]
    assert registry.reconcile_completed() == []
    assert registry.list_handles()[0]["reconciled"] is True


def test_task_runner_backgrounds_slow_subagent_and_keeps_task_running(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store, budgets={"subagent_foreground_seconds": 0})
    subagent_done = threading.Event()
    agent = Agent(llm=FakeLLM(subagent_done), tools=[AgentTool()], max_rounds=4)
    runner = TaskRunner(agent=agent, store=store)

    response = runner.run(task.id, "Approved plan context")

    assert response == "main agent continues"
    assert store.load(task.id).status == "running"
    registry = BackgroundRegistry(task.id, root)
    handle = _wait_for(
        lambda: next(
            (item for item in registry.list_handles() if item["status"] == "completed"),
            None,
        )
    )
    assert subagent_done.is_set()
    assert handle["reconciled"] is False
    checkpoint = CheckpointStore(root).load(task.id)
    assert checkpoint.background_subagents[0]["id"] == handle["id"]


def test_task_runner_reconciles_completed_background_result_once(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)
    store.update_step(task.id, "S1", "completed", notes="Background result exists.")
    store.update_step(task.id, "S2", "completed", notes="Ready to finish.")
    registry = BackgroundRegistry(task.id, root)
    def run_subagent():
        time.sleep(0.05)
        return "stored background result"

    registry.run_with_foreground_budget(
        description="Explore code",
        prompt="slow background research",
        foreground_seconds=0.001,
        run=run_subagent,
        format_foreground_result=lambda value: value,
    )
    _wait_for(lambda: registry.list_handles()[0]["status"] == "completed")

    agent = Agent(llm=FakeLLM(), tools=[], max_rounds=2)
    runner = TaskRunner(agent=agent, store=store)

    runner.run(task.id, "Continue approved task")
    messages_with_notification = [
        message
        for message in agent.messages
        if "<task-notification>" in (message.get("content") or "")
    ]

    assert len(messages_with_notification) == 1
    assert "stored background result" in messages_with_notification[0]["content"]
    assert registry.list_handles()[0]["reconciled"] is True
    assert registry.reconcile_completed() == []


def test_background_registry_cancel_marks_handle_without_result_notification(tmp_path):
    registry = BackgroundRegistry("task-a", tmp_path)
    blocker = threading.Event()
    started = threading.Event()

    def run_subagent():
        started.set()
        blocker.wait(1.0)
        return "late result"

    registry.run_with_foreground_budget(
        description="Explore code",
        prompt="slow background research",
        foreground_seconds=0,
        run=run_subagent,
        format_foreground_result=lambda value: value,
    )
    assert started.wait(1.0)
    handle = registry.list_handles()[0]

    cancelled = registry.cancel(handle["id"], reason="User requested cancel")

    assert cancelled["status"] == "cancelled"
    assert cancelled["reconciled"] is True
    assert registry.reconcile_completed() == []
    blocker.set()


def test_task_summary_and_logs_show_background_status(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)
    registry = BackgroundRegistry(task.id, root)

    def run_subagent():
        time.sleep(0.05)
        return "summary output"

    registry.run_with_foreground_budget(
        description="Explore code",
        prompt="slow background research",
        foreground_seconds=0.001,
        run=run_subagent,
        format_foreground_result=lambda value: value,
    )
    handle = _wait_for(
        lambda: next(
            (item for item in registry.list_handles() if item["status"] == "completed"),
            None,
        )
    )

    summary = _format_task_summary(task, registry.list_handles())
    logs = registry.read_logs(handle["id"])

    assert "Background agents:" in summary
    assert "completed pending reconciliation" in summary
    assert handle["transcript_path"] in summary
    assert "summary output" in logs
