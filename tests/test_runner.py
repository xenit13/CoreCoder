from pathlib import Path

from corecoder import checkpoint as checkpoint_module
from corecoder import events as events_module
from corecoder import runner as runner_module
from corecoder import tasks as tasks_module
from corecoder.agent import Agent
from corecoder.checkpoint import CheckpointStore
from corecoder.events import EventLog
from corecoder.llm import LLMResponse, ToolCall
from corecoder.runner import TaskRunner
from corecoder.tasks import TaskStep, TaskStore
from corecoder.tools.base import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Return the provided value."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def execute(self, value: str) -> str:
        return f"echo:{value}"


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model = "gpt-5"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)


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
        user_goal="Implement approved plan",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="runner-task",
        steps=[
            TaskStep(id="S1", title="Use a tool", acceptance="Tool result is observed."),
            TaskStep(id="S2", title="Report done", acceptance="Final response is saved."),
        ],
        budgets=budgets,
    )
    store.update_status(task.id, "awaiting_approval")
    return store.update_status(task.id, "running")


def test_runner_completes_task_only_after_explicit_progress(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)
    llm = FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="update_task_progress",
                        arguments={"step_id": "S1", "status": "in_progress"},
                    ),
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="update_task_progress",
                        arguments={
                            "step_id": "S1",
                            "status": "completed",
                            "notes": "Tool behavior is covered.",
                            "evidence": ["uv run pytest tests/test_runner.py"],
                        },
                    ),
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-3",
                        name="update_task_progress",
                        arguments={"step_id": "S2", "status": "in_progress"},
                    ),
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call-4",
                        name="update_task_progress",
                        arguments={
                            "step_id": "S2",
                            "status": "completed",
                            "notes": "Final response is saved.",
                            "evidence": ["checkpoint contains messages"],
                        },
                    ),
                ]
            ),
            LLMResponse(content="done"),
        ]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], max_rounds=8)
    runner = TaskRunner(agent=agent, store=store)

    result = runner.run(task.id, "Approved plan context")

    assert result == "done"
    reloaded = store.load(task.id)
    assert reloaded.status == "completed"
    assert reloaded.current_step is None
    assert [step.status for step in reloaded.steps] == ["completed", "completed"]
    tool_names = {tool["function"]["name"] for tool in llm.calls[0]["tools"]}
    assert "update_task_progress" in tool_names

    checkpoint = CheckpointStore(root).load(task.id)
    assert checkpoint is not None
    assert checkpoint.tool_counters == {"rounds": 5, "tool_calls": 4}
    assert checkpoint.messages[0]["content"] == "Approved plan context"

    events = [event["type"] for event in EventLog(task.id, root).read()]
    assert "step_started" in events
    assert events.count("step_completed") == 2
    assert "task_completed" in events


def test_runner_blocks_task_when_tool_budget_is_exceeded(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store, budgets={"max_tool_calls": 0})
    llm = FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"value": "too-many"}),
                ]
            ),
        ]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], max_rounds=3)
    runner = TaskRunner(agent=agent, store=store)

    result = runner.run(task.id, "Approved plan context")

    assert "tool call budget exceeded" in result
    reloaded = store.load(task.id)
    assert reloaded.status == "blocked"
    assert "tool call budget exceeded" in reloaded.last_error
    checkpoint = CheckpointStore(root).load(task.id)
    assert checkpoint.tool_counters == {"rounds": 1, "tool_calls": 1}
    assert checkpoint.task.status == "blocked"
    events = [event["type"] for event in EventLog(task.id, root).read()]
    assert "task_blocked" in events


def test_runner_blocks_when_agent_returns_before_all_steps_complete(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)
    llm = FakeLLM([LLMResponse(content="done without progress")])
    agent = Agent(llm=llm, tools=[EchoTool()], max_rounds=3)
    runner = TaskRunner(agent=agent, store=store)

    result = runner.run(task.id, "Approved plan context")

    assert "before completing all plan steps" in result
    reloaded = store.load(task.id)
    assert reloaded.status == "blocked"
    assert reloaded.last_error == "Agent returned before completing all plan steps"
    assert [step.status for step in reloaded.steps] == ["pending", "pending"]


def test_runner_injects_light_progress_reminder_after_inactive_rounds(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore()
    task = _running_task(store)
    llm = FakeLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"value": "one"}),
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(id="call-2", name="echo", arguments={"value": "two"}),
                ]
            ),
            LLMResponse(content="done"),
        ]
    )
    agent = Agent(llm=llm, tools=[EchoTool()], max_rounds=5)
    runner = TaskRunner(agent=agent, store=store, progress_reminder_rounds=2)

    runner.run(task.id, "Approved plan context")

    third_call_messages = llm.calls[2]["messages"]
    reminder = third_call_messages[-1]["content"]
    assert "<system-reminder>" in reminder
    assert "Current task progress:" in reminder
    assert "update_task_progress" in reminder
    assert "S1 [pending]" in reminder


def test_runner_marks_task_failed_when_agent_raises(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)

    class FailingAgent:
        def __init__(self):
            self.messages = []

        def chat(self, prompt, on_token=None, on_tool=None, on_round=None):
            self.messages.append({"role": "user", "content": prompt})
            raise RuntimeError("agent failed")

    runner = TaskRunner(agent=FailingAgent(), store=store)

    try:
        runner.run(task.id, "Approved plan context")
    except RuntimeError as exc:
        assert str(exc) == "agent failed"
    else:
        raise AssertionError("expected agent failure")

    reloaded = store.load(task.id)
    assert reloaded.status == "failed"
    assert reloaded.last_error == "agent failed"
    checkpoint = CheckpointStore(root).load(task.id)
    assert checkpoint.task.status == "failed"
    events = [event["type"] for event in EventLog(task.id, root).read()]
    assert "task_failed" in events


def test_runner_restores_latest_checkpoint_for_session(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore(root)
    task = _running_task(store)
    task = store.update_status(task.id, "running", current_step="S2")
    store.update_step(task.id, "S1", "completed")
    messages = [{"role": "user", "content": "Approved plan context"}]
    CheckpointStore(root).save(
        task=store.load(task.id),
        messages=messages,
        tool_counters={"rounds": 3, "tool_calls": 4},
    )
    agent = Agent(llm=FakeLLM([]), tools=[EchoTool()])
    runner = TaskRunner(agent=agent, store=store)

    checkpoint = runner.restore_latest_for_session("session-a")

    assert checkpoint is not None
    assert agent.messages[0] == messages[0]
    assert "Current task progress:" in agent.messages[-1]["content"]
    assert "update_task_progress" in agent.messages[-1]["content"]
    assert checkpoint.task.current_step == "S2"
    assert checkpoint.task.steps[0].status == "completed"
