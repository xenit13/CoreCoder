import json
from pathlib import Path

from corecoder.agent import Agent
from corecoder import cli as cli_module
from corecoder import checkpoint as checkpoint_module
from corecoder import events as events_module
from corecoder import planner as planner_module
from corecoder import tasks as tasks_module
from corecoder.cli import _format_plan_approval, _format_task_summary, _repl
from corecoder.config import Config
from corecoder.events import EventLog
from corecoder.llm import LLMResponse, ToolCall
from corecoder.plan_files import plan_file_path, read_plan_file
from corecoder.planner import Planner
from corecoder.tasks import TaskStatusError, TaskStore
from corecoder.tools.read import ReadFileTool


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append({"messages": messages, "tools": tools})
        return LLMResponse(content=self.content)


def _patch_task_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / ".corecoder" / "tasks"
    monkeypatch.setattr(tasks_module, "TASKS_DIR", root)
    monkeypatch.setattr(events_module, "TASKS_DIR", root)
    monkeypatch.setattr(checkpoint_module, "TASKS_DIR", root)
    monkeypatch.setattr(planner_module, "TASKS_DIR", root)
    return root


def _plan_response() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "id": "S1",
                    "title": "Add planner tests",
                    "acceptance": "Plan mode behavior is covered.",
                },
                {
                    "id": "S2",
                    "title": "Wire CLI approval",
                    "acceptance": "Approval starts normal foreground execution.",
                },
            ]
        }
    )


def test_generate_plan_creates_awaiting_approval_task_with_readonly_tools(monkeypatch, tmp_path):
    root = _patch_task_root(monkeypatch, tmp_path)
    llm = FakeLLM(_plan_response())
    planner = Planner(session_id="session-a")

    planner.enable_plan_mode()
    task = planner.generate_plan(
        llm=llm,
        user_goal="Add plan mode",
        model="gpt-5",
        cwd="/data/CoreCoder",
    )

    assert task.status == "awaiting_approval"
    assert planner.plan_mode_enabled is True
    assert planner.active_task_id == task.id
    assert [step.id for step in task.steps] == ["S1", "S2"]
    assert [step.status for step in task.steps] == ["pending", "pending"]
    plan_path = plan_file_path(task.id, root)
    assert plan_path.exists()
    plan_content = read_plan_file(task.id, root)
    assert "Add plan mode" in plan_content
    assert "Add planner tests" in plan_content
    assert "Plan mode behavior is covered." in plan_content
    tool_names = {tool["function"]["name"] for tool in llm.calls[0]["tools"]}
    assert tool_names == {"glob", "grep", "read_file"}
    assert "Do not edit files" in llm.calls[0]["messages"][0]["content"]
    assert EventLog(task.id).read()[0]["type"] == "plan_generated"


def test_restricted_agent_rejects_tools_not_in_instance_allowlist():
    class ToolCallingLLM:
        def __init__(self):
            self.responses = [
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="call-1", name="edit_file", arguments={}),
                    ]
                ),
                LLMResponse(content="done"),
            ]

        def chat(self, messages, tools=None, on_token=None):
            return self.responses.pop(0)

    agent = Agent(llm=ToolCallingLLM(), tools=[ReadFileTool()], max_rounds=2)

    assert agent.chat("try editing") == "done"
    assert agent.messages[2]["role"] == "tool"
    assert "unknown tool 'edit_file'" in agent.messages[2]["content"]


def test_approve_plan_marks_running_and_disables_plan_mode(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    planner = Planner(session_id="session-a")
    planner.enable_plan_mode()
    task = planner.generate_plan(
        llm=FakeLLM(_plan_response()),
        user_goal="Add plan mode",
        model="gpt-5",
        cwd="/data/CoreCoder",
    )

    approved = planner.approve_plan()

    assert approved.id == task.id
    assert approved.status == "running"
    assert planner.plan_mode_enabled is False
    assert [event["type"] for event in EventLog(task.id).read()] == [
        "plan_generated",
        "plan_approved",
    ]


def test_reject_plan_records_reason_and_disables_plan_mode(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    planner = Planner(session_id="session-a")
    planner.enable_plan_mode()
    task = planner.generate_plan(
        llm=FakeLLM(_plan_response()),
        user_goal="Add plan mode",
        model="gpt-5",
        cwd="/data/CoreCoder",
    )

    rejected = planner.reject_plan(reason="Too broad")

    assert rejected.id == task.id
    assert rejected.status == "cancelled"
    assert rejected.last_error == "Too broad"
    assert planner.plan_mode_enabled is False
    assert EventLog(task.id).read()[-1]["type"] == "plan_rejected"


def test_task_store_rejects_invalid_status_transition(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    store = TaskStore()
    task = store.create_task(
        session_id="default",
        user_goal="Add plan mode",
        model="gpt-5",
        cwd="/data/CoreCoder",
        task_id="plan-task",
    )

    try:
        store.update_status(task.id, "running")
    except TaskStatusError as exc:
        assert "Invalid task status transition" in str(exc)
    else:
        raise AssertionError("expected invalid task transition to fail")


def test_cli_formats_active_task_and_plan_choices(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    planner = Planner(session_id="session-a")
    planner.enable_plan_mode()
    task = planner.generate_plan(
        llm=FakeLLM(_plan_response()),
        user_goal="Add plan mode",
        model="gpt-5",
        cwd="/data/CoreCoder",
    )

    assert "[Approve]  [Reject]  [Revise]" in _format_plan_approval(task)
    summary = _format_task_summary(task)
    assert task.id in summary
    assert "awaiting_approval" in summary
    assert "S1" in summary
    assert "Add planner tests" in summary


def test_cli_uses_approved_plan_as_foreground_execution_context(monkeypatch, tmp_path):
    _patch_task_root(monkeypatch, tmp_path)
    inputs = iter(["Implement phase2", "approve", "quit"])
    monkeypatch.setattr(cli_module, "pt_prompt", lambda *args, **kwargs: next(inputs))

    class ForegroundAgent:
        def __init__(self):
            self.llm = FakeLLM(_plan_response())
            self.prompts = []

        def chat(self, prompt, on_token=None, on_tool=None, on_round=None):
            self.prompts.append(prompt)
            return "done"

    planner = Planner(session_id="session-a")
    planner.enable_plan_mode()
    agent = ForegroundAgent()

    _repl(agent, Config(model="gpt-5"), planner)

    assert len(agent.prompts) == 1
    execution_prompt = agent.prompts[0]
    assert "Approved plan" in execution_prompt
    assert "Plan file:" in execution_prompt
    assert "update_task_progress" in execution_prompt
    assert "Original request:" in execution_prompt
    assert "Implement phase2" in execution_prompt
    assert "Add planner tests" in execution_prompt
    assert "Plan mode behavior is covered." in execution_prompt
    assert planner.plan_mode_enabled is False
