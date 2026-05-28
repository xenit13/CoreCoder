"""Sub-agent spawning (inspired by Claude Code's AgentTool, 1397 lines).

The idea: for complex sub-tasks, spawn an independent agent with its own
conversation history and tool access. This lets the main agent delegate
work like "go research this codebase and report back" without polluting
its own context window.

The sub-agent runs to completion and returns a text summary.
"""

from .base import Tool


class AgentTool(Tool):
    name = "agent"
    description = (
        "Spawn a sub-agent to handle a complex sub-task independently. "
        "The sub-agent has its own context and tool access. Use this for: "
        "researching a codebase, implementing a multi-step change in isolation, "
        "or any task that would benefit from a fresh context window."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
        },
        "required": ["task"],
    }

    # set by Agent.__init__ after construction
    _parent_agent = None
    _background_registry = None
    _background_foreground_seconds = None

    def execute(self, task: str) -> str:
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        registry = self._background_registry
        if registry is not None:
            return registry.run_with_foreground_budget(
                description=task,
                prompt=task,
                foreground_seconds=float(self._background_foreground_seconds or 0),
                run=lambda: self._run_subagent(task, read_only=True),
                format_foreground_result=lambda result: f"[Sub-agent completed]\n{result}",
                format_foreground_error=lambda exc: f"Sub-agent error: {exc}",
            )

        try:
            result = self._run_subagent(task, read_only=False)
            return f"[Sub-agent completed]\n{result}"
        except Exception as e:
            return f"Sub-agent error: {e}"

    def configure_background(self, registry=None, foreground_seconds=None) -> None:
        self._background_registry = registry
        self._background_foreground_seconds = foreground_seconds

    def _run_subagent(self, task: str, *, read_only: bool) -> str:
        # import here to avoid circular dep
        from ..agent import Agent

        parent = self._parent_agent
        sub = Agent(
            llm=parent.llm,
            tools=_subagent_tools(parent.tools, read_only=read_only),
            max_context_tokens=parent.context.max_tokens,
            max_rounds=20,
        )

        result = sub.chat(task)
        # trim long results to avoid blowing up parent context
        if len(result) > 5000:
            result = result[:4500] + "\n... (sub-agent output truncated)"
        return result


def _subagent_tools(tools, *, read_only: bool):
    tool_list = [tool for tool in tools if tool.name not in {"agent", "update_task_progress"}]
    if not read_only:
        return tool_list
    read_only_names = {"read_file", "glob", "grep"}
    return [tool for tool in tool_list if tool.name in read_only_names]
