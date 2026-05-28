"""Interactive REPL - the user-facing terminal interface."""

import sys
import os
import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from .agent import Agent
from .llm import LLM, LiteLLM
from .config import Config
from .session import save_session, load_session, list_sessions
from .planner import PlanError, Planner, build_approved_plan_context
from .runner import TaskRunner
from .tasks import TaskState
from . import __version__

console = Console()


def _parse_args():
    p = argparse.ArgumentParser(
        prog="corecoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $CORECODER_MODEL or gpt-4o)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


def main():
    args = _parse_args()
    config = Config.from_env()

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or CORECODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)

    planner = Planner(session_id=args.resume or "default")
    runner = TaskRunner(agent=agent, store=planner.store)

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
            checkpoint = runner.restore_latest_for_session(args.resume)
            if checkpoint is not None:
                planner.active_task_id = checkpoint.task_id
                console.print(f"[green]Restored task checkpoint: {checkpoint.task_id}[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # one-shot mode
    if args.prompt:
        _run_once(agent, args.prompt)
        return

    # interactive REPL
    _repl(agent, config, planner, runner)


def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    print()


def _repl(
    agent: Agent,
    config: Config,
    planner: Planner | None = None,
    runner: TaskRunner | None = None,
):
    """Interactive read-eval-print loop."""
    planner = planner or Planner()
    runner = runner or TaskRunner(agent=agent, store=planner.store)
    console.print(Panel(
        f"[bold]CoreCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    hist_path = os.path.expanduser("~/.corecoder_history")
    history = FileHistory(hist_path)

    # Enter submits, Escape+Enter inserts a newline (for pasting code blocks etc.)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    while True:
        try:
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/plan_mode":
            planner.enable_plan_mode()
            console.print("[cyan]Plan mode enabled for the next request.[/cyan]")
            continue
        if user_input == "/task":
            task = planner.active_task()
            if task is None:
                console.print("[dim]No active task.[/dim]")
            else:
                console.print(_format_task_summary(task), markup=False)
            continue
        if user_input == "/tasks":
            console.print(
                _format_tasks_summary(planner.store.list_tasks()),
                markup=False,
            )
            continue
        if user_input == "/reset":
            agent.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/save":
            sid = save_session(agent.messages, config.model)
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: corecoder -r {sid}")
            continue
        if user_input == "/diff":
            from .tools.edit import _changed_files
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue

        execution_input = user_input
        run_task_id: str | None = None

        if planner.plan_mode_enabled:
            try:
                task = planner.generate_plan(
                    llm=agent.llm,
                    user_goal=user_input,
                    model=config.model,
                    cwd=os.getcwd(),
                )
            except PlanError as e:
                console.print(f"[red]Plan failed: {e}[/red]")
                continue

            console.print(_format_plan_approval(task), markup=False)
            choice = pt_prompt("Plan choice [approve/reject/revise] > ").strip().lower()
            if choice in {"approve", "a", "yes", "y"}:
                task = planner.approve_plan(task.id)
                run_task_id = task.id
                execution_input = build_approved_plan_context(task, user_input, planner.store.root)
                console.print(f"[green]Plan approved: {task.id}[/green]")
            elif choice in {"reject", "r", "no", "n"}:
                reason = pt_prompt("Reject reason > ").strip()
                task = planner.reject_plan(task.id, reason=reason)
                console.print(f"[yellow]Plan rejected: {task.id}[/yellow]")
                continue
            elif choice in {"revise", "v"}:
                console.print("[yellow]Plan mode stays enabled for revision.[/yellow]")
                continue
            else:
                console.print("[yellow]Unknown choice; plan mode stays enabled.[/yellow]")
                continue

        if run_task_id is None:
            active_task = planner.active_task()
            if active_task is not None and active_task.status in {"running", "paused"}:
                run_task_id = active_task.id

        # call the agent
        streamed: list[str] = []

        def on_token(tok):
            streamed.append(tok)
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        try:
            if run_task_id is not None:
                response = runner.run(
                    run_task_id,
                    execution_input,
                    on_token=on_token,
                    on_tool=on_tool,
                )
            else:
                response = agent.chat(execution_input, on_token=on_token, on_tool=on_tool)
            if streamed:
                print()  # newline after streamed tokens
            else:
                # response wasn't streamed (came after tool calls)
                console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  quit           Exit CoreCoder\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="CoreCoder Help",
        border_style="dim",
    ))


def _format_plan_approval(task: TaskState) -> str:
    lines = [f"Plan generated for task {task.id}", ""]
    for index, step in enumerate(task.steps, start=1):
        lines.append(f"{index}. {step.title}")
        if step.acceptance:
            lines.append(f"   Verify: {step.acceptance}")
    lines.extend(["", "[Approve]  [Reject]  [Revise]"])
    return "\n".join(lines)


def _format_task_summary(task: TaskState) -> str:
    lines = [f"Task {task.id}", f"Status: {task.status}", f"Goal: {task.user_goal}"]
    if task.current_step:
        lines.append(f"Current step: {task.current_step}")
    if task.steps:
        lines.append("Steps:")
        for step in task.steps:
            lines.append(f"- {step.id} [{step.status}] {step.title}")
    if task.last_error:
        lines.append(f"Last error: {task.last_error}")
    return "\n".join(lines)


def _format_tasks_summary(tasks: list[TaskState]) -> str:
    if not tasks:
        return "No tasks."
    lines = []
    for task in tasks:
        lines.append(f"{task.id}  {task.status}  {task.user_goal}")
    return "\n".join(lines)


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
