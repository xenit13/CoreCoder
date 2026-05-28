"""Local background sub-agent registry for durable tasks."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .tasks import TASKS_DIR, _write_json_atomic, task_dir

BackgroundRun = Callable[[], str]
ForegroundFormatter = Callable[[str], str]
ErrorFormatter = Callable[[BaseException], str]

ACTIVE_BACKGROUND_STATUSES = {"foreground", "backgrounded"}
PENDING_RECONCILIATION_STATUSES = {"completed", "failed"}


class BackgroundRegistry:
    """Track background sub-agents for one durable task."""

    def __init__(self, task_id: str, root: Path | None = None):
        self.task_id = task_id
        self.root = root or TASKS_DIR
        self._lock = threading.RLock()

    @property
    def task_dir(self) -> Path:
        return task_dir(self.task_id, self.root)

    @property
    def background_dir(self) -> Path:
        return self.task_dir / "background"

    @property
    def index_path(self) -> Path:
        return self.background_dir / "background.json"

    def run_with_foreground_budget(
        self,
        *,
        description: str,
        prompt: str,
        foreground_seconds: float,
        run: BackgroundRun,
        format_foreground_result: ForegroundFormatter | None = None,
        format_foreground_error: ErrorFormatter | None = None,
    ) -> str:
        """Run a sub-agent synchronously until the foreground budget expires."""

        formatter = format_foreground_result or (lambda value: value)
        error_formatter = format_foreground_error or (lambda exc: f"Sub-agent error: {exc}")
        finished = threading.Event()
        state: dict[str, Any] = {"agent_id": None, "completed": False}
        state_lock = threading.Lock()

        def worker() -> None:
            try:
                result = run()
            except BaseException as exc:
                with state_lock:
                    state.update({"completed": True, "error": exc})
                    agent_id = state["agent_id"]
                finished.set()
                if agent_id:
                    self._finish_background(agent_id, error=exc)
                return
            with state_lock:
                state.update({"completed": True, "result": result})
                agent_id = state["agent_id"]
            finished.set()
            if agent_id:
                self._finish_background(agent_id, result=result)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        if finished.wait(max(0.0, foreground_seconds)):
            thread.join(timeout=0)
            if "error" in state:
                return error_formatter(state["error"])
            return formatter(state.get("result", ""))

        handle = self._create_handle(description=description, prompt=prompt)
        with state_lock:
            state["agent_id"] = handle["id"]
            completed = state["completed"]
            result = state.get("result")
            error = state.get("error")
        if completed:
            if error is not None:
                self._finish_background(handle["id"], error=error)
            else:
                self._finish_background(handle["id"], result=result or "")
        return _format_background_handle(handle)

    def list_handles(self) -> list[dict[str, Any]]:
        return self._load_index()

    def has_active_or_unreconciled(self) -> bool:
        for handle in self.list_handles():
            status = handle.get("status")
            if status in ACTIVE_BACKGROUND_STATUSES:
                return True
            if status in PENDING_RECONCILIATION_STATUSES and not handle.get("reconciled"):
                return True
        return False

    def reconcile_completed(self) -> list[str]:
        notifications: list[str] = []
        with self._lock:
            handles = self._load_index()
            changed = False
            for handle in handles:
                if handle.get("status") not in PENDING_RECONCILIATION_STATUSES:
                    continue
                if handle.get("reconciled"):
                    continue
                notifications.append(self._notification_for(handle))
                handle["reconciled"] = True
                handle["updated_at"] = _now()
                changed = True
                self._append_transcript(handle["id"], "reconciled")
            if changed:
                self._save_index(handles)
        return notifications

    def cancel(self, agent_id: str, *, reason: str = "Cancelled") -> dict[str, Any]:
        with self._lock:
            handles = self._load_index()
            for handle in handles:
                if handle["id"] != agent_id:
                    continue
                if handle["status"] not in ACTIVE_BACKGROUND_STATUSES:
                    return handle
                handle["status"] = "cancelled"
                handle["last_error"] = reason
                handle["reconciled"] = True
                handle["updated_at"] = _now()
                self._write_result(handle, status="cancelled", error=reason)
                self._append_transcript(agent_id, "cancelled", reason=reason)
                self._save_index(handles)
                return handle
        raise KeyError(f"Unknown background agent: {agent_id}")

    def cancel_all(self, *, reason: str = "Cancelled") -> list[dict[str, Any]]:
        cancelled = []
        for handle in self.list_handles():
            if handle.get("status") in ACTIVE_BACKGROUND_STATUSES:
                cancelled.append(self.cancel(handle["id"], reason=reason))
        return cancelled

    def mark_interrupted(self, *, reason: str = "Interrupted by process restart") -> list[dict[str, Any]]:
        interrupted = []
        with self._lock:
            handles = self._load_index()
            changed = False
            for handle in handles:
                if handle.get("status") not in ACTIVE_BACKGROUND_STATUSES:
                    continue
                handle["status"] = "failed"
                handle["last_error"] = reason
                handle["reconciled"] = False
                handle["updated_at"] = _now()
                self._write_result(handle, status="failed", error=reason)
                self._append_transcript(handle["id"], "interrupted", reason=reason)
                interrupted.append(handle)
                changed = True
            if changed:
                self._save_index(handles)
        return interrupted

    def read_logs(self, agent_id: str | None = None) -> str:
        handles = self.list_handles()
        if not handles:
            return "No background agents."
        selected = handles
        if agent_id is not None:
            selected = [handle for handle in handles if handle["id"] == agent_id]
            if not selected:
                raise KeyError(f"Unknown background agent: {agent_id}")

        chunks = []
        for handle in selected:
            chunks.append(f"Background agent {handle['id']} [{_display_status(handle)}]")
            transcript_path = self.task_dir / handle["transcript_path"]
            if transcript_path.exists():
                chunks.append(transcript_path.read_text(encoding="utf-8").strip())
            result_path = self.task_dir / handle["result_path"]
            if result_path.exists():
                chunks.append(result_path.read_text(encoding="utf-8").strip())
        return "\n".join(chunk for chunk in chunks if chunk)

    def _create_handle(self, *, description: str, prompt: str) -> dict[str, Any]:
        agent_id = f"agent-{uuid.uuid4().hex[:12]}"
        now = _now()
        handle = {
            "id": agent_id,
            "task_id": self.task_id,
            "description": description,
            "prompt": prompt,
            "status": "backgrounded",
            "started_at": now,
            "updated_at": now,
            "transcript_path": f"background/{agent_id}.jsonl",
            "result_path": f"background/{agent_id}.result.json",
            "reconciled": False,
            "last_error": None,
        }
        with self._lock:
            handles = self._load_index()
            handles.append(handle)
            self._save_index(handles)
            self._append_transcript(agent_id, "backgrounded", prompt=prompt)
        return handle

    def _finish_background(
        self,
        agent_id: str,
        *,
        result: str | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        with self._lock:
            handles = self._load_index()
            for handle in handles:
                if handle["id"] != agent_id:
                    continue
                if handle.get("status") == "cancelled":
                    self._append_transcript(agent_id, "completion_ignored_after_cancel")
                    return
                if error is not None:
                    error_text = str(error)
                    handle["status"] = "failed"
                    handle["last_error"] = error_text
                    self._write_result(handle, status="failed", error=error_text)
                    self._append_transcript(agent_id, "failed", error=error_text)
                else:
                    handle["status"] = "completed"
                    handle["last_error"] = None
                    self._write_result(handle, status="completed", result=result or "")
                    self._append_transcript(agent_id, "completed")
                handle["reconciled"] = False
                handle["updated_at"] = _now()
                self._save_index(handles)
                return

    def _notification_for(self, handle: dict[str, Any]) -> str:
        result = self._read_result(handle)
        status = result.get("status", handle.get("status", "completed"))
        body = result.get("result") or result.get("error") or ""
        lines = [
            "<task-notification>",
            f"Background sub-agent completed: {handle['id']}",
            f"Task: {handle.get('description') or handle.get('prompt') or ''}",
            f"Status: {status}",
            f"Result path: {handle.get('result_path')}",
            "Result:",
            str(body),
            "Use this result to continue the approved task. Do not call the same sub-agent again unless more information is needed.",
            "</task-notification>",
        ]
        return "\n".join(lines)

    def _read_result(self, handle: dict[str, Any]) -> dict[str, Any]:
        path = self.task_dir / handle["result_path"]
        if not path.exists():
            return {"status": handle.get("status"), "result": ""}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_result(
        self,
        handle: dict[str, Any],
        *,
        status: str,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "id": handle["id"],
            "task_id": self.task_id,
            "description": handle.get("description", ""),
            "prompt": handle.get("prompt", ""),
            "status": status,
            "result": result,
            "error": error,
            "completed_at": _now(),
        }
        _write_json_atomic(self.task_dir / handle["result_path"], payload)

    def _append_transcript(self, agent_id: str, event_type: str, **fields: Any) -> None:
        path = self.background_dir / f"{agent_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"type": event_type, "time": _now(), **fields}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return list(data)
        return list(data.get("agents", []))

    def _save_index(self, handles: list[dict[str, Any]]) -> None:
        _write_json_atomic(self.index_path, {"agents": handles})


def _format_background_handle(handle: dict[str, Any]) -> str:
    return "\n".join(
        [
            "[Sub-agent backgrounded]",
            f"agent_id: {handle['id']}",
            "status: backgrounded",
            f"transcript_path: {handle['transcript_path']}",
            f"result_path: {handle['result_path']}",
            "The sub-agent is still running. Its result will be injected later as <task-notification>.",
        ]
    )


def _display_status(handle: dict[str, Any]) -> str:
    status = handle.get("status", "unknown")
    if status in PENDING_RECONCILIATION_STATUSES and not handle.get("reconciled"):
        return f"{status} pending reconciliation"
    return status


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
