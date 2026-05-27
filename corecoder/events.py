"""Append-only task event log."""

import json
import time
from pathlib import Path
from typing import Any

from .tasks import TASKS_DIR, task_dir


class EventLog:
    """Append and read JSONL task events."""

    def __init__(self, task_id: str, root: Path | None = None):
        self.task_id = task_id
        self.root = root or TASKS_DIR

    @property
    def path(self) -> Path:
        return task_dir(self.task_id, self.root) / "events.jsonl"

    def append(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            **fields,
            "type": event_type,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events
