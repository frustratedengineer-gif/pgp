"""Append-only audit log: every deletion (or status change) logs its
reason. Required by the architecture figure's "self-compaction, forget +
audit log" box, and a reviewer's first question about any memory system
that deletes things ("how do I know why X was forgotten?").

Plain JSONL on disk, append-only by construction (open in "a" mode, never
truncated/rewritten) -- deliberately not a database: the audit trail's
whole value is being simple enough to inspect by eye or `grep`.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

VALID_EVENT_TYPES = ("forgotten", "compacted", "updated", "created")


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, memory_id: str, reason: str, extra: dict | None = None) -> None:
        assert event_type in VALID_EVENT_TYPES, f"bad event_type: {event_type}"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "memory_id": memory_id,
            "reason": reason,
            "extra": extra or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
