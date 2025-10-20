"""Structured telemetry output helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .logger import get_logger


@dataclass(slots=True)
class AuditSink:
    """Persist decisions and health signals for observability."""

    decision_log_path: Path
    heartbeat_path: Path

    def __post_init__(self) -> None:
        self._log = get_logger(__name__)
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    def record_decision(self, payload: dict[str, Any]) -> None:
        enriched = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            **payload,
        }
        self._append_json_line(self.decision_log_path, enriched)

    def record_execution(self, payload: dict[str, Any]) -> None:
        enriched = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            **payload,
        }
        self._append_json_line(self.decision_log_path, enriched)

    def write_heartbeat(self, status: str, *, details: dict[str, Any] | None = None) -> None:
        heartbeat = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "status": status,
            "details": details or {},
        }
        self.heartbeat_path.write_text(json.dumps(heartbeat, indent=2))

    def _append_json_line(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload))
                handle.write("\n")
        except Exception as exc:
            self._log.error("Failed to write telemetry to %s: %s", path, exc)
