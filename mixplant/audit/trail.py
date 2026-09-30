"""Durable audit trail of page queries.

Page serving must be audited, but a read must never mutate the production
journal it reads from -- otherwise walking the stream would append an audit
record per page and feed records back into the walk.  Page events therefore go
to a trail of their own: one json line per event, append only, in call order.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ValidationError


@dataclass(frozen=True, slots=True)
class TrailEvent:
    """One audited page serving decision."""

    seq: int
    name: str
    tick: int
    detail: Mapping[str, Any]

    def describe(self) -> dict[str, Any]:
        return {"seq": self.seq, "name": self.name, "tick": self.tick, "detail": dict(self.detail)}


class PageAuditTrail:
    """Append only sink for page serving events, backed by one jsonl file."""

    def __init__(self, path: Path | str | None = None, *, clock: Any | None = None) -> None:
        self._path = None if path is None else Path(path)
        self._clock = clock
        self._events: list[TrailEvent] = []
        self._lock = threading.Lock()
        self._handle = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load()
            self._handle = self._path.open("a", encoding="utf-8")

    def _load(self) -> None:
        assert self._path is not None
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                    self._events.append(
                        TrailEvent(
                            seq=int(document["seq"]),
                            name=str(document["name"]),
                            tick=int(document.get("tick", 0)),
                            detail=dict(document.get("detail") or {}),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValidationError(
                        "page audit trail contains an unreadable line",
                        path=str(self._path),
                        line=number,
                    ) from exc

    def record(self, name: str, detail: Mapping[str, Any], *, tick: int | None = None) -> TrailEvent:
        """Append one event durably and return it."""

        with self._lock:
            when = self._clock.now() if tick is None and self._clock is not None else (tick or 0)
            event = TrailEvent(seq=len(self._events) + 1, name=name, tick=when, detail=dict(detail))
            line = json.dumps(event.describe(), sort_keys=True, ensure_ascii=False)
            self._events.append(event)
            if self._handle is not None:
                self._handle.write(line + "\n")
                self._handle.flush()
        return event

    def events(self) -> tuple[TrailEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return [event.describe() for event in self._events[-limit:]]

    def describe(self, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            return {"count": len(self._events), "events": [event.describe() for event in self._events[-limit:]]}

    def close(self) -> None:
        with self._lock:
            if self._handle is not None and not self._handle.closed:
                self._handle.close()
