"""Cursor based, forward only paging over the committed stream.

The query in :mod:`mixplant.decision.query` returns "the newest N rows": every
call rescans the whole stream and pagination can only walk backwards, which
duplicates and skips rows when several records share a tick or when a rollback
rewrites the effective view.  This module fixes both problems on purpose:

* ordering is fixed to ``(tick, seq)`` and a cursor names the last position the
  caller saw, so continuation reads strictly after it.  The stream appends in
  seq order, so ``(tick, seq)`` is a total order and the same cursor resumed
  concurrently returns the same rows - reads are idempotent without sessions;
* the rollback policy is decided by the query type, not by the caller.  The
  ``journal`` view reads the append only committed stream and a cursor survives
  every rollback (history already paged can never be silently rewritten); the
  ``effective`` view hides tombstones, so a cursor stamped with an older epoch
  is refused with :class:`~mixplant.errors.StaleCursorError` and the reader
  restarts from the beginning.  ``VIEW_POLICIES`` is the single place that maps
  a query type to its policy, and nothing may override it per request.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..errors import CursorError, StaleCursorError, ValidationError
from ..store.epoch import Epoch, epoch_from_document, stream_epoch
from ..store.records import INTERNAL_KINDS, Record
from ..store.tombstone import effective_records

JOURNAL_VIEW = "journal"
EFFECTIVE_VIEW = "effective"
VIEWS = (JOURNAL_VIEW, EFFECTIVE_VIEW)
DEFAULT_VIEW = JOURNAL_VIEW
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 500
# Station local signing key.  Cursors only need to be tamper evident inside one
# deployment; rotate the configuration value to retire every outstanding one.
DEFAULT_CURSOR_KEY = "mixplant.page-cursor.v1"


@dataclass(frozen=True, slots=True)
class ViewPolicy:
    """How one query type treats rollbacks.

    ``invalidate_on_rollback`` decides whether an old cursor is refused
    (effective view) or reused (journal view); ``exposes_internal`` decides
    whether tombstones and rollback markers are readable rows.
    """

    view: str
    invalidate_on_rollback: bool
    exposes_internal: bool


# The rollback policy map.  This is fixed by query type: callers cannot ask an
# effective query to keep a stale cursor or ask a journal query to invalidate.
VIEW_POLICIES: Mapping[str, ViewPolicy] = {
    JOURNAL_VIEW: ViewPolicy(JOURNAL_VIEW, invalidate_on_rollback=False, exposes_internal=True),
    EFFECTIVE_VIEW: ViewPolicy(EFFECTIVE_VIEW, invalidate_on_rollback=True, exposes_internal=False),
}


def policy_for(view: str) -> ViewPolicy:
    """Return the fixed rollback policy of a query type."""

    policy = VIEW_POLICIES.get(view)
    if policy is None:
        raise ValidationError("unknown paging view", view=view, known=list(VIEW_POLICIES))
    return policy


@dataclass(frozen=True, slots=True)
class PageQuery:
    """The filter and the fixed order a paged read applies to the stream."""

    view: str = DEFAULT_VIEW
    kind: str | None = None
    origin: str | None = None
    since_tick: int | None = None
    include_tombstones: bool | None = None
    page_size: int = DEFAULT_PAGE_SIZE

    def __post_init__(self) -> None:
        policy_for(self.view)
        if self.page_size < 1:
            raise ValidationError("page_size must be positive", page_size=self.page_size)
        if self.page_size > MAX_PAGE_SIZE:
            raise ValidationError("page_size is too large", page_size=self.page_size, limit=MAX_PAGE_SIZE)
        if self.since_tick is not None and self.since_tick < 0:
            raise ValidationError("query tick must not be negative", since_tick=self.since_tick)
        if self.include_tombstones is None:
            # The journal is the whole append only stream, tombstones included;
            # the effective view can never expose internal records.
            object.__setattr__(self, "include_tombstones", self.view == JOURNAL_VIEW)
        elif self.include_tombstones and self.view != JOURNAL_VIEW:
            raise ValidationError(
                "only the append only journal view can expose tombstones",
                view=self.view,
            )
    def identity(self) -> dict[str, Any]:
        """The filter part embedded in a cursor so it cannot silently change."""

        return {
            "view": self.view,
            "kind": self.kind,
            "origin": self.origin,
            "include_tombstones": self.include_tombstones,
        }

    def matches(self, record: Record) -> bool:
        if record.kind in INTERNAL_KINDS and not self.include_tombstones:
            return False
        if self.kind is not None and record.kind != self.kind:
            return False
        if self.origin is not None and record.origin != self.origin:
            return False
        return True


@dataclass(frozen=True, slots=True)
class CursorToken:
    """The last position a caller saw, together with its consistency epoch."""

    view: str
    after_tick: int
    after_seq: int
    epoch: Epoch
    filter_id: Mapping[str, Any]

    @staticmethod
    def _prefix() -> bytes:
        return b"mixplant-cursor-v1"

    def encode(self, key: str) -> str:
        body = {
            "view": self.view,
            "after_tick": self.after_tick,
            "after_seq": self.after_seq,
            "epoch": self.epoch.describe(),
            "filter": dict(self.filter_id),
        }
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload = base64.urlsafe_b64encode(raw).rstrip(b"=")
        digest = hmac.new(key.encode("utf-8"), self._prefix() + b"." + payload, hashlib.sha256).digest()
        signature = base64.urlsafe_b64encode(digest).rstrip(b"=")
        return f"{payload.decode('ascii')}.{signature.decode('ascii')}"

    @classmethod
    def decode(cls, token: str, key: str) -> "CursorToken":
        try:
            payload_text, signature_text = token.split(".", 1)
            padding = "=" * (-len(payload_text) % 4)
            payload = base64.urlsafe_b64decode(payload_text + padding)
            signature_padding = "=" * (-len(signature_text) % 4)
            signature = base64.urlsafe_b64decode(signature_text + signature_padding)
        except (ValueError, TypeError) as exc:
            raise CursorError("cursor is not well formed") from exc
        expected = hmac.new(
            key.encode("utf-8"), cls._prefix() + b"." + payload_text.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(signature, expected):
            raise CursorError("cursor signature does not match")
        try:
            body = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CursorError("cursor payload is not readable") from exc
        try:
            return cls(
                view=str(body["view"]),
                after_tick=int(body["after_tick"]),
                after_seq=int(body["after_seq"]),
                epoch=epoch_from_document(body.get("epoch") or {}),
                filter_id=dict(body.get("filter") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CursorError("cursor payload is missing fields") from exc


@dataclass(frozen=True, slots=True)
class Page:
    """One fixed order slice plus everything needed to continue or audit it."""

    query: PageQuery
    rows: tuple[Record, ...]
    next_cursor: str | None
    issued_epoch: Epoch
    current_epoch: Epoch
    from_seq: int | None
    to_seq: int | None
    scanned: int

    @property
    def has_more(self) -> bool:
        """Whether another page can be read after this one."""

        return self.next_cursor is not None

    @property
    def epoch_changed(self) -> bool:
        """A rollback landed between the cursor's epoch and this read."""

        return not self.issued_epoch.equals(self.current_epoch)

    def describe(self) -> dict[str, Any]:
        return {
            "view": self.query.view,
            "kind": self.query.kind,
            "origin": self.query.origin,
            "include_tombstones": self.query.include_tombstones,
            "page_size": self.query.page_size,
            "returned": len(self.rows),
            "scanned": self.scanned,
            "has_more": self.next_cursor is not None,
            "next_cursor": self.next_cursor,
            "order": "tick:asc,seq:asc",
            "seq_range": {"from": self.from_seq, "to": self.to_seq},
            "epoch": self.current_epoch.describe(),
            "cursor_epoch": self.issued_epoch.describe(),
            "epoch_changed": self.epoch_changed,
            "rows": [_describe_row(record) for record in self.rows],
        }


def _describe_row(record: Record) -> dict[str, Any]:
    row = {
        "seq": record.seq,
        "kind": record.kind,
        "origin": record.origin,
        "generation": record.generation,
        "tick": record.tick,
        "payload": dict(record.payload),
    }
    if record.tombstone_of is not None:
        row["tombstone_of"] = record.tombstone_of
    return row


def page_query_from_params(params: Mapping[str, Any]) -> PageQuery:
    """Build a paged query from the loose string parameters a console sends."""

    raw_view = params.get("view")
    raw_flag = params.get("include_tombstones")
    raw_limit = params.get("page_size", params.get("limit"))
    raw_tick = params.get("since_tick")
    return PageQuery(
        view=DEFAULT_VIEW if raw_view in (None, "") else str(raw_view),
        kind=_optional_text(params.get("kind")),
        origin=_optional_text(params.get("origin")),
        since_tick=None if raw_tick in (None, "") else int(raw_tick),
        include_tombstones=None if raw_flag in (None, "") else str(raw_flag).lower() in {"1", "true", "yes"},
        page_size=DEFAULT_PAGE_SIZE if raw_limit in (None, "") else int(raw_limit),
    )


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def read_page(
    committed: Iterable[Record],
    query: PageQuery,
    *,
    cursor: str | None = None,
    key: str = DEFAULT_CURSOR_KEY,
) -> Page:
    """Read one fixed order page, continuing from ``cursor`` when supplied."""

    materialised = tuple(committed)
    current_epoch = stream_epoch(materialised)

    token: CursorToken | None = None
    if cursor:
        token = CursorToken.decode(cursor, key)
        if token.view != query.view:
            raise CursorError("cursor was issued for another view", cursor_view=token.view, view=query.view)
        if dict(token.filter_id) != query.identity():
            raise CursorError("cursor filter does not match this query")
        policy = policy_for(query.view)
        if policy.invalidate_on_rollback and not token.epoch.equals(current_epoch):
            raise StaleCursorError(
                "the effective view was rolled back after this cursor was issued; restart from the beginning",
                view=query.view,
                cursor_epoch=token.epoch.describe(),
                current_epoch=current_epoch.describe(),
            )

    if token is not None:
        after_tick, after_seq, issued_epoch = token.after_tick, token.after_seq, token.epoch
    else:
        after_tick = query.since_tick if query.since_tick is not None else 0
        after_seq = 0
        issued_epoch = current_epoch

    candidates = _ordered_candidates(materialised, query)
    scanned = 0
    selected: list[Record] = []
    for record in candidates:
        if (record.tick, record.seq) <= (after_tick, after_seq):
            continue
        scanned += 1
        if query.matches(record):
            selected.append(record)
            if len(selected) == query.page_size:
                break

    has_more = len(selected) == query.page_size and _has_more_after(candidates, selected[-1], query)
    next_cursor: str | None = None
    if selected and has_more:
        last = selected[-1]
        next_cursor = CursorToken(
            view=query.view,
            after_tick=last.tick,
            after_seq=last.seq,
            epoch=issued_epoch,
            filter_id=query.identity(),
        ).encode(key)

    return Page(
        query=query,
        rows=tuple(selected),
        next_cursor=next_cursor,
        issued_epoch=issued_epoch,
        current_epoch=current_epoch,
        from_seq=selected[0].seq if selected else None,
        to_seq=selected[-1].seq if selected else None,
        scanned=scanned,
    )


def _ordered_candidates(committed: tuple[Record, ...], query: PageQuery) -> tuple[Record, ...]:
    policy = policy_for(query.view)
    candidates = committed if policy.exposes_internal else effective_records(committed)
    return tuple(sorted(candidates, key=lambda record: (record.tick, record.seq)))


def _has_more_after(candidates: tuple[Record, ...], last: Record, query: PageQuery) -> bool:
    last_position = (last.tick, last.seq)
    return any(
        (record.tick, record.seq) > last_position and query.matches(record) for record in candidates
    )
