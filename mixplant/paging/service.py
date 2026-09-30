"""Incremental, fixed order paging over the committed stream.

A page walk is opened once and continued with the opaque cursor the previous
page returned.  Ordering is always the journal's time order -- records are read
in strictly increasing ``seq`` and the cursor is a *seek position* (``after
which seq``), never an offset, so newly appended records cannot shift rows the
caller has not seen yet and re-requesting the same cursor cannot repeat one.

Three guarantees are enforced here, independently of storage:

* **Idempotent continuation** -- ``(walk_id, after_seq)`` is the key of a
  served page.  Replaying the same cursor concurrently returns the exact page
  already served; the walk position is read and advanced under a lock.  A
  cursor that does not match its own walk is refused, never remapped.

* **History is immutable** -- every page is produced from a view pinned when
  the walk opened.  Records the caller already received cannot be rewritten by
  a later rollback: a stream walk keeps the tombstoned originals in place
  (flagged ``active: false`` with the neutralising seq), a ledger walk freezes
  its pin, and a replay walk is frozen at one tick.

* **Fork policy is fixed per query type** -- see :mod:`mixplant.paging.policy`.
  The server never recomputes a cursor; when the policy says invalidate, the
  caller gets ``cursor_forked`` plus the watermark it needs to open a new walk.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..clock import Clock
from ..errors import CursorExpiredError, CursorForkedError, InvalidCursorError, ValidationError
from ..ids import SequenceIds
from ..store.records import Record
from ..store.repository import RecordRepository
from .cursor import CursorCodec, PageCursor
from .policy import (
    POLICY_CONTINUE,
    POLICY_INVALIDATE,
    POLICY_PINNED,
    SPECS,
    QueryType,
    get_spec,
)

DEFAULT_LIMIT = 20
MAX_LIMIT = 200
WALK_SCOPE = "walk"


@dataclass(frozen=True, slots=True)
class PageFilter:
    """The optional subset a walk is constrained to."""

    kind: str | None = None
    origin: str | None = None

    def accepts(self, record: Record) -> bool:
        if self.kind is not None and record.kind != self.kind:
            return False
        if self.origin is not None and record.origin != self.origin:
            return False
        return True

    @property
    def key(self) -> tuple[str | None, str | None]:
        return (self.kind, self.origin)


@dataclass(frozen=True, slots=True)
class Page:
    """One slice of a walk."""

    qtype: str
    policy: str
    walk_id: str
    rows: tuple[dict[str, Any], ...]
    after_seq: int
    next_after_seq: int
    limit: int
    view: Mapping[str, Any]
    kind: str | None
    origin: str | None
    pinned_end_seq: int
    has_more: bool
    at_pin_end: bool
    tail_open: bool
    watermark: Mapping[str, Any]
    cached: bool = False

    def describe(self, codec: CursorCodec) -> dict[str, Any]:
        next_cursor = PageCursor(
            qtype=self.qtype,
            after_seq=self.next_after_seq,
            view=dict(self.view),
            walk_id=self.walk_id,
            limit=self.limit,
            kind=self.kind,
            origin=self.origin,
        )
        return {
            "query_type": self.qtype,
            "policy": self.policy,
            "walk_id": self.walk_id,
            "rows": list(self.rows),
            "returned": len(self.rows),
            "limit": self.limit,
            "after_seq": self.after_seq,
            "next_after_seq": self.next_after_seq,
            "has_more": self.has_more,
            "at_pin_end": self.at_pin_end,
            "tail_open": self.tail_open,
            "pinned_end_seq": self.pinned_end_seq,
            "cursor": codec.encode(next_cursor),
            "watermark": dict(self.watermark),
            "idempotent": self.cached,
        }


AuditHook = Callable[[str, Mapping[str, Any]], None]


class PageService:
    """Opens and continues cursor walks for one station."""

    def __init__(
        self,
        repository: RecordRepository,
        clock: Clock,
        codec: CursorCodec,
        ids: SequenceIds,
        *,
        audit_hook: AuditHook | None = None,
        page_cache_size: int = 4096,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._codec = codec
        self._ids = ids
        self._audit_hook = audit_hook
        self._lock = threading.RLock()
        self._served: OrderedDict[str, Page] = OrderedDict()
        self._page_cache_size = page_cache_size

    @property
    def codec(self) -> CursorCodec:
        return self._codec

    def query_types(self) -> list[dict[str, str]]:
        """Describe the fixed policy table, useful for the console and audits."""

        return [spec.describe() for spec in SPECS.values()]

    def open_page(
        self,
        qtype: str,
        *,
        limit: int | None = None,
        kind: str | None = None,
        origin: str | None = None,
        as_of_tick: int | None = None,
    ) -> Page:
        """Open a fresh walk and return its first page."""

        spec = get_spec(qtype)
        page_limit = self._valid_limit(limit)
        page_filter = PageFilter(kind=kind or None, origin=origin or None)
        with self._lock:
            walk_id = self._ids.next(WALK_SCOPE)
            view = self._open_view(spec, as_of_tick)
            cursor = PageCursor(
                qtype=qtype,
                after_seq=0,
                view=view,
                walk_id=walk_id,
                limit=page_limit,
                kind=page_filter.kind,
                origin=page_filter.origin,
            )
            return self._serve(cursor, spec, page_filter, opening=True)

    def continue_page(self, token: str) -> Page:
        """Continue a walk from the cursor a previous page returned."""

        with self._lock:
            # Verify first; never trust an unsigned hint.
            cursor = self._codec.decode(token)
            spec = get_spec(cursor.qtype)
            page_filter = PageFilter(kind=cursor.kind, origin=cursor.origin)
            return self._serve(cursor, spec, page_filter, opening=False)

    # -- views -----------------------------------------------------------

    def _open_view(self, spec: QueryType, as_of_tick: int | None) -> dict[str, Any]:
        committed = self._repository.committed_records()
        if spec.policy == POLICY_CONTINUE:
            return {"pin": "open", "opened_seq": committed[-1].seq if committed else 0}
        if spec.policy == POLICY_INVALIDATE:
            hidden = sorted(rec.tombstone_of for rec in committed if rec.is_tombstone)
            return {
                "pin": "seq",
                "pin_seq": committed[-1].seq if committed else 0,
                "hidden": hidden,
            }
        if spec.policy == POLICY_PINNED:
            now = self._clock.now()
            if as_of_tick is None:
                tick = now
            else:
                tick = as_of_tick
                if tick < 0:
                    raise ValidationError("as_of_tick must not be negative", as_of_tick=tick)
                if tick > now:
                    raise ValidationError("as_of_tick is in the future", as_of_tick=tick, now=now)
            return {"pin": "tick", "as_of_tick": tick}
        raise ValidationError("query type has no pinning rule", qtype=spec.name)  # pragma: no cover

    # -- serving ---------------------------------------------------------

    def _serve(self, cursor: PageCursor, spec: QueryType, page_filter: PageFilter, *, opening: bool) -> Page:
        if (cursor.kind, cursor.origin) != page_filter.key:  # pragma: no cover - defensive
            raise InvalidCursorError("cursor filter does not match the request")
        served_key = f"{cursor.walk_id}:{cursor.after_seq}"
        cached_page = self._served.get(served_key)
        if cached_page is not None:
            if cached_page.qtype != spec.name:  # pragma: no cover - walk ids are unique
                raise InvalidCursorError("cursor walk id belongs to another query type")
            self._served.move_to_end(served_key)
            replay = self._with_flag(cached_page, cached=True)
            self._audit(replay, opening=opening)
            return replay

        view = dict(cursor.view)
        pin_end, tail_open = self._check_view(spec, view)
        candidate_records = self._view_records(spec, view)
        selected = tuple(
            rec for rec in candidate_records if rec.seq > cursor.after_seq and page_filter.accepts(rec)
        )
        page_records = selected[: cursor.limit]
        next_after, at_pin_end = self._advance_position(
            spec, cursor.after_seq, page_records, selected, candidate_records, pin_end
        )
        has_more = len(selected) > len(page_records)
        hidden = self._hidden_seqs(spec, view)
        rows = tuple(self._describe_row(rec, hidden) for rec in page_records)
        page = Page(
            qtype=spec.name,
            policy=spec.policy,
            walk_id=cursor.walk_id,
            rows=rows,
            after_seq=cursor.after_seq,
            next_after_seq=next_after,
            limit=cursor.limit,
            view=view,
            kind=cursor.kind,
            origin=cursor.origin,
            pinned_end_seq=pin_end,
            has_more=has_more,
            at_pin_end=at_pin_end,
            tail_open=tail_open,
            watermark=self._repository.watermark().describe(),
            cached=False,
        )
        # Frozen pages: every bounded pin, and every non-empty open page, are
        # remembered so a concurrent replay is byte identical.  An empty page
        # of an open ended walk is a live tail probe and must not freeze.
        if not (tail_open and not page_records):
            self._remember(served_key, page)
        self._audit(page, opening=opening)
        return page

    @staticmethod
    def _with_flag(page: Page, *, cached: bool) -> Page:
        return Page(
            qtype=page.qtype,
            policy=page.policy,
            walk_id=page.walk_id,
            rows=page.rows,
            after_seq=page.after_seq,
            next_after_seq=page.next_after_seq,
            limit=page.limit,
            view=page.view,
            kind=page.kind,
            origin=page.origin,
            pinned_end_seq=page.pinned_end_seq,
            has_more=page.has_more,
            at_pin_end=page.at_pin_end,
            tail_open=page.tail_open,
            watermark=page.watermark,
            cached=cached,
        )

    def _remember(self, key: str, page: Page) -> None:
        self._served[key] = page
        self._served.move_to_end(key)
        while len(self._served) > self._page_cache_size:
            self._served.popitem(last=False)

    # -- row shaping -----------------------------------------------------

    def _hidden_seqs(self, spec: QueryType, view: Mapping[str, Any]) -> frozenset[int]:
        """Return the sequences a row must be flagged ``active: false`` for."""

        committed = self._repository.committed_records()
        if spec.policy == POLICY_CONTINUE:
            # The timeline keeps originals forever; only the *active* flag is
            # derived from the current tombstone set, so a served row keeps its
            # immutable body while rollback becomes visible as a marker.
            return frozenset(rec.tombstone_of for rec in committed if rec.is_tombstone)
        if spec.policy == POLICY_INVALIDATE:
            return frozenset(int(seq) for seq in view["hidden"])
        if spec.policy == POLICY_PINNED:
            as_of_tick = int(view["as_of_tick"])
            return frozenset(
                rec.tombstone_of for rec in committed if rec.is_tombstone and rec.tick <= as_of_tick
            )
        return frozenset()  # pragma: no cover

    def _describe_row(self, record: Record, hidden: frozenset[int]) -> dict[str, Any]:
        # Tombstones are themselves active records: only a surviving original
        # can be hidden, and never just because its seq happens to equal the
        # target of some other tombstone.
        active = True if record.is_tombstone else record.seq not in hidden
        return {
            "seq": record.seq,
            "kind": record.kind,
            "origin": record.origin,
            "generation": record.generation,
            "tick": record.tick,
            "active": active,
            "tombstone_of": record.tombstone_of,
            "payload": dict(record.payload),
        }

    # -- view resolution and fork policy --------------------------------

    def _check_view(self, spec: QueryType, view: Mapping[str, Any]) -> tuple[int, bool]:
        """Validate the pinned view against the live stream.

        Returns ``(end_seq, tail_open)`` -- ``tail_open`` marks a walk whose end
        moves forward as more records commit.
        """

        pin = str(view.get("pin"))
        committed = self._repository.committed_records()
        tip = committed[-1].seq if committed else 0
        if spec.policy == POLICY_CONTINUE:
            if pin != "open":
                raise InvalidCursorError("stream cursor is missing its open pin")
            return tip, True
        if spec.policy == POLICY_PINNED:
            as_of_tick = view.get("as_of_tick")
            if pin != "tick" or not isinstance(as_of_tick, int):
                raise InvalidCursorError("replay cursor is missing its as-of tick")
            if as_of_tick > self._clock.now():
                raise CursorExpiredError("the pinned tick is gone after a clock rewind", as_of_tick=as_of_tick)
            within = tuple(rec for rec in committed if rec.tick <= as_of_tick)
            return (within[-1].seq if within else 0), False
        if spec.policy == POLICY_INVALIDATE:
            pin_seq = view.get("pin_seq")
            hidden = view.get("hidden")
            if pin != "seq" or not isinstance(pin_seq, int) or not isinstance(hidden, list):
                raise InvalidCursorError("ledger cursor is missing its pin")
            current_hidden = frozenset(rec.tombstone_of for rec in committed if rec.is_tombstone)
            pinned_hidden = frozenset(int(seq) for seq in hidden)
            if current_hidden != pinned_hidden:
                raise CursorForkedError(
                    "the ledger rolled back inside the pinned window; open a new walk",
                    pinned_seq=pin_seq,
                    committed_seq=self._repository.last_seq(),
                    hidden_added=sorted(current_hidden - pinned_hidden),
                    hidden_removed=sorted(pinned_hidden - current_hidden),
                    restart_hint="re-open the query without a cursor",
                )
            return int(pin_seq), False
        raise InvalidCursorError("cursor names a policy the server does not implement")  # pragma: no cover

    def _view_records(self, spec: QueryType, view: Mapping[str, Any]) -> tuple[Record, ...]:
        committed = self._repository.committed_records()
        if spec.policy == POLICY_CONTINUE:
            return tuple(committed)
        if spec.policy == POLICY_INVALIDATE:
            pin_seq = int(view["pin_seq"])
            pinned_hidden = frozenset(int(seq) for seq in view["hidden"])
            return tuple(
                rec
                for rec in committed
                if rec.seq <= pin_seq and not rec.is_tombstone and rec.seq not in pinned_hidden
            )
        if spec.policy == POLICY_PINNED:
            as_of_tick = int(view["as_of_tick"])
            within = tuple(rec for rec in committed if rec.tick <= as_of_tick)
            hidden = frozenset(rec.tombstone_of for rec in within if rec.is_tombstone)
            return tuple(rec for rec in within if not rec.is_tombstone and rec.seq not in hidden)
        raise ValidationError("query type has no read rule", qtype=spec.name)  # pragma: no cover

    @staticmethod
    def _advance_position(
        spec: QueryType,
        after_seq: int,
        page_records: tuple[Record, ...],
        matched: tuple[Record, ...],
        candidates: tuple[Record, ...],
        pin_end: int,
    ) -> tuple[int, bool]:
        """Pick the seq the next cursor seeks from and whether the walk ended.

        The cursor is a keyset seek over the rows that matched the filter.
        With rows in hand the position is the last returned seq.  On an empty
        bounded page the position jumps to the last examined record (matched or
        not), so a continuation never rescans the same non matching gap.

        ``matched`` holds the rows still passing the filter after ``after_seq``;
        ``candidates`` holds the whole pinned view.  The walk ends when no
        matching row remains -- trailing non matching records up to the pin do
        not keep it artificially open.  An open ended stream never reports an
        end; it reports a live tail.
        """

        if page_records:
            last_seq = page_records[-1].seq
            if spec.policy == POLICY_CONTINUE:
                return last_seq, False
            return last_seq, not any(rec.seq > last_seq for rec in matched)
        if spec.policy == POLICY_CONTINUE:
            return after_seq, False
        examined = tuple(rec for rec in candidates if rec.seq > after_seq)
        return (examined[-1].seq if examined else after_seq), True

    # -- helpers ---------------------------------------------------------

    def _valid_limit(self, limit: int | None) -> int:
        if limit is None:
            return DEFAULT_LIMIT
        if limit < 1:
            raise ValidationError("page limit must be positive", limit=limit)
        if limit > MAX_LIMIT:
            raise ValidationError("page limit is too large", limit=limit, max=MAX_LIMIT)
        return int(limit)

    def _audit(self, page: Page, *, opening: bool) -> None:
        if self._audit_hook is None:
            return
        self._audit_hook(
            "page.served",
            {
                "query_type": page.qtype,
                "policy": page.policy,
                "walk_id": page.walk_id,
                "opening": opening,
                "idempotent": page.cached,
                "after_seq": page.after_seq,
                "next_after_seq": page.next_after_seq,
                "returned": len(page.rows),
                "at_pin_end": page.at_pin_end,
                "tail_open": page.tail_open,
            },
        )
