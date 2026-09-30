"""Cursor based paging: fixed order, rollback epochs, idempotency and audit."""

from __future__ import annotations

import json

import pytest

from mixplant.clock import ManualClock
from mixplant.console.pages import PageStore
from mixplant.console.router import Request
from mixplant.console.server import ConsoleServer
from mixplant.decision.paging import (
    DEFAULT_VIEW,
    EFFECTIVE_VIEW,
    JOURNAL_VIEW,
    VIEW_POLICIES,
    CursorToken,
    PageQuery,
    page_query_from_params,
    policy_for,
    read_page,
)
from mixplant.errors import CursorError, StaleCursorError, ValidationError
from mixplant.ids import SequenceIds
from mixplant.store.epoch import stream_epoch
from mixplant.store.repository import RecordRepository
from tests.support import run


def _store(*, ticks: bool = False) -> RecordRepository:
    clock = ManualClock()
    store = RecordRepository.in_memory(clock, SequenceIds())

    def publish(kind: str, payload: dict | None = None, *, origin: str = "agg") -> None:
        store.publish(kind, origin, 1, payload or {"active": True})
        if ticks:
            clock.advance(1)

    publish("agg.weigh", {"active": True, "actual_kg": 10})
    publish("agg.persist")
    publish("mix.mixer", origin="mix")
    publish("agg.weigh", {"active": True, "actual_kg": 20})
    publish("agg.release", {"residual_kg": 3.0})
    return store


def _walk(store: RecordRepository, query: PageQuery) -> tuple[list, list[str | None]]:
    rows: list = []
    cursors: list[str | None] = []
    token: str | None = None
    for _ in range(100):
        page = read_page(store.committed_records(), query, cursor=token)
        rows.extend(page.rows)
        token = page.next_cursor
        cursors.append(token)
        if token is None:
            break
    else:  # pragma: no cover - the walk must terminate
        raise AssertionError("paging did not terminate")
    return rows, cursors


def test_pages_walk_forward_in_fixed_tick_seq_order() -> None:
    store = _store(ticks=True)
    rows, _ = _walk(store, PageQuery(page_size=2))
    assert [record.seq for record in rows] == [1, 2, 3, 4, 5]
    positions = [(record.tick, record.seq) for record in rows]
    assert positions == sorted(positions)


def test_rows_sharing_a_tick_are_not_duplicated_or_skipped() -> None:
    store = _store(ticks=False)
    assert len({record.tick for record in store.committed_records()}) == 1
    for size in (1, 2, 3, 5):
        rows, _ = _walk(store, PageQuery(page_size=size))
        assert sorted(record.seq for record in rows) == [1, 2, 3, 4, 5]
        assert len({record.seq for record in rows}) == 5


def test_resuming_the_same_cursor_concurrently_is_idempotent() -> None:
    store = _store(ticks=True)
    first = read_page(store.committed_records(), PageQuery(page_size=2))
    assert first.next_cursor is not None
    one = read_page(store.committed_records(), PageQuery(page_size=2), cursor=first.next_cursor)
    two = read_page(store.committed_records(), PageQuery(page_size=2), cursor=first.next_cursor)
    assert [record.seq for record in one.rows] == [3, 4]
    assert [record.seq for record in two.rows] == [3, 4]
    assert one.next_cursor == two.next_cursor


def test_filtered_walk_visits_every_match_once() -> None:
    store = _store(ticks=True)
    rows, _ = _walk(store, PageQuery(kind="agg.weigh", page_size=1))
    assert [record.seq for record in rows] == [1, 4]


def test_last_page_carries_no_cursor() -> None:
    store = _store(ticks=True)
    page = read_page(store.committed_records(), PageQuery(page_size=10))
    assert page.next_cursor is None
    assert page.has_more is False


def test_view_policies_are_pinned_by_query_type() -> None:
    assert VIEW_POLICIES[JOURNAL_VIEW].invalidate_on_rollback is False
    assert VIEW_POLICIES[JOURNAL_VIEW].exposes_internal is True
    assert VIEW_POLICIES[EFFECTIVE_VIEW].invalidate_on_rollback is True
    assert VIEW_POLICIES[EFFECTIVE_VIEW].exposes_internal is False
    with pytest.raises(ValidationError):
        policy_for("something-else")
    with pytest.raises(ValidationError):
        PageQuery(view=EFFECTIVE_VIEW, include_tombstones=True)


def test_journal_cursor_is_reused_after_a_rollback_and_flags_the_epoch() -> None:
    store = _store(ticks=True)
    first = read_page(store.committed_records(), PageQuery(view=JOURNAL_VIEW, page_size=2))
    cursor = first.next_cursor
    assert cursor is not None
    assert [record.seq for record in first.rows] == [1, 2]

    store.rollback_after(2, reason="operator rollback", origin="console", generation=0)
    # History already paged is still physically there: the journal is immutable.
    assert [record.seq for record in store.committed_records()[:2]] == [1, 2]

    second = read_page(store.committed_records(), PageQuery(view=JOURNAL_VIEW, page_size=2), cursor=cursor)
    assert second.epoch_changed is True
    assert second.issued_epoch.revision == 0
    assert second.current_epoch.revision == 1
    # The cursor resumes strictly after seq 2; the rollback's own records show
    # up in the journal instead of rewriting anything.
    assert [record.seq for record in second.rows] == [3, 4]
    rest, _ = _walk(store, PageQuery(view=JOURNAL_VIEW, page_size=2))
    assert rest  # walking still terminates and reaches the new tail


def test_effective_cursor_is_invalidated_after_a_rollback() -> None:
    store = _store(ticks=True)
    first = read_page(store.committed_records(), PageQuery(view=EFFECTIVE_VIEW, page_size=2))
    assert first.next_cursor is not None
    store.rollback_after(2, reason="operator rollback", origin="console", generation=0)
    with pytest.raises(StaleCursorError) as exc:
        read_page(store.committed_records(), PageQuery(view=EFFECTIVE_VIEW, page_size=2), cursor=first.next_cursor)
    assert exc.value.http_status == 409
    # Restarting without a cursor recomputes against the new epoch.
    restarted = read_page(store.committed_records(), PageQuery(view=EFFECTIVE_VIEW, page_size=10))
    assert [record.seq for record in restarted.rows] == [1, 2]
    assert restarted.epoch_changed is False


def test_effective_view_hides_tombstones_markers_and_targets() -> None:
    store = _store(ticks=True)
    store.rollback_after(2, reason="operator rollback", origin="console", generation=0)
    rows, _ = _walk(store, PageQuery(view=EFFECTIVE_VIEW, page_size=4))
    kinds = {record.kind for record in rows}
    assert "store.tombstone" not in kinds
    assert "store.rollback" not in kinds
    assert [record.seq for record in rows] == [1, 2]
    assert store.state().get("store") is None


def test_journal_view_keeps_internal_history_but_can_hide_it() -> None:
    store = _store(ticks=True)
    store.rollback_after(2, reason="operator rollback", origin="console", generation=0)
    shown = read_page(
        store.committed_records(),
        PageQuery(view=JOURNAL_VIEW, page_size=50),
    )
    hidden = read_page(
        store.committed_records(),
        PageQuery(view=JOURNAL_VIEW, include_tombstones=False, page_size=50),
    )
    assert any(record.is_tombstone for record in shown.rows)
    assert any(record.kind == "store.rollback" for record in shown.rows)
    assert not any(record.is_tombstone for record in hidden.rows)
    assert not any(record.kind == "store.rollback" for record in hidden.rows)


def test_epoch_is_derived_from_the_stream_and_survives_replay(tmp_path) -> None:
    store = RecordRepository.open(tmp_path, ManualClock(), SequenceIds())
    store.publish("agg.weigh", "agg", 1, {"active": True})
    store.publish("agg.persist", "agg", 1, {"active": True})
    store.rollback_after(1, reason="operator rollback", origin="console", generation=0)
    expected = store.epoch()
    assert expected.revision == 1
    assert expected.boundary == 1

    reopened = RecordRepository.open(tmp_path, ManualClock(), SequenceIds())
    assert reopened.epoch().equals(expected)
    assert [record.kind for record in reopened.visible()] == ["agg.weigh"]


def test_cursor_tampering_and_reuse_mistakes_are_refused() -> None:
    store = _store(ticks=True)
    page = read_page(store.committed_records(), PageQuery(kind="agg.weigh", page_size=1))
    token = page.next_cursor
    assert token is not None

    mutilated = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    with pytest.raises(CursorError):
        read_page(store.committed_records(), PageQuery(kind="agg.weigh", page_size=1), cursor=mutilated)
    with pytest.raises(CursorError):
        CursorToken.decode(token, key="a-different-deployment-key")
    with pytest.raises(CursorError):
        read_page(store.committed_records(), PageQuery(view=EFFECTIVE_VIEW, page_size=1), cursor=token)
    with pytest.raises(CursorError):
        read_page(store.committed_records(), PageQuery(kind="mix.mixer", page_size=1), cursor=token)


def test_journal_walk_bridges_a_rollback_into_a_republish_without_gaps() -> None:
    store = _store(ticks=True)
    store.rollback_after(2, reason="operator rollback", origin="console", generation=0)
    store.publish("agg.weigh", "agg", 1, {"active": True, "actual_kg": 99})

    rows, _ = _walk(store, PageQuery(view=JOURNAL_VIEW, page_size=3))
    # The five originals, three tombstones, one marker and the new record.
    assert [record.seq for record in rows] == list(range(1, 11))
    effective, _ = _walk(store, PageQuery(view=EFFECTIVE_VIEW, page_size=3))
    assert [record.seq for record in effective] == [1, 2, 10]


def test_legacy_query_hides_internal_records_unless_asked_for() -> None:
    from mixplant.decision.query import RecordQuery, run_query

    store = _store(ticks=True)
    store.rollback_after(2, reason="operator rollback", origin="console", generation=0)
    default_rows = run_query(store.committed_records(), RecordQuery())
    assert all(not record.is_tombstone and record.kind != "store.rollback" for record in default_rows.rows)
    with_internals = run_query(store.committed_records(), RecordQuery(include_tombstones=True))
    kinds = {record.kind for record in with_internals.rows}
    assert "store.tombstone" in kinds
    assert "store.rollback" in kinds


def test_repeated_full_walks_are_stable_without_server_session_state() -> None:
    store = _store(ticks=True)
    query = PageQuery(page_size=2)
    first, _ = _walk(store, query)
    second, _ = _walk(store, query)
    third, _ = _walk(store, query)
    assert [record.seq for record in first] == list(range(1, 6))
    assert [record.seq for record in second] == [record.seq for record in first]
    assert [record.seq for record in third] == [record.seq for record in first]


def test_default_view_is_the_immutable_journal() -> None:
    assert DEFAULT_VIEW == JOURNAL_VIEW
    query = page_query_from_params({"page_size": "3", "kind": "agg.weigh"})
    assert query.view == JOURNAL_VIEW
    assert query.page_size == 3
    assert query.kind == "agg.weigh"


def test_runtime_pages_are_served_and_audited_without_touching_the_stream(runtime) -> None:
    run(runtime, "recipe.publish", {"name": "r1", "grade": "C30", "factors": {"cement": 300.0}})
    before = runtime.store.last_seq()
    page_one = runtime.page_records({"view": EFFECTIVE_VIEW, "page_size": 5})
    cursor = page_one["next_cursor"]
    page_two = runtime.page_records({"view": EFFECTIVE_VIEW, "page_size": 5, "cursor": cursor})
    # Reads must not consume sequences, otherwise resuming a cursor would drift.
    assert runtime.store.last_seq() == before
    # The audit entry ranges match exactly the rows that were served.
    reads = [entry for entry in runtime.audit.ledger.entries() if entry.name == "audit.page_read"]
    assert len(reads) == 2
    assert reads[1].detail["seq_range"] == page_two["seq_range"]
    assert reads[1].detail["epoch"] == page_two["epoch"]
    assert page_one["order"] == "tick:asc,seq:asc"
    assert page_one["seq_range"]["from"] == page_one["rows"][0]["seq"]


def test_stale_effective_cursor_is_refused_with_409_and_audited(runtime) -> None:
    run(runtime, "recipe.publish", {"name": "r1", "grade": "C30", "factors": {"cement": 300.0}})
    run(runtime, "recipe.select", {"name": "r1"})
    page = runtime.page_records({"view": EFFECTIVE_VIEW, "page_size": 1})
    runtime.rollback(1)
    server = ConsoleServer(runtime, pages=PageStore(runtime.config.web_dir))
    response = server.handle(
        Request(method="GET", path="/api/records/page", query={"view": EFFECTIVE_VIEW, "cursor": page["next_cursor"]})
    )
    assert response.status == 409
    assert json.loads(response.text)["code"] == "stale_cursor"
    refusals = [entry for entry in runtime.audit.ledger.entries() if entry.name == "audit.page_refused"]
    assert len(refusals) == 1
    assert refusals[0].detail["current_epoch"]["revision"] == 1


def test_console_exposes_pages_and_the_audit_ledger(runtime) -> None:
    run(runtime, "recipe.publish", {"name": "r1", "grade": "C30", "factors": {"cement": 300.0}})
    server = ConsoleServer(runtime, pages=PageStore(runtime.config.web_dir))
    response = server.handle(
        Request(method="GET", path="/api/records/page", query={"view": JOURNAL_VIEW, "page_size": "3"})
    )
    assert response.ok
    payload = json.loads(response.text)
    assert payload["order"] == "tick:asc,seq:asc"
    assert "next_cursor" in payload
    ledger = server.handle(Request(method="GET", path="/api/audit/ledger"))
    names = [entry["name"] for entry in json.loads(ledger.text)["entries"]]
    assert "audit.page_read" in names


def test_replay_cursor_and_audit_share_one_reconcilable_epoch(runtime) -> None:
    run(runtime, "recipe.publish", {"name": "r1", "grade": "C30", "factors": {"cement": 300.0}})
    page = runtime.page_records({"view": EFFECTIVE_VIEW, "page_size": 2})
    derived = stream_epoch(runtime.store.committed_records())
    assert derived.equals(
        type(derived)(
            revision=page["epoch"]["revision"],
            revision_seq=page["epoch"]["revision_seq"],
            boundary=page["epoch"]["boundary"],
            fingerprint=page["epoch"]["fingerprint"],
        )
    )
    audit_entry = [entry for entry in runtime.audit.ledger.entries() if entry.name == "audit.page_read"][-1]
    assert audit_entry.detail["epoch"] == derived.describe()
    assert audit_entry.detail["seq_range"]["from"] == page["rows"][0]["seq"]
    assert audit_entry.detail["seq_range"]["to"] == page["rows"][-1]["seq"]
