"""Fixed order cursor paging: incremental walks, forks and audit."""

from __future__ import annotations

import threading

import pytest

from mixplant.errors import (
    CursorForkedError,
    InvalidCursorError,
    ValidationError,
)
from mixplant.paging.cursor import CursorCodec, PageCursor
from mixplant.paging.policy import LEDGER, REPLAY, STREAM, SPECS
from mixplant.runtime import PlantRuntime


@pytest.fixture()
def plant() -> PlantRuntime:
    return PlantRuntime.in_memory()


def _notes(plant: PlantRuntime, count: int) -> None:
    for index in range(count):
        plant.dispatch("audit.note", {"name": f"evt.{index}", "detail": {"i": index}})


def _all_pages(plant: PlantRuntime, first: dict) -> list[int]:
    seqs = [row["seq"] for row in first["rows"]]
    page = first
    while page["has_more"]:
        page = plant.continue_page(page["cursor"])
        seqs.extend(row["seq"] for row in page["rows"])
    return seqs


def test_policy_table_is_fixed_per_query_type() -> None:
    assert tuple(SPECS) == (STREAM, LEDGER, REPLAY)
    assert SPECS[STREAM].policy == "continue"
    assert SPECS[LEDGER].policy == "invalidate"
    assert SPECS[REPLAY].policy == "pinned"
    rt = PlantRuntime.in_memory()
    names = {item["name"]: item["policy"] for item in rt.query_types()["query_types"]}
    assert names == {STREAM: "continue", LEDGER: "invalidate", REPLAY: "pinned"}


def test_stream_walks_in_fixed_order_without_duplicates_or_gaps(plant: PlantRuntime) -> None:
    _notes(plant, 6)
    first = plant.open_page({"type": STREAM, "limit": "2"})
    assert _all_pages(plant, first) == [1, 2, 3, 4, 5, 6]


def test_same_cursor_concurrently_is_idempotent(plant: PlantRuntime) -> None:
    _notes(plant, 4)
    head = plant.open_page({"type": STREAM, "limit": "1"})
    head_rows = [row["seq"] for row in head["rows"]]
    assert head_rows == [1]

    replies: list[list[int]] = []
    lock = threading.Lock()

    def worker() -> None:
        page = plant.continue_page(head["cursor"])
        with lock:
            replies.append([row["seq"] for row in page["rows"]])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert replies == [[2]] * 8
    # the walk only advanced once; following the next cursor yields no overlap
    again = plant.continue_page(head["cursor"])
    assert again["idempotent"] is True
    after = plant.continue_page(again["cursor"])
    assert [row["seq"] for row in after["rows"]] == [3]


def test_stream_cursor_survives_rollback_and_keeps_history(plant: PlantRuntime) -> None:
    _notes(plant, 6)
    page_two = plant.open_page({"type": STREAM, "limit": "2"})
    middle = plant.continue_page(page_two["cursor"])  # serves 3,4
    served = [dict(row) for row in middle["rows"]]

    plant.rollback(4, reason="operator changed their mind")
    replay = plant.continue_page(page_two["cursor"])  # replay served position 2
    assert replay["idempotent"] is True
    assert replay["rows"] == served

    # walking forward crosses the fork: originals stay in the timeline, flagged
    onward = plant.continue_page(replay["cursor"])
    assert [row["seq"] for row in onward["rows"]] == [5, 6]
    assert all(row["active"] is False for row in onward["rows"])
    assert onward["rows"][0]["payload"]["detail"] == {"i": 4}  # body immutable
    tombstones = plant.continue_page(onward["cursor"])
    assert {row["tombstone_of"] for row in tombstones["rows"]} == {5, 6}
    assert all(row["active"] for row in tombstones["rows"])


def test_ledger_cursor_is_invalidated_by_a_fork(plant: PlantRuntime) -> None:
    _notes(plant, 6)
    walk = plant.open_page({"type": LEDGER, "limit": "50", "kind": "audit.entry"})
    assert [row["seq"] for row in walk["rows"]] == [1, 2, 3, 4, 5, 6]
    plant.rollback(4)
    try:
        plant.continue_page(walk["cursor"])
    except CursorForkedError as error:
        assert error.context["hidden_added"] == [5, 6]
        assert error.context["restart_hint"]
    else:  # pragma: no cover
        raise AssertionError("expected the ledger cursor to be invalidated")

    restarted = plant.open_page({"type": LEDGER, "limit": "50", "kind": "audit.entry"})
    original = [row["seq"] for row in restarted["rows"] if row["seq"] <= 6]
    assert original == [1, 2, 3, 4]
    assert all(row["active"] for row in restarted["rows"])

    # the rollback's own marker is new history appended after the cut, not a
    # resurrection of a rolled back row
    full = plant.open_page({"type": LEDGER, "limit": "50"})
    names = [row["payload"].get("name") for row in full["rows"]]
    assert "store.rolled_back" in names
    assert "evt.5" not in names


def test_ledger_without_a_fork_pages_gaplessly(plant: PlantRuntime) -> None:
    _notes(plant, 5)
    first = plant.open_page({"type": LEDGER, "limit": "2"})
    second = plant.continue_page(first["cursor"])
    third = plant.continue_page(second["cursor"])
    pages = first["rows"] + second["rows"] + third["rows"]
    assert [row["seq"] for row in pages] == [1, 2, 3, 4, 5]
    assert third["at_pin_end"] is True


def test_replay_walk_is_frozen_at_its_tick(plant: PlantRuntime) -> None:
    plant.dispatch("audit.note", {"name": "before"})
    pin_tick = plant.clock.now()
    plant.dispatch("clock.advance", {"ticks": 3})
    plant.dispatch("audit.note", {"name": "after"})

    first = plant.open_page({"type": REPLAY, "as_of_tick": str(pin_tick), "limit": "50"})
    names = [row["payload"]["name"] for row in first["rows"]]
    assert names == ["before"]

    # a later rollback must not retroactively hide an earlier record
    plant.rollback(2, reason="late correction")
    continued = plant.continue_page(first["cursor"])
    assert continued["at_pin_end"] is True
    reopened = plant.open_page({"type": REPLAY, "as_of_tick": str(pin_tick), "limit": "50"})
    assert reopened["rows"] == first["rows"]
    assert first["rows"][0]["active"] is True


def test_filter_is_part_of_the_walk_and_keysets_over_gaps(plant: PlantRuntime) -> None:
    plant.dispatch("audit.note", {"name": "keep.one"})  # audit.entry
    plant.dispatch("clock.advance", {"ticks": 1})       # store.clock (filtered out)
    plant.dispatch("audit.note", {"name": "keep.two"})  # audit.entry

    by_kind = plant.open_page({"type": LEDGER, "kind": "audit.entry", "limit": "1"})
    page_two = plant.continue_page(by_kind["cursor"])
    assert [row["seq"] for row in by_kind["rows"]] == [1]
    assert [row["seq"] for row in page_two["rows"]] == [3]
    assert page_two["at_pin_end"] is True
    assert page_two["has_more"] is False

    by_origin = plant.open_page({"type": LEDGER, "origin": "console", "limit": "50"})
    kinds = {row["kind"] for row in by_origin["rows"]}
    assert "store.clock" in kinds and "audit.entry" not in kinds


def test_cursor_is_signed_and_bound_to_station_and_type(plant: PlantRuntime) -> None:
    _notes(plant, 1)
    token = plant.open_page({"type": STREAM, "limit": "1"})["cursor"]
    body, key_id, signature = token.split(".")

    with pytest.raises(InvalidCursorError):
        plant.continue_page(f"{body}AAAA.{key_id}.{signature}")
    with pytest.raises(InvalidCursorError):
        plant.continue_page(f"{body}.{key_id}.{signature[:-6]}AAAAAA")
    with pytest.raises(InvalidCursorError):
        plant.continue_page("not-a-cursor")

    foreign = CursorCodec("another-station", plant.config.page_secret)
    other = foreign.encode(
        PageCursor(qtype=STREAM, after_seq=0, view={"pin": "open"}, walk_id="w", limit=1)
    )
    with pytest.raises(InvalidCursorError):
        plant.continue_page(other)


def test_query_parameters_are_validated(plant: PlantRuntime) -> None:
    _notes(plant, 1)
    with pytest.raises(ValidationError):
        plant.open_page({"type": STREAM, "limit": "0"})
    with pytest.raises(ValidationError):
        plant.open_page({"type": STREAM, "limit": "99999"})
    with pytest.raises(ValidationError):
        plant.open_page({"type": "unknown"})
    with pytest.raises(ValidationError):
        plant.open_page({"type": REPLAY, "as_of_tick": "99999999"})


def test_page_serving_is_audited_off_stream(plant: PlantRuntime) -> None:
    _notes(plant, 3)
    head = plant.open_page({"type": STREAM, "limit": "1"})
    plant.continue_page(head["cursor"])
    plant.continue_page(head["cursor"])  # idempotent replay

    events = plant.page_trail.events()
    names = [event.name for event in events]
    assert names.count("page.served") == 3
    assert any(event.detail["idempotent"] for event in events)
    # reads never append into the journal they page over
    production = {entry.name for entry in plant.audit.ledger.entries()}
    assert "page.served" not in production


def test_rollback_is_audited_with_its_targets(plant: PlantRuntime) -> None:
    _notes(plant, 4)
    plant.rollback(2, reason="bad batch")
    entries = [entry for entry in plant.audit.ledger.entries() if entry.name == "store.rolled_back"]
    assert entries
    detail = entries[-1].detail
    assert detail["after_seq"] == 2
    assert detail["tombstoned"] == [3, 4]
    assert detail["reason"] == "bad batch"


def test_file_backed_runtime_writes_the_trail_under_data_dir(tmp_path) -> None:
    from mixplant.config import StationConfig

    config = StationConfig(data_dir=tmp_path, web_dir="/workspace/web")
    plant = PlantRuntime(config, storage="file")
    try:
        plant.dispatch("audit.note", {"name": "x"})
        plant.open_page({"type": STREAM, "limit": "1"})
        assert (tmp_path / "page-audit.jsonl").exists()
    finally:
        plant.close()


def test_walk_ids_never_repeat_after_a_restart(tmp_path) -> None:
    from mixplant.config import StationConfig

    config = StationConfig(data_dir=tmp_path, web_dir="/workspace/web")
    first = PlantRuntime(config, storage="file")
    try:
        first.dispatch("audit.note", {"name": "x"})
        page = first.open_page({"type": STREAM, "limit": "1"})
        walk_id = page["walk_id"]
    finally:
        first.close()

    second = PlantRuntime(config, storage="file")
    try:
        second.bootstrap()
        reopened = second.open_page({"type": STREAM, "limit": "1"})
        assert reopened["walk_id"] != walk_id
        # ids are numeric and strictly greater
        assert int(reopened["walk_id"].split("-", 1)[1]) > int(walk_id.split("-", 1)[1])
    finally:
        second.close()


def test_page_audit_trail_is_durable_and_replays(tmp_path) -> None:
    from mixplant.audit.trail import PageAuditTrail
    from mixplant.clock import ManualClock

    clock = ManualClock()
    path = tmp_path / "page-audit.jsonl"
    trail = PageAuditTrail(path, clock=clock)
    trail.record("page.served", {"walk_id": "walk-1", "returned": 2})
    clock.advance(3)
    trail.record("page.served", {"walk_id": "walk-1", "idempotent": True})
    trail.close()

    reopened = PageAuditTrail(path, clock=clock)
    events = reopened.events()
    assert [event.seq for event in events] == [1, 2]
    assert events[0].tick == 0 and events[1].tick == 3
    assert events[1].detail["idempotent"] is True
    assert len(reopened.recent(1)) == 1
