"""Console routes for the cursor query API."""

from __future__ import annotations

import json

from mixplant.console.pages import PageStore
from mixplant.console.router import Request
from mixplant.console.server import ConsoleServer
from mixplant.errors import CursorForkedError
from mixplant.paging.policy import LEDGER, STREAM
from mixplant.runtime import PlantRuntime


def _server(runtime: PlantRuntime) -> ConsoleServer:
    return ConsoleServer(runtime, pages=PageStore(runtime.config.web_dir))


def _get(server: ConsoleServer, path: str, query: dict[str, str] | None = None):
    return server.handle(Request(method="GET", path=path, query=query or {}))


def test_query_types_route_exposes_the_policy_table(runtime: PlantRuntime) -> None:
    response = _get(_server(runtime), "/api/query/types")
    assert response.ok
    payload = json.loads(response.text)["query_types"]
    assert {row["name"]: row["policy"] for row in payload} == {
        "stream": "continue",
        "ledger": "invalidate",
        "replay": "pinned",
    }


def test_open_and_next_walk_the_stream(runtime: PlantRuntime) -> None:
    runtime.dispatch("audit.note", {"name": "one"})
    runtime.dispatch("audit.note", {"name": "two"})
    server = _server(runtime)

    opened = _get(server, "/api/query/open", {"type": STREAM, "limit": "1"})
    first = json.loads(opened.text)
    assert first["returned"] == 1
    assert first["cursor"]
    assert first["watermark"]["committed_seq"] >= 2

    nxt = _get(server, "/api/query/next", {"cursor": first["cursor"]})
    second = json.loads(nxt.text)
    assert second["rows"][0]["seq"] == first["rows"][0]["seq"] + 1

    # a replay of the same cursor is flagged idempotent and returns the same page
    replay = json.loads(_get(server, "/api/query/next", {"cursor": first["cursor"]}).text)
    assert replay["idempotent"] is True
    assert replay["rows"] == second["rows"]


def test_next_without_a_cursor_is_a_validation_error(runtime: PlantRuntime) -> None:
    response = _get(_server(runtime), "/api/query/next")
    assert response.status == 400
    assert json.loads(response.text)["code"] == "validation_error"


def test_tampered_cursor_is_refused(runtime: PlantRuntime) -> None:
    runtime.dispatch("audit.note", {"name": "x"})
    token = json.loads(
        _get(_server(runtime), "/api/query/open", {"type": STREAM}).text
    )["cursor"]
    body, key_id, signature = token.split(".")
    response = _get(
        _server(runtime),
        "/api/query/next",
        {"cursor": f"{body}AAAA.{key_id}.{signature}"},
    )
    assert response.status == 400
    assert json.loads(response.text)["code"] == "invalid_cursor"


def test_ledger_fork_returns_conflict(runtime: PlantRuntime) -> None:
    for _ in range(4):
        runtime.dispatch("audit.note", {"name": "event"})
    server = _server(runtime)
    walk = json.loads(
        _get(server, "/api/query/open", {"type": LEDGER, "limit": "50", "kind": "audit.entry"}).text
    )
    opened_seqs = [row["seq"] for row in walk["rows"]]
    hidden = opened_seqs[2:]
    runtime.rollback(opened_seqs[1])
    response = _get(server, "/api/query/next", {"cursor": walk["cursor"]})
    assert response.status == CursorForkedError.http_status
    payload = json.loads(response.text)
    assert payload["code"] == "cursor_forked"
    assert payload["context"]["hidden_added"] == hidden


def test_query_page_renders(runtime: PlantRuntime) -> None:
    response = _get(_server(runtime), "/pages/query")
    assert response.ok
    assert "<!DOCTYPE html>" in response.text
    assert "/api/query/open" in response.text


def test_page_audit_route_lists_servings(runtime: PlantRuntime) -> None:
    runtime.dispatch("audit.note", {"name": "x"})
    server = _server(runtime)
    _get(server, "/api/query/open", {"type": STREAM})
    response = _get(server, "/api/query/audit")
    assert response.ok
    payload = json.loads(response.text)
    assert payload["count"] >= 1
    assert payload["events"][0]["name"] == "page.served"
