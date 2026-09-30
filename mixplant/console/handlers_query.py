"""Routes for fixed order, cursor based queries.

``GET /api/query/open`` opens a walk and returns its first page plus an opaque
cursor; ``GET /api/query/next?cursor=…`` continues it.  The cursor is the only
state the client holds, it is signed by the station, and the server -- never
the client -- decides whether it still survives a rollback.
"""

from __future__ import annotations

from typing import Any

from ..errors import ValidationError
from .response import inject_snapshot
from .router import Request, Response, Router


def register(router: Router, runtime: Any, pages: Any) -> None:
    """Register the cursor query routes."""

    router.add(
        "GET",
        "/pages/query",
        lambda _: Response.html(inject_snapshot(pages.load("query"), runtime.state())),
    )
    router.add("GET", "/api/query/types", lambda _: Response.json(runtime.query_types()))
    router.add("GET", "/api/query/open", _open(runtime))
    router.add("GET", "/api/query/next", _next(runtime))
    router.add("GET", "/api/query/audit", _audit(runtime))


def _audit(runtime: Any):
    def handler(request: Request) -> Response:
        raw = request.field("limit", "50")
        try:
            limit = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError("limit must be an integer", limit=str(raw)) from exc
        return Response.json(runtime.page_audit(limit))

    return handler


def _open(runtime: Any):
    def handler(request: Request) -> Response:
        params = dict(request.query)
        try:
            return Response.json(runtime.open_page(params))
        except ValueError as exc:
            raise ValidationError("query parameter must be an integer", detail=str(exc)) from exc

    return handler


def _next(runtime: Any):
    def handler(request: Request) -> Response:
        token = request.require_field("cursor")
        return Response.json(runtime.continue_page(str(token)))

    return handler
