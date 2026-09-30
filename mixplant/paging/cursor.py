"""Opaque, signed page cursors.

A cursor is the *position* a walk has reached inside one fixed ordering of the
stream.  It is handed back verbatim by the caller; the server treats it as an
opaque token and never trusts its contents.

Every cursor pins the three things a continuation has to agree on:

``qtype``
    The query type that opened the walk.  What happens at a rollback fork is
    decided per query type, so a cursor can never be replayed against a
    different type.
``after_seq``
    The last sequence already handed out.  The next page starts strictly after
    it (a keyset/seek position -- never an offset, so rows cannot shift under
    the caller).
``view``
    The read view the walk is pinned to (an as-of tick, a boundary seq, …).
    A walk always reads the same frozen view it started on, which is what keeps
    a page that has already been served from being silently rewritten after a
    rollback.

The token is ``base64url(json).signature`` where the signature is an
``HMAC-SHA256`` keyed by the station's page secret.  A cursor that is decoded
from, or whose signature/key id/station does not verify, is refused rather than
recomputed: callers must restart the walk from the first page.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import InvalidCursorError, ValidationError

CURSOR_VERSION = 1


@dataclass(frozen=True, slots=True)
class PageCursor:
    """A position inside one pinned, fixed-order walk."""

    qtype: str
    after_seq: int
    view: Mapping[str, Any]
    walk_id: str
    limit: int
    kind: str | None = None
    origin: str | None = None

    def __post_init__(self) -> None:
        if not self.qtype:
            raise ValidationError("a cursor needs a query type")
        if self.after_seq < 0:
            raise ValidationError("cursor after_seq must not be negative", after_seq=self.after_seq)
        if self.limit < 1:
            raise ValidationError("cursor limit must be positive", limit=self.limit)
        if not self.walk_id:
            raise ValidationError("a cursor needs a walk id")

    @property
    def filter_key(self) -> tuple[str | None, str | None]:
        """The immutable filter the walk was opened with."""

        return (self.kind, self.origin)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "v": CURSOR_VERSION,
            "q": self.qtype,
            "a": self.after_seq,
            "w": dict(self.view),
            "id": self.walk_id,
            "n": self.limit,
            "k": self.kind,
            "o": self.origin,
        }

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "PageCursor":
        try:
            version = int(document["v"])
            qtype = str(document["q"])
            after_seq = int(document["a"])
            view = document["w"]
            walk_id = str(document["id"])
            limit = int(document["n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidCursorError("cursor is missing fields") from exc
        if version != CURSOR_VERSION:
            raise InvalidCursorError("cursor version is unsupported", version=version, supported=CURSOR_VERSION)
        if not isinstance(view, Mapping):
            raise InvalidCursorError("cursor view must be an object")
        raw_kind = document.get("k")
        raw_origin = document.get("o")
        kind = None if raw_kind is None else str(raw_kind)
        origin = None if raw_origin is None else str(raw_origin)
        return cls(
            qtype=qtype,
            after_seq=after_seq,
            view=dict(view),
            walk_id=walk_id,
            limit=limit,
            kind=kind,
            origin=origin,
        )


class CursorCodec:
    """Signs and verifies cursors for one station."""

    __slots__ = ("_station", "_secret", "_key_id")

    def __init__(self, station: str, secret: str | bytes) -> None:
        if not station:
            raise ValidationError("a cursor codec needs a station")
        if not secret:
            raise ValidationError("a cursor codec needs a signing secret")
        self._station = station
        self._secret = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        self._key_id = hashlib.sha256(self._secret).digest()[:4].hex()

    @property
    def key_id(self) -> str:
        return self._key_id

    def _signing_input(self, body: str) -> bytes:
        return b"\x1f".join((self._station.encode("utf-8"), self._key_id.encode("ascii"), body.encode("utf-8")))

    def encode(self, cursor: PageCursor) -> str:
        body = json.dumps(cursor.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        body_b64 = base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii")
        digest = hmac.new(self._secret, self._signing_input(body_b64), hashlib.sha256).digest()
        signature = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return f"{body_b64}.{self._key_id}.{signature}"

    def decode(self, token: str, *, qtype: str | None = None) -> PageCursor:
        if not token or token.count(".") != 2:
            raise InvalidCursorError("cursor is malformed")
        body_b64, key_id, signature = token.split(".")
        if not hmac.compare_digest(key_id, self._key_id):
            raise InvalidCursorError("cursor was signed with another key")
        expected = hmac.new(
            self._secret, self._signing_input(body_b64), hashlib.sha256
        ).digest()
        padded = f"{signature}{'=' * (-len(signature) % 4)}"
        try:
            provided = base64.urlsafe_b64decode(padded.encode("ascii"))
        except ValueError as exc:
            raise InvalidCursorError("cursor signature is malformed") from exc
        if not hmac.compare_digest(provided, expected):
            raise InvalidCursorError("cursor signature does not verify")
        try:
            padded_body = body_b64 + "=" * (-len(body_b64) % 4)
            body = base64.urlsafe_b64decode(padded_body.encode("ascii")).decode("utf-8")
            document = json.loads(body)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidCursorError("cursor body is unreadable") from exc
        cursor = PageCursor.from_mapping(document)
        if qtype is not None and cursor.qtype != qtype:
            raise InvalidCursorError(
                "cursor belongs to another query type", cursor_qtype=cursor.qtype, requested=qtype
            )
        return cursor
