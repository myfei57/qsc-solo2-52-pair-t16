"""Consistency epochs of the record stream.

Paging serves two very different notions of "the ledger", and the difference
only shows up when a rollback appends tombstones:

* the committed journal is append only.  Nothing ever disappears from it, so a
  cursor handed out before a rollback keeps addressing the same immutable
  positions afterwards;
* the effective (tombstone filtered) view is rewritten by a rollback, so a
  cursor issued in an earlier epoch cannot continue there.

An epoch describes exactly that: which rollback revision the stream is at,
where the revision marker landed and a fingerprint of its boundary.  Every
value is derived from the committed records themselves, which keeps replay,
cursor validation and the audit trail comparable after a restart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .records import ROLLBACK_MARKER_KIND, Record


@dataclass(frozen=True, slots=True)
class Epoch:
    """The rollback state of a stream at one point in time.

    ``revision`` counts committed rollbacks, ``revision_seq`` is the sequence
    of the newest rollback marker and ``boundary`` is the sequence that
    rollback kept (nothing above it survived).
    """

    revision: int = 0
    revision_seq: int = 0
    boundary: int = 0
    fingerprint: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "revision_seq": self.revision_seq,
            "boundary": self.boundary,
            "fingerprint": self.fingerprint,
        }

    def equals(self, other: "Epoch") -> bool:
        """Return whether both epochs describe the same rollback state."""

        return (
            self.revision == other.revision
            and self.revision_seq == other.revision_seq
            and self.boundary == other.boundary
            and self.fingerprint == other.fingerprint
        )


def stream_epoch(records: Iterable[Record]) -> Epoch:
    """Derive the current epoch by walking ``records`` in write order."""

    revision = 0
    revision_seq = 0
    boundary = 0
    for record in records:
        if record.kind == ROLLBACK_MARKER_KIND:
            revision += 1
            revision_seq = record.seq
            boundary = int(record.payload.get("boundary", 0))
    return Epoch(
        revision=revision,
        revision_seq=revision_seq,
        boundary=boundary,
        fingerprint=_fingerprint(revision, revision_seq, boundary),
    )


def _fingerprint(revision: int, revision_seq: int, boundary: int) -> str:
    document = json.dumps(
        {"revision": revision, "revision_seq": revision_seq, "boundary": boundary},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()[:16]


def epoch_from_document(document: Mapping[str, Any]) -> Epoch:
    """Rebuild an epoch carried by a cursor or an audit entry."""

    return Epoch(
        revision=int(document.get("revision", 0)),
        revision_seq=int(document.get("revision_seq", 0)),
        boundary=int(document.get("boundary", 0)),
        fingerprint=str(document.get("fingerprint", "")),
    )
