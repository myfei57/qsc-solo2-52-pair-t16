"""The fixed policy of every supported query type.

Cursor handling after a rollback is *per query type*, decided here once and for
all -- a continuation never improvises and the server never recomputes a cursor
on the caller's behalf.

``stream`` -- policy ``continue``
    The raw time ordered journal: every committed record in ``seq`` order,
    tombstones included (an original row stays in the timeline, flagged
    ``active: false`` once a later tombstone names it).  ``seq`` is gapless for
    the life of the stream, so a cursor always keeps working after a rollback
    and simply continues forward; already served rows are immutable facts.

``ledger`` -- policy ``invalidate``
    The live console ledger: only records that currently survive tombstoning.
    A rollback really removes rows from this view, which is why the walk is
    opened against a frozen pin (``pin_seq`` plus the hidden set as it stood at
    open time).  When a rollback touches the pinned window the cursor is
    refused with ``cursor_forked``; the caller opens a fresh walk.  It is never
    silently translated.

``replay`` -- policy ``pinned``
    A point in time replay bound to one ``as_of_tick`` chosen when the walk
    opens.  Only records with ``tick <= as_of_tick`` belong to the walk and a
    tombstone counts only if it was already in effect at that tick, so the walk
    answers identically forever and the cursor never invalidates.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ValidationError

STREAM = "stream"
LEDGER = "ledger"
REPLAY = "replay"

POLICY_CONTINUE = "continue"
POLICY_INVALIDATE = "invalidate"
POLICY_PINNED = "pinned"


@dataclass(frozen=True, slots=True)
class QueryType:
    """One row of the cursor policy table."""

    name: str
    policy: str
    description: str
    shows_tombstones: bool

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "policy": self.policy, "description": self.description}


SPECS: dict[str, QueryType] = {
    STREAM: QueryType(
        name=STREAM,
        policy=POLICY_CONTINUE,
        description="time ordered journal; cursor keeps moving across rollbacks",
        shows_tombstones=True,
    ),
    LEDGER: QueryType(
        name=LEDGER,
        policy=POLICY_INVALIDATE,
        description="effective ledger; a rollback inside the pin invalidates the cursor",
        shows_tombstones=False,
    ),
    REPLAY: QueryType(
        name=REPLAY,
        policy=POLICY_PINNED,
        description="frozen as-of-tick replay; the cursor never invalidates",
        shows_tombstones=False,
    ),
}

QUERY_TYPES: tuple[str, ...] = tuple(SPECS)


def get_spec(qtype: str) -> QueryType:
    """Return the fixed policy for ``qtype`` or refuse an unknown type."""

    spec = SPECS.get(qtype)
    if spec is None:
        raise ValidationError("unknown query type", qtype=qtype, known=list(SPECS))
    return spec
