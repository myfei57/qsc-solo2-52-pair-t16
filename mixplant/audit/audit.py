"""Record kinds used by the audit side."""

from __future__ import annotations

ENTRY_KIND = "audit.entry"
ROLLUP_KIND = "audit.rollup"

# Names of the read only ledger entries written when a console page is served.
PAGE_READ = "audit.page_read"
PAGE_REFUSED = "audit.page_refused"
