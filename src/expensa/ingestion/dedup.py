"""Content-hash based deduplication for incremental CSV ingestion.

We hash a stable subset of fields so that re-importing an overlapping
export produces zero new rows. Banks sometimes alter the verwendungszweck
slightly across exports (added trailing whitespace, encoded references), so
we normalize first and truncate to a fixed prefix.
"""

from __future__ import annotations

import hashlib
from datetime import date

from expensa.ingestion.csv_loader import ParsedRow
from expensa.ingestion.normalizer import (
    normalize_counterparty,
    normalize_verwendungszweck,
)

# Number of leading characters of the normalized verwendungszweck used in the
# dedup hash. Long enough to disambiguate; short enough that minor cosmetic
# changes don't break dedup.
VZ_PREFIX_LEN = 120


def _date_str(d: date | None) -> str:
    return d.isoformat() if d else ""


def compute_dedup_hash(row: ParsedRow) -> str:
    counterparty = row.zahlungsempfaenger or row.zahlungspflichtiger
    counterparty_norm = normalize_counterparty(counterparty)
    vz_norm = normalize_verwendungszweck(row.verwendungszweck)[:VZ_PREFIX_LEN]
    parts = [
        _date_str(row.buchungsdatum),
        _date_str(row.wertstellung),
        str(row.betrag_cents),
        row.iban or "",
        counterparty_norm,
        vz_norm,
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
