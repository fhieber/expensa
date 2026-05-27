"""PayPal enrichment engine (simplified).

Matches EnrichmentRecords against bank expenses by absolute amount +
nearest date within ±N days.  On a match, rewrites the expense's
``verwendungszweck`` directly to the string carried by
``EnrichmentRecord.description`` (e.g. "Etsy Inc"), then
re-normalises and optionally re-embeds the row.

Idempotency: matched records carry their PayPal Transaktionscode in
``enrichment_ref``; rows with a non-NULL ``enrichment_ref`` are
skipped on re-run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from expensa.features.embeddings import Embedder, store_embeddings
from expensa.ingestion.normalizer import (
    combined_text,
    normalize_counterparty,
    normalize_verwendungszweck,
)
from expensa.ingestion.sources import EnrichmentRecord, SourceAdapter

DEFAULT_DATE_WINDOW_DAYS = 4


@dataclass
class EnrichReport:
    source: str
    parsed: int = 0          # records in the secondary CSV
    already_enriched: int = 0  # records skipped — ref already in DB
    matched: int = 0         # bank rows updated in this run
    ambiguous: int = 0       # records with tied nearest-date candidates
    unmatched: int = 0       # eligible records with no matching bank row
    reembedded: int = 0
    enriched_ids: list[int] = field(default_factory=list)


@dataclass
class _MatchResult:
    matches: list[tuple[dict, EnrichmentRecord]]
    ambiguous: int   # records dropped due to tied nearest-date candidates
    unmatched: int   # eligible records with no bank row in window


def _match(
    candidates: list[dict],
    records: list[EnrichmentRecord],
    adapter: SourceAdapter,
    date_window_days: int,
) -> _MatchResult:
    """Return one-to-one (candidate, record) pairs by |amount| + nearest date.

    Eligible candidates are those passing ``adapter.candidate_filter``.
    If two records tie for nearest date, neither is matched (ambiguous).
    Each record is consumed by at most one candidate.
    """
    eligible = [c for c in candidates if adapter.candidate_filter(c)]

    by_cents: dict[int, list[EnrichmentRecord]] = {}
    for rec in records:
        by_cents.setdefault(abs(rec.amount_cents), []).append(rec)

    consumed: set[str] = set()
    ambiguous_refs: set[str] = set()
    matches: list[tuple[dict, EnrichmentRecord]] = []

    for cand in sorted(eligible, key=lambda c: (c["buchungsdatum"], c["id"])):
        pool = [
            rec
            for rec in by_cents.get(abs(int(cand["betrag_cents"])), [])
            if rec.source_ref not in consumed
            and abs((cand["buchungsdatum"] - rec.txn_date).days) <= date_window_days
        ]
        if not pool:
            continue
        best = min(abs((cand["buchungsdatum"] - r.txn_date).days) for r in pool)
        closest = [r for r in pool if abs((cand["buchungsdatum"] - r.txn_date).days) == best]
        if len(closest) > 1:
            for r in closest:
                ambiguous_refs.add(r.source_ref)
            continue
        rec = closest[0]
        consumed.add(rec.source_ref)
        matches.append((cand, rec))

    matched_refs = {rec.source_ref for _, rec in matches}
    unmatched = sum(
        1 for rec in records
        if rec.source_ref not in matched_refs
        and rec.source_ref not in ambiguous_refs
    )
    return _MatchResult(matches=matches, ambiguous=len(ambiguous_refs), unmatched=unmatched)


def enrich_from_records(
    conn: sqlite3.Connection,
    records: list[EnrichmentRecord],
    adapter: SourceAdapter,
    embedder: Embedder | None = None,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> EnrichReport:
    """Match records against the DB and write enriched Verwendungszweck.

    Safe to call repeatedly — expenses with a non-NULL ``enrichment_ref``
    are excluded from matching, so re-running on the same data is a no-op.
    """
    report = EnrichReport(source=adapter.name, parsed=len(records))

    used_refs = {
        r["enrichment_ref"]
        for r in conn.execute(
            "SELECT enrichment_ref FROM expenses WHERE enrichment_ref IS NOT NULL"
        ).fetchall()
    }
    already_enriched = [r for r in records if r.source_ref in used_refs]
    eligible = [r for r in records if r.source_ref not in used_refs]
    report.already_enriched = len(already_enriched)

    candidates = [
        dict(row)
        for row in conn.execute(
            "SELECT id, buchungsdatum, betrag_cents, "
            "zahlungsempfaenger, zahlungspflichtiger "
            "FROM expenses WHERE enrichment_ref IS NULL"
        ).fetchall()
    ]

    result = _match(candidates, eligible, adapter, date_window_days)
    matches = result.matches
    report.matched = len(matches)
    report.ambiguous = result.ambiguous
    report.unmatched = result.unmatched

    for cand, rec in matches:
        eid = int(cand["id"])
        new_vz = rec.description          # pre-formatted by the adapter
        new_vz_norm = normalize_verwendungszweck(new_vz)
        cp_norm = normalize_counterparty(rec.counterparty)
        new_combined = combined_text(cp_norm, new_vz_norm)
        conn.execute(
            """
            UPDATE expenses
               SET verwendungszweck            = ?,
                   verwendungszweck_normalized = ?,
                   counterparty_normalized     = ?,
                   combined_text               = ?,
                   enrichment_source           = ?,
                   enrichment_ref              = ?
             WHERE id = ?
            """,
            (new_vz, new_vz_norm, cp_norm, new_combined, adapter.name, rec.source_ref, eid),
        )
        report.enriched_ids.append(eid)

    if embedder is not None and report.enriched_ids:
        ph = ",".join("?" * len(report.enriched_ids))
        conn.execute(
            f"DELETE FROM embeddings WHERE expense_id IN ({ph})",
            report.enriched_ids,
        )
        rows = conn.execute(
            f"SELECT id, combined_text FROM expenses WHERE id IN ({ph})",
            report.enriched_ids,
        ).fetchall()
        report.reembedded = store_embeddings(
            conn, embedder, [(r["id"], r["combined_text"] or "") for r in rows]
        )

    return report


# --- Dry-run preview (no DB writes) -----------------------------------------


@dataclass
class EnrichmentPreview:
    vz_before: str
    vz_after: str


@dataclass
class PreviewReport:
    source: str
    matched: int
    previews: list[EnrichmentPreview] = field(default_factory=list)


def preview_enrichment(
    rows,  # list[ParsedRow]
    records: list[EnrichmentRecord],
    adapter: SourceAdapter,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> PreviewReport:
    """Match parsed bank rows against secondary records in memory.

    Returns a before/after view for each match without writing to the DB.
    """
    candidates = [
        {
            "id": i,
            "buchungsdatum": row.buchungsdatum,
            "betrag_cents": row.betrag_cents,
            "zahlungsempfaenger": row.zahlungsempfaenger,
            "zahlungspflichtiger": row.zahlungspflichtiger,
        }
        for i, row in enumerate(rows)
    ]

    result = _match(candidates, records, adapter, date_window_days)
    previews = [
        EnrichmentPreview(
            vz_before=rows[cand["id"]].verwendungszweck,
            vz_after=rec.description,
        )
        for cand, rec in result.matches
    ]
    return PreviewReport(
        source=adapter.name,
        matched=len(result.matches),
        previews=previews,
    )
