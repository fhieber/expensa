"""Generic secondary-source enrichment engine.

Matches a list of :class:`~expense_analyzer.ingestion.sources.EnrichmentRecord`
objects (produced by some adapter) against expenses by **absolute amount +
date proximity**, then writes the enrichment onto the matched expense, rebuilds
its ``combined_text`` and re-embeds it so the richer description actually
improves classification.

The engine is source-agnostic: it only ever sees ``EnrichmentRecord`` and the
adapter's ``name`` / ``candidate_filter``. Adding a new source needs no change
here. The same matching core powers a no-DB ``preview_enrichment`` used by the
``ingest --dry-run`` showcase.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from expense_analyzer.features.embeddings import Embedder, store_embeddings
from expense_analyzer.ingestion.csv_loader import ParsedRow
from expense_analyzer.ingestion.normalizer import (
    combined_text,
    normalize_counterparty,
    normalize_verwendungszweck,
)
from expense_analyzer.ingestion.sources import EnrichmentRecord, SourceAdapter

DEFAULT_DATE_WINDOW_DAYS = 4


@dataclass
class EnrichReport:
    source: str
    parsed: int = 0
    candidate_expenses: int = 0
    matched: int = 0
    ambiguous: int = 0
    unmatched_expenses: int = 0
    unused_records: int = 0
    reembedded: int = 0
    enriched_ids: list[int] = field(default_factory=list)


@dataclass
class _Candidate:
    """A transaction that could be enriched, abstracted away from whether it
    came from the DB or a freshly-parsed bank CSV."""

    key: int                 # DB expense id, or list index for a parsed bank row
    txn_date: date
    betrag_cents: int
    filter_row: Mapping[str, object]  # passed to adapter.candidate_filter
    verwendungszweck: str


@dataclass
class _MatchResult:
    matches: list[tuple[_Candidate, EnrichmentRecord]]
    candidate_count: int
    ambiguous: int
    unmatched: int
    consumed_refs: set[str]


def _match_candidates(
    candidates: list[_Candidate],
    records: list[EnrichmentRecord],
    adapter: SourceAdapter,
    date_window_days: int,
) -> _MatchResult:
    """Pure matching: amount (abs cents) + nearest date within the window,
    one-to-one, ambiguous ties left unmatched. No DB / side effects."""
    eligible_cands = [c for c in candidates if adapter.candidate_filter(c.filter_row)]

    by_cents: dict[int, list[EnrichmentRecord]] = {}
    for rec in records:
        by_cents.setdefault(abs(rec.amount_cents), []).append(rec)

    consumed: set[str] = set()
    matches: list[tuple[_Candidate, EnrichmentRecord]] = []
    ambiguous = 0
    unmatched = 0

    for cand in sorted(eligible_cands, key=lambda c: (c.txn_date, c.key)):
        pool = [
            rec
            for rec in by_cents.get(abs(cand.betrag_cents), [])
            if rec.source_ref not in consumed
            and abs((cand.txn_date - rec.txn_date).days) <= date_window_days
        ]
        if not pool:
            unmatched += 1
            continue
        best = min(abs((cand.txn_date - r.txn_date).days) for r in pool)
        closest = [r for r in pool if abs((cand.txn_date - r.txn_date).days) == best]
        if len(closest) > 1:
            ambiguous += 1
            continue
        rec = closest[0]
        consumed.add(rec.source_ref)
        matches.append((cand, rec))

    return _MatchResult(
        matches=matches,
        candidate_count=len(eligible_cands),
        ambiguous=ambiguous,
        unmatched=unmatched,
        consumed_refs=consumed,
    )


def _rebuilt_combined_text(verwendungszweck: str, rec: EnrichmentRecord) -> str:
    cp = normalize_counterparty(rec.counterparty)
    vz = normalize_verwendungszweck(f"{verwendungszweck or ''} {rec.description or ''}")
    return combined_text(cp, vz)


_CANDIDATE_SELECT = """
    SELECT id, buchungsdatum, betrag_cents, verwendungszweck,
           zahlungsempfaenger, zahlungspflichtiger, enrichment_ref
    FROM expenses
"""


def enrich_from_records(
    conn: sqlite3.Connection,
    records: list[EnrichmentRecord],
    adapter: SourceAdapter,
    embedder: Embedder | None = None,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> EnrichReport:
    report = EnrichReport(source=adapter.name, parsed=len(records))

    # Idempotency: records whose source_ref is already attached to some
    # expense are considered applied; expenses already enriched are not
    # re-matched.
    used_refs = {
        r["enrichment_ref"]
        for r in conn.execute(
            "SELECT enrichment_ref FROM expenses WHERE enrichment_ref IS NOT NULL"
        ).fetchall()
    }
    eligible = [r for r in records if r.source_ref not in used_refs]

    candidates = [
        _Candidate(
            key=int(row["id"]),
            txn_date=row["buchungsdatum"],
            betrag_cents=int(row["betrag_cents"]),
            filter_row=dict(row),
            verwendungszweck=row["verwendungszweck"] or "",
        )
        for row in conn.execute(_CANDIDATE_SELECT).fetchall()
        if row["enrichment_ref"] is None
    ]

    result = _match_candidates(candidates, eligible, adapter, date_window_days)
    report.candidate_expenses = result.candidate_count
    report.ambiguous = result.ambiguous
    report.unmatched_expenses = result.unmatched
    report.matched = len(result.matches)
    report.unused_records = len(eligible) - len(result.consumed_refs)

    for cand, rec in result.matches:
        eid = int(cand.key)
        conn.execute(
            """
            UPDATE expenses
               SET enrichment_source = ?,
                   enrichment_ref = ?,
                   enriched_counterparty = ?,
                   enriched_description = ?,
                   enriched_at = CURRENT_TIMESTAMP,
                   combined_text = ?
             WHERE id = ?
            """,
            (
                adapter.name,
                rec.source_ref,
                rec.counterparty,
                rec.description,
                _rebuilt_combined_text(cand.verwendungszweck, rec),
                eid,
            ),
        )
        report.enriched_ids.append(eid)

    if embedder is not None and report.enriched_ids:
        ph = ",".join("?" * len(report.enriched_ids))
        # combined_text changed, so drop stale vectors first --
        # store_embeddings skips ids that already have one.
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


# --- Dry-run preview (no DB) -------------------------------------------------


@dataclass
class EnrichmentPreview:
    """One matched bank row shown with and without enrichment."""

    row: ParsedRow
    record: EnrichmentRecord
    combined_before: str
    combined_after: str


@dataclass
class PreviewReport:
    source: str
    parsed_records: int
    candidate_rows: int
    matched: int
    ambiguous: int
    unmatched: int
    previews: list[EnrichmentPreview] = field(default_factory=list)


def _row_combined_text(row: ParsedRow) -> str:
    """combined_text exactly as plain ingestion would compute it (no
    enrichment) -- mirrors ingestion._row_to_params."""
    counterparty = row.zahlungsempfaenger or row.zahlungspflichtiger
    return combined_text(
        normalize_counterparty(counterparty),
        normalize_verwendungszweck(row.verwendungszweck),
    )


def preview_enrichment(
    rows: list[ParsedRow],
    records: list[EnrichmentRecord],
    adapter: SourceAdapter,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> PreviewReport:
    """Match parsed bank rows against secondary records purely in memory and
    return a before/after view for each match. No DB, no writes."""
    # Key by position in the combined list -- source_row restarts per file,
    # so it isn't unique when several bank CSVs are previewed together.
    candidates = [
        _Candidate(
            key=i,
            txn_date=row.buchungsdatum,
            betrag_cents=row.betrag_cents,
            filter_row={
                "zahlungsempfaenger": row.zahlungsempfaenger,
                "zahlungspflichtiger": row.zahlungspflichtiger,
            },
            verwendungszweck=row.verwendungszweck,
        )
        for i, row in enumerate(rows)
    ]

    result = _match_candidates(candidates, records, adapter, date_window_days)
    previews = [
        EnrichmentPreview(
            row=rows[cand.key],
            record=rec,
            combined_before=_row_combined_text(rows[cand.key]),
            combined_after=_rebuilt_combined_text(cand.verwendungszweck, rec),
        )
        for cand, rec in result.matches
    ]
    return PreviewReport(
        source=adapter.name,
        parsed_records=len(records),
        candidate_rows=result.candidate_count,
        matched=len(result.matches),
        ambiguous=result.ambiguous,
        unmatched=result.unmatched,
        previews=previews,
    )
