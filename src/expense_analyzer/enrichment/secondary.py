"""Generic secondary-source enrichment engine.

Matches a list of :class:`~expense_analyzer.ingestion.sources.EnrichmentRecord`
objects (produced by some adapter) against expenses already in the DB, by
**absolute amount + date proximity**, then writes the enrichment onto the
matched expense, rebuilds its ``combined_text`` and re-embeds it so the richer
description actually improves classification.

The engine is source-agnostic: it only ever sees ``EnrichmentRecord`` and the
adapter's ``name`` / ``candidate_filter``. Adding a new source needs no change
here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from expense_analyzer.features.embeddings import Embedder, store_embeddings
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


_CANDIDATE_SELECT = """
    SELECT id, buchungsdatum, betrag_cents, verwendungszweck,
           zahlungsempfaenger, zahlungspflichtiger, enrichment_ref
    FROM expenses
"""


def _rebuilt_combined_text(verwendungszweck: str, rec: EnrichmentRecord) -> str:
    cp = normalize_counterparty(rec.counterparty)
    vz = normalize_verwendungszweck(f"{verwendungszweck or ''} {rec.description or ''}")
    return combined_text(cp, vz)


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
        row
        for row in conn.execute(_CANDIDATE_SELECT).fetchall()
        if row["enrichment_ref"] is None and adapter.candidate_filter(row)
    ]
    report.candidate_expenses = len(candidates)

    # Index eligible records by absolute amount in cents.
    by_cents: dict[int, list[EnrichmentRecord]] = {}
    for rec in eligible:
        by_cents.setdefault(abs(rec.amount_cents), []).append(rec)

    consumed: set[str] = set()
    matches: list[tuple[sqlite3.Row, EnrichmentRecord]] = []

    # Deterministic order: oldest expense first.
    for exp in sorted(candidates, key=lambda r: (r["buchungsdatum"], r["id"])):
        pool = [
            rec
            for rec in by_cents.get(abs(int(exp["betrag_cents"])), [])
            if rec.source_ref not in consumed
            and abs((exp["buchungsdatum"] - rec.txn_date).days) <= date_window_days
        ]
        if not pool:
            report.unmatched_expenses += 1
            continue
        best = min(abs((exp["buchungsdatum"] - r.txn_date).days) for r in pool)
        closest = [
            r for r in pool
            if abs((exp["buchungsdatum"] - r.txn_date).days) == best
        ]
        if len(closest) > 1:
            report.ambiguous += 1
            continue
        rec = closest[0]
        consumed.add(rec.source_ref)
        matches.append((exp, rec))

    for exp, rec in matches:
        eid = int(exp["id"])
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
                _rebuilt_combined_text(exp["verwendungszweck"], rec),
                eid,
            ),
        )
        report.enriched_ids.append(eid)

    report.matched = len(matches)
    report.unused_records = len(eligible) - len(consumed)

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
