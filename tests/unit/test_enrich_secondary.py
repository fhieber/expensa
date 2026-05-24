"""Tests for the generic secondary-source enrichment engine, exercised via
the PayPal adapter and the toy fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.enrichment.secondary import enrich_from_records
from expense_analyzer.features.embeddings import HashEmbedder, load_embeddings
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ingestion.sources.paypal import PaypalAdapter


def _ingest(tmp_db: sqlite3.Connection, fixtures_dir: Path, embedder=None):
    ingest_csv(tmp_db, fixtures_dir / "sample_de_paypal.csv", embedder=embedder)


def test_match_counts(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    _ingest(tmp_db, fixtures_dir)
    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    rep = enrich_from_records(tmp_db, records, adapter, date_window_days=4)

    assert rep.parsed == 6
    # 4 PayPal Lastschrift lines are candidates (the REWE line is filtered out).
    assert rep.candidate_expenses == 4
    assert rep.matched == 2          # Etsy (-19,80) and Steam (-45,00)
    assert rep.ambiguous == 1        # two -50,00 records equidistant from the bank line
    assert rep.unmatched_expenses == 1  # the -99,99 line has no record
    assert rep.unused_records == 4   # the two ambiguous + bookshop decoy + faraway


def test_non_paypal_row_not_enriched(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    _ingest(tmp_db, fixtures_dir)
    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    enrich_from_records(tmp_db, records, adapter, date_window_days=4)
    # The REWE -30,00 line shares an amount with the bookshop PayPal record
    # but must NOT be enriched (candidate_filter excludes non-PayPal rows).
    row = tmp_db.execute(
        "SELECT enrichment_ref FROM expenses WHERE counterparty = 'REWE Markt GmbH'"
    ).fetchone()
    assert row["enrichment_ref"] is None


def test_enrichment_written_and_combined_text_rebuilt(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    _ingest(tmp_db, fixtures_dir)
    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    enrich_from_records(tmp_db, records, adapter, date_window_days=4)

    row = tmp_db.execute(
        "SELECT enrichment_source, enriched_counterparty, enriched_description, "
        "combined_text FROM expenses WHERE betrag_cents = -1980"
    ).fetchone()
    assert row["enrichment_source"] == "paypal"
    assert row["enriched_counterparty"] == "Etsy Inc"
    assert "Tasse" in row["enriched_description"]
    # The real merchant now drives the embedding input.
    assert "etsy" in row["combined_text"].lower()


def test_reembeds_enriched_rows(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    emb = HashEmbedder(dim=64)
    _ingest(tmp_db, fixtures_dir, embedder=emb)
    etsy_id = tmp_db.execute(
        "SELECT id FROM expenses WHERE betrag_cents = -1980"
    ).fetchone()["id"]
    _, before = load_embeddings(tmp_db, emb.model_name, [etsy_id])

    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    rep = enrich_from_records(tmp_db, records, adapter, embedder=emb, date_window_days=4)
    assert rep.reembedded == 2

    _, after = load_embeddings(tmp_db, emb.model_name, [etsy_id])
    # combined_text changed, so the cached vector must have changed too.
    assert not (before == after).all()


def test_idempotent_rerun(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    emb = HashEmbedder(dim=64)
    _ingest(tmp_db, fixtures_dir, embedder=emb)
    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")

    first = enrich_from_records(tmp_db, records, adapter, embedder=emb, date_window_days=4)
    assert first.matched == 2

    second = enrich_from_records(tmp_db, records, adapter, embedder=emb, date_window_days=4)
    assert second.matched == 0
    assert second.reembedded == 0
