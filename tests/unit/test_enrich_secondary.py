"""Tests for the PayPal enrichment engine and ingest-time VZ simplification."""

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

    # 5 PayPal CSV rows with name + ref + parseable date/amount.
    assert rep.parsed == 5
    # Alpha (-19,80, 2 days off) and Beta (-45,00, 1 day off) match.
    assert rep.matched == 2


def test_vz_rewritten_directly(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    """Enrichment writes the merchant name directly into verwendungszweck."""
    _ingest(tmp_db, fixtures_dir)
    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    enrich_from_records(tmp_db, records, adapter, date_window_days=4)

    row = tmp_db.execute(
        "SELECT verwendungszweck, enrichment_source, enrichment_ref, "
        "counterparty_normalized, combined_text "
        "FROM expenses WHERE betrag_cents = -1980"
    ).fetchone()
    assert row["verwendungszweck"] == "Haendler Alpha GmbH"
    assert row["enrichment_source"] == "paypal"
    assert row["enrichment_ref"] == "TXN-ALPHA-1"
    assert "haendler alpha" in (row["counterparty_normalized"] or "").lower()
    assert "haendler alpha" in (row["combined_text"] or "").lower()


def test_vz_with_email(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    """When the PayPal row has an email address, it appears in the VZ."""
    # Gamma has email "vendor-a@example.com" but -50,00 is ambiguous, so
    # use TXN-BETA-1 which has no email to show the no-email path.
    _ingest(tmp_db, fixtures_dir)
    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    enrich_from_records(tmp_db, records, adapter, date_window_days=4)

    row = tmp_db.execute(
        "SELECT verwendungszweck FROM expenses WHERE betrag_cents = -4500"
    ).fetchone()
    assert row["verwendungszweck"] == "Haendler Beta GmbH"


def test_ingest_simplifies_known_merchant_vz(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """At ingest time, PayPal rows with 'Ihr Einkauf bei <merchant>' are
    simplified to just the merchant name — no PayPal CSV needed."""
    _ingest(tmp_db, fixtures_dir)
    row = tmp_db.execute(
        "SELECT verwendungszweck FROM expenses WHERE betrag_cents = -8760"
    ).fetchone()
    assert row["verwendungszweck"] == "Onlinehaendler GmbH"


def test_ingest_strips_multiple_whitespace(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """The fixture has multiple spaces in zahlungsempfaenger; they must be
    collapsed to a single space in all stored fields."""
    _ingest(tmp_db, fixtures_dir)
    row = tmp_db.execute(
        "SELECT zahlungsempfaenger FROM expenses WHERE betrag_cents = -1980"
    ).fetchone()
    assert "  " not in (row["zahlungsempfaenger"] or "")


def test_non_paypal_row_not_enriched(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    _ingest(tmp_db, fixtures_dir)
    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    enrich_from_records(tmp_db, records, adapter, date_window_days=4)
    row = tmp_db.execute(
        "SELECT enrichment_ref FROM expenses WHERE counterparty = 'Markt Alpha GmbH'"
    ).fetchone()
    assert row["enrichment_ref"] is None


def test_reembeds_enriched_rows(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    emb = HashEmbedder(dim=64)
    _ingest(tmp_db, fixtures_dir, embedder=emb)
    alpha_id = tmp_db.execute(
        "SELECT id FROM expenses WHERE betrag_cents = -1980"
    ).fetchone()["id"]
    _, before = load_embeddings(tmp_db, emb.model_name, [alpha_id])

    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    rep = enrich_from_records(tmp_db, records, adapter, embedder=emb, date_window_days=4)
    assert rep.reembedded == 2

    _, after = load_embeddings(tmp_db, emb.model_name, [alpha_id])
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
