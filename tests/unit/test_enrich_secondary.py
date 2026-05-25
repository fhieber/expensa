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

    assert rep.parsed == 9
    # 4 PayPal Lastschrift lines are candidates (the REWE line is filtered out).
    assert rep.candidate_expenses == 4
    assert rep.matched == 2          # Etsy (-19,80) and Steam (-45,00)
    assert rep.ambiguous == 1        # two -50,00 records equidistant from the bank line
    assert rep.unmatched_expenses == 1  # the -99,99 line has no record
    # Two ambiguous Shop A/B + bookshop decoy + faraway + refund (+15,00) +
    # the wrapper-resolved betterplace pair (both at |14,00| with no bank
    # match in this fixture).
    assert rep.unused_records == 7


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
    # The PayPal Beschreibung is a merchant-supplied memo in this row;
    # it must surface in the enriched description (and downstream in
    # combined_text) so the classifier sees the real item, not the
    # generic "PP.1234.PP Ihr Einkauf" the bank gave us.
    assert "Tasse" in row["enriched_description"]
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


def test_enrichment_rewrites_counterparty_normalized_to_merchant(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """Before this change, every PayPal Lastschrift normalized to the
    same "paypal europe sarl et cie..." string, so vendor_exact_match
    could never accumulate across PayPal-routed expenses. Enrichment
    now overwrites ``counterparty_normalized`` with the merchant's
    normalized name -- so the cascade's vendor_exact_match stage can
    actually pool labels across Etsy purchases regardless of how
    they were funded."""
    _ingest(tmp_db, fixtures_dir)
    adapter = PaypalAdapter()
    records = adapter.parse(fixtures_dir / "sample_paypal.csv")
    enrich_from_records(tmp_db, records, adapter, date_window_days=4)

    row = tmp_db.execute(
        "SELECT counterparty, enriched_counterparty, counterparty_normalized "
        "FROM expenses WHERE betrag_cents = -1980"
    ).fetchone()
    # Bank column unchanged ("PayPal Europe ..."); enriched_counterparty
    # holds the merchant; counterparty_normalized now follows the merchant.
    assert "paypal" in (row["counterparty"] or "").lower()
    assert row["enriched_counterparty"] == "Etsy Inc"
    assert "etsy" in (row["counterparty_normalized"] or "").lower()
    # And no "paypal" residue in the normalized form -- the normalizer's
    # processor-noise filter strips it, but the test pins the contract.
    assert "paypal" not in (row["counterparty_normalized"] or "").lower()


def test_unmatched_rows_keep_their_bank_counterparty_normalized(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """Defensive: rows that didn't match must NOT have their
    counterparty_normalized clobbered. The COALESCE guard in the
    UPDATE only fires for matched ids, but we pin it here so a
    future refactor can't quietly break the invariant."""
    _ingest(tmp_db, fixtures_dir)
    before = tmp_db.execute(
        "SELECT id, counterparty_normalized FROM expenses "
        "WHERE counterparty = 'REWE Markt GmbH'"
    ).fetchone()
    adapter = PaypalAdapter()
    enrich_from_records(
        tmp_db, adapter.parse(fixtures_dir / "sample_paypal.csv"),
        adapter, date_window_days=4,
    )
    after = tmp_db.execute(
        "SELECT counterparty_normalized FROM expenses WHERE id = ?",
        (before["id"],),
    ).fetchone()
    assert after["counterparty_normalized"] == before["counterparty_normalized"]


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
