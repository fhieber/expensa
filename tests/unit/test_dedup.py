"""Dedup-hash + ingestion tests against the synthetic German fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ingestion.csv_loader import parse_csv
from expense_analyzer.ingestion.dedup import compute_dedup_hash


def test_dedup_hash_stable_across_calls(fixtures_dir: Path) -> None:
    rows = parse_csv(fixtures_dir / "sample_de.csv")
    h1 = [compute_dedup_hash(r) for r in rows]
    h2 = [compute_dedup_hash(r) for r in rows]
    assert h1 == h2


def test_dedup_hash_unique_per_row(fixtures_dir: Path) -> None:
    rows = parse_csv(fixtures_dir / "sample_de.csv")
    hashes = [compute_dedup_hash(r) for r in rows]
    assert len(set(hashes)) == len(hashes), "no two rows in the fixture should hash equal"


def test_ingest_inserts_all_new(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    report = ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    assert report.parsed == 50
    assert report.inserted == 50
    assert report.duplicates == 0
    assert report.embedded == 0  # no embedder passed -> no embeddings computed
    assert len(report.new_ids) == 50
    n = tmp_db.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
    assert n == 50


def test_ingest_with_embedder_computes_embeddings(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    from expense_analyzer.features.embeddings import HashEmbedder

    emb = HashEmbedder(dim=64)
    report = ingest_csv(tmp_db, fixtures_dir / "sample_de.csv", embedder=emb)
    assert report.embedded == 50
    # Re-ingesting the same file yields 0 new rows AND 0 new embeddings.
    report2 = ingest_csv(tmp_db, fixtures_dir / "sample_de.csv", embedder=emb)
    assert report2.inserted == 0
    assert report2.embedded == 0
    # Embeddings table is populated.
    n = tmp_db.execute(
        "SELECT COUNT(*) AS n FROM embeddings WHERE model_name = ?", (emb.model_name,)
    ).fetchone()["n"]
    assert n == 50


def test_reingest_same_file_zero_new(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    report = ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    assert report.inserted == 0
    assert report.duplicates == 50


def test_ingest_overlapping_file(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    report = ingest_csv(tmp_db, fixtures_dir / "sample_overlap.csv")
    # sample_overlap has 13 rows; 6 overlap with sample_de, 7 are new.
    assert report.parsed == 13
    assert report.inserted == 7
    assert report.duplicates == 6
    n = tmp_db.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
    assert n == 50 + 7


def test_ingested_rows_have_normalized_text(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    row = tmp_db.execute(
        "SELECT counterparty_normalized, combined_text FROM expenses "
        "WHERE zahlungsempfaenger='Markt Alpha GmbH' LIMIT 1"
    ).fetchone()
    assert row["counterparty_normalized"] == "markt alpha"
    assert "markt alpha" in row["combined_text"]


def test_ingested_rows_have_iban_features(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    foreign = tmp_db.execute(
        "SELECT iban_country, iban_is_foreign FROM expenses WHERE zahlungsempfaenger='Onlinehandel SARL' LIMIT 1"
    ).fetchone()
    assert foreign["iban_country"] == "LU"
    assert foreign["iban_is_foreign"] == 1

    domestic = tmp_db.execute(
        "SELECT iban_country, iban_is_foreign FROM expenses WHERE zahlungsempfaenger='Vermieter GmbH' LIMIT 1"
    ).fetchone()
    assert domestic["iban_country"] == "DE"
    assert domestic["iban_is_foreign"] == 0
