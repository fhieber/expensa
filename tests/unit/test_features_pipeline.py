"""Feature pipeline tests using the HashEmbedder (no model download)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expense_analyzer.features.embeddings import (
    HashEmbedder,
    load_embeddings,
    store_embeddings,
)
from expense_analyzer.features.pipeline import (
    add_calendar_features,
    add_log_amount,
    add_temporal_recurrence,
    base_dataframe,
    build_full_features,
)
from expense_analyzer.features.similarity import best_fuzzy_match_known_vendor
from expense_analyzer.features.temporal import (
    basic_calendar_features,
    is_likely_recurring,
)
from expense_analyzer.ingestion import ingest_csv


def test_basic_calendar_features() -> None:
    from datetime import date

    f = basic_calendar_features(date(2026, 2, 28))  # a Saturday
    assert f["year"] == 2026
    assert f["month"] == 2
    assert f["quarter"] == 1
    assert f["day_of_week"] == 5
    assert f["is_weekend"] == 1
    assert f["is_month_end"] == 1


def test_base_dataframe_after_ingest(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = base_dataframe(tmp_db)
    assert len(df) == 50
    assert "combined_text" in df.columns
    assert df["combined_text"].notna().all()


def test_calendar_features_added(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = add_calendar_features(base_dataframe(tmp_db))
    for c in ("year", "month", "day_of_week", "is_weekend"):
        assert c in df.columns


def test_log_amount_added(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = add_log_amount(base_dataframe(tmp_db))
    assert "log_abs_amount" in df.columns
    assert (df["log_abs_amount"] >= 0).all()


def test_recurrence_features(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = add_temporal_recurrence(tmp_db, base_dataframe(tmp_db))
    # The Feb-1 rent has a Jan-1 prior.
    feb_rent = df[
        (df["counterparty_normalized"] == "vermieter")
        & (df["buchungsdatum"].astype(str) == "2026-02-01")
    ].iloc[0]
    assert feb_rent["days_since_prev_same_cp"] == 31
    # 31 days ago is outside the 30-day window but within 90/365.
    assert feb_rent["count_same_cp_30d"] == 0
    assert feb_rent["count_same_cp_90d"] >= 1


def test_is_likely_recurring_needs_three_months(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """Rent appears in Jan and Feb only - 2 months, so NOT yet recurring per heuristic."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    feb_rent_id = tmp_db.execute(
        "SELECT id FROM expenses "
        "WHERE counterparty_normalized = 'vermieter' "
        "ORDER BY buchungsdatum DESC LIMIT 1"
    ).fetchone()["id"]
    assert is_likely_recurring(tmp_db, feb_rent_id) is False


def test_hash_embedder_deterministic() -> None:
    e = HashEmbedder(dim=128)
    a = e.encode(["markt alpha | einkauf"])
    b = e.encode(["markt alpha | einkauf"])
    assert a.shape == (1, 128)
    assert (a == b).all()


def test_store_and_load_embeddings(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    e = HashEmbedder(dim=64)
    rows = tmp_db.execute("SELECT id, combined_text FROM expenses").fetchall()
    n = store_embeddings(e, *[]) if False else store_embeddings(
        tmp_db, e, [(r["id"], r["combined_text"]) for r in rows]
    )
    assert n == 50
    ids, mat = load_embeddings(tmp_db, e.model_name)
    assert len(ids) == 50
    assert mat.shape == (50, 64)
    # Re-running should be a no-op (already cached).
    n2 = store_embeddings(
        tmp_db, e, [(r["id"], r["combined_text"]) for r in rows]
    )
    assert n2 == 0


def test_build_full_features_with_embeddings(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    e = HashEmbedder(dim=64)
    df, mat = build_full_features(tmp_db, embedder=e)
    assert mat is not None
    assert mat.shape == (len(df), 64)
    assert "log_abs_amount" in df.columns
    assert "days_since_prev_same_cp" in df.columns


def test_fuzzy_match_known_vendor() -> None:
    known = ["markt alpha", "markt beta", "vermieter"]
    name, score = best_fuzzy_match_known_vendor("markt alpha filiale", known)
    assert name == "markt alpha"
    assert score >= 80
    name, score = best_fuzzy_match_known_vendor("totally unknown vendor 1234", known)
    assert score < 80


def test_fuzzy_match_empty_inputs() -> None:
    assert best_fuzzy_match_known_vendor("", ["foo"]) == (None, 0)
    assert best_fuzzy_match_known_vendor("foo", []) == (None, 0)
