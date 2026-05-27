"""Equivalence test: bulk SQL pass must match the per-row helpers.

The per-row helpers (kept for testability and the rare single-row
caller) define the canonical semantics; the bulk path is the
production code path. Pin them together so future refactors of the
window-function SQL don't drift silently."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from expensa.features.temporal import (
    amount_zscore_within_counterparty,
    compute_temporal_features_bulk,
    count_to_same_counterparty,
    days_since_prev_to_same_counterparty,
    is_likely_recurring,
)
from expensa.ingestion import ingest_csv


@pytest.fixture
def populated_db(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> sqlite3.Connection:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    return tmp_db


def test_bulk_matches_per_row(populated_db: sqlite3.Connection) -> None:
    feats = compute_temporal_features_bulk(populated_db)
    ids = [r["id"] for r in populated_db.execute("SELECT id FROM expenses").fetchall()]
    for eid in ids:
        row = feats.get(eid)
        # Rows with empty counterparty_normalized don't appear in `same_cp`
        # CTE so they're missing from `feats` -- that matches the per-row
        # helpers, which return None / 0 for those rows.
        if row is None:
            assert days_since_prev_to_same_counterparty(populated_db, eid) is None
            continue

        assert row["days_since_prev_same_cp"] == days_since_prev_to_same_counterparty(
            populated_db, eid
        )
        assert row["count_same_cp_30d"] == count_to_same_counterparty(populated_db, eid, 30)
        assert row["count_same_cp_90d"] == count_to_same_counterparty(populated_db, eid, 90)
        assert row["count_same_cp_365d"] == count_to_same_counterparty(populated_db, eid, 365)

        bulk_z = row["amount_zscore_within_cp"]
        ref_z = amount_zscore_within_counterparty(populated_db, eid)
        if ref_z is None:
            assert bulk_z is None
        else:
            assert bulk_z == pytest.approx(ref_z, abs=1e-9)

        assert bool(row["is_likely_recurring"]) == is_likely_recurring(populated_db, eid)


def test_bulk_filter_by_ids(populated_db: sqlite3.Connection) -> None:
    """Asking for a subset only returns those rows."""
    all_ids = [
        int(r["id"]) for r in populated_db.execute("SELECT id FROM expenses").fetchall()
    ]
    subset = all_ids[:5]
    feats = compute_temporal_features_bulk(populated_db, expense_ids=subset)
    # Rows with empty counterparty_normalized aren't in the CTE; the bulk
    # function legitimately omits them. Just check we don't get extras.
    assert set(feats.keys()).issubset(set(subset))
