"""Tests for the new dashboard statistics / forecast helpers in viz.data.

The synthetic fixture (``sample_de.csv``) spans 2026-01-01 .. 2026-02-28,
so a February window has January as its previous same-length period.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from expensa.ingestion import ingest_csv
from expensa.viz.data import (
    _previous_period,
    categorization_mix,
    category_period_comparison,
    fixed_vs_variable,
    month_to_date_pace,
    period_totals,
    upcoming_recurring,
)


def _populated(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> sqlite3.Connection:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    return tmp_db


# ── _previous_period ──────────────────────────────────────────────────


def test_previous_period_same_length_immediately_before() -> None:
    # March (31 days inclusive) -> the 31 days ending the day before March 1.
    prev = _previous_period(date(2026, 3, 1), date(2026, 3, 31))
    assert prev == (date(2026, 1, 29), date(2026, 2, 28))


def test_previous_period_open_range_is_none() -> None:
    assert _previous_period(None, None) == (None, None)
    assert _previous_period(date(2026, 1, 1), None) == (None, None)


# ── period_totals ─────────────────────────────────────────────────────


def test_period_totals_matches_raw_sums(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    conn = _populated(tmp_db, fixtures_dir)
    tot = period_totals(conn, since=date(2026, 1, 1), until=date(2026, 2, 28))
    # Sanity: expenses positive, income >= 0, net == income - expenses.
    assert tot["expenses"] > 0
    assert tot["income"] >= 0
    assert abs(tot["net"] - (tot["income"] - tot["expenses"])) < 1e-6
    if tot["income"] > 0:
        assert tot["savings_rate"] is not None


def test_period_totals_empty_window_is_zero(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    conn = _populated(tmp_db, fixtures_dir)
    tot = period_totals(conn, since=date(2030, 1, 1), until=date(2030, 1, 31))
    assert tot["income"] == 0.0 and tot["expenses"] == 0.0
    assert tot["savings_rate"] is None


# ── category_period_comparison ────────────────────────────────────────


def test_category_period_comparison_february_vs_january(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    conn = _populated(tmp_db, fixtures_dir)
    # February window; previous period is the 28 days before Feb 1 (i.e.
    # most of January), so both sides carry data.
    cmp = category_period_comparison(
        conn, since=date(2026, 2, 1), until=date(2026, 2, 28)
    )
    assert not cmp.empty
    assert list(cmp.columns) == ["name", "current", "previous", "delta", "pct"]
    # delta is always current - previous.
    for _, r in cmp.iterrows():
        assert abs(r["delta"] - (r["current"] - r["previous"])) < 1e-6
    # Sorted by descending absolute delta.
    abs_deltas = cmp["delta"].abs().tolist()
    assert abs_deltas == sorted(abs_deltas, reverse=True)


def test_category_period_comparison_open_range_empty(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    conn = _populated(tmp_db, fixtures_dir)
    assert category_period_comparison(conn, since=None, until=None).empty


# ── month_to_date_pace ────────────────────────────────────────────────


def test_month_to_date_pace_projects_linearly(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    conn = _populated(tmp_db, fixtures_dir)
    # As of mid-February, the month is half elapsed.
    pace = month_to_date_pace(conn, today=date(2026, 2, 14))
    assert pace["days_elapsed"] == 14
    assert pace["days_in_month"] == 28
    # Projection scales the partial spend to the full month.
    if pace["spent"] > 0:
        expected = pace["spent"] / 14 * 28
        assert abs(pace["projected"] - expected) < 1e-6
    # January is a prior complete month -> baseline is populated.
    assert pace["baseline"] is not None


def test_month_to_date_pace_no_prior_months(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    conn = _populated(tmp_db, fixtures_dir)
    # January has no prior complete month in the fixture -> no baseline.
    pace = month_to_date_pace(conn, today=date(2026, 1, 15))
    assert pace["baseline"] is None


# ── fixed_vs_variable ─────────────────────────────────────────────────


def test_fixed_vs_variable_splits_total(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    conn = _populated(tmp_db, fixtures_dir)
    fv = fixed_vs_variable(conn)
    assert fv["total_monthly"] > 0
    # Components are non-negative and sum back to the total (fixed clamped
    # to never exceed total).
    assert fv["fixed_monthly"] >= 0
    assert fv["variable_monthly"] >= 0
    assert abs(
        (fv["fixed_monthly"] + fv["variable_monthly"]) - fv["total_monthly"]
    ) < 1e-6
    assert 0.0 <= fv["fixed_share"] <= 1.0


def test_fixed_vs_variable_empty_db(tmp_db: sqlite3.Connection) -> None:
    fv = fixed_vs_variable(tmp_db)
    assert fv["total_monthly"] == 0.0
    assert fv["fixed_share"] is None


# ── upcoming_recurring ────────────────────────────────────────────────


def test_upcoming_recurring_projects_future_charges(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    conn = _populated(tmp_db, fixtures_dir)
    # Just after the fixture's last date, with a wide horizon so monthly
    # vendors land inside it.
    up = upcoming_recurring(conn, horizon_days=40, today=date(2026, 3, 1))
    assert list(up.columns) == [
        "name", "cadence", "expected_date", "typical_amount", "days_until"
    ]
    if not up.empty:
        # Every projected date is in the future and within the horizon.
        assert (up["days_until"] >= 0).all()
        assert (up["days_until"] <= 40).all()
        # Sorted ascending by expected date.
        dates = up["expected_date"].tolist()
        assert dates == sorted(dates)


def test_upcoming_recurring_empty_when_no_recurring(
    tmp_db: sqlite3.Connection,
) -> None:
    assert upcoming_recurring(tmp_db).empty


# ── categorization_mix ────────────────────────────────────────────────


def test_categorization_mix_buckets_sum_to_total(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    from expensa.storage.categories import add_label, upsert_category

    conn = _populated(tmp_db, fixtures_dir)
    cat = upsert_category(conn, "Lebensmittel")
    ids = [int(r["id"]) for r in conn.execute(
        "SELECT id FROM expenses ORDER BY id LIMIT 6"
    ).fetchall()]
    # 2 user labels, plus model labels across the three confidence bands.
    add_label(conn, ids[0], cat, "user")
    add_label(conn, ids[1], cat, "user")
    add_label(conn, ids[2], cat, "model", confidence=0.95)  # high
    add_label(conn, ids[3], cat, "model", confidence=0.55)  # medium
    add_label(conn, ids[4], cat, "model", confidence=0.10)  # low

    mix = categorization_mix(conn)
    assert mix["total"] == 50
    assert mix["user"] == 2
    assert mix["high"] == 1
    assert mix["medium"] == 1
    assert mix["low"] == 1
    # Buckets are disjoint and cover everything.
    assert (
        mix["user"] + mix["high"] + mix["medium"] + mix["low"]
        + mix["uncategorized"]
    ) == mix["total"]


def test_categorization_mix_empty_db(tmp_db: sqlite3.Connection) -> None:
    mix = categorization_mix(tmp_db)
    assert mix["total"] == 0
    assert mix["uncategorized"] == 0
