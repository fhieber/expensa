"""Tests for the three new Dashboard insight data fns in viz/data.py."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.storage.categories import add_label, upsert_category
from expense_analyzer.viz import (
    anomalies,
    monthly_flow_by_category,
    monthly_income_vs_expense,
    recurring_subscriptions,
    weekly_by_category,
)

# ---------------------------------------------------------------- recurring


RECURRING_COLS = {
    "name", "cadence", "last_seen", "typical_amount",
    "charges_per_year", "annualised", "n_charges",
}


def test_recurring_subscriptions_empty_db_returns_empty(
    tmp_db: sqlite3.Connection,
) -> None:
    df = recurring_subscriptions(tmp_db)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert set(df.columns) == RECURRING_COLS


def test_recurring_subscriptions_detects_recurring_vendors(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """sample_de.csv has multiple vendors with >=3 transactions
    (REWE / Edeka / etc.). The cadence detector should surface them."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = recurring_subscriptions(tmp_db, min_charges=3)
    assert not df.empty
    assert set(df.columns) == RECURRING_COLS
    # Each vendor has the floor number of charges.
    assert (df["n_charges"] >= 3).all()
    # Cadence is one of the supported labels.
    assert df["cadence"].isin({
        "weekly", "bi-weekly", "monthly", "quarterly",
        "semi-annual", "annual",
    }).all()
    # annualised = typical_amount * charges_per_year (within float tol).
    for _, r in df.iterrows():
        assert abs(r["annualised"]
                   - r["typical_amount"] * r["charges_per_year"]) < 1e-6
    # Sorted DESC by annualised cost.
    assert (df["annualised"].diff().dropna() <= 0).all()


def test_recurring_subscriptions_min_charges_floor(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    n_at_3 = len(recurring_subscriptions(tmp_db, min_charges=3))
    n_at_99 = len(recurring_subscriptions(tmp_db, min_charges=99))
    assert n_at_99 == 0
    assert n_at_3 >= n_at_99


def test_weekly_by_category_groups_by_iso_week(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = weekly_by_category(tmp_db)
    assert not df.empty
    assert set(df.columns) == {"w", "name", "color", "amount"}
    # All week labels match the YYYY-Www pattern.
    import re

    pat = re.compile(r"^\d{4}-W\d{2}$")
    assert df["w"].map(lambda x: bool(pat.match(x))).all()
    # Amounts are non-negative (we already take ABS in SQL).
    assert (df["amount"] >= 0).all()


def test_classify_cadence_snaps_to_buckets() -> None:
    from expense_analyzer.viz.data import _classify_cadence

    # Each of the canonical buckets should map back to itself.
    label, cpy = _classify_cadence(7)
    assert label == "weekly"
    assert abs(cpy - 365.25 / 7) < 1e-6

    label, cpy = _classify_cadence(30)  # ~monthly
    assert label == "monthly"

    label, cpy = _classify_cadence(91)
    assert label == "quarterly"

    label, cpy = _classify_cadence(365)
    assert label == "annual"

    # Edge / degenerate inputs.
    assert _classify_cadence(0)[0] == "irregular"
    assert _classify_cadence(-5)[0] == "irregular"
    assert _classify_cadence(None)[0] == "irregular"


# ----------------------------------------------------- income vs expense


def test_monthly_income_vs_expense_shape(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = monthly_income_vs_expense(tmp_db)
    assert not df.empty
    assert set(df.columns) == {
        "ym", "income", "expenses", "net", "savings_rate",
    }
    # Income and expenses are both expressed as non-negative magnitudes
    # in this representation (the SUM of |betrag_cents| for expenses
    # already strips the sign).
    assert (df["income"] >= 0).all()
    assert (df["expenses"] >= 0).all()
    # net = income - expenses; check the invariant.
    assert ((df["income"] - df["expenses"] - df["net"]).abs() < 1e-6).all()


def test_monthly_income_vs_expense_savings_rate_formula(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = monthly_income_vs_expense(tmp_db)
    for _, r in df.iterrows():
        if r["income"] > 0:
            expected = (r["income"] - r["expenses"]) / r["income"]
            assert abs(r["savings_rate"] - expected) < 1e-9
        else:
            assert pd.isna(r["savings_rate"])


def test_monthly_income_vs_expense_excludes_internal_transfers(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """When exclude_internal=True (default), expense rows whose IBAN is
    in `own_ibans` should not count as expenses."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # Flag every row as internal -- the bluntest possible exclusion.
    tmp_db.execute("UPDATE expenses SET iban_is_known_self = 1")
    df_excl = monthly_income_vs_expense(tmp_db, exclude_internal=True)
    df_incl = monthly_income_vs_expense(tmp_db, exclude_internal=False)
    if df_excl.empty:
        # Everything filtered out as expected.
        assert not df_incl.empty
    else:
        # If there are any rows at all in `df_excl`, every income / expense
        # column must be zero (since all rows are internal).
        assert df_excl["income"].sum() == 0
        assert df_excl["expenses"].sum() == 0


# ----------------------------------------------------------- anomalies


def test_anomalies_empty_when_no_data(tmp_db: sqlite3.Connection) -> None:
    df = anomalies(tmp_db)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_anomalies_returns_only_above_threshold(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """Synthesize a clear anomaly: insert several identical small REWE
    charges, then one giant REWE charge. The big one should surface."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # Drop everything we don't care about to make the assertion tight.
    tmp_db.execute("DELETE FROM expenses WHERE counterparty_normalized != 'rewe markt'")
    # Now add an outlier row -- 5x the typical REWE bill.
    typical = tmp_db.execute(
        "SELECT AVG(ABS(betrag_cents)) FROM expenses"
    ).fetchone()[0]
    outlier_cents = int(typical * 5)
    cur = tmp_db.execute(
        """
        INSERT INTO expenses (
            buchungsdatum, betrag_cents, counterparty,
            counterparty_normalized, is_income, dedup_hash
        )
        VALUES (?, ?, 'REWE Markt', 'rewe markt', 0, ?)
        """,
        (date.today().isoformat(), -outlier_cents, "synthetic_outlier_hash"),
    )
    new_eid = cur.lastrowid

    df = anomalies(tmp_db, z_threshold=2.0, min_history=3)
    assert not df.empty
    # The synthetic outlier should be one of the results.
    assert (df["id"] == new_eid).any()
    # All returned rows have z above threshold.
    assert (df["zscore"] > 2.0).all()
    # vs_typical computed correctly.
    for _, r in df.iterrows():
        assert abs(r["vs_typical"] - r["amount"] / r["typical"]) < 1e-6


def test_anomalies_respects_min_history(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """A vendor with < min_history records cannot produce anomalies."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = anomalies(tmp_db, min_history=1000)  # absurdly high
    assert df.empty


def test_anomalies_columns_include_category_when_labeled(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    cid = upsert_category(tmp_db, "Lebensmittel")
    # Label every REWE row to ensure the JOIN finds something.
    for r in tmp_db.execute(
        "SELECT id FROM expenses WHERE counterparty_normalized = 'rewe markt'"
    ):
        add_label(tmp_db, int(r["id"]), cid, "user")
    # Synthesize an outlier so the table isn't empty.
    typical = tmp_db.execute(
        "SELECT AVG(ABS(betrag_cents)) FROM expenses "
        "WHERE counterparty_normalized = 'rewe markt'"
    ).fetchone()[0]
    cur = tmp_db.execute(
        """
        INSERT INTO expenses (
            buchungsdatum, betrag_cents, counterparty,
            counterparty_normalized, is_income, dedup_hash
        )
        VALUES (?, ?, 'REWE Markt', 'rewe markt', 0, ?)
        """,
        (date.today().isoformat(), -int(typical * 5), "synthetic_outlier_2"),
    )
    # Label the synthetic outlier as Lebensmittel too so the JOIN picks
    # up the category for this row.
    add_label(tmp_db, int(cur.lastrowid), cid, "user")

    df = anomalies(tmp_db, z_threshold=2.0, min_history=3)
    assert not df.empty
    assert "category" in df.columns
    # At least one of the anomalies should have the category we just set.
    assert (df["category"] == "Lebensmittel").any()


# ----------------- monthly_flow_by_category × monthly_income_vs_expense


def test_monthly_flow_excludes_internal_transfers_by_default(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """When `exclude_internal=True` (default), the per-category monthly
    flow must NOT count transactions whose iban is in own_ibans. This
    keeps it numerically consistent with monthly_income_vs_expense."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # Mark every row as internal — the bluntest exclusion.
    tmp_db.execute("UPDATE expenses SET iban_is_known_self = 1")

    df_excluded = monthly_flow_by_category(tmp_db)        # default True
    df_included = monthly_flow_by_category(tmp_db, exclude_internal=False)

    # All rows internal -> default call should return empty (no per-cat
    # rows survive the filter).
    assert df_excluded.empty or df_excluded["amount"].abs().sum() == 0
    # Disabling the filter brings them back.
    assert not df_included.empty


def test_monthly_flow_and_income_vs_expense_agree_on_net(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """The per-category stacked chart and the income/expense summary
    chart should expose the SAME net flow per month -- modulo how rows
    are grouped (by category vs by sign).  This only holds if BOTH
    apply the same internal-transfer filter, which is the point of
    this PR: ``monthly_flow_by_category`` now defaults to
    ``exclude_internal=True``, matching ``monthly_income_vs_expense``.

    Note: pos / neg can NOT be matched per row, because the by-category
    fn nets each category's rows together. A category with a refund
    (+) and regular spend (-) becomes one row with the net amount, so
    the positive vs negative buckets won't reconcile per-row across
    the two queries. Net totals always reconcile.
    """
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")

    by_cat = monthly_flow_by_category(tmp_db)              # excl_internal=True
    ivex = monthly_income_vs_expense(tmp_db)               # excl_internal=True

    net_by_month_cat = by_cat.groupby("ym")["amount"].sum()
    ivex_indexed = ivex.set_index("ym")

    all_months = set(net_by_month_cat.index) | set(ivex_indexed.index)
    for ym in all_months:
        net_cat = float(net_by_month_cat.get(ym, 0.0))
        net_ive = (
            float(ivex_indexed.loc[ym, "net"])
            if ym in ivex_indexed.index else 0.0
        )
        assert abs(net_cat - net_ive) < 0.01, (
            f"net mismatch for {ym}: by_cat={net_cat} vs ivex={net_ive}"
        )


def test_monthly_flow_net_changes_when_internals_included(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    """When internal transfers exist, ``exclude_internal=True`` should
    yield a different (smaller-in-magnitude) net than
    ``exclude_internal=False`` -- proving the flag actually filters."""
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    # Mark every salary row (positive amounts only) as internal so we
    # know which side will move when we toggle the flag.
    tmp_db.execute("UPDATE expenses SET iban_is_known_self = 1 WHERE is_income = 1")
    excl = monthly_flow_by_category(tmp_db, exclude_internal=True)
    incl = monthly_flow_by_category(tmp_db, exclude_internal=False)
    # `incl` should have at least one more row OR a larger absolute net.
    incl_total = float(incl["amount"].sum())
    excl_total = float(excl["amount"].sum())
    assert incl_total != excl_total
