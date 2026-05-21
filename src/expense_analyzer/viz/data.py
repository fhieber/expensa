"""SQL-driven views over the expenses + labels tables for visualization.

Each function returns a :class:`pandas.DataFrame` ready to feed a chart.
We use the most-recent label per expense (``user`` or ``model``) so that
predicted categories show up in dashboards even before the user reviews them.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd

from expense_analyzer.storage.sql import JOIN_LATEST_LABEL

# Category names that represent transfers between the user's own accounts
# (rather than real income / consumption). Treated as NEUTRAL by all the
# Dashboard income / expense aggregates: a row labelled with one of these
# names is excluded from both the income side and the expense side of
# ``monthly_income_vs_expense`` and from per-category sums. Pass a custom
# tuple to override on a per-call basis.
DEFAULT_SAVINGS_CATEGORIES: tuple[str, ...] = ("Sparen",)


def _savings_clause(
    savings_categories: tuple[str, ...] | None,
) -> tuple[str, list]:
    """Build an ``AND COALESCE(c.name, '') NOT IN (?, ?, ...)`` clause for
    excluding rows in any of ``savings_categories``. Returns ``("", [])``
    when no categories are supplied (no filter)."""
    if not savings_categories:
        return "", []
    ph = ",".join("?" * len(savings_categories))
    return f" AND COALESCE(c.name, '') NOT IN ({ph})", list(savings_categories)


def _date_filter_clause(
    column: str, since: date | None, until: date | None
) -> tuple[str, list]:
    parts: list[str] = []
    params: list = []
    if since is not None:
        parts.append(f"{column} >= ?")
        params.append(since.isoformat())
    if until is not None:
        parts.append(f"{column} <= ?")
        params.append(until.isoformat())
    if not parts:
        return "", []
    return " AND " + " AND ".join(parts), params


def spend_by_category(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    include_income: bool = False,
) -> pd.DataFrame:
    """Sum of |betrag| per category. Returns columns: name, color, amount."""
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    income_clause = "" if include_income else " AND e.is_income = 0"
    sql = f"""
        SELECT c.name, c.color, SUM(ABS(e.betrag_cents)) / 100.0 AS amount
        FROM expenses e
        {JOIN_LATEST_LABEL}
        WHERE 1=1 {income_clause} {extra}
        GROUP BY COALESCE(c.id, -1), c.name, c.color
        ORDER BY amount DESC
    """
    rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["name", "color", "amount"])
    df["name"] = df["name"].fillna("(unkategorisiert)")
    df["color"] = df["color"].fillna("#bbbbbb")
    return df


def monthly_flow_by_category(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    exclude_internal: bool = True,
    savings_categories: tuple[str, ...] = DEFAULT_SAVINGS_CATEGORIES,
) -> pd.DataFrame:
    """Monthly sums per category, signed. Useful for stacked / line charts.

    ``exclude_internal`` (default ``True``) drops rows where
    ``iban_is_known_self = 1`` (transfers between the user's own
    accounts).

    ``savings_categories`` (default ``("Sparen",)``) drops rows labelled
    as savings -- the user marks money-moved-to-own-bank-accounts with
    that category so it doesn't count as consumption. Combined with
    ``exclude_internal`` the view stays numerically consistent with
    ``monthly_income_vs_expense``.

    Set either flag to ``False`` / empty to see those flows.
    """
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    internal = (
        " AND COALESCE(e.iban_is_known_self, 0) = 0" if exclude_internal else ""
    )
    savings_sql, savings_params = _savings_clause(savings_categories)
    sql = f"""
        SELECT strftime('%Y-%m', e.buchungsdatum) AS ym,
               COALESCE(c.name, '(unkategorisiert)') AS name,
               COALESCE(c.color, '#bbbbbb') AS color,
               SUM(e.betrag_cents) / 100.0 AS amount
        FROM expenses e
        {JOIN_LATEST_LABEL}
        WHERE 1=1 {extra} {internal} {savings_sql}
        GROUP BY ym, name
        ORDER BY ym
    """
    rows = conn.execute(sql, params + savings_params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["ym", "name", "color", "amount"])
    return df


def amount_distribution(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    include_income: bool = False,
) -> pd.DataFrame:
    """One row per expense: amount + category. For histograms."""
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    income_clause = "" if include_income else " AND e.is_income = 0"
    sql = f"""
        SELECT ABS(e.betrag_cents) / 100.0 AS amount,
               COALESCE(c.name, '(unkategorisiert)') AS name
        FROM expenses e
        {JOIN_LATEST_LABEL}
        WHERE 1=1 {income_clause} {extra}
    """
    rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["amount", "name"])
    return df


def top_counterparties(
    conn: sqlite3.Connection,
    n: int = 15,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    extra, params = _date_filter_clause("buchungsdatum", since, until)
    sql = f"""
        SELECT counterparty AS name,
               SUM(ABS(betrag_cents)) / 100.0 AS amount,
               COUNT(*) AS n_tx
        FROM expenses
        WHERE is_income = 0 {extra}
        GROUP BY counterparty
        ORDER BY amount DESC
        LIMIT ?
    """
    rows = conn.execute(sql, params + [n]).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["name", "amount", "n_tx"])
    return df


def daily_calendar(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    """One row per (date) with total expense magnitude. For calendar heatmap."""
    extra, params = _date_filter_clause("buchungsdatum", since, until)
    sql = f"""
        SELECT buchungsdatum AS d, SUM(ABS(betrag_cents)) / 100.0 AS amount
        FROM expenses
        WHERE is_income = 0 {extra}
        GROUP BY buchungsdatum
        ORDER BY buchungsdatum
    """
    rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["d", "amount"])
    if not df.empty:
        df["d"] = pd.to_datetime(df["d"]).dt.date
    return df


# Cadence buckets used by recurring_subscriptions. Median day-gap snaps
# to the geometrically-nearest of these. ``charges_per_year`` is what we
# multiply ``typical_amount`` by to get the annualised cost.
_CADENCE_BUCKETS: tuple[tuple[str, float], ...] = (
    ("weekly",      7.0),
    ("bi-weekly",   14.0),
    ("monthly",     30.4),    # avg days/month
    ("quarterly",   91.3),
    ("semi-annual", 182.6),
    ("annual",      365.25),
)


def _classify_cadence(median_gap_days: float) -> tuple[str, float]:
    """Snap a median inter-charge gap to the geometrically-closest
    cadence bucket. Returns ``(label, charges_per_year)``."""
    import math

    if median_gap_days is None or median_gap_days <= 0:
        return ("irregular", 0.0)
    best_label, best_days = min(
        _CADENCE_BUCKETS,
        key=lambda b: abs(math.log(median_gap_days / b[1])),
    )
    return (best_label, 365.25 / best_days)


def recurring_subscriptions(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    min_charges: int = 3,
) -> pd.DataFrame:
    """Vendors that show a consistent **cadence** (weekly / bi-weekly /
    monthly / quarterly / semi-annual / annual) over their transaction
    history. Cadence is inferred from the median day-gap between
    consecutive charges; ``charges_per_year`` falls out of that, and
    ``annualised = typical_amount * charges_per_year``.

    Returns one row per vendor with at least ``min_charges`` transactions,
    sorted DESC by annualised cost.

    Columns: ``name``, ``cadence``, ``last_seen``, ``typical_amount``,
    ``charges_per_year``, ``annualised``, ``n_charges``.
    """
    extra, params = _date_filter_clause("buchungsdatum", since, until)
    sql = f"""
        SELECT counterparty_normalized AS cpn,
               counterparty AS name,
               buchungsdatum AS d,
               ABS(betrag_cents) / 100.0 AS amount
        FROM expenses
        WHERE counterparty_normalized IS NOT NULL
          AND counterparty_normalized <> ''
          AND is_income = 0
          {extra}
        ORDER BY counterparty_normalized, buchungsdatum
    """
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return pd.DataFrame(
            columns=["name", "cadence", "last_seen", "typical_amount",
                     "charges_per_year", "annualised", "n_charges"]
        )
    raw = pd.DataFrame([dict(r) for r in rows])
    raw["d"] = pd.to_datetime(raw["d"])

    out: list[dict] = []
    for cpn, g in raw.groupby("cpn"):
        if len(g) < min_charges:
            continue
        dates = g["d"].sort_values().reset_index(drop=True)
        gaps = dates.diff().dt.days.dropna()
        if gaps.empty:
            continue
        median_gap = float(gaps.median())
        cadence_label, cpy = _classify_cadence(median_gap)
        if cpy <= 0:
            continue
        typical = float(g["amount"].median())
        annualised = typical * cpy
        out.append({
            # Prefer the most-recent display-friendly counterparty string;
            # falls back to the normalised key if absent.
            "name": str(g.sort_values("d", ascending=False).iloc[0]["name"]
                        or cpn),
            "cadence": cadence_label,
            "last_seen": dates.iloc[-1].date(),
            "typical_amount": typical,
            "charges_per_year": cpy,
            "annualised": annualised,
            "n_charges": int(len(g)),
        })
    if not out:
        return pd.DataFrame(
            columns=["name", "cadence", "last_seen", "typical_amount",
                     "charges_per_year", "annualised", "n_charges"]
        )
    df = pd.DataFrame(out).sort_values("annualised", ascending=False)
    return df.reset_index(drop=True)


def savings_flow(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    savings_categories: tuple[str, ...] = DEFAULT_SAVINGS_CATEGORIES,
) -> pd.DataFrame:
    """Per-month money the user moved to / from their own savings.

    A row counts as savings flow when its **category** is in
    ``savings_categories`` (the user's own labelling). ``is_income``
    decides direction:

    * ``is_income = 0`` -> ``to_savings`` (money leaving this account
      to a savings account)
    * ``is_income = 1`` -> ``from_savings`` (money coming back from a
      savings account; common when funnelling between sub-accounts)

    Returns columns: ``ym``, ``to_savings``, ``from_savings``, ``net``
    (= ``to_savings - from_savings``, the actual net amount you put
    aside this month).
    """
    if not savings_categories:
        return pd.DataFrame(
            columns=["ym", "to_savings", "from_savings", "net"]
        )
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    ph = ",".join("?" * len(savings_categories))
    sql = f"""
        SELECT
            strftime('%Y-%m', e.buchungsdatum) AS ym,
            SUM(CASE WHEN e.is_income = 0 THEN ABS(e.betrag_cents) ELSE 0 END)
                / 100.0 AS to_savings,
            SUM(CASE WHEN e.is_income = 1 THEN e.betrag_cents ELSE 0 END)
                / 100.0 AS from_savings
        FROM expenses e
        {JOIN_LATEST_LABEL}
        WHERE COALESCE(c.name, '') IN ({ph}) {extra}
        GROUP BY ym
        ORDER BY ym
    """
    rows = conn.execute(sql, list(savings_categories) + params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return pd.DataFrame(
            columns=["ym", "to_savings", "from_savings", "net"]
        )
    df["net"] = df["to_savings"] - df["from_savings"]
    return df


def monthly_income_vs_expense(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    exclude_internal: bool = True,
    savings_categories: tuple[str, ...] = DEFAULT_SAVINGS_CATEGORIES,
) -> pd.DataFrame:
    """Per-month income vs expense totals.

    Returns: ``ym``, ``income``, ``expenses`` (both positive), ``net``,
    ``savings_rate`` ((income − expenses) / income).

    Rows that look like internal account-to-account transfers are
    excluded by default so the savings rate reflects real income vs
    real consumption. Two complementary filters apply:

    * ``iban_is_known_self`` -- automatic match against registered own
      IBANs. Catches transfers where the user has wired up
      ``Settings → My Accounts``.
    * ``savings_categories`` -- the user marks Sparen-bound charges
      with a designated category name (default ``("Sparen",)``). Catches
      transfers to / from own banks even when the destination IBAN
      isn't registered. Income rows in this category are also dropped
      (a return-trip from savings isn't fresh income).
    """
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    internal = (
        " AND COALESCE(e.iban_is_known_self, 0) = 0" if exclude_internal else ""
    )
    savings_sql, savings_params = _savings_clause(savings_categories)
    sql = f"""
        SELECT
            strftime('%Y-%m', e.buchungsdatum) AS ym,
            SUM(CASE WHEN e.is_income = 1 THEN e.betrag_cents ELSE 0 END) / 100.0
                AS income,
            SUM(CASE WHEN e.is_income = 0 THEN ABS(e.betrag_cents) ELSE 0 END) / 100.0
                AS expenses
        FROM expenses e
        {JOIN_LATEST_LABEL}
        WHERE 1=1 {extra} {internal} {savings_sql}
        GROUP BY ym
        ORDER BY ym
    """
    rows = conn.execute(sql, params + savings_params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return pd.DataFrame(
            columns=["ym", "income", "expenses", "net", "savings_rate"]
        )
    df["net"] = df["income"] - df["expenses"]
    df["savings_rate"] = df.apply(
        lambda r: (r["net"] / r["income"]) if r["income"] > 0 else None,
        axis=1,
    )
    return df


def anomalies(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    z_threshold: float = 2.0,
    min_history: int = 3,
    limit: int = 25,
) -> pd.DataFrame:
    """Recent expenses whose |amount| is unusually high for the vendor.

    Statistics (mean, variance, count) are computed over the WHOLE
    history of each counterparty -- not just the visible date range --
    so a wider baseline gives more confident anomaly scores. Anomalies
    themselves are filtered to ``since..until`` and the top ``limit``
    most-recent + most-deviant rows are returned.

    A row qualifies as an anomaly when:
        * the vendor has ``min_history`` or more prior records,
        * the across-history standard deviation is non-zero, and
        * ``(|amount| - mean) / stddev > z_threshold``.

    Returns columns: ``id``, ``date``, ``counterparty``, ``category``,
    ``amount``, ``typical``, ``vs_typical`` (``amount / typical``),
    ``zscore``, ``n_history``.
    """
    # SQLite's stdlib build doesn't ship SQRT (the math extension isn't
    # compiled in by default). Pull AVG / mean-of-squares / count from
    # SQL, do the z-score arithmetic + threshold filter in pandas.
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    sql = f"""
        WITH cp_stats AS (
            SELECT counterparty_normalized,
                   AVG(ABS(betrag_cents)) AS mean_cents,
                   AVG(ABS(betrag_cents) * ABS(betrag_cents)) AS msq_cents,
                   COUNT(*) AS n
            FROM expenses
            WHERE counterparty_normalized IS NOT NULL
              AND counterparty_normalized <> ''
              AND is_income = 0
            GROUP BY counterparty_normalized
            HAVING n >= ?
        )
        SELECT
            e.id,
            e.buchungsdatum AS date,
            e.counterparty AS counterparty,
            COALESCE(c.name, '(unkategorisiert)') AS category,
            ABS(e.betrag_cents) / 100.0 AS amount,
            s.mean_cents / 100.0 AS typical,
            s.mean_cents AS _mean_cents,
            s.msq_cents AS _msq_cents,
            ABS(e.betrag_cents) AS _abs_cents,
            s.n AS n_history
        FROM expenses e
        JOIN cp_stats s ON s.counterparty_normalized = e.counterparty_normalized
        {JOIN_LATEST_LABEL}
        WHERE e.is_income = 0
          AND s.msq_cents - s.mean_cents * s.mean_cents > 0
          {extra}
    """
    full_params = [min_history] + params
    rows = conn.execute(sql, full_params).fetchall()
    if not rows:
        return pd.DataFrame(
            columns=["id", "date", "counterparty", "category",
                     "amount", "typical", "vs_typical", "zscore", "n_history"]
        )
    df = pd.DataFrame([dict(r) for r in rows])
    # Population variance: msq - mean^2 (> 0 by SQL guard above).
    var = (df["_msq_cents"] - df["_mean_cents"] ** 2).clip(lower=0)
    std = var ** 0.5
    df["zscore"] = (df["_abs_cents"] - df["_mean_cents"]) / std
    df = df[df["zscore"] > z_threshold].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["id", "date", "counterparty", "category",
                     "amount", "typical", "vs_typical", "zscore", "n_history"]
        )
    df["vs_typical"] = df["amount"] / df["typical"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["date", "zscore"], ascending=[False, False])
    df = df.head(limit).reset_index(drop=True)
    return df[
        ["id", "date", "counterparty", "category",
         "amount", "typical", "vs_typical", "zscore", "n_history"]
    ]


def weekly_by_category(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    """One row per (ISO-ish week, category) with the spend total.

    Less fine-grained than ``daily_by_category`` -- the daily bars get
    unreadably narrow over multi-month ranges, which is the common
    Dashboard case. Week label is ``YYYY-Www`` using
    ``strftime('%Y-W%W')`` (Monday-based week number, zero-padded),
    sorts correctly as a string.
    """
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    sql = f"""
        SELECT strftime('%Y-W%W', e.buchungsdatum) AS w,
               COALESCE(c.name, '(unkategorisiert)') AS name,
               COALESCE(c.color, '#bbbbbb') AS color,
               SUM(ABS(e.betrag_cents)) / 100.0 AS amount
        FROM expenses e
        {JOIN_LATEST_LABEL}
        WHERE e.is_income = 0 {extra}
        GROUP BY w, name, color
        ORDER BY w, name
    """
    rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["w", "name", "color", "amount"])
    return df


def daily_by_category(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
) -> pd.DataFrame:
    """One row per (date, category) with the spend total. Used by the
    Dashboard's stacked daily-spend bar chart.

    Uncategorized rows fall under "(unkategorisiert)" with a neutral grey.
    """
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    sql = f"""
        SELECT e.buchungsdatum AS d,
               COALESCE(c.name, '(unkategorisiert)') AS name,
               COALESCE(c.color, '#bbbbbb') AS color,
               SUM(ABS(e.betrag_cents)) / 100.0 AS amount
        FROM expenses e
        {JOIN_LATEST_LABEL}
        WHERE e.is_income = 0 {extra}
        GROUP BY d, name, color
        ORDER BY d, name
    """
    rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["d", "name", "color", "amount"])
    if not df.empty:
        df["d"] = pd.to_datetime(df["d"]).dt.date
    return df
