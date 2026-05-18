"""SQL-driven views over the expenses + labels tables for visualization.

Each function returns a :class:`pandas.DataFrame` ready to feed a chart.
We use the most-recent label per expense (``user`` or ``model``) so that
predicted categories show up in dashboards even before the user reviews them.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd

_LATEST_LABEL_CTE = """
WITH latest_label AS (
    SELECT l.expense_id, l.category_id
    FROM labels l
    JOIN (
        SELECT expense_id, MAX(id) AS max_id
        FROM labels GROUP BY expense_id
    ) m ON l.id = m.max_id
)
"""


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
    sql = (
        _LATEST_LABEL_CTE
        + f"""
        SELECT c.name, c.color, SUM(ABS(e.betrag_cents)) / 100.0 AS amount
        FROM expenses e
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c ON c.id = ll.category_id
        WHERE 1=1 {income_clause} {extra}
        GROUP BY COALESCE(c.id, -1), c.name, c.color
        ORDER BY amount DESC
        """
    )
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
) -> pd.DataFrame:
    """Monthly sums per category, signed. Useful for stacked / line charts."""
    extra, params = _date_filter_clause("e.buchungsdatum", since, until)
    sql = (
        _LATEST_LABEL_CTE
        + f"""
        SELECT strftime('%Y-%m', e.buchungsdatum) AS ym,
               COALESCE(c.name, '(unkategorisiert)') AS name,
               COALESCE(c.color, '#bbbbbb') AS color,
               SUM(e.betrag_cents) / 100.0 AS amount
        FROM expenses e
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c ON c.id = ll.category_id
        WHERE 1=1 {extra}
        GROUP BY ym, name
        ORDER BY ym
        """
    )
    rows = conn.execute(sql, params).fetchall()
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
    sql = (
        _LATEST_LABEL_CTE
        + f"""
        SELECT ABS(e.betrag_cents) / 100.0 AS amount,
               COALESCE(c.name, '(unkategorisiert)') AS name
        FROM expenses e
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c ON c.id = ll.category_id
        WHERE 1=1 {income_clause} {extra}
        """
    )
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


def recurring_subscriptions(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    min_months: int = 3,
) -> pd.DataFrame:
    """Vendors that look like recurring subscriptions: appear in
    ``min_months`` or more distinct calendar months within the visible
    date range. Returns one row per vendor sorted by annualised cost.

    Columns: ``name``, ``last_seen``, ``typical_amount``, ``n_months``,
    ``annualised``.

    The schema has an ``is_likely_recurring`` column, but the ingestion
    pipeline doesn't populate it (the per-row heuristic is recomputed
    on demand in ``features/temporal.py``). So we re-detect here via a
    pure SQL aggregate -- ``HAVING COUNT(DISTINCT ym) >= ?`` -- and use
    the per-month mean as the typical-amount estimate.
    """
    extra, params = _date_filter_clause("buchungsdatum", since, until)
    sql = f"""
        WITH cp_monthly AS (
            SELECT counterparty_normalized,
                   strftime('%Y-%m', buchungsdatum) AS ym,
                   AVG(ABS(betrag_cents)) AS avg_cents
            FROM expenses
            WHERE counterparty_normalized IS NOT NULL
              AND counterparty_normalized <> ''
              AND is_income = 0
              {extra}
            GROUP BY counterparty_normalized, ym
        ),
        cp_stats AS (
            SELECT counterparty_normalized,
                   COUNT(DISTINCT ym) AS n_months,
                   AVG(avg_cents) AS typical_cents
            FROM cp_monthly
            GROUP BY counterparty_normalized
            HAVING n_months >= ?
        )
        SELECT
            e.counterparty AS name,
            MAX(e.buchungsdatum) AS last_seen,
            s.typical_cents / 100.0 AS typical_amount,
            s.n_months,
            (s.typical_cents * 12) / 100.0 AS annualised
        FROM expenses e
        JOIN cp_stats s ON s.counterparty_normalized = e.counterparty_normalized
        WHERE e.is_income = 0 {extra}
        GROUP BY e.counterparty_normalized, s.typical_cents, s.n_months
        ORDER BY annualised DESC
    """
    full_params = params + [min_months] + params
    rows = conn.execute(sql, full_params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return pd.DataFrame(
            columns=["name", "last_seen", "typical_amount",
                     "n_months", "annualised"]
        )
    df["last_seen"] = pd.to_datetime(df["last_seen"]).dt.date
    return df


def monthly_income_vs_expense(
    conn: sqlite3.Connection,
    since: date | None = None,
    until: date | None = None,
    exclude_internal: bool = True,
) -> pd.DataFrame:
    """Per-month income vs expense totals.

    Returns: ``ym``, ``income``, ``expenses`` (both positive), ``net``,
    ``savings_rate`` ((income − expenses) / income).

    ``iban_is_known_self`` rows are excluded by default so that money
    moved between your own accounts isn't counted as either income or
    expense (which would inflate both sides and skew the savings rate).
    """
    extra, params = _date_filter_clause("buchungsdatum", since, until)
    internal = (
        " AND COALESCE(iban_is_known_self, 0) = 0" if exclude_internal else ""
    )
    sql = f"""
        SELECT
            strftime('%Y-%m', buchungsdatum) AS ym,
            SUM(CASE WHEN is_income = 1 THEN betrag_cents ELSE 0 END) / 100.0
                AS income,
            SUM(CASE WHEN is_income = 0 THEN ABS(betrag_cents) ELSE 0 END) / 100.0
                AS expenses
        FROM expenses
        WHERE 1=1 {extra} {internal}
        GROUP BY ym
        ORDER BY ym
    """
    rows = conn.execute(sql, params).fetchall()
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
    sql = (
        _LATEST_LABEL_CTE
        + f"""
        , cp_stats AS (
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
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c ON c.id = ll.category_id
        WHERE e.is_income = 0
          AND s.msq_cents - s.mean_cents * s.mean_cents > 0
          {extra}
    """
    )
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
    sql = (
        _LATEST_LABEL_CTE
        + f"""
        SELECT e.buchungsdatum AS d,
               COALESCE(c.name, '(unkategorisiert)') AS name,
               COALESCE(c.color, '#bbbbbb') AS color,
               SUM(ABS(e.betrag_cents)) / 100.0 AS amount
        FROM expenses e
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c ON c.id = ll.category_id
        WHERE e.is_income = 0 {extra}
        GROUP BY d, name, color
        ORDER BY d, name
        """
    )
    rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        df = pd.DataFrame(columns=["d", "name", "color", "amount"])
    if not df.empty:
        df["d"] = pd.to_datetime(df["d"]).dt.date
    return df
