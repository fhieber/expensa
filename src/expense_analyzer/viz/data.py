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
