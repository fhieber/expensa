"""Time-based features and recurrence proxies.

These run as a SQL pass over the `expenses` table so we can compute them
without dragging full pandas dataframes around for each new ingest.
"""

from __future__ import annotations

import sqlite3
from datetime import date


def _isodate(d: date) -> str:
    return d.isoformat()


def basic_calendar_features(d: date) -> dict[str, int]:
    return {
        "year": d.year,
        "month": d.month,
        "quarter": (d.month - 1) // 3 + 1,
        "week": int(d.strftime("%V")),
        "day_of_month": d.day,
        "day_of_week": d.weekday(),  # 0=Mon
        "is_weekend": int(d.weekday() >= 5),
        "is_month_end": int(d.day >= 25),
    }


def days_since_prev_to_same_counterparty(
    conn: sqlite3.Connection, expense_id: int
) -> int | None:
    """Days between this expense and the most recent prior one to the same
    counterparty_normalized. Returns None if no prior match."""
    row = conn.execute(
        """
        SELECT counterparty_normalized, buchungsdatum
        FROM expenses WHERE id = ?
        """,
        (expense_id,),
    ).fetchone()
    if row is None or not row["counterparty_normalized"]:
        return None
    prior = conn.execute(
        """
        SELECT MAX(buchungsdatum) AS d
        FROM expenses
        WHERE counterparty_normalized = ?
          AND id <> ?
          AND buchungsdatum < ?
        """,
        (row["counterparty_normalized"], expense_id, row["buchungsdatum"]),
    ).fetchone()
    if prior is None or prior["d"] is None:
        return None
    prior_date = date.fromisoformat(str(prior["d"]))
    this_date = date.fromisoformat(str(row["buchungsdatum"]))
    return (this_date - prior_date).days


def count_to_same_counterparty(
    conn: sqlite3.Connection, expense_id: int, days: int
) -> int:
    """Count of prior expenses to the same counterparty within `days` days
    before this one."""
    row = conn.execute(
        """
        SELECT counterparty_normalized, buchungsdatum
        FROM expenses WHERE id = ?
        """,
        (expense_id,),
    ).fetchone()
    if row is None or not row["counterparty_normalized"]:
        return 0
    res = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM expenses
        WHERE counterparty_normalized = ?
          AND id <> ?
          AND buchungsdatum < ?
          AND julianday(?) - julianday(buchungsdatum) <= ?
        """,
        (
            row["counterparty_normalized"],
            expense_id,
            row["buchungsdatum"],
            row["buchungsdatum"],
            days,
        ),
    ).fetchone()
    return int(res["n"])


def amount_zscore_within_counterparty(
    conn: sqlite3.Connection, expense_id: int
) -> float | None:
    """Z-score of this expense's |amount| against the distribution of past
    expenses to the same counterparty. Returns None when n<2."""
    row = conn.execute(
        """
        SELECT counterparty_normalized, betrag_cents, buchungsdatum
        FROM expenses WHERE id = ?
        """,
        (expense_id,),
    ).fetchone()
    if row is None or not row["counterparty_normalized"]:
        return None
    stats = conn.execute(
        """
        SELECT AVG(ABS(betrag_cents)) AS mean,
               AVG(ABS(betrag_cents) * ABS(betrag_cents)) AS msq,
               COUNT(*) AS n
        FROM expenses
        WHERE counterparty_normalized = ?
          AND id <> ?
          AND buchungsdatum < ?
        """,
        (row["counterparty_normalized"], expense_id, row["buchungsdatum"]),
    ).fetchone()
    if stats is None or stats["n"] is None or stats["n"] < 2:
        return None
    mean = float(stats["mean"])
    var = float(stats["msq"]) - mean * mean
    if var <= 0:
        return 0.0
    std = var ** 0.5
    return (abs(row["betrag_cents"]) - mean) / std


def is_likely_recurring(conn: sqlite3.Connection, expense_id: int) -> bool:
    """Heuristic: same counterparty appears in >=3 distinct prior months
    with amount within 10% of this one."""
    row = conn.execute(
        """
        SELECT counterparty_normalized, betrag_cents, buchungsdatum
        FROM expenses WHERE id = ?
        """,
        (expense_id,),
    ).fetchone()
    if row is None or not row["counterparty_normalized"]:
        return False
    cents = abs(row["betrag_cents"])
    if cents == 0:
        return False
    res = conn.execute(
        """
        SELECT COUNT(DISTINCT strftime('%Y-%m', buchungsdatum)) AS m
        FROM expenses
        WHERE counterparty_normalized = ?
          AND id <> ?
          AND buchungsdatum < ?
          AND ABS(ABS(betrag_cents) - ?) <= ? * 0.10
        """,
        (row["counterparty_normalized"], expense_id, row["buchungsdatum"], cents, cents),
    ).fetchone()
    return int(res["m"] or 0) >= 3
