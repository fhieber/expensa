"""Time-based features and recurrence proxies.

Two surfaces:

1. **Per-row helpers** -- used by tests and the rare caller that wants a
   single record's feature value. Each issues one SQL query.

2. **Bulk computation** -- :func:`compute_temporal_features_bulk` runs a
   *single* SQL pass with window functions to compute every recurrence /
   counterparty-zscore feature for every row in one shot. This is what
   :mod:`expense_analyzer.features.pipeline` calls during training and
   prediction. The N+1 per-row loop it used to run was the dominant cost
   on real DBs (5,000 rows * 6 queries = 30,000 roundtrips).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date


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


# ---------------------------------------------------------------------------
# Bulk computation -- one SQL pass for every per-row feature.
# ---------------------------------------------------------------------------


_BULK_FEATURES_SQL = """
WITH same_cp AS (
    -- Per row, capture:
    --   * the most recent prior buchungsdatum to the same vendor
    --     (for days_since_prev_same_cp)
    --   * running mean / mean-of-squares / count of ABS(betrag_cents) of
    --     PRIOR rows to the same vendor (for amount_zscore_within_cp)
    --   * the number of months in the rolling 12-month window where a
    --     same-amount-±10% charge occurred (for is_likely_recurring)
    -- All computed with window functions partitioned by counterparty_normalized
    -- and ordered by buchungsdatum (tie-broken on id so two rows on the same
    -- day get a deterministic order). ROWS BETWEEN UNBOUNDED PRECEDING AND
    -- 1 PRECEDING gives us "strictly prior rows" per the historic per-row
    -- helpers' semantics.
    SELECT
        e.id,
        e.buchungsdatum,
        e.counterparty_normalized AS cpn,
        ABS(e.betrag_cents) AS abs_cents,
        LAG(e.buchungsdatum) OVER w_chrono AS prev_date,
        AVG(ABS(e.betrag_cents)) OVER w_prior AS prior_mean,
        AVG(CAST(ABS(e.betrag_cents) AS REAL) * ABS(e.betrag_cents))
            OVER w_prior AS prior_msq,
        COUNT(*) OVER w_prior AS prior_n
    FROM expenses e
    WINDOW
        w_chrono AS (
            PARTITION BY e.counterparty_normalized
            ORDER BY e.buchungsdatum, e.id
        ),
        w_prior AS (
            PARTITION BY e.counterparty_normalized
            ORDER BY e.buchungsdatum, e.id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
),
-- count_same_cp_<N>d: number of prior rows to the same vendor within
-- the last N calendar days. SQLite has no rolling RANGE-based window
-- over julianday differences, so we self-join the expenses table to
-- itself bounded by `id <> e.id AND prev.buchungsdatum < e.buchungsdatum
-- AND julianday(e.bd) - julianday(prev.bd) <= N`.
cp_counts AS (
    SELECT
        e.id,
        SUM(CASE WHEN julianday(e.buchungsdatum) - julianday(p.buchungsdatum) <= 30
                 THEN 1 ELSE 0 END) AS n30,
        SUM(CASE WHEN julianday(e.buchungsdatum) - julianday(p.buchungsdatum) <= 90
                 THEN 1 ELSE 0 END) AS n90,
        SUM(CASE WHEN julianday(e.buchungsdatum) - julianday(p.buchungsdatum) <= 365
                 THEN 1 ELSE 0 END) AS n365
    FROM expenses e
    LEFT JOIN expenses p
        ON p.counterparty_normalized = e.counterparty_normalized
        AND p.id <> e.id
        AND p.buchungsdatum < e.buchungsdatum
        AND julianday(e.buchungsdatum) - julianday(p.buchungsdatum) <= 365
    WHERE e.counterparty_normalized IS NOT NULL AND e.counterparty_normalized <> ''
    GROUP BY e.id
),
-- is_likely_recurring: 1 iff the vendor has charged a similar amount
-- (within ±10%) in ≥3 distinct prior calendar months.
recurring AS (
    SELECT
        e.id,
        COUNT(DISTINCT strftime('%Y-%m', p.buchungsdatum)) AS n_months
    FROM expenses e
    JOIN expenses p
        ON p.counterparty_normalized = e.counterparty_normalized
        AND p.id <> e.id
        AND p.buchungsdatum < e.buchungsdatum
        AND ABS(ABS(p.betrag_cents) - ABS(e.betrag_cents)) <= ABS(e.betrag_cents) * 0.10
    WHERE e.counterparty_normalized IS NOT NULL AND e.counterparty_normalized <> ''
      AND ABS(e.betrag_cents) > 0
    GROUP BY e.id
)
SELECT
    s.id,
    CASE WHEN s.cpn IS NULL OR s.cpn = '' OR s.prev_date IS NULL
         THEN NULL
         ELSE CAST(julianday(s.buchungsdatum) - julianday(s.prev_date) AS INTEGER)
    END AS days_since_prev_same_cp,
    COALESCE(c.n30, 0)  AS count_same_cp_30d,
    COALESCE(c.n90, 0)  AS count_same_cp_90d,
    COALESCE(c.n365, 0) AS count_same_cp_365d,
    CASE
        WHEN s.cpn IS NULL OR s.cpn = '' OR s.prior_n IS NULL OR s.prior_n < 2 THEN NULL
        WHEN s.prior_msq - s.prior_mean * s.prior_mean <= 0 THEN 0.0
        ELSE (s.abs_cents - s.prior_mean)
             / sqrt(s.prior_msq - s.prior_mean * s.prior_mean)
    END AS amount_zscore_within_cp,
    CASE WHEN COALESCE(r.n_months, 0) >= 3 THEN 1 ELSE 0 END AS is_likely_recurring
FROM same_cp s
LEFT JOIN cp_counts c ON c.id = s.id
LEFT JOIN recurring r ON r.id = s.id
"""


def compute_temporal_features_bulk(
    conn: sqlite3.Connection, expense_ids: Sequence[int] | None = None
) -> dict[int, dict[str, int | float | None]]:
    """Compute every per-row temporal feature for every (or a subset of)
    expense rows in a single SQL pass.

    Returns ``{expense_id: {feature_name: value}}``. Features:

      * ``days_since_prev_same_cp``  (int | None)
      * ``count_same_cp_30d``        (int)
      * ``count_same_cp_90d``        (int)
      * ``count_same_cp_365d``       (int)
      * ``amount_zscore_within_cp``  (float | None)
      * ``is_likely_recurring``      (0 | 1)

    SQLite's stdlib build doesn't expose ``sqrt`` -- we register it once
    on the connection here. Cheap and idempotent.
    """
    # Register sqrt only once per connection; the stdlib build lacks it.
    try:
        import math

        conn.create_function("sqrt", 1, lambda x: math.sqrt(x) if x is not None else None)
    except sqlite3.NotSupportedError:
        # Already registered with conflicting signature; safe to ignore.
        pass

    rows = conn.execute(_BULK_FEATURES_SQL).fetchall()
    by_id: dict[int, dict[str, int | float | None]] = {
        int(r["id"]): {
            "days_since_prev_same_cp": (
                int(r["days_since_prev_same_cp"])
                if r["days_since_prev_same_cp"] is not None
                else None
            ),
            "count_same_cp_30d":  int(r["count_same_cp_30d"]),
            "count_same_cp_90d":  int(r["count_same_cp_90d"]),
            "count_same_cp_365d": int(r["count_same_cp_365d"]),
            "amount_zscore_within_cp": (
                float(r["amount_zscore_within_cp"])
                if r["amount_zscore_within_cp"] is not None
                else None
            ),
            "is_likely_recurring": int(r["is_likely_recurring"] or 0),
        }
        for r in rows
    }
    if expense_ids is None:
        return by_id
    wanted = {int(i) for i in expense_ids}
    return {eid: feats for eid, feats in by_id.items() if eid in wanted}


# ---------------------------------------------------------------------------
# Per-row helpers (legacy / single-record callers). Each is one query.
# ---------------------------------------------------------------------------


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
