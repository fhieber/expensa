"""Time-based features and recurrence proxies.

Two surfaces:

1. **Per-row helpers** -- used by tests and the rare caller that wants a
   single record's feature value. Each issues one SQL query.

2. **Bulk computation** -- :func:`compute_temporal_features_bulk` runs a
   *single* SQL pass with window functions to compute every recurrence /
   counterparty-zscore feature for every row in one shot. This is what
   :mod:`expensa.features.pipeline` calls during training and
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
WITH global_amt AS (
    -- Running mean / mean-of-squares / count of ABS(betrag_cents) over all
    -- STRICTLY PRIOR rows (any vendor), ordered chronologically. Gives a
    -- global amount z-score that exists from the 3rd row onward -- a
    -- backstop for `amount_zscore_within_cp`, which is NULL until a vendor
    -- has >=2 prior charges (so first-time / rare vendors have no signal).
    -- Leak-free: only prior rows feed each row's statistics.
    SELECT
        e.id,
        ABS(e.betrag_cents) AS g_abs,
        AVG(ABS(e.betrag_cents)) OVER w_g AS g_mean,
        AVG(CAST(ABS(e.betrag_cents) AS REAL) * ABS(e.betrag_cents))
            OVER w_g AS g_msq,
        COUNT(*) OVER w_g AS g_n
    FROM expenses e
    WINDOW w_g AS (
        ORDER BY e.buchungsdatum, e.id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),
same_cp AS (
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
-- recurring_months_12: how many of the trailing 12 calendar months had a
--   similar-amount (±10%) charge to the same vendor -- a strength signal
--   (12 = every month, a hard subscription; 3 = barely qualifies).
-- recurring_exact: 1 iff every prior similar-amount charge was the EXACT
--   same cents (fixed subscription vs noisy variable charge).
recurring AS (
    SELECT
        e.id,
        COUNT(DISTINCT strftime('%Y-%m', p.buchungsdatum)) AS n_months,
        COUNT(DISTINCT CASE
            WHEN julianday(e.buchungsdatum) - julianday(p.buchungsdatum) <= 365
            THEN strftime('%Y-%m', p.buchungsdatum) END) AS n_months_12,
        SUM(CASE WHEN ABS(p.betrag_cents) = ABS(e.betrag_cents) THEN 1 ELSE 0 END) AS n_exact,
        COUNT(*) AS n_similar
    FROM expenses e
    JOIN expenses p
        ON p.counterparty_normalized = e.counterparty_normalized
        AND p.id <> e.id
        AND p.buchungsdatum < e.buchungsdatum
        AND ABS(ABS(p.betrag_cents) - ABS(e.betrag_cents)) <= ABS(e.betrag_cents) * 0.10
    WHERE e.counterparty_normalized IS NOT NULL AND e.counterparty_normalized <> ''
      AND ABS(e.betrag_cents) > 0
    GROUP BY e.id
),
-- iban_count_before: number of prior rows (by date) sharing this row's
-- IBAN. Transaction-frequency only -- NOT label-conditioned -- so it is
-- leak-free under cross-validation. Bridges merchants that vary their
-- counterparty name but keep a stable IBAN (REWE / REWE MARKT / ...).
iban_counts AS (
    SELECT
        e.id,
        COUNT(*) AS n_iban_before
    FROM expenses e
    JOIN expenses p
        ON p.iban = e.iban
        AND p.id <> e.id
        AND p.buchungsdatum < e.buchungsdatum
    WHERE e.iban IS NOT NULL AND e.iban <> ''
    GROUP BY e.id
),
-- glaeubiger_count_before: number of prior rows (by date) sharing this
-- row's SEPA creditor id (Gläubiger-ID). The creditor id is a stable,
-- globally-unique merchant identifier that survives BOTH name and IBAN
-- variation, so it catches recurring direct-debit merchants the other
-- two keys miss. Leak-free (frequency only).
glaeubiger_counts AS (
    SELECT
        e.id,
        COUNT(*) AS n_gid_before
    FROM expenses e
    JOIN expenses p
        ON p.glaeubiger_id = e.glaeubiger_id
        AND p.id <> e.id
        AND p.buchungsdatum < e.buchungsdatum
    WHERE e.glaeubiger_id IS NOT NULL AND e.glaeubiger_id <> ''
    GROUP BY e.id
),
-- recurring_stable_key: like `recurring` above but partitioned on the
-- most STABLE merchant key available -- the Gläubiger-ID when present,
-- else the IBAN -- instead of the (drifting) counterparty name. Catches
-- subscriptions whose display name changes between exports but whose
-- creditor id / IBAN is constant. ``stable_key`` is built per row and a
-- self-join matches rows sharing it.
stable_keyed AS (
    SELECT e.id,
           CASE WHEN e.glaeubiger_id IS NOT NULL AND e.glaeubiger_id <> ''
                THEN 'g:' || e.glaeubiger_id
                WHEN e.iban IS NOT NULL AND e.iban <> ''
                THEN 'i:' || e.iban
                ELSE NULL END AS sk,
           e.buchungsdatum AS bd,
           ABS(e.betrag_cents) AS abs_cents
    FROM expenses e
),
recurring_stable AS (
    SELECT
        e.id,
        COUNT(DISTINCT strftime('%Y-%m', p.bd)) AS n_months_stable
    FROM stable_keyed e
    JOIN stable_keyed p
        ON p.sk = e.sk
        AND p.id <> e.id
        AND p.bd < e.bd
        AND ABS(p.abs_cents - e.abs_cents) <= e.abs_cents * 0.10
    WHERE e.sk IS NOT NULL AND e.abs_cents > 0
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
    CASE
        WHEN g.g_n IS NULL OR g.g_n < 2 THEN NULL
        WHEN g.g_msq - g.g_mean * g.g_mean <= 0 THEN 0.0
        ELSE (g.g_abs - g.g_mean) / sqrt(g.g_msq - g.g_mean * g.g_mean)
    END AS amount_zscore_global,
    CASE WHEN COALESCE(r.n_months, 0) >= 3 THEN 1 ELSE 0 END AS is_likely_recurring,
    COALESCE(r.n_months_12, 0) AS recurring_months_12,
    CASE WHEN COALESCE(r.n_similar, 0) > 0 AND r.n_exact = r.n_similar
         THEN 1 ELSE 0 END AS recurring_is_exact_amount,
    COALESCE(ic.n_iban_before, 0) AS iban_count_before,
    COALESCE(gc.n_gid_before, 0) AS glaeubiger_count_before,
    CASE WHEN COALESCE(rs.n_months_stable, 0) >= 3 THEN 1 ELSE 0 END
        AS is_recurring_stable_key
FROM same_cp s
LEFT JOIN global_amt g ON g.id = s.id
LEFT JOIN cp_counts c ON c.id = s.id
LEFT JOIN recurring r ON r.id = s.id
LEFT JOIN iban_counts ic ON ic.id = s.id
LEFT JOIN glaeubiger_counts gc ON gc.id = s.id
LEFT JOIN recurring_stable rs ON rs.id = s.id
"""


def compute_temporal_features_bulk(
    conn: sqlite3.Connection, expense_ids: Sequence[int] | None = None
) -> dict[int, dict[str, int | float | None]]:
    """Compute every per-row temporal feature for every (or a subset of)
    expense rows in a single SQL pass.

    Returns ``{expense_id: {feature_name: value}}``. Features:

      * ``days_since_prev_same_cp``    (int | None)
      * ``count_same_cp_30d``          (int)
      * ``count_same_cp_90d``          (int)
      * ``count_same_cp_365d``         (int)
      * ``amount_zscore_within_cp``    (float | None)
      * ``amount_zscore_global``       (float | None)  z vs all prior rows
      * ``is_likely_recurring``        (0 | 1)
      * ``recurring_months_12``        (int)  months of last 12 with a similar charge
      * ``recurring_is_exact_amount``  (0 | 1)  all prior similar charges identical
      * ``iban_count_before``          (int)  prior rows sharing this IBAN
      * ``glaeubiger_count_before``    (int)  prior rows sharing this creditor id
      * ``is_recurring_stable_key``    (0 | 1)  recurrence keyed on Gläubiger-ID/IBAN

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
            "amount_zscore_global": (
                float(r["amount_zscore_global"])
                if r["amount_zscore_global"] is not None
                else None
            ),
            "is_likely_recurring": int(r["is_likely_recurring"] or 0),
            "recurring_months_12": int(r["recurring_months_12"] or 0),
            "recurring_is_exact_amount": int(r["recurring_is_exact_amount"] or 0),
            "iban_count_before": int(r["iban_count_before"] or 0),
            "glaeubiger_count_before": int(r["glaeubiger_count_before"] or 0),
            "is_recurring_stable_key": int(r["is_recurring_stable_key"] or 0),
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
