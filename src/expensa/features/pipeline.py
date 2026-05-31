"""Build the per-expense feature DataFrame fed to the ML pipeline.

The DB stores most features as columns; the rest (temporal recurrence
proxies, embeddings) are materialized here on demand.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd

from expensa.features.embeddings import Embedder, load_embeddings, store_embeddings
from expensa.features.numeric import (
    UMSATZTYP_BUCKETS,
    amount_ends_99,
    cyclical,
    digit_ratio,
    has_cents,
    is_small_verification,
    log_abs_amount,
    text_length,
    token_count,
    umsatztyp_bucket,
)
from expensa.features.temporal import (
    basic_calendar_features,
    compute_temporal_features_bulk,
)

_BASE_COLUMNS = [
    "id",
    "buchungsdatum",
    "betrag_cents",
    "iban",
    "glaeubiger_id",
    "counterparty",
    "counterparty_normalized",
    "verwendungszweck_normalized",
    "combined_text",
    "is_income",
    "is_round",
    "amount_bucket",
    "umsatztyp",
    "iban_country",
    "iban_is_foreign",
    "iban_is_known_self",
    "has_glaeubiger_id",
    "mandatsreferenz_present",
]


def base_dataframe(
    conn: sqlite3.Connection, expense_ids: Sequence[int] | None = None
) -> pd.DataFrame:
    """Pull stored columns into a DataFrame indexed by expense id."""
    cols = ", ".join(_BASE_COLUMNS)
    if expense_ids is not None:
        if not expense_ids:
            return pd.DataFrame(columns=_BASE_COLUMNS).set_index("id")
        ph = ",".join("?" * len(expense_ids))
        rows = conn.execute(f"SELECT {cols} FROM expenses WHERE id IN ({ph})", expense_ids).fetchall()
    else:
        rows = conn.execute(f"SELECT {cols} FROM expenses").fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return pd.DataFrame(columns=_BASE_COLUMNS).set_index("id")
    df["buchungsdatum"] = pd.to_datetime(df["buchungsdatum"]).dt.date
    return df.set_index("id")


def _calendar_columns(d: date) -> dict[str, int]:
    return basic_calendar_features(d)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cal = pd.DataFrame([_calendar_columns(d) for d in df["buchungsdatum"]], index=df.index)
    return df.join(cal)


def add_temporal_recurrence(conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    """Attach recurrence + same-vendor count + amount-zscore features.

    Runs as a single SQL pass using window functions, regardless of how
    many rows are in ``df``. Replaces a former N+1 loop that issued
    6 queries per row.
    """
    if df.empty:
        return df
    df = df.copy()
    feats = compute_temporal_features_bulk(conn, list(df.index))
    df["days_since_prev_same_cp"] = [
        feats.get(eid, {}).get("days_since_prev_same_cp") for eid in df.index
    ]
    df["count_same_cp_30d"] = [
        feats.get(eid, {}).get("count_same_cp_30d", 0) for eid in df.index
    ]
    df["count_same_cp_90d"] = [
        feats.get(eid, {}).get("count_same_cp_90d", 0) for eid in df.index
    ]
    df["count_same_cp_365d"] = [
        feats.get(eid, {}).get("count_same_cp_365d", 0) for eid in df.index
    ]
    df["amount_zscore_within_cp"] = [
        feats.get(eid, {}).get("amount_zscore_within_cp") for eid in df.index
    ]
    df["is_likely_recurring"] = [
        int(feats.get(eid, {}).get("is_likely_recurring", 0)) for eid in df.index
    ]
    df["recurring_months_12"] = [
        int(feats.get(eid, {}).get("recurring_months_12", 0)) for eid in df.index
    ]
    df["recurring_is_exact_amount"] = [
        int(feats.get(eid, {}).get("recurring_is_exact_amount", 0)) for eid in df.index
    ]
    df["iban_count_before"] = [
        int(feats.get(eid, {}).get("iban_count_before", 0)) for eid in df.index
    ]
    df["glaeubiger_count_before"] = [
        int(feats.get(eid, {}).get("glaeubiger_count_before", 0)) for eid in df.index
    ]
    df["is_recurring_stable_key"] = [
        int(feats.get(eid, {}).get("is_recurring_stable_key", 0)) for eid in df.index
    ]
    df["amount_zscore_global"] = [
        feats.get(eid, {}).get("amount_zscore_global") for eid in df.index
    ]
    return df


def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["log_abs_amount"] = df["betrag_cents"].apply(
        lambda c: log_abs_amount(Decimal(c) / Decimal(100))
    )
    return df


def add_amount_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap integer-cents amount-shape flags (item 7)."""
    if df.empty:
        return df
    df = df.copy()
    cents = df["betrag_cents"].astype("int64")
    df["has_cents"] = cents.apply(has_cents)
    df["is_small_verification"] = cents.apply(is_small_verification)
    df["amount_ends_99"] = cents.apply(amount_ends_99)
    return df


def add_umsatztyp_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot the folded umsatztyp bucket into a stable set of columns
    (item 4). ``umsatztyp_<bucket>`` is 1 for the matching bucket, 0 else."""
    if df.empty:
        return df
    df = df.copy()
    buckets = df["umsatztyp"].apply(umsatztyp_bucket)
    for b in UMSATZTYP_BUCKETS:
        df[f"umsatztyp_{b}"] = (buckets == b).astype("int64")
    return df


def add_text_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Structural features over the normalised purpose text (item 8)."""
    if df.empty:
        return df
    df = df.copy()
    vz = df["verwendungszweck_normalized"]
    df["vz_length"] = vz.apply(text_length)
    df["vz_token_count"] = vz.apply(token_count)
    df["vz_digit_ratio"] = vz.apply(digit_ratio)
    return df


def add_cyclical_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """(sin, cos) encodings of month / day-of-week / day-of-month so a
    linear model sees the true circular distance (item 2). Requires the
    raw calendar columns from :func:`add_calendar_features`."""
    if df.empty:
        return df
    df = df.copy()
    for col, period in (("month", 12.0), ("day_of_week", 7.0), ("day_of_month", 31.0)):
        if col not in df.columns:
            continue
        pairs = df[col].apply(lambda v, p=period: cyclical(float(v), p))
        df[f"{col}_sin"] = [s for s, _ in pairs]
        df[f"{col}_cos"] = [c for _, c in pairs]
    return df


def ensure_embeddings(
    conn: sqlite3.Connection, embedder: Embedder, df: pd.DataFrame
) -> int:
    """Make sure every row in `df` has an embedding stored. Returns number added."""
    if df.empty:
        return 0
    pairs = [(eid, str(t or "")) for eid, t in df["combined_text"].items()]
    return store_embeddings(conn, embedder, pairs)


def attach_embeddings(
    conn: sqlite3.Connection, embedder: Embedder, df: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (df, embedding_matrix) aligned by row order."""
    ensure_embeddings(conn, embedder, df)
    ids, matrix = load_embeddings(conn, embedder.model_name, list(df.index))
    # Reorder matrix to df order:
    id_to_pos = {eid: i for i, eid in enumerate(ids)}
    order = [id_to_pos[eid] for eid in df.index if eid in id_to_pos]
    if len(order) != len(df):
        # Some rows missing embeddings (shouldn't happen after ensure). Be defensive.
        keep = [eid for eid in df.index if eid in id_to_pos]
        df = df.loc[keep]
        order = [id_to_pos[eid] for eid in keep]
    return df, matrix[order]


def build_full_features(
    conn: sqlite3.Connection,
    embedder: Embedder | None = None,
    expense_ids: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Top-level orchestrator. If `embedder` is None, embeddings are skipped
    (useful for cheap reports / visualizations)."""
    df = base_dataframe(conn, expense_ids)
    df = add_calendar_features(df)
    df = add_cyclical_calendar_features(df)
    df = add_temporal_recurrence(conn, df)
    df = add_log_amount(df)
    df = add_amount_pattern_features(df)
    df = add_umsatztyp_features(df)
    df = add_text_shape_features(df)
    if embedder is None:
        return df, None
    df, matrix = attach_embeddings(conn, embedder, df)
    return df, matrix
