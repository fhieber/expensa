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

from expense_analyzer.features.embeddings import Embedder, load_embeddings, store_embeddings
from expense_analyzer.features.numeric import log_abs_amount
from expense_analyzer.features.temporal import (
    amount_zscore_within_counterparty,
    basic_calendar_features,
    count_to_same_counterparty,
    days_since_prev_to_same_counterparty,
    is_likely_recurring,
)

_BASE_COLUMNS = [
    "id",
    "buchungsdatum",
    "betrag_cents",
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
    "cluster_id",
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
    if df.empty:
        return df
    df = df.copy()
    df["days_since_prev_same_cp"] = [
        days_since_prev_to_same_counterparty(conn, eid) for eid in df.index
    ]
    df["count_same_cp_30d"] = [count_to_same_counterparty(conn, eid, 30) for eid in df.index]
    df["count_same_cp_90d"] = [count_to_same_counterparty(conn, eid, 90) for eid in df.index]
    df["count_same_cp_365d"] = [count_to_same_counterparty(conn, eid, 365) for eid in df.index]
    df["amount_zscore_within_cp"] = [
        amount_zscore_within_counterparty(conn, eid) for eid in df.index
    ]
    df["is_likely_recurring"] = [int(is_likely_recurring(conn, eid)) for eid in df.index]
    return df


def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["log_abs_amount"] = df["betrag_cents"].apply(
        lambda c: log_abs_amount(Decimal(c) / Decimal(100))
    )
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
    df = add_temporal_recurrence(conn, df)
    df = add_log_amount(df)
    if embedder is None:
        return df, None
    df, matrix = attach_embeddings(conn, embedder, df)
    return df, matrix
