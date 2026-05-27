"""Fuzzy-string features over the labeled vendor index."""

from __future__ import annotations

import sqlite3

from rapidfuzz import fuzz, process


def best_fuzzy_match_known_vendor(
    candidate: str, known_vendors: list[str]
) -> tuple[str | None, int]:
    """Return (best_match, score 0-100). RapidFuzz token_set_ratio is
    robust to word reordering, which matters for noisy verwendungszwecks."""
    if not candidate or not known_vendors:
        return None, 0
    match = process.extractOne(
        candidate, known_vendors, scorer=fuzz.token_set_ratio
    )
    if match is None:
        return None, 0
    name, score, _idx = match
    return name, int(score)


def labeled_vendor_index(conn: sqlite3.Connection) -> list[str]:
    """All distinct counterparty_normalized values that have at least one
    user-supplied label. Used as the fuzzy index."""
    rows = conn.execute(
        """
        SELECT DISTINCT e.counterparty_normalized
        FROM expenses e
        JOIN labels l ON l.expense_id = e.id
        WHERE l.source = 'user'
          AND e.counterparty_normalized IS NOT NULL
          AND e.counterparty_normalized <> ''
        """
    ).fetchall()
    return [r["counterparty_normalized"] for r in rows]
