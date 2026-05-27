"""CRUD for the user's *own* IBANs and the per-expense
``iban_is_known_self`` flag derived from them.

The flag is normally computed at ingest time by
``features.iban.classify_iban``. Mutating ``own_ibans`` from the UI / CLI
without re-ingesting would leave existing rows stale, so every mutation
helper here also re-runs the classifier over every row in ``expenses``.

IBANs are normalised to *upper-case without whitespace* before storage,
so lookups are direct equality. ``schwifty`` validates the input where
it's installed; otherwise a permissive country-code check is used.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class OwnIban:
    iban: str
    label: str | None


@dataclass(frozen=True)
class ReflagReport:
    """How many rows changed when re-running ``iban_is_known_self`` over
    the whole ``expenses`` table after an own-IBAN add / remove."""
    n_now_self: int
    n_was_self: int

    @property
    def n_changed(self) -> int:
        return self.n_now_self + self.n_was_self


def _normalise(iban: str) -> str:
    return (iban or "").replace(" ", "").upper()


def _validate(iban_normalised: str) -> None:
    """Raise ``ValueError`` if the IBAN doesn't look right.

    Schwifty (when available) catches the full IBAN checksum + bank-code
    structure. As a permissive fallback (lib missing or its parser is
    cranky about a country) we require at least two letters + six digits
    so the row is at least vaguely IBAN-shaped.
    """
    if not iban_normalised:
        raise ValueError("IBAN is empty")
    try:
        from schwifty import IBAN  # type: ignore

        try:
            IBAN(iban_normalised)
            return
        except Exception as e:
            raise ValueError(f"invalid IBAN: {e}") from None
    except ImportError:
        pass
    # Permissive fallback.
    if len(iban_normalised) < 8:
        raise ValueError("IBAN too short")
    if not (iban_normalised[:2].isalpha() and iban_normalised[2:4].isdigit()):
        raise ValueError("IBAN must start with 2 country letters + 2 check digits")


# --------------------------------------------------------------------- read

def list_own_ibans(conn: sqlite3.Connection) -> list[OwnIban]:
    rows = conn.execute(
        "SELECT iban, label FROM own_ibans ORDER BY iban"
    ).fetchall()
    return [OwnIban(iban=r["iban"], label=r["label"]) for r in rows]


# --------------------------------------------------------- per-IBAN reflag

def _reflag_one(conn: sqlite3.Connection, iban_normalised: str, value: int) -> int:
    """Set ``iban_is_known_self = value`` on every expense whose IBAN
    matches ``iban_normalised`` (after the same normalisation). Returns
    the number of rows actually changed (i.e. excludes rows that were
    already at ``value``)."""
    cur = conn.execute(
        """
        UPDATE expenses
        SET iban_is_known_self = ?
        WHERE UPPER(REPLACE(COALESCE(iban, ''), ' ', '')) = ?
          AND COALESCE(iban_is_known_self, 0) <> ?
        """,
        (value, iban_normalised, value),
    )
    return cur.rowcount or 0


# -------------------------------------------------------------------- write

def add_own_iban(
    conn: sqlite3.Connection, iban: str, label: str | None = None
) -> ReflagReport:
    """Insert (or upsert the label) of an own IBAN, then retroactively
    mark every matching expense as ``iban_is_known_self = 1``.

    Returns a :class:`ReflagReport` so the caller can show e.g. "flagged
    N transactions as internal transfers". Raises ``ValueError`` on a
    malformed IBAN.
    """
    norm = _normalise(iban)
    _validate(norm)
    lbl = (label or "").strip() or None
    conn.execute(
        """
        INSERT INTO own_ibans(iban, label) VALUES (?, ?)
        ON CONFLICT(iban) DO UPDATE SET label = excluded.label
        """,
        (norm, lbl),
    )
    n_set = _reflag_one(conn, norm, 1)
    return ReflagReport(n_now_self=n_set, n_was_self=0)


def remove_own_iban(conn: sqlite3.Connection, iban: str) -> ReflagReport:
    """Delete an own-IBAN row and retroactively clear
    ``iban_is_known_self`` for every matching expense (since no other
    own-IBAN can match the same value -- ``iban`` is the table's primary
    key, so removal makes the match impossible).
    """
    norm = _normalise(iban)
    if not norm:
        raise ValueError("IBAN is empty")
    cur = conn.execute("DELETE FROM own_ibans WHERE iban = ?", (norm,))
    if (cur.rowcount or 0) == 0:
        return ReflagReport(n_now_self=0, n_was_self=0)
    n_clear = _reflag_one(conn, norm, 0)
    return ReflagReport(n_now_self=0, n_was_self=n_clear)


def update_label(
    conn: sqlite3.Connection, iban: str, label: str | None
) -> bool:
    """Change the friendly label for an existing own-IBAN. Returns True
    if a row was updated."""
    norm = _normalise(iban)
    lbl = (label or "").strip() or None
    cur = conn.execute(
        "UPDATE own_ibans SET label = ? WHERE iban = ?", (lbl, norm)
    )
    return bool(cur.rowcount)


def reflag_all_expenses(conn: sqlite3.Connection) -> ReflagReport:
    """Recompute ``iban_is_known_self`` for every row in ``expenses``
    against the current ``own_ibans`` table. Useful one-shot if the
    flag drifted out of sync (e.g. after a manual SQL edit)."""
    own_set_sql = "SELECT iban FROM own_ibans"
    set_to_one = conn.execute(
        f"""
        UPDATE expenses
        SET iban_is_known_self = 1
        WHERE UPPER(REPLACE(COALESCE(iban, ''), ' ', '')) IN ({own_set_sql})
          AND COALESCE(iban_is_known_self, 0) <> 1
        """
    ).rowcount or 0
    set_to_zero = conn.execute(
        f"""
        UPDATE expenses
        SET iban_is_known_self = 0
        WHERE UPPER(REPLACE(COALESCE(iban, ''), ' ', '')) NOT IN ({own_set_sql})
          AND COALESCE(iban_is_known_self, 0) <> 0
        """
    ).rowcount or 0
    return ReflagReport(n_now_self=set_to_one, n_was_self=set_to_zero)
