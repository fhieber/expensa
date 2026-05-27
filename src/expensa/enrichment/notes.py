"""User-supplied notes per expense."""

from __future__ import annotations

import sqlite3


def set_note(conn: sqlite3.Connection, expense_id: int, text: str) -> None:
    """Replace the note for an expense. Empty string deletes."""
    if not text.strip():
        conn.execute("DELETE FROM notes WHERE expense_id = ?", (expense_id,))
        return
    conn.execute(
        """
        INSERT INTO notes(expense_id, text) VALUES (?, ?)
        ON CONFLICT(expense_id) DO UPDATE SET
            text=excluded.text, updated_at=CURRENT_TIMESTAMP
        """,
        (expense_id, text.strip()),
    )


def get_note(conn: sqlite3.Connection, expense_id: int) -> str | None:
    r = conn.execute("SELECT text FROM notes WHERE expense_id = ?", (expense_id,)).fetchone()
    return r["text"] if r else None


def delete_note(conn: sqlite3.Connection, expense_id: int) -> None:
    conn.execute("DELETE FROM notes WHERE expense_id = ?", (expense_id,))
