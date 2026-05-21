"""Shared SQL fragments used by query builders across the codebase.

These constants exist so callers don't re-spell the "most-recent label
per expense" join in six different files. The actual definition lives
as a SQL VIEW in ``schema.sql`` (``latest_label``); these strings are
the join clauses that consume it.
"""

from __future__ import annotations

# Join the most-recent label (and the matching category row) onto an
# ``expenses`` alias ``e``. Pair with ``SELECT ... FROM expenses e ...``
# and refer to ``ll.category_id`` / ``ll.label_source`` /
# ``ll.confidence`` / ``c.name`` / ``c.color``.
JOIN_LATEST_LABEL = """
LEFT JOIN latest_label ll ON ll.expense_id = e.id
LEFT JOIN categories c ON c.id = ll.category_id
"""
