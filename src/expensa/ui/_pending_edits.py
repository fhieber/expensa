"""Pending-edit state for the Data tab.

The Data tab maintains three orthogonal pending-edit stashes that overlay
onto the SQL-fetched rows before they're shown in AgGrid:

  * ``user_typed``: user-typed category cell changes (eid -> cat_name or
    None for an explicit clear). Saved as ``source='user'``.
  * ``autolabel``:  auto-label predictions awaiting confirmation
    (eid -> {cat, conf, stage}). Saved as ``source='model'``.
  * ``promote``:    set of eids whose current category should be
    re-stamped as ``source='user'`` on Save.

This module is a thin facade over ``st.session_state`` -- it owns the
key names and the merge / clear / read semantics so the Data tab body
doesn't have to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import streamlit as st

# Session-state keys, defined once so a typo doesn't silently create a
# new stash.
_USER_TYPED_KEY = "data_user_typed_edits"
_AUTOLABEL_KEY = "data_autolabel_stage"
_PROMOTE_KEY = "data_promote_stage"


class AutolabelEntry(TypedDict):
    cat: str         # category name (may be empty)
    conf: str        # formatted "0.87" string ready for the grid
    stage: str       # cascade stage tag (e.g. "vendor_exact_match")


@dataclass
class PendingEdits:
    """Snapshot of the three pending-edit stashes. Created at the top of
    each render via :func:`load`; the Data tab mutates the dict copies
    and writes them back via :func:`save`."""

    user_typed: dict[int, str | None] = field(default_factory=dict)
    autolabel: dict[int, AutolabelEntry] = field(default_factory=dict)
    promote: set[int] = field(default_factory=set)

    # ---- read accessors ------------------------------------------------

    def has_user_typed(self, eid: int) -> bool:
        return eid in self.user_typed

    def autolabel_for(self, eid: int) -> AutolabelEntry | None:
        return self.autolabel.get(eid)

    def is_promoted(self, eid: int) -> bool:
        return eid in self.promote

    def total_pending(self) -> int:
        return len(self.user_typed) + len(self.autolabel) + len(self.promote)


def load() -> PendingEdits:
    """Read the three stashes out of ``st.session_state`` into mutable
    copies. Mutate the returned object freely; call :func:`save` to
    persist."""
    return PendingEdits(
        user_typed=dict(st.session_state.get(_USER_TYPED_KEY, {})),
        autolabel=dict(st.session_state.get(_AUTOLABEL_KEY, {})),
        promote=set(st.session_state.get(_PROMOTE_KEY, set()) or set()),
    )


def save(p: PendingEdits) -> None:
    st.session_state[_USER_TYPED_KEY] = p.user_typed
    st.session_state[_AUTOLABEL_KEY] = p.autolabel
    st.session_state[_PROMOTE_KEY] = p.promote


def clear_all() -> None:
    """Drop every stash. Used by Save (after commit) and Revert."""
    for k in (_USER_TYPED_KEY, _AUTOLABEL_KEY, _PROMOTE_KEY):
        st.session_state.pop(k, None)


def merge_user_typed(new: dict[int, str | None]) -> None:
    """Update the user-typed stash with ``new`` entries, keeping any
    pre-existing entries that aren't in ``new``. Used by the bulk-edit
    propagation and to preserve manual edits across an action that
    bumps the AgGrid key."""
    if not new:
        return
    existing = dict(st.session_state.get(_USER_TYPED_KEY, {}))
    existing.update(new)
    st.session_state[_USER_TYPED_KEY] = existing
