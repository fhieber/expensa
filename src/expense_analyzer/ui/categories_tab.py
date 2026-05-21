"""Categories tab: list / edit / add / delete categories.

Each row is an inline editor: text inputs auto-save on blur/Enter, the
color picker auto-saves on pick. Delete is a two-step prompt when the
category has labels referencing it.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from expense_analyzer.config import packaged_default_categories
from expense_analyzer.storage.admin import category_removal_impact, remove_category
from expense_analyzer.storage.categories import (
    import_categories_from_yaml,
    upsert_category,
)
from expense_analyzer.storage.stats import category_stats, uncategorized_stat
from expense_analyzer.ui._shared import get_conn
from expense_analyzer.utils.colors import random_hex_color


def _save_cat_name(cat_id: int) -> None:
    conn = get_conn()
    new_name = (st.session_state.get(f"cat_{cat_id}_name") or "").strip()
    if not new_name:
        st.session_state[f"cat_{cat_id}_error"] = "name cannot be empty"
        return
    try:
        conn.execute("UPDATE categories SET name=? WHERE id=?", (new_name, cat_id))
        st.session_state.pop(f"cat_{cat_id}_error", None)
    except sqlite3.IntegrityError as e:
        st.session_state[f"cat_{cat_id}_error"] = f"name conflict: {e}"


def _save_cat_desc(cat_id: int) -> None:
    conn = get_conn()
    new_desc = (st.session_state.get(f"cat_{cat_id}_desc") or "").strip()
    conn.execute("UPDATE categories SET description=? WHERE id=?", (new_desc, cat_id))


def _save_cat_color(cat_id: int) -> None:
    conn = get_conn()
    new_color = st.session_state.get(f"cat_{cat_id}_color") or "#888888"
    conn.execute("UPDATE categories SET color=? WHERE id=?", (new_color, cat_id))


def render() -> None:
    conn = get_conn()
    st.header("Categories")

    stats = category_stats(conn)

    # --- Empty-state bootstrap ------------------------------------------
    if not stats:
        st.info(
            "No categories yet. Install the bundled German defaults to get "
            "started, or add your own below."
        )
        if st.button("Install default German categories", type="primary"):
            n = import_categories_from_yaml(conn, packaged_default_categories())
            st.success(f"installed {n} categories")
            st.rerun()

    # --- Existing categories table --------------------------------------
    if stats:
        st.caption(
            "Edit any cell to save immediately. Name and description commit on "
            "blur/Enter; color commits when you pick a new one. Click ✕ to "
            "delete (cascade prompt if labels reference it)."
        )

        widths = [2, 3, 1, 1, 1, 1.2, 0.5]
        h = st.columns(widths)
        h[0].markdown("**Name**")
        h[1].markdown("**Description**")
        h[2].markdown("**Color**")
        h[3].markdown("**# Records**")
        h[4].markdown("**Abs total €**")
        h[5].markdown("**Last seen**")
        h[6].markdown("")

        for s in stats:
            row = st.columns(widths)
            with row[0]:
                st.text_input(
                    "name",
                    value=s.name,
                    key=f"cat_{s.id}_name",
                    label_visibility="collapsed",
                    on_change=_save_cat_name,
                    args=(s.id,),
                )
            with row[1]:
                st.text_input(
                    "desc",
                    value=s.description,
                    key=f"cat_{s.id}_desc",
                    label_visibility="collapsed",
                    on_change=_save_cat_desc,
                    args=(s.id,),
                    placeholder="(used as zero-shot hypothesis)",
                )
            with row[2]:
                st.color_picker(
                    "color",
                    value=s.color,
                    key=f"cat_{s.id}_color",
                    label_visibility="collapsed",
                    on_change=_save_cat_color,
                    args=(s.id,),
                )
            row[3].write(s.n_expenses)
            row[4].write(f"{s.abs_total_eur:.2f}")
            row[5].write(s.last_seen or "—")
            with row[6]:
                if st.button("✕", key=f"cat_{s.id}_del", help=f"Delete {s.name!r}"):
                    impact = category_removal_impact(conn, s.name)
                    if impact.n_labels == 0:
                        remove_category(conn, s.name)
                        st.rerun()
                    else:
                        st.session_state[f"cat_{s.id}_confirm_delete"] = True

            err = st.session_state.get(f"cat_{s.id}_error")
            if err:
                st.error(f"`{s.name}` — {err}")

            if st.session_state.get(f"cat_{s.id}_confirm_delete"):
                impact = category_removal_impact(conn, s.name)
                with st.container(border=True):
                    st.warning(
                        f"Deleting **{s.name}** will cascade-delete "
                        f"{impact.n_labels} label(s). Continue?"
                    )
                    cc = st.columns([1, 1, 6])
                    if cc[0].button("Yes, delete", key=f"cat_{s.id}_del_yes",
                                    type="secondary"):
                        remove_category(conn, s.name)
                        st.session_state.pop(f"cat_{s.id}_confirm_delete", None)
                        st.rerun()
                    if cc[1].button("Cancel", key=f"cat_{s.id}_del_no"):
                        st.session_state.pop(f"cat_{s.id}_confirm_delete", None)
                        st.rerun()

        # Uncategorized note
        uncat = uncategorized_stat(conn)
        if uncat.n_expenses > 0:
            st.info(
                f"**{uncat.n_expenses}** record(s) currently have no category "
                f"(total |€| {uncat.abs_total_eur:.2f})."
            )

    # --- Add a new category --------------------------------------------
    st.divider()
    st.markdown("**Add a new category**")
    if "new_cat_color" not in st.session_state:
        st.session_state.new_cat_color = random_hex_color()

    add_widths = [2, 3, 1, 1]
    add_cols = st.columns(add_widths)
    with add_cols[0]:
        st.text_input(
            "new name", key="new_cat_name", label_visibility="collapsed",
            placeholder="Name (e.g. Lebensmittel)",
        )
    with add_cols[1]:
        st.text_input(
            "new desc", key="new_cat_desc", label_visibility="collapsed",
            placeholder="Description (optional; used as zero-shot hypothesis)",
        )
    with add_cols[2]:
        st.color_picker(
            "new color", key="new_cat_color", label_visibility="collapsed",
        )
    with add_cols[3]:
        if st.button("➕ Add", type="primary"):
            new_name = (st.session_state.get("new_cat_name") or "").strip()
            new_desc = (st.session_state.get("new_cat_desc") or "").strip()
            new_color = st.session_state.get("new_cat_color") or random_hex_color()
            if not new_name:
                st.error("name is required")
            else:
                try:
                    upsert_category(conn, new_name, new_desc, new_color)
                    for k in ("new_cat_name", "new_cat_desc", "new_cat_color"):
                        st.session_state.pop(k, None)
                    st.session_state.new_cat_color = random_hex_color()
                    st.rerun()
                except sqlite3.IntegrityError as e:
                    st.error(f"could not save: {e}")
