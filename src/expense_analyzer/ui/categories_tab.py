"""Categories tab: list / edit / add / delete categories.

Each row is an inline editor: text inputs auto-save on blur/Enter, the
color picker auto-saves on pick. Delete is a two-step prompt when the
category has labels referencing it.

**Cross-account safety.** Every per-row widget key embeds the active
account id at render time (``cat_{account_id}_{cat_id}_name`` etc.),
and every ``on_change`` handler receives that same account id as a
bound arg and refuses to write if the active account has changed since.

The race this guards against:
  1. User edits a category text input on Account A (no commit yet).
  2. User clicks the account picker, switching to Account B.
  3. Streamlit reruns. ``on_change`` callbacks fire in widget-declaration
     order. The picker is declared above the tabs, so its ``on_change``
     runs first -- flipping the active account and clearing the cached
     DB connection.
  4. The text input's ``on_change`` then runs. Without this guard,
     ``get_conn()`` would now return Account B's connection and the
     ``UPDATE WHERE id=?`` would corrupt B's same-id category with A's
     intended value.

The fix is both belt and braces:
  * Different widget keys per account (so cross-account session_state
    can't collide either; ``clear_tab_state()`` would also pop them,
    but the unique keys make any leftover handler firing harmless).
  * Active-account assertion inside the handler. If the assertion
    trips we drop a one-shot toast on the next render via session_state
    so the user knows something was suppressed.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from expense_analyzer.config import packaged_default_categories
from expense_analyzer.ml import embedding_viz_cache
from expense_analyzer.ml.embedding_viz import project_labeled_embeddings
from expense_analyzer.storage.admin import category_removal_impact, remove_category
from expense_analyzer.storage.categories import (
    import_categories_from_yaml,
    list_categories,
    set_category_savings,
    upsert_category,
)
from expense_analyzer.storage.stats import category_stats, uncategorized_stat
from expense_analyzer.ui._components import chart_expander
from expense_analyzer.ui._shared import (
    get_active_account,
    get_config,
    get_conn,
)
from expense_analyzer.utils.colors import random_hex_color
from expense_analyzer.viz import category_embedding_scatter

# session_state key the handlers use to signal "I was suppressed because
# the active account changed mid-edit". `render()` checks this on each
# pass and shows a single toast.
_SUPPRESSED_KEY = "_cat_edit_suppressed_msg"

# session_state slots that hold the latest embedding scatter run.
# Cleared on account switch by clear_tab_state() (prefix matches the
# default `cat_` allow-list).
_CAT_PROJ_KEY = "cat_embedding_projection"
_CAT_PROJ_META_KEY = "cat_embedding_meta"
_CAT_PROJ_SAVED_AT_KEY = "cat_embedding_saved_at"


def _active_account_matches(bound_account_id: str) -> bool:
    """Returns True iff the active account is still the one this widget
    was rendered under. When False the on_change handler must abort to
    avoid writing one account's edit to another account's DB."""
    if get_active_account().id == bound_account_id:
        return True
    st.session_state[_SUPPRESSED_KEY] = (
        "Discarded a pending category edit -- you switched accounts "
        "before it committed. Re-apply it on the right account."
    )
    return False


def _save_cat_name(account_id: str, cat_id: int) -> None:
    if not _active_account_matches(account_id):
        return
    conn = get_conn()
    key = f"cat_{account_id}_{cat_id}_name"
    err_key = f"cat_{account_id}_{cat_id}_error"
    new_name = (st.session_state.get(key) or "").strip()
    if not new_name:
        st.session_state[err_key] = "name cannot be empty"
        return
    try:
        conn.execute("UPDATE categories SET name=? WHERE id=?", (new_name, cat_id))
        st.session_state.pop(err_key, None)
    except sqlite3.IntegrityError as e:
        st.session_state[err_key] = f"name conflict: {e}"


def _save_cat_desc(account_id: str, cat_id: int) -> None:
    if not _active_account_matches(account_id):
        return
    conn = get_conn()
    new_desc = (
        st.session_state.get(f"cat_{account_id}_{cat_id}_desc") or ""
    ).strip()
    conn.execute("UPDATE categories SET description=? WHERE id=?", (new_desc, cat_id))


def _save_cat_color(account_id: str, cat_id: int) -> None:
    if not _active_account_matches(account_id):
        return
    conn = get_conn()
    new_color = (
        st.session_state.get(f"cat_{account_id}_{cat_id}_color") or "#888888"
    )
    conn.execute("UPDATE categories SET color=? WHERE id=?", (new_color, cat_id))


def _save_cat_savings(account_id: str, cat_id: int) -> None:
    if not _active_account_matches(account_id):
        return
    conn = get_conn()
    is_savings = bool(st.session_state.get(f"cat_{account_id}_{cat_id}_savings"))
    set_category_savings(conn, cat_id, is_savings)


def render() -> None:
    conn = get_conn()
    account_id = get_active_account().id
    st.header("Categories")

    # Surface any cross-account-edit suppression from the previous run.
    if (msg := st.session_state.pop(_SUPPRESSED_KEY, None)):
        st.warning(msg)

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
            "blur/Enter; color commits when you pick a new one. Tick **Sparen** "
            "to mark a category as savings — its rows are treated as neutral on "
            "the Dashboard (excluded from income/expenses, drive the "
            "*To savings* tile). Click ✕ to delete (cascade prompt if labels "
            "reference it)."
        )

        widths = [2, 3, 1, 0.7, 1, 1, 1.2, 0.5]
        h = st.columns(widths)
        h[0].markdown("**Name**")
        h[1].markdown("**Description**")
        h[2].markdown("**Color**")
        h[3].markdown("**Sparen?**")
        h[4].markdown("**# Records**")
        h[5].markdown("**Abs total €**")
        h[6].markdown("**Last seen**")
        h[7].markdown("")

        for s in stats:
            # Every widget key embeds the active account id, so that:
            #   (a) cross-account session_state can't collide, and
            #   (b) a stale on_change from a previous account's render
            #       would address a session_state key that no longer
            #       exists (the new render only wired up the current
            #       account's keys), preventing a wrong-DB write.
            name_key  = f"cat_{account_id}_{s.id}_name"
            desc_key  = f"cat_{account_id}_{s.id}_desc"
            color_key = f"cat_{account_id}_{s.id}_color"
            sav_key   = f"cat_{account_id}_{s.id}_savings"
            del_key   = f"cat_{account_id}_{s.id}_del"
            del_y_key = f"cat_{account_id}_{s.id}_del_yes"
            del_n_key = f"cat_{account_id}_{s.id}_del_no"
            confirm_key = f"cat_{account_id}_{s.id}_confirm_delete"
            err_key   = f"cat_{account_id}_{s.id}_error"

            row = st.columns(widths)
            with row[0]:
                st.text_input(
                    "name",
                    value=s.name,
                    key=name_key,
                    label_visibility="collapsed",
                    on_change=_save_cat_name,
                    args=(account_id, s.id),
                )
            with row[1]:
                st.text_input(
                    "desc",
                    value=s.description,
                    key=desc_key,
                    label_visibility="collapsed",
                    on_change=_save_cat_desc,
                    args=(account_id, s.id),
                    placeholder="(used as zero-shot hypothesis)",
                )
            with row[2]:
                st.color_picker(
                    "color",
                    value=s.color,
                    key=color_key,
                    label_visibility="collapsed",
                    on_change=_save_cat_color,
                    args=(account_id, s.id),
                )
            with row[3]:
                st.checkbox(
                    "savings",
                    value=s.is_savings,
                    key=sav_key,
                    label_visibility="collapsed",
                    help="Treat this category as savings (neutral on the Dashboard).",
                    on_change=_save_cat_savings,
                    args=(account_id, s.id),
                )
            row[4].write(s.n_expenses)
            row[5].write(f"{s.abs_total_eur:.2f}")
            row[6].write(s.last_seen or "—")
            with row[7]:
                if st.button("✕", key=del_key, help=f"Delete {s.name!r}"):
                    # Re-verify the active account at click time. Button
                    # bodies run during the script run (after on_change
                    # callbacks), so the active account here matches
                    # what `render()` saw above. But guard explicitly
                    # so a future refactor can't sneak this open again.
                    if get_active_account().id != account_id:
                        st.session_state[_SUPPRESSED_KEY] = (
                            "Discarded a delete -- account changed."
                        )
                        st.rerun()
                    impact = category_removal_impact(conn, s.name)
                    if impact.n_labels == 0:
                        remove_category(conn, s.name)
                        st.rerun()
                    else:
                        st.session_state[confirm_key] = True

            err = st.session_state.get(err_key)
            if err:
                st.error(f"`{s.name}` — {err}")

            if st.session_state.get(confirm_key):
                impact = category_removal_impact(conn, s.name)
                with st.container(border=True):
                    st.warning(
                        f"Deleting **{s.name}** will cascade-delete "
                        f"{impact.n_labels} label(s). Continue?"
                    )
                    cc = st.columns([1, 1, 6])
                    if cc[0].button("Yes, delete", key=del_y_key,
                                    type="secondary"):
                        if get_active_account().id != account_id:
                            st.session_state[_SUPPRESSED_KEY] = (
                                "Discarded a delete -- account changed."
                            )
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                        remove_category(conn, s.name)
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                    if cc[1].button("Cancel", key=del_n_key):
                        st.session_state.pop(confirm_key, None)
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

    # --- Embedding separation plot (bottom of tab) -----------------------
    _render_embedding_separation(conn)


def _render_embedding_separation(conn) -> None:
    """2D scatter of labeled-expense embeddings, coloured by category.

    Lets the user eyeball *before* a Quality-tab cross-validation
    whether their categories form distinguishable clusters in the
    embedding space. Overlapping clusters → the cascade will mix
    those categories up; a sharp gap → they should classify cleanly.

    Results are cached on disk (per account) so a re-render of the
    tab doesn't re-run the projection; the user explicitly clicks
    "Generate" to refresh. Same UX pattern as the Quality tab cache.
    """
    st.divider()
    st.subheader("Embedding separation")
    st.caption(
        "Project every user-labeled expense's embedding to 2D and "
        "colour by category. Cluster overlap here predicts which "
        "categories the cascade will confuse — same diagnosis as the "
        "Quality tab's confusion matrix, but visible **before** you "
        "spend the CV budget. Click a legend entry to hide/show a "
        "category."
    )

    cfg = get_config()
    account = get_active_account()

    # Hydrate from disk once per session (or after account switch).
    if st.session_state.get(_CAT_PROJ_KEY) is None:
        cached = embedding_viz_cache.load(account.data_dir)
        if cached is not None:
            st.session_state[_CAT_PROJ_KEY] = cached.projection
            st.session_state[_CAT_PROJ_META_KEY] = cached.meta
            st.session_state[_CAT_PROJ_SAVED_AT_KEY] = cached.saved_at.isoformat(
                timespec="minutes"
            )

    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    with c1:
        method = st.selectbox(
            "Projection",
            ["pca", "tsne"],
            index=0,
            key="cat_embedding_method",
            help=(
                "**PCA** is deterministic, fast, and preserves global "
                "distance — best for an at-a-glance "
                "*are-my-categories-separable?* read. **t-SNE** often "
                "shows finer sub-clusters but distorts global distances, "
                "and is much slower on big DBs."
            ),
        )
    with c2:
        seed = st.number_input(
            "Seed", min_value=0, value=0, step=1, key="cat_embedding_seed",
        )
    with c3:
        generate = st.button(
            "🎨 Generate embedding visualization",
            type="primary",
            key="cat_embedding_run",
        )

    if generate:
        with st.spinner("Loading embeddings and projecting…"):
            try:
                projection = project_labeled_embeddings(
                    conn,
                    model_name=cfg.embedding_model,
                    method=method,
                    seed=int(seed),
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"projection failed: {e}")
                return
        if projection is None:
            st.info(
                "Not enough labeled data with stored embeddings to "
                "project. Label some expenses in the **Review** tab and "
                "re-run; embeddings are computed on demand."
            )
            return
        from datetime import datetime

        meta = {"method": projection.method, "model_name": cfg.embedding_model, "seed": int(seed)}
        st.session_state[_CAT_PROJ_KEY] = projection
        st.session_state[_CAT_PROJ_META_KEY] = meta
        st.session_state[_CAT_PROJ_SAVED_AT_KEY] = datetime.now().isoformat(timespec="minutes")
        try:
            embedding_viz_cache.save(account.data_dir, projection, meta)
        except Exception as e:  # noqa: BLE001
            st.warning(f"could not persist projection: {e}")
        st.rerun()

    projection = st.session_state.get(_CAT_PROJ_KEY)
    if projection is None:
        return

    saved_at = st.session_state.get(_CAT_PROJ_SAVED_AT_KEY)
    meta = st.session_state.get(_CAT_PROJ_META_KEY) or {}
    if saved_at:
        pretty = saved_at.replace("T", " ")
        cap_bits = [f"📌 **{pretty}**"]
        if meta.get("method"):
            cap_bits.append(f"method: `{meta['method']}`")
        if meta.get("model_name"):
            cap_bits.append(f"model: `{meta['model_name']}`")
        st.caption(" · ".join(cap_bits))

    cats = list_categories(conn)
    id_to_name = {c.id: c.name for c in cats}
    color_map = {c.id: c.color for c in cats}

    fig = category_embedding_scatter(
        xy=projection.xy,
        category_ids=projection.category_ids,
        id_to_name=id_to_name,
        color_map=color_map,
        method=projection.method,
    )
    chart_expander(
        "Category embeddings — 2D projection",
        fig,
        expanded=True,
        key="cat_embedding_chart",
    )
    info_bits: list[str] = []
    if projection.n_categories:
        info_bits.append(f"{projection.n_categories} categor(y/ies)")
    if projection.n_dropped_singletons:
        info_bits.append(
            f"{projection.n_dropped_singletons} singleton label(s) excluded"
        )
    if projection.notes:
        info_bits.append(projection.notes)
    if info_bits:
        st.caption(" · ".join(info_bits))
