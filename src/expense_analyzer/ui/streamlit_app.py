"""Streamlit UI orchestrator — local-only (binds to 127.0.0.1 via the
CLI launcher).

This file is intentionally thin: it sets the page config + global CSS,
renders the top header bar, and dispatches into one of four tab
modules. All tab logic lives in dedicated modules so this file stays
under ~150 lines and the tabs can be edited / tested independently.

Tabs, in order:
  1. dashboard.py        -- overview stats, charts, records table.
  2. review_tab.py       -- active-learning queue + bulk Predict-all.
  3. data_tab.py         -- AgGrid + Auto-Label flow + record dialog.
  4. categories_tab.py   -- types table with per-category stats.
  5. settings.py         -- model picker, privacy, own IBANs, DB admin.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from expense_analyzer.ui import (
    categories_tab,
    dashboard,
    data_tab,
    eval_tab,
    review_tab,
    settings,
)
from expense_analyzer.ui._shared import (
    add_account_via_ui,
    clear_tab_state,
    get_active_account,
    get_config,
    get_conn,
    get_registry,
    is_unlocked,
    remove_account_via_ui,
    rename_account_via_ui,
    set_active_account,
    unlock,
)

# ---------------------------------------------------------------------------
# Page setup -- CSS + header bar live in this file because they're cross-tab.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="expense-analyzer-de", layout="wide", page_icon="💶")

# Density tweaks: slightly smaller fonts, tighter spacing everywhere.
# Target streamlit's stable data-testid hooks and base elements; keep
# adjustments subtle (~10–15% reduction) so things still feel comfortable.
st.markdown(
    """
    <style>
      /* Hide Streamlit's default top toolbar (Deploy/menu) — it overlays
         our own status bar. We don't need it for a local-only app. */
      header[data-testid="stHeader"] { display: none !important; }
      [data-testid="stToolbar"] { display: none !important; }
      [data-testid="stDecoration"] { display: none !important; }
      .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
      }
      h1 { font-size: 1.35rem !important; margin: 0.25rem 0 0.4rem 0 !important; }
      h2 { font-size: 1.10rem !important; margin: 0.2rem 0 0.3rem 0 !important; }
      h3 { font-size: 1.00rem !important; margin: 0.2rem 0 0.25rem 0 !important; }
      h4, h5 { font-size: 0.92rem !important; margin: 0.15rem 0 0.2rem 0 !important; }
      p, li, span, label { font-size: 0.88rem; }
      [data-testid="stVerticalBlock"] { gap: 0.4rem !important; }
      [data-testid="stHorizontalBlock"] { gap: 0.4rem !important; }
      [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
      [data-testid="stMetricLabel"] { font-size: 0.72rem !important; }
      [data-testid="stCaptionContainer"], .stCaption { font-size: 0.74rem !important; }
      .stButton button { padding: 0.25rem 0.7rem !important; font-size: 0.85rem !important; }
      .stTabs [data-baseweb="tab"] { padding: 0.35rem 0.9rem !important; }
      .stTabs [data-baseweb="tab"] p { font-size: 0.92rem !important; }
      div[data-testid="stExpander"] summary p { font-size: 0.88rem !important; }
      hr { margin: 0.4rem 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Account picker (sits above the header metrics).
# ---------------------------------------------------------------------------


def _on_account_changed() -> None:
    """Selectbox on_change handler. Streamlit writes the new label into
    `account_picker`; we map it back to a slug, switch, wipe stale
    tab state, and let the natural rerun re-render every tab against
    the new DB."""
    picked_label = st.session_state.get("account_picker")
    if not picked_label:
        return
    slug = st.session_state.get("_picker_label_to_id", {}).get(picked_label)
    if slug is None or slug == get_active_account().id:
        return
    set_active_account(slug)
    clear_tab_state()


@st.dialog("Add account")
def _add_account_dialog() -> None:
    """Name input -> register row, init DB, seed German defaults,
    switch active. The rerun is the only thing that changes UI state
    so the failure mode (validation error) leaves the user where they
    were."""
    st.caption(
        "A new account gets its own SQLite DB under "
        "`accounts/<slug>/db.sqlite`. ML models, vendor lookup and the "
        "Streamlit config are shared across accounts."
    )
    name = st.text_input(
        "Account name", placeholder="e.g. Business",
        key="add_account_name",
    )
    with_defaults = st.checkbox(
        "Install default German categories", value=True,
        key="add_account_with_defaults",
    )
    cols = st.columns([1, 1, 4])
    if cols[0].button("Add", type="primary", disabled=not name.strip(),
                       key="add_account_confirm"):
        try:
            info = add_account_via_ui(name, with_defaults=with_defaults)
        except ValueError as e:
            st.error(f"refusing: {e}")
            return
        set_active_account(info.id)
        clear_tab_state()
        # Clear the dialog's inputs so the next open is blank.
        for k in ("add_account_name", "add_account_with_defaults"):
            st.session_state.pop(k, None)
        st.rerun()
    if cols[1].button("Cancel", key="add_account_cancel"):
        st.rerun()


@st.dialog("Rename account")
def _rename_account_dialog() -> None:
    active = get_active_account()
    st.caption(
        f"Rename **{active.name}** (slug `{active.id}`). The slug and "
        "data directory stay put -- this is purely cosmetic."
    )
    new_name = st.text_input(
        "New name", value=active.name,
        key="rename_account_new_name",
    )
    cols = st.columns([1, 1, 4])
    if cols[0].button(
        "Rename", type="primary",
        disabled=not new_name.strip() or new_name.strip() == active.name,
        key="rename_account_confirm",
    ):
        try:
            rename_account_via_ui(active.id, new_name)
        except ValueError as e:
            st.error(f"refusing: {e}")
            return
        st.session_state.pop("rename_account_new_name", None)
        st.rerun()
    if cols[1].button("Cancel", key="rename_account_cancel"):
        st.rerun()


@st.dialog("Remove account")
def _remove_account_dialog() -> None:
    active = get_active_account()
    registry = get_registry()
    rows = registry.all()
    if len(rows) <= 1:
        st.warning(
            "Refusing: you'd be left with zero registered accounts. "
            "Add another account first."
        )
        if st.button("Close", key="remove_account_close"):
            st.rerun()
        return
    st.warning(
        f"Remove **{active.name}** from the registry?\n\n"
        f"This does **not** delete files on disk. The data directory "
        f"`{active.data_dir}` stays put -- you can re-register it later "
        "by editing `accounts.yaml` manually, or just leave it."
    )
    cols = st.columns([1, 1, 4])
    if cols[0].button("Remove", type="primary",
                       key="remove_account_confirm"):
        try:
            removed = remove_account_via_ui(active.id)
        except KeyError as e:
            st.error(f"already removed: {e}")
            return
        # Switch to the first remaining account.
        remaining = get_registry().all()
        if remaining:
            set_active_account(remaining[0].id)
        clear_tab_state()
        st.toast(f"removed {removed.name}; files still at {removed.data_dir}")
        st.rerun()
    if cols[1].button("Cancel", key="remove_account_cancel"):
        st.rerun()


def _render_account_picker() -> None:
    """Account selectbox + Add / Rename / Remove buttons. Sits above
    the header metrics so it reads as a single row of "what am I
    looking at, and what tools do I have on it"."""
    registry = get_registry()
    rows = registry.all()

    # On first render with no registered accounts, show only the +Add
    # button. Picker would be empty otherwise.
    if not rows:
        cols = st.columns([4, 1])
        with cols[0]:
            st.caption(
                "No accounts registered yet. Add one to get started -- "
                "the bundled German default categories are seeded "
                "automatically."
            )
        if cols[1].button("➕ Add account", type="primary",
                           key="add_first_account_btn"):
            _add_account_dialog()
        return

    # The selectbox uses display labels (name); we map back via the
    # _picker_label_to_id dict. Storing the mapping on session_state
    # lets the on_change handler resolve without re-walking the
    # registry.
    labels = [a.name for a in rows]
    st.session_state["_picker_label_to_id"] = {a.name: a.id for a in rows}
    active = get_active_account()
    try:
        index = labels.index(active.name)
    except ValueError:
        index = 0

    picker_col, add_col, rename_col, remove_col = st.columns([2, 0.5, 0.5, 0.5], width=350, gap='small')
    with picker_col:
        st.selectbox(
            "Account",
            labels,
            index=index,
            key="account_picker",
            on_change=_on_account_changed,
            label_visibility="collapsed",
        )
    if add_col.button("➕", key="account_add_btn",
                       help="Create another account."):
        _add_account_dialog()
    if rename_col.button("✏", key="account_rename_btn",
                          help="Rename the active account."):
        _rename_account_dialog()
    if remove_col.button("🗑", key="account_remove_btn",
                          help="Drop the active account from the registry "
                               "(files on disk stay).",
                          disabled=len(rows) <= 1):
        _remove_account_dialog()


# ---------------------------------------------------------------------------
# Unlock gate for encrypted accounts.
# ---------------------------------------------------------------------------


def _render_unlock_gate() -> bool:
    """Gate the rest of the page behind a password when the active account
    is encrypted and not yet unlocked this session.

    Returns True when the account is accessible (plaintext or already
    unlocked); otherwise renders a password prompt and returns False so the
    caller can ``st.stop()`` before any tab touches the locked DB. Switching
    to an encrypted account triggers a rerun that lands here, which is what
    makes the UI ask for the password on account switch."""
    active = get_active_account()
    if is_unlocked(active):
        return True

    from expense_analyzer.storage.crypto import encryption_available

    st.title("🔒 Locked account")
    st.write(
        f"Account **{active.name}** is encrypted. Enter its password to continue."
    )
    if not encryption_available():
        st.error(
            "This database is encrypted but the SQLCipher dependency isn't "
            "installed in the running environment. Install it with "
            "`pip install -e '.[encryption]'` (from the repo root) and restart the UI."
        )
        return False
    with st.form("unlock_form"):
        pw = st.text_input("Password", type="password", key="unlock_pw")
        if st.form_submit_button("Unlock", type="primary"):
            if unlock(active, pw):
                st.session_state.pop("unlock_pw", None)
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


# ---------------------------------------------------------------------------
# Top header bar (replaces the old sidebar).
# ---------------------------------------------------------------------------


def _render_header() -> None:
    cfg = get_config()
    conn = get_conn()
    try:
        n_exp = conn.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
        n_lab = conn.execute(
            "SELECT COUNT(DISTINCT expense_id) AS n FROM labels WHERE source='user'"
        ).fetchone()["n"]
        n_cat = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
        # Categorized = at least one label of any source (user or model).
        n_categorized = conn.execute(
            "SELECT COUNT(DISTINCT expense_id) AS n FROM labels"
        ).fetchone()["n"]
    except sqlite3.OperationalError:
        n_exp = n_lab = n_cat = n_categorized = 0
    pct_cat = (n_categorized / n_exp * 100.0) if n_exp else 0.0
    # >=90% green, <50% red, in-between amber. Signal via a coloured dot
    # so we stay in plain st.metric() and the typography matches the other
    # metrics exactly.
    if pct_cat >= 90:
        dot = "🟢"
    elif pct_cat < 50:
        dot = "🔴"
    else:
        dot = "🟡"
    # Review queue size as a header metric instead of in the tab label
    # -- the tab label has to stay constant across reruns (see the
    # explanation around the `st.tabs(...)` call below).
    try:
        n_review = review_tab.queue_size(conn)
    except Exception:
        n_review = 0
    try:
        db_size_mb = (
            cfg.db_path.stat().st_size / (1024 * 1024)
            if cfg.db_path.exists() else 0.0
        )
    except OSError:
        db_size_mb = 0.0
    bar = st.container()
    with bar:
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1.3, 1, 1, 1])
        with c1:
            st.metric("Expenses", n_exp)
        with c2:
            st.metric("User-labeled", n_lab)
        with c3:
            st.metric(
                "Categorized", f"{dot} {pct_cat:.0f}%",
                help=f"{n_categorized} of {n_exp} expenses have a "
                     "label (user or model). 🟢 ≥90% · 🟡 50-90% · 🔴 <50%.",
            )
        with c4:
            st.metric(
                "To review", n_review,
                help="Expenses needing a user label or a mid-confidence "
                     "confirmation. See the Review tab.",
            )
        with c5:
            st.metric("Categories", n_cat)
        with c6:
            st.metric("DB size", f"{db_size_mb:.1f} MB",
                      help=f"Path: {cfg.db_path}")
    st.divider()


_render_account_picker()
if not _render_unlock_gate():
    st.stop()
_render_header()

tab_dash, tab_review, tab_data, tab_cats, tab_quality, tab_settings = st.tabs(
    ["Dashboard", "Review", "Data", "Categories", "Quality", "Settings"]
)

# Render order matches tab order so a future side-effect from an
# earlier tab (DB write, session_state mutation) is consistently visible
# to a later tab in the same rerun.
with tab_dash:
    dashboard.render()
with tab_review:
    review_tab.render()
with tab_data:
    data_tab.render()
with tab_cats:
    categories_tab.render()
with tab_quality:
    eval_tab.render()
with tab_settings:
    settings.render()

# The Data tab parks an expense id in session_state when the user clicks
# "See Details". Open the dialog here so it's rendered at the document
# root and not inside a tab container.
data_tab.maybe_show_inspect_dialog()
