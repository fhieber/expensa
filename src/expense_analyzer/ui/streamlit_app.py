"""Streamlit UI orchestrator — local-only (binds to 127.0.0.1 via the
CLI launcher).

This file is intentionally thin: it sets the page config + global CSS,
renders the top header bar, and dispatches into one of four tab
modules. All tab logic lives in dedicated modules so this file stays
under ~150 lines and the tabs can be edited / tested independently.

Tabs, in order:
  1. dashboard.py        -- overview stats, charts, records table.
  2. categories_tab.py   -- types table with per-category stats.
  3. data_tab.py         -- AgGrid + Auto-Label flow + record dialog.
  4. settings.py         -- model picker, privacy, own IBANs, DB admin.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from expense_analyzer.ui import (
    categories_tab,
    dashboard,
    data_tab,
    settings,
)
from expense_analyzer.ui._shared import get_config, get_conn

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
    try:
        db_size_mb = (
            cfg.db_path.stat().st_size / (1024 * 1024)
            if cfg.db_path.exists() else 0.0
        )
    except OSError:
        db_size_mb = 0.0
    bar = st.container()
    with bar:
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1.3, 1, 1])
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
            st.metric("Categories", n_cat)
        with c5:
            st.metric("DB size", f"{db_size_mb:.1f} MB",
                      help=f"Path: {cfg.db_path}")
    st.divider()


_render_header()

tab_dash, tab_cats, tab_data, tab_settings = st.tabs(
    ["Dashboard", "Categories", "Data", "Settings"]
)

with tab_dash:
    dashboard.render()
with tab_cats:
    categories_tab.render()
with tab_data:
    data_tab.render()
with tab_settings:
    settings.render()

# The Data tab parks an expense id in session_state when the user clicks
# "See Details". Open the dialog here so it's rendered at the document
# root and not inside a tab container.
data_tab.maybe_show_inspect_dialog()
