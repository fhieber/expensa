"""Streamlit UI — local-only (binds to 127.0.0.1 via the CLI launcher).

Tabs, in order:
  1. Dashboard  — overview stats, inline-editable records list.
  2. Categories — types table with per-category stats, add/remove.
  3. Data       — sortable/filterable table; inline category edit;
                  selection-driven Auto-Label flow. Upload+ingest lives in
                  a collapsed "Import CSV" expander at the top.
  4. Settings   — model info, privacy, danger zone.

No sidebar by design — everything lives in tabs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from expense_analyzer.config import Config, load_config
from expense_analyzer.enrichment.notes import get_note, set_note
from expense_analyzer.features.embeddings import (
    Embedder,
    SentenceTransformerEmbedder,
)
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.storage.admin import (
    category_removal_impact,
    remove_category,
    reset_all,
    reset_data,
)
from expense_analyzer.storage.categories import (
    add_label,
    list_categories,
    upsert_category,
)
from expense_analyzer.storage.database import get_or_create_database
from expense_analyzer.storage.stats import category_stats, uncategorized_stat
from expense_analyzer.viz import (
    bar_spend_by_category,
    monthly_flow_by_category,
    spend_by_category,
    stacked_monthly_by_category,
)

# Shared CTE used by the Dashboard drill-down queries.
_LATEST_LABEL_CTE_DASHBOARD = """
    WITH latest_label AS (
        SELECT l.expense_id, l.category_id
        FROM labels l
        JOIN (
            SELECT expense_id, MAX(id) AS max_id
            FROM labels GROUP BY expense_id
        ) m ON l.id = m.max_id
    )
    SELECT
        e.id, e.buchungsdatum, e.counterparty, e.verwendungszweck,
        e.betrag_cents / 100.0 AS "betrag_€",
        COALESCE(c.name, '(unkategorisiert)') AS category,
        e.iban
    FROM expenses e
    LEFT JOIN latest_label ll ON ll.expense_id = e.id
    LEFT JOIN categories c ON c.id = ll.category_id
"""


# ---------------------------------------------------------------------------
# Boot
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


@st.cache_resource
def _load_config_cached() -> Config:
    return load_config()


@st.cache_resource
def _connect_cached(db_path_str: str) -> sqlite3.Connection:
    return get_or_create_database(Path(db_path_str))


@st.cache_resource
def _real_embedder(model_name: str, device: str, batch_size: int) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=model_name, device=device, batch_size=batch_size, verbose=False
    )


cfg = _load_config_cached()
conn = _connect_cached(str(cfg.db_path))


def _embedder() -> Embedder:
    """The configured local sentence-transformer (no cloud calls)."""
    return _real_embedder(cfg.embedding_model, cfg.device, cfg.embedding_batch_size)


# ---------------------------------------------------------------------------
# Top header bar (replaces the old sidebar)
# ---------------------------------------------------------------------------

def _render_header() -> None:
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
        db_size_mb = cfg.db_path.stat().st_size / (1024 * 1024) \
            if cfg.db_path.exists() else 0.0
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
            st.metric("Categorized", f"{dot} {pct_cat:.0f}%",
                      help=f"{n_categorized} of {n_exp} expenses have a "
                           "label (user or model). 🟢 ≥90% · 🟡 50-90% · 🔴 <50%.")
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


# ---------------------------------------------------------------------------
# Helpers used by multiple tabs
# ---------------------------------------------------------------------------

def _category_options(include_unlabeled: bool = True) -> list[tuple[int | None, str]]:
    """[(category_id, name), ...] for dropdowns. `None` represents 'unlabeled'."""
    out: list[tuple[int | None, str]] = []
    for c in list_categories(conn):
        out.append((c.id, c.name))
    if include_unlabeled:
        out.append((None, "(unkategorisiert)"))
    return out


def _set_user_label(expense_id: int, category_id: int) -> None:
    add_label(conn, expense_id, category_id, "user")


def _format_eur(cents: int) -> str:
    return f"{cents / 100:>9,.2f} €"


# ---------------------------------------------------------------------------
# Tab 1: Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_PRESETS = [
    # Default is "Last 3 months" -- a 30/90/180-day cluster of short
    # windows at the front, followed by the annual-ish bucket
    # (YTD -> Last year -> Last 2 years), with "All-time" and Custom
    # at the end. Order picked so adjacent options are semantically
    # close.
    "Last month", "Last 3 months", "Last 6 months",
    "YTD", "Last year", "Last 2 years", "All-time", "Custom",
]
_DASHBOARD_DEFAULT_PRESET = "Last 3 months"


def _dashboard_date_range(preset: str, custom_from=None, custom_to=None):
    from datetime import date as _date
    from datetime import timedelta

    today = _date.today()
    if preset == "All-time":
        return None, None
    if preset == "YTD":
        return _date(today.year, 1, 1), today
    if preset == "Last month":
        return today - timedelta(days=30), today
    if preset == "Last 3 months":
        return today - timedelta(days=90), today
    if preset == "Last 6 months":
        return today - timedelta(days=180), today
    if preset == "Last year":
        return today - timedelta(days=365), today
    if preset == "Last 2 years":
        return today - timedelta(days=730), today
    if preset == "Custom":
        return custom_from, custom_to
    return None, None


with tab_dash:
    from expense_analyzer.utils.colors import readable_text_color
    from expense_analyzer.viz import (
        stacked_weekly_by_category,
        weekly_by_category,
    )

    st.header("Dashboard")
    n_exp = conn.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
    if n_exp == 0:
        st.info("Import a CSV from the **Data** tab's *Import CSV* expander to get started.")
    else:
        # Radio + (when Custom) From/To inputs on a SINGLE row. Giving
        # the radio the wide column keeps the horizontal labels from
        # wrapping at typical viewport widths; From/To get just enough
        # room for a `YYYY-MM-DD` input each.
        preset_col, from_col, to_col = st.columns([6, 1.5, 1.5])
        with preset_col:
            preset = st.radio(
                "Date range",
                _DASHBOARD_PRESETS,
                index=_DASHBOARD_PRESETS.index(_DASHBOARD_DEFAULT_PRESET),
                horizontal=True,
                key="dashboard_date_preset",
            )
        if preset == "Custom":
            with from_col:
                custom_from = st.date_input(
                    "From", value=None, key="dashboard_from"
                )
            with to_col:
                custom_to = st.date_input(
                    "To", value=None, key="dashboard_to"
                )
            since, until = _dashboard_date_range(preset, custom_from, custom_to)
        else:
            since, until = _dashboard_date_range(preset)

        # ====================================================================
        # Headline stats live AT THE TOP, immediately below the date range
        # selector. The user said: when scanning the dashboard the savings
        # rate / income / expenses totals should be the first thing seen,
        # not buried beneath the per-category charts.
        # ====================================================================
        from expense_analyzer.viz import (
            DEFAULT_SAVINGS_CATEGORIES,
            anomalies,
            income_vs_expense_chart,
            monthly_income_vs_expense,
            recurring_subscriptions,
            savings_flow,
        )

        ivex_df = monthly_income_vs_expense(conn, since=since, until=until)
        sav_df = savings_flow(conn, since=since, until=until)
        total_income = float(ivex_df["income"].sum()) if not ivex_df.empty else 0.0
        total_exp = float(ivex_df["expenses"].sum()) if not ivex_df.empty else 0.0
        total_to_sav = float(sav_df["to_savings"].sum()) if not sav_df.empty else 0.0
        total_from_sav = float(sav_df["from_savings"].sum()) if not sav_df.empty else 0.0
        net_to_sav = total_to_sav - total_from_sav
        if total_income > 0:
            savings_rate = (total_income - total_exp) / total_income
            pct = savings_rate * 100
            if pct >= 20:
                dot = "🟢"
            elif pct < 0:
                dot = "🔴"
            else:
                dot = "🟡"
            sr_value = f"{dot} {pct:.0f}%"
        else:
            sr_value = "—"

        def _de_eur(v: float) -> str:
            """Format a euro amount in DE locale: thousands `.`, decimal `,`."""
            return (
                f"{v:,.2f} €"
                .replace(",", "X").replace(".", ",").replace("X", ".")
            )

        # Bounded container so the headline tiles read as a single
        # unit, matching the chart-expander frames below.
        with st.container(border=True):
            sr_cols = st.columns(4)
            sr_cols[0].metric("Savings rate", sr_value)
            sr_cols[1].metric("Income", _de_eur(total_income))
            sr_cols[2].metric("Expenses", _de_eur(total_exp))
            sr_cols[3].metric(
                "💰 To savings (net)",
                _de_eur(net_to_sav),
                help=(
                    f"Money moved to your own accounts (category Sparen) "
                    f"minus what came back. Gross out: {_de_eur(total_to_sav)} · "
                    f"gross in: {_de_eur(total_from_sav)}."
                ),
            )
            st.caption(
                "Reflects the currently selected date range above. "
                "🟢 ≥20% · 🟡 0–20% · 🔴 negative. "
                "Rows categorised **Sparen** and rows matching a registered "
                "own IBAN (`iban_is_known_self`) are treated as neutral on "
                "both the income and expense sides — so the savings rate "
                "captures real income vs real consumption."
            )

        # ---- Build a single category-color map used by every chart -----
        # The categories table is the source of truth. Falls back to grey
        # for the synthetic "(unkategorisiert)" bucket.
        dash_cats = list_categories(conn)
        category_color_map: dict[str, str] = {c.name: c.color for c in dash_cats}
        category_color_map["(unkategorisiert)"] = "#bbbbbb"

        # ---- Charts (display-only, each in a collapsible expander) -----
        # The expander label IS the chart title, so we suppress the
        # in-chart title to avoid double-rendering. By default only the
        # two "structural" charts (Summe Ausgaben + Monatlicher Saldo)
        # are expanded; Wöchentliche + Einkommen-vs-Ausgaben start
        # collapsed so the page is scannable at first glance.
        # Sparen is pre-hidden on the per-category charts because it
        # represents transfers to the user's own accounts; including it
        # would inflate the "consumption" picture. Re-enableable via the
        # chart legend once a chart is expanded.
        def _chart_expander(label: str, fig, expanded: bool, key: str) -> None:
            with st.expander(label, expanded=expanded):
                # `title=None` renders as the literal string "undefined"
                # in some Plotly/Streamlit version combos. Setting an
                # empty title with zero top margin removes both the
                # text and the reserved gap above the chart.
                fig.update_layout(
                    title_text="",
                    margin={"t": 20, "b": 20, "l": 0, "r": 0},
                )
                st.plotly_chart(fig, width="stretch", key=key)

        _chart_expander(
            "Summe Ausgaben nach Kategorie",
            bar_spend_by_category(
                spend_by_category(conn, since=since, until=until),
                color_map=category_color_map,
                hidden_categories=DEFAULT_SAVINGS_CATEGORIES,
            ),
            expanded=True,
            key="dashboard_bar_chart",
        )
        _chart_expander(
            "Wöchentliche Ausgaben nach Kategorie",
            stacked_weekly_by_category(
                weekly_by_category(conn, since=since, until=until),
                color_map=category_color_map,
                hidden_categories=DEFAULT_SAVINGS_CATEGORIES,
            ),
            expanded=False,
            key="dashboard_weekly_chart",
        )
        # Diverging stacked bar (income above 0, expenses below 0).
        # Sparen rows are already excluded upstream by
        # `monthly_flow_by_category(savings_categories=...)`, so no
        # `hidden_categories` plumbing is needed here.
        _chart_expander(
            "Monatlicher Saldo je Kategorie",
            stacked_monthly_by_category(
                monthly_flow_by_category(conn, since=since, until=until),
                color_map=category_color_map,
            ),
            expanded=True,
            key="dashboard_trend_chart",
        )
        # Income vs Expenses chart over the visible date range. Used to
        # live under an "Insights" subheader; the subheader was removed
        # because the section above is now headline-level on its own.
        _chart_expander(
            "Einkommen vs Ausgaben (monatlich)",
            income_vs_expense_chart(ivex_df),
            expanded=False,
            key="dashboard_ivex_chart",
        )

        st.divider()

        # ---- Single records table --------------------------------------
        # Read-only (relabel happens on the Data tab). Date-range driven.
        # Category cells are coloured with the user-chosen category
        # colour from the Categories tab; text colour is auto-picked
        # (black/white) to stay legible on any background.
        params: list = []
        clauses = ["e.is_income = 0"]
        if since is not None:
            clauses.append("e.buchungsdatum >= ?")
            params.append(since.isoformat())
        if until is not None:
            clauses.append("e.buchungsdatum <= ?")
            params.append(until.isoformat())

        sql = (
            _LATEST_LABEL_CTE_DASHBOARD
            + " WHERE " + " AND ".join(clauses)
            + " ORDER BY e.buchungsdatum DESC, e.id DESC LIMIT 5000"
        )
        full_df = pd.read_sql_query(sql, conn, params=params)
        if not full_df.empty:
            full_df["buchungsdatum"] = pd.to_datetime(full_df["buchungsdatum"])
            full_df["category"] = full_df["category"].fillna("(unkategorisiert)")

        date_label = ("all dates" if since is None and until is None
                      else f"{since} … {until}")
        st.caption(
            f"{len(full_df)} record(s) · date range: {date_label}"
        )

        # Cell-level colouring via pandas Styler. `applymap` is the right
        # tool here -- per-cell function -> CSS string. Streamlit's
        # st.dataframe consumes Stylers directly. The category lookup
        # falls back to neutral grey for "(unkategorisiert)" so empty
        # cells still get *some* fill rather than a jarring transparent
        # gap mid-table.
        def _style_category(val: str) -> str:
            name = (val or "").strip()
            if not name or name == "(unkategorisiert)":
                return ""
            bg = category_color_map.get(name, "#bbbbbb")
            fg = readable_text_color(bg)
            return f"background-color: {bg}; color: {fg};"

        if full_df.empty:
            styled = full_df
        else:
            styled = full_df.style.map(_style_category, subset=["category"])

        st.dataframe(
            styled,
            hide_index=True,
            width="stretch",
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "buchungsdatum": st.column_config.DateColumn(
                    "Date", format="DD.MM.YYYY"
                ),
                "counterparty": st.column_config.TextColumn("Counterparty"),
                "verwendungszweck": st.column_config.TextColumn("Verwendungszweck"),
                "betrag_€": st.column_config.NumberColumn(
                    "Amount €", format="%.2f"
                ),
                "category": st.column_config.TextColumn("Category"),
                "iban": st.column_config.TextColumn("IBAN"),
            },
            key="dashboard_records_table",
        )

        # ---- All-time helper tables: recurring subs + anomalies ---------
        # Placed BELOW the records table because they use the full
        # history regardless of the dashboard date range, so they don't
        # belong inline with the date-range-scoped views above. Both
        # collapsed by default -- they're reference views, not always
        # needed.
        with st.expander("Recurring expenses (all-time)", expanded=False):
            st.caption(
                "Vendors with a detectable cadence (weekly / bi-weekly / "
                "monthly / quarterly / semi-annual / annual). Cadence is "
                "inferred from the median day-gap between charges; "
                "annualised cost = typical amount × charges/year. Sorted "
                "DESC by annualised cost."
            )
            recurring_df = recurring_subscriptions(conn)
            if recurring_df.empty:
                st.info(
                    "No vendors with ≥3 charges yet -- ingest more data "
                    "so the cadence detector has gaps to measure."
                )
            else:
                st.dataframe(
                    recurring_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "name": st.column_config.TextColumn("Vendor"),
                        "cadence": st.column_config.TextColumn("Cadence"),
                        "last_seen": st.column_config.DateColumn(
                            "Last seen", format="DD.MM.YYYY"
                        ),
                        "typical_amount": st.column_config.NumberColumn(
                            "Typical (€)", format="%.2f"
                        ),
                        "charges_per_year": st.column_config.NumberColumn(
                            "Charges/yr", format="%.1f"
                        ),
                        "annualised": st.column_config.NumberColumn(
                            "Annualised (€)", format="%.2f"
                        ),
                        "n_charges": st.column_config.NumberColumn(
                            "Seen", format="%d"
                        ),
                    },
                )

        with st.expander("Unusual amounts (all-time)", expanded=False):
            st.caption(
                "Rows whose amount is more than 2σ above the vendor's "
                "historical average. Surfaces price hikes, double-charges "
                "and suspected fraud. Baseline statistics use the vendor's "
                "full history."
            )
            anom_df = anomalies(conn)
            if anom_df.empty:
                st.info(
                    "No anomalies above z=2 in this view. Either every "
                    "expense is in line with its vendor's typical amount, "
                    "or there's not enough history yet — vendors need "
                    "≥3 prior records to score."
                )
            else:
                # Drop columns we don't need to show in the cramped view.
                display = anom_df.drop(columns=["id", "n_history"])
                st.dataframe(
                    display,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "date": st.column_config.DateColumn(
                            "Date", format="DD.MM.YYYY"
                        ),
                        "counterparty": st.column_config.TextColumn("Vendor"),
                        "category": st.column_config.TextColumn("Category"),
                        "amount": st.column_config.NumberColumn(
                            "Amount (€)", format="%.2f"
                        ),
                        "typical": st.column_config.NumberColumn(
                            "Typical (€)", format="%.2f"
                        ),
                        "vs_typical": st.column_config.NumberColumn(
                            "× typical", format="%.1fx"
                        ),
                        "zscore": st.column_config.NumberColumn(
                            "z", format="%.1f"
                        ),
                    },
                )


# ---------------------------------------------------------------------------
# Tab 2: Categories
# ---------------------------------------------------------------------------

def _save_cat_name(cat_id: int) -> None:
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
    new_desc = (st.session_state.get(f"cat_{cat_id}_desc") or "").strip()
    conn.execute("UPDATE categories SET description=? WHERE id=?", (new_desc, cat_id))


def _save_cat_color(cat_id: int) -> None:
    new_color = st.session_state.get(f"cat_{cat_id}_color") or "#888888"
    conn.execute("UPDATE categories SET color=? WHERE id=?", (new_color, cat_id))


with tab_cats:
    from expense_analyzer.config import packaged_default_categories
    from expense_analyzer.storage.categories import import_categories_from_yaml
    from expense_analyzer.utils.colors import random_hex_color

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

        # Header row
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
                    # Clear inputs for the next addition, refresh suggested color.
                    for k in ("new_cat_name", "new_cat_desc", "new_cat_color"):
                        st.session_state.pop(k, None)
                    st.session_state.new_cat_color = random_hex_color()
                    st.rerun()
                except sqlite3.IntegrityError as e:
                    st.error(f"could not save: {e}")


# ---------------------------------------------------------------------------
# Tab 3: Data
# ---------------------------------------------------------------------------
#
# Upload + ingest now lives in an "Import CSV" expander at the top of this
# tab. After a successful ingest, the new expense IDs are pinned in
# session_state["data_pinned_ids"] so the filter shows exactly those rows.
# Labeling happens through the selection-driven Auto-Label flow below.

def _build_data_query(
    date_from, date_to, cats: list[str], source: str,
    search: str, amount_min: float, amount_max: float,
    include_income: bool,
    pinned_ids: list[int] | None = None,
) -> tuple[str, list]:
    """Return (SQL, params) for the Data table given filter widgets.

    If ``pinned_ids`` is provided, ONLY rows with those IDs are shown
    (other filters are ignored). Used right after a CSV ingest to scope
    the Data table to the just-imported records.
    """
    parts: list[str] = []
    params: list = []
    if pinned_ids:
        ph_pin = ",".join("?" * len(pinned_ids))
        parts.append(f"e.id IN ({ph_pin})")
        params.extend(int(x) for x in pinned_ids)
        # Build SELECT and return early -- other filters don't apply when
        # we're pinning to a specific id set.
        where = " WHERE " + " AND ".join(parts)
        sql = (
            """
            WITH latest_label AS (
                SELECT l.expense_id, l.category_id, l.source, l.confidence
                FROM labels l
                JOIN (
                    SELECT expense_id, MAX(id) AS max_id
                    FROM labels GROUP BY expense_id
                ) m ON l.id = m.max_id
            )
            SELECT
                e.id, e.buchungsdatum,
                e.counterparty, e.zahlungspflichtiger, e.verwendungszweck,
                e.betrag_cents / 100.0 AS "betrag_€",
                c.name AS category, ll.category_id AS category_id,
                ll.source AS label_source, ll.confidence,
                e.umsatztyp, e.iban, e.iban_is_foreign,
                e.has_glaeubiger_id, e.mandatsreferenz_present
            FROM expenses e
            LEFT JOIN latest_label ll ON ll.expense_id = e.id
            LEFT JOIN categories c ON c.id = ll.category_id
            """
            + where
            + " ORDER BY e.buchungsdatum DESC, e.id DESC"
        )
        return sql, params
    if date_from is not None:
        parts.append("e.buchungsdatum >= ?")
        params.append(date_from.isoformat())
    if date_to is not None:
        parts.append("e.buchungsdatum <= ?")
        params.append(date_to.isoformat())
    if not include_income:
        parts.append("e.is_income = 0")
    if amount_min is not None:
        parts.append("ABS(e.betrag_cents) >= ?")
        params.append(int(amount_min * 100))
    if amount_max is not None:
        parts.append("ABS(e.betrag_cents) <= ?")
        params.append(int(amount_max * 100))
    if search:
        like = f"%{search.lower()}%"
        parts.append(
            "(LOWER(e.counterparty) LIKE ? OR LOWER(e.verwendungszweck) LIKE ?)"
        )
        params.extend([like, like])
    if cats:
        unlabeled_picked = "(unkategorisiert)" in cats
        named = [c for c in cats if c != "(unkategorisiert)"]
        cat_conds = []
        if named:
            ph = ",".join("?" * len(named))
            cat_conds.append(f"c.name IN ({ph})")
            params.extend(named)
        if unlabeled_picked:
            cat_conds.append("c.id IS NULL")
        parts.append("(" + " OR ".join(cat_conds) + ")")
    if source == "user":
        parts.append("ll.source = 'user'")
    elif source == "model":
        parts.append("ll.source = 'model'")
    elif source == "unlabeled":
        parts.append("ll.expense_id IS NULL")

    where = (" WHERE " + " AND ".join(parts)) if parts else ""
    sql = (
        """
        WITH latest_label AS (
            SELECT l.expense_id, l.category_id, l.source, l.confidence
            FROM labels l
            JOIN (
                SELECT expense_id, MAX(id) AS max_id
                FROM labels GROUP BY expense_id
            ) m ON l.id = m.max_id
        )
        SELECT
            e.id, e.buchungsdatum,
            e.counterparty, e.zahlungspflichtiger, e.verwendungszweck,
            e.betrag_cents / 100.0 AS "betrag_€",
            c.name AS category, ll.category_id AS category_id,
            ll.source AS label_source, ll.confidence,
            e.umsatztyp, e.iban, e.iban_is_foreign,
            e.has_glaeubiger_id, e.mandatsreferenz_present
        FROM expenses e
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c ON c.id = ll.category_id
        """
        + where
        + " ORDER BY e.buchungsdatum DESC, e.id DESC"
    )
    return sql, params


with tab_data:
    st.header("Data")

    # --- Import CSV (collapsed) ---------------------------------------------
    with st.expander("Import CSV", expanded=False):
        st.caption(
            "Drop one or more German bank-export CSVs (`;` separator, comma "
            "decimal). On Ingest each new row's text/IBAN/numeric features "
            "and sentence-transformer embedding are computed and stored; the "
            "table below then pins to those new rows so you can review and "
            "label them with the Auto-Label flow."
        )
        files = st.file_uploader(
            "CSV file(s)", accept_multiple_files=True, type=["csv"],
            key="data_import_files",
        )
        ingest_clicked = st.button(
            "Ingest", type="primary", disabled=not files, key="data_ingest_btn",
        )
        if ingest_clicked and files:
            import tempfile

            emb = _embedder()
            new_ids: list[int] = []
            with st.status("Importing…", expanded=True) as status:
                progress = st.progress(0.0, text="starting…")
                for f in files:
                    status.write(f"parsing {f.name}…")
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        tmp.write(f.read())
                        p = Path(tmp.name)

                    def _cb(phase: str, done: int, total: int, fname: str = f.name) -> None:
                        if total <= 0:
                            return
                        # Coarse phase labels for the user; the bar fills
                        # within each phase per file.
                        label_map = {
                            "parse": "parsing",
                            "insert": "inserting rows",
                            "embed": "computing embeddings",
                        }
                        phase_label = label_map.get(phase, phase)
                        progress.progress(
                            min(1.0, done / total),
                            text=f"{fname}: {phase_label} {done}/{total}",
                        )

                    r = ingest_csv(conn, p, embedder=emb, progress_callback=_cb)
                    new_ids.extend(r.new_ids)
                    status.write(
                        f"{f.name}: parsed={r.parsed} · new={r.inserted} · "
                        f"duplicate={r.duplicates} · embedded={r.embedded}"
                    )
                progress.empty()
                status.update(
                    label=f"Imported {len(new_ids)} new row(s).",
                    state="complete",
                )
            if new_ids:
                st.session_state["data_pinned_ids"] = new_ids
                st.rerun()
            else:
                st.info("Nothing new — all records were duplicates.")

    pinned_ids: list[int] = st.session_state.get("data_pinned_ids") or []
    if pinned_ids:
        pin_cols = st.columns([5, 1])
        pin_cols[0].info(
            f"📌 Showing **{len(pinned_ids)} record(s)** from your last import. "
            "Filters below are ignored while pinned."
        )
        if pin_cols[1].button("Unpin", key="data_unpin_btn"):
            st.session_state.pop("data_pinned_ids", None)
            st.rerun()

    # Date range: same preset row as the Dashboard tab so the UI stays
    # consistent. Default is "All-time" here (the Data tab is for
    # exploration / relabelling, where loading the full history is the
    # natural starting point), whereas the Dashboard defaults to a
    # 3-month window for glance-readability.
    data_preset_col, data_from_col, data_to_col = st.columns([6, 1.5, 1.5])
    with data_preset_col:
        data_preset = st.radio(
            "Date range",
            _DASHBOARD_PRESETS,
            index=_DASHBOARD_PRESETS.index("All-time"),
            horizontal=True,
            key="data_date_preset",
        )
    if data_preset == "Custom":
        with data_from_col:
            custom_from = st.date_input(
                "From", value=None, key="data_from"
            )
        with data_to_col:
            custom_to = st.date_input(
                "To", value=None, key="data_to"
            )
        date_from, date_to = _dashboard_date_range(
            data_preset, custom_from, custom_to
        )
    else:
        date_from, date_to = _dashboard_date_range(data_preset)

    # Free-form search lives on its own row below the date picker so the
    # picker isn't squeezed when the preset row stays full-width.
    search_text = st.text_input(
        "Search all fields",
        value="",
        key="data_quick_search",
        placeholder="e.g. food, *aldi*, REWE*Berlin",
        help=(
            "Case-insensitive substring match across Counterparty, "
            "Verwendungszweck, Category, Source, IBAN, Umsatztyp, ID "
            "and Amount. Use `*` as a wildcard (e.g. `rewe*berlin`). "
            "Multiple terms separated by space must all match somewhere "
            "in the row."
        ),
    )
    include_income = True  # always loaded; filter via Amount column header

    # Extended-columns toggle was here but moved into the top action bar so
    # it sits next to the staging actions. We read its value out of session
    # state since the widget is rendered AFTER the grid is configured.
    extended = bool(st.session_state.get("data_extended", False))

    # Bump the AgGrid widget key when the free-form search text changes so
    # the grid re-initializes with the new (smaller) row set instead of
    # holding onto its prior data via the client_wins sync.
    if st.session_state.get("data_quick_search_prev") != search_text:
        st.session_state["data_quick_search_prev"] = search_text
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )
    # Same trick for the date-range preset: one radio click can swap
    # multiple SQL params at once, so force the grid to reload instead
    # of carrying over the prior date window's rows.
    date_range_signature = (
        data_preset,
        date_from.isoformat() if date_from else None,
        date_to.isoformat() if date_to else None,
    )
    if st.session_state.get("data_date_range_prev") != date_range_signature:
        st.session_state["data_date_range_prev"] = date_range_signature
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )
    # Bump the AgGrid widget key when the toggle flips so column visibility
    # actually changes (client_wins sync otherwise ignores the new options).
    if st.session_state.get("data_extended_prev") != extended:
        st.session_state["data_extended_prev"] = extended
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )

    # AgGrid handles category/source/text/amount filters via per-column header
    # filters — no need for them up here.
    sql, params = _build_data_query(
        date_from or None, date_to or None,
        [],          # picked_cats handled by AgGrid header filter
        "all",       # source handled by AgGrid header filter
        "",          # search handled by AgGrid header filter
        None, None,  # amount min/max handled by AgGrid header filter
        include_income,
        pinned_ids=pinned_ids if pinned_ids else None,
    )
    df = pd.read_sql_query(sql, conn, params=params)
    if not df.empty:
        df["buchungsdatum"] = pd.to_datetime(df["buchungsdatum"]).dt.strftime("%Y-%m-%d")
        df["category"] = df["category"].fillna("(unkategorisiert)")
        df["confidence"] = df["confidence"].apply(
            lambda v: f"{float(v):.2f}" if v is not None and v == v else ""
        )
        df["label_source"] = df["label_source"].fillna("")
        df["src"] = df["label_source"].map(
            {"user": "✅ user", "model": "🤖 model"}
        ).fillna("")  # unlabeled rows -> empty Source cell
    else:
        df["src"] = ""
        df["label_source"] = ""
    # Carry the original (DB) category alongside so the JS cellStyle and our
    # Python diff can spot pending edits without a second query.
    df["_orig_category"] = df["category"]

    # Show the empty-string sentinel rather than "(unkategorisiert)" in the
    # editable cell so the dropdown starts blank for unlabeled rows.
    df.loc[df["category"] == "(unkategorisiert)", "category"] = ""

    # --- Free-form search: case-insensitive substring across many columns
    # with `*` as a wildcard. Filter the DataFrame BEFORE we hand it to
    # AgGrid so the grid never sees the rows the user filtered out.
    if search_text and search_text.strip() and not df.empty:
        import re
        text_cols = [
            "id", "buchungsdatum", "counterparty", "zahlungspflichtiger",
            "verwendungszweck", "betrag_€", "category", "_orig_category",
            "src", "umsatztyp", "iban",
        ]
        cols_present = [c for c in text_cols if c in df.columns]

        # Stringify the haystack once per row. df.astype(str) is unreliable
        # when columns contain NaN / None / mixed dtypes (it can leak a
        # float into the .agg(' · '.join), so coerce per-cell explicitly).
        def _safe_str(v) -> str:
            if v is None:
                return ""
            try:
                if isinstance(v, float) and pd.isna(v):
                    return ""
            except Exception:
                pass
            return str(v)

        haystack = df[cols_present].apply(
            lambda row: " · ".join(_safe_str(v) for v in row),
            axis=1,
        ).str.lower()
        mask = pd.Series([True] * len(df), index=df.index)
        # Lowercase the search terms to match the lowercased haystack.
        for term in search_text.strip().lower().split():
            pattern = re.escape(term).replace(r"\*", ".*")
            try:
                term_mask = haystack.str.contains(pattern, regex=True, na=False)
            except re.error:
                term_mask = haystack.str.contains(re.escape(term), regex=True, na=False)
            mask &= term_mask
        df = df[mask].reset_index(drop=True)

    # Category lookups
    _all_cat_objs = list_categories(conn)
    all_cat_names = sorted(c.name for c in _all_cat_objs)
    cat_id_by_name = {c.name: c.id for c in _all_cat_objs}
    cat_name_by_id = {c.id: c.name for c in _all_cat_objs}

    def _autolabel_predictions(target_ids: list[int], label_text: str):
        """Run cascade on `target_ids`, return predictions. No DB writes."""
        if not target_ids:
            return []
        with st.status(label_text, expanded=True) as status:
            status.write(f"loading embedding model `{cfg.embedding_model}`…")
            emb = _embedder()
            cascade = CategorizationCascade(conn, cfg, emb)
            status.write("fitting cascade on the latest user labels…")
            try:
                cascade.fit()
            except Exception as e:
                status.write(f"  fit skipped: {e}")
            status.write(f"predicting {len(target_ids)} record(s)…")
            progress = st.progress(0, text=f"0 / {len(target_ids)}")

            def _cb(done: int, total: int) -> None:
                progress.progress(done / total, text=f"{done} / {total}")

            preds = cascade.predict_batch(target_ids, progress_callback=_cb)
            progress.empty()
            from collections import Counter
            stages = Counter(p.stage for p in preds)
            n_with_cat = sum(1 for p in preds if p.category_id is not None)
            status.update(
                label=(
                    f"staged {n_with_cat}/{len(preds)} prediction(s) · "
                    + ", ".join(f"{k}={v}" for k, v in stages.items())
                    + " — review highlighted cells and click Save changes to commit"
                ),
                state="complete",
            )
        return preds

    # --- AgGrid setup ------------------------------------------------------
    from st_aggrid import (  # local import keeps non-UI imports light
        AgGrid,
        DataReturnMode,
        GridOptionsBuilder,
        GridUpdateMode,
    )
    from st_aggrid.shared import JsCode

    # ---- Pending-edit stashes (persist across reruns; cleared on Save) ----
    # 1) User-typed cell edits: {eid -> cat_name or None (clear)}. These
    #    represent the user's manual choice — saved as `source='user'`.
    # 2) Auto-label predictions: {eid -> {cat, conf, stage}}. Saved as
    #    `source='model'` with the recorded confidence.
    # 3) Promote-to-user staging: set[eid]. Re-saves the row's current
    #    (model) category as a `source='user'` label.
    user_typed: dict[int, str | None] = dict(
        st.session_state.get("data_user_typed_edits", {})
    )
    autolabel_stage: dict[int, dict] = dict(
        st.session_state.get("data_autolabel_stage", {})
    )
    promote_stage: set[int] = set(
        st.session_state.get("data_promote_stage", set()) or set()
    )

    # Hidden helper columns the JS valueGetters / cellStyles read.
    df["_user_pending"] = False
    df["_stage_cat"] = ""
    df["_stage_conf"] = ""
    df["_stage_stage"] = ""
    df["_promote"] = False

    if not df.empty:
        for i in range(len(df)):
            eid = int(df.iloc[i]["id"])
            # Priority: user-typed > auto-label stash. Promote is orthogonal
            # (doesn't change the category cell, only the source tag).
            if eid in user_typed:
                val = user_typed[eid]
                df.iat[i, df.columns.get_loc("category")] = val if val is not None else ""
                df.iat[i, df.columns.get_loc("_user_pending")] = True
            elif eid in autolabel_stage:
                item = autolabel_stage[eid]
                cat_name = item.get("cat", "") or ""
                df.iat[i, df.columns.get_loc("category")] = cat_name
                df.iat[i, df.columns.get_loc("_stage_cat")] = cat_name
                df.iat[i, df.columns.get_loc("_stage_conf")] = item.get("conf", "")
                df.iat[i, df.columns.get_loc("_stage_stage")] = item.get("stage", "")
            if eid in promote_stage:
                df.iat[i, df.columns.get_loc("_promote")] = True

    # Selection to restore right after a key-bumped rerun (auto-label /
    # promote / extended toggle). The onFirstDataRendered JS handler below
    # reads `_pre_selected` off each row and ticks it.
    pre_select_eids: set[int] = set(
        st.session_state.pop("data_aggrid_pre_select_eids", set()) or set()
    )
    df["_pre_selected"] = df["id"].astype(int).isin(pre_select_eids) \
        if not df.empty else False

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        editable=False,
        floatingFilter=False,
    )
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        header_checkbox=True,
        header_checkbox_filtered_only=True,
    )
    gb.configure_grid_options(
        rowHeight=28,
        headerHeight=34,
        animateRows=False,
        suppressFieldDotNotation=True,
        domLayout="normal",
    )

    # Hide internal/duplicate columns.
    for hid in ("_orig_category", "category_id", "label_source", "_pre_selected",
                "_user_pending", "_stage_cat", "_stage_conf", "_stage_stage",
                "_promote"):
        if hid in df.columns:
            gb.configure_column(hid, hide=True)

    # Re-tick rows whose data has `_pre_selected: true` once the grid is
    # ready. Used to restore selection after a key bump (e.g. auto-label).
    gb.configure_grid_options(onFirstDataRendered=JsCode("""
        function(params) {
          var nodesToSelect = [];
          params.api.forEachNode(function(node) {
            if (node.data && node.data._pre_selected === true) {
              nodesToSelect.push(node);
            }
          });
          if (nodesToSelect.length) {
            params.api.setNodesSelected({nodes: nodesToSelect, newValue: true});
          }
        }
    """))

    gb.configure_column("id", header_name="ID", width=80, filter="agNumberColumnFilter")
    gb.configure_column("buchungsdatum", header_name="Date", width=110,
                        filter="agDateColumnFilter")
    gb.configure_column("counterparty", header_name="Counterparty", width=200,
                        filter="agTextColumnFilter")
    gb.configure_column("zahlungspflichtiger", header_name="Payer", width=170,
                        filter="agTextColumnFilter")
    gb.configure_column("verwendungszweck", header_name="Verwendungszweck",
                        width=320, filter="agTextColumnFilter")
    amount_cell_style = JsCode("""
        function(p){
          if (p.value == null) return null;
          var v = Number(p.value);
          if (v < 0) return {'color': '#d62728', 'fontWeight': '500'};
          if (v > 0) return {'color': '#157347', 'fontWeight': '500'};
          return null;
        }
    """)
    gb.configure_column(
        "betrag_€", header_name="Amount €", width=110,
        type=["numericColumn"],
        filter="agNumberColumnFilter",
        valueFormatter=JsCode(
            "function(p){"
            "  if (p.value == null) return '';"
            "  return Number(p.value).toLocaleString('de-DE', "
            "    {minimumFractionDigits: 2, maximumFractionDigits: 2});"
            "}"
        ),
        cellStyle=amount_cell_style,
    )
    # Category: editable single-click dropdown + yellow highlight when changed.
    # Highlight every pending state: user-typed edit, auto-label stash, OR
    # promote-to-user (the last one doesn't change the cell value, only the
    # source tag — we still want a visual cue on the row).
    pending_cell_style = JsCode("""
        function(params) {
          if (!params.data) return null;
          var d = params.data;
          if (d._user_pending === true) return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          if (d._stage_cat && d._stage_cat !== '') return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          if (d._promote === true) return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          return null;
        }
    """)
    cat_cell_style = pending_cell_style
    # Plain agSelectCellEditor: opens an inline native-style dropdown of all
    # categories. Once open you can press the first letter(s) of a category
    # to jump to it (browser-native typeahead on <select>), then Enter or
    # click to commit. agRichSelectCellEditor offers fuzzy filter-as-you-
    # type but its popup gets clipped by the streamlit-aggrid iframe.
    import json as _json
    _valid_cats_js = _json.dumps([""] + all_cat_names)
    category_value_parser = JsCode(
        "function(params){"
        f"  var valid = {_valid_cats_js};"
        "  if (valid.indexOf(params.newValue) >= 0) return params.newValue;"
        "  return params.oldValue;"
        "}"
    )
    gb.configure_column(
        "category", header_name="Category", width=170,
        editable=True,
        singleClickEdit=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": [""] + all_cat_names},
        valueParser=category_value_parser,
        cellStyle=cat_cell_style,
        filter="agTextColumnFilter",
    )
    # Source column: distinguishes the three pending kinds so the user can
    # see EXACTLY what Save will do. If the visible category differs from
    # the staged prediction, the user has overridden it -- show "→ user"
    # instead of the cascade stage.
    src_value_getter = JsCode("""
        function(params) {
          if (!params.data) return '';
          var d = params.data;
          if (d._user_pending === true) {
            var v = d.category == null ? '' : d.category;
            return v === '' ? '📝 → clear' : '📝 → user';
          }
          if (d._stage_cat && d._stage_cat !== '') {
            var cur = d.category == null ? '' : d.category;
            if (cur !== d._stage_cat) {
              return '📝 → user';   // user overrode the prediction
            }
            // Untouched staged prediction -- include the cascade stage so
            // it's obvious WHICH part of the pipeline fired.
            return '🤖 ' + (d._stage_stage || 'model');
          }
          if (d._promote === true) {
            return '📝 → user';
          }
          return d.src || '';
        }
    """)
    gb.configure_column("src", header_name="Source", width=170,
                        filter="agTextColumnFilter",
                        valueGetter=src_value_getter,
                        cellStyle=pending_cell_style)

    # Confidence column: show the staged confidence ONLY when the row's
    # category still equals the prediction. User overrode -> blank.
    conf_value_getter = JsCode("""
        function(params) {
          if (!params.data) return '';
          var d = params.data;
          if (d._user_pending === true) return '';
          if (d._stage_cat && d._stage_cat !== '') {
            var cur = d.category == null ? '' : d.category;
            if (cur !== d._stage_cat) return '';   // user override
            return d._stage_conf || '';
          }
          return d.confidence || '';
        }
    """)
    gb.configure_column("confidence", header_name="Conf", width=80,
                        filter="agNumberColumnFilter",
                        valueGetter=conf_value_getter,
                        cellStyle=pending_cell_style)

    # IBAN: always visible (used to filter / search for a specific account).
    # Country-code-only column was dropped: the country is encoded in the
    # first two characters of the full IBAN, and `iban_is_foreign` already
    # serves the "foreign vs DE" classifier signal.
    gb.configure_column("iban", header_name="IBAN", width=220,
                        filter="agTextColumnFilter")

    ext_columns = ("umsatztyp", "iban_is_foreign",
                   "has_glaeubiger_id", "mandatsreferenz_present")
    if extended:
        gb.configure_column("umsatztyp", header_name="Umsatztyp", width=120)
        gb.configure_column("iban_is_foreign", header_name="Foreign?", width=80)
        gb.configure_column("has_glaeubiger_id", header_name="Gläubiger?", width=90)
        gb.configure_column("mandatsreferenz_present", header_name="Mandat?", width=90)
    else:
        for c in ext_columns:
            if c in df.columns:
                gb.configure_column(c, hide=True)

    # Restore filter + sort + column widths/visibility across key bumps
    # (auto-label / save / revert / extended toggle). AG-Grid's initialState
    # is applied once on grid init; we capture it after each render and
    # re-pass it on the next one. Selection is intentionally NOT restored
    # from the saved state -- it's managed separately via
    # `data_aggrid_pre_select_eids` only when an action requires it.
    _saved_grid_state = st.session_state.get("data_aggrid_grid_state")
    if isinstance(_saved_grid_state, dict) and _saved_grid_state:
        _init_state = {k: v for k, v in _saved_grid_state.items()
                       if k != "rowSelection"}
        if _init_state:
            gb.configure_grid_options(initialState=_init_state)

    grid_options = gb.build()

    # ------ Action bar (rendered TWICE: above + below the table) ----------
    # Both bars use the CURRENT render's counts so their button states stay
    # perfectly in sync. The top bar lives in a placeholder reserved here
    # but filled AFTER the grid renders (so we know edited_df / sel_ids).

    def _render_action_buttons(prefix: str, counts: dict):
        """Render the 6-button action bar (no caption).
        Returns (save, revert, autolabel, promote, see_details) booleans."""
        n_pend = int(counts.get("n_pending", 0))
        n_sel = int(counts.get("n_selected", 0))
        can_prom = bool(counts.get("can_promote", False))
        can_insp = bool(counts.get("can_inspect", False))

        cols = st.columns([1.6, 1.6, 1.5, 1.9, 1.4, 1.4, 0.6])
        save_c = cols[0].button(
            f"💾 Save Changes ({n_pend})" if n_pend else "💾 Save Changes",
            type="tertiary", disabled=n_pend == 0,
            key=f"data_save_{prefix}_btn",
            help="Commit highlighted rows.",
        )
        revert_c = cols[1].button(
            f"↩ Revert Changes ({n_pend})" if n_pend else "↩ Revert Changes",
            type="tertiary", disabled=n_pend == 0,
            key=f"data_revert_{prefix}_btn",
            help="Discard every highlighted pending change without saving.",
        )
        auto_c = cols[2].button(
            f"🤖 Auto Label ({n_sel})" if n_sel else "🤖 Auto Label",
            type="tertiary", disabled=n_sel == 0,
            key=f"data_autolabel_{prefix}_btn",
            help=(
                "Run the cascade on selected rows. Predictions appear "
                "highlighted; user-labeled rows are skipped."
            ),
        )
        promote_c = cols[3].button(
            f"⬆️ Promote to User Label ({n_sel})" if n_sel else "⬆️ Promote to User Label",
            type="tertiary", disabled=not can_prom,
            key=f"data_promote_{prefix}_btn",
            help=(
                "Re-save selected rows as `source='user'`. Disabled if any "
                "selected row is uncategorized."
            ),
        )
        see_c = cols[4].button(
            "👁 See Details",
            type="tertiary", disabled=not can_insp,
            key=f"data_see_details_{prefix}_btn",
            help="Full record popup. Active only when exactly one row is selected.",
        )
        # Extended Columns toggle (two widgets share master state via on_change).
        cols[5].toggle(
            "Extended Columns",
            value=bool(st.session_state.get("data_extended", False)),
            key=f"data_extended_{prefix}_toggle",
            on_change=_sync_extended_from,
            args=(f"data_extended_{prefix}_toggle",),
            help="Reveal Umsatztyp / IBAN / SEPA flag columns.",
        )
        return save_c, revert_c, auto_c, promote_c, see_c

    def _render_caption(counts: dict) -> None:
        n_pend = int(counts.get("n_pending", 0))
        n_sel = int(counts.get("n_selected", 0))
        n_row = int(counts.get("n_rows", 0))
        st.markdown(
            f"<div style='text-align:center; font-size:0.78rem; "
            f"opacity:0.75; margin: 0.25rem 0 0.6rem 0;'>"
            f"{n_pend} unsaved changes  ·  {n_sel} of {n_row} selected"
            f"</div>",
            unsafe_allow_html=True,
        )

    def _sync_extended_from(src_key: str) -> None:
        """Mirror a toggle's value into the master state (data_extended)."""
        st.session_state["data_extended"] = bool(st.session_state.get(src_key, False))

    # Force both Extended toggles to reflect the master value BEFORE either
    # widget is instantiated this run (Streamlit forbids writing to a widget
    # key after the widget has rendered, but allows it before).
    _master_ext = bool(st.session_state.get("data_extended", False))
    st.session_state["data_extended_top_toggle"] = _master_ext
    st.session_state["data_extended_bot_toggle"] = _master_ext

    # AgGrid widget key. Bumped on action clicks to force a fresh init.
    aggrid_key = "data_aggrid_v" + str(st.session_state.get("data_aggrid_seed", 0))

    # ---- Top bar: render INLINE before the grid using the LATEST state ---
    # st.empty()-based placeholder caused a visible "wipe + refill" flash
    # because the slot starts at 0px height each rerun. Rendering inline
    # lets Streamlit's React reconciler update the buttons in place across
    # reruns -- no clear, no shift.
    #
    # streamlit-aggrid's React component writes its latest grid state
    # (nodes + isSelected + cell values) into st.session_state[aggrid_key]
    # IMMEDIATELY when the user interacts (before the rerun fires). So
    # reading it here at the top of the script gives counts that already
    # reflect the click that triggered this rerun -- no lag.
    def _counts_from_grid_state(key: str) -> dict:
        out = {"n_pending": 0, "n_selected": 0, "n_rows": 0,
               "can_promote": False, "can_inspect": False, "sel_single_eid": None}
        raw = st.session_state.get(key)
        if not isinstance(raw, dict):
            return out
        nodes = raw.get("nodes") or []
        if not isinstance(nodes, list):
            return out
        out["n_rows"] = len(nodes)
        sel_eids: list[int] = []
        n_pending = 0
        for node in nodes:
            if not isinstance(node, dict):
                continue
            d = node.get("data") or {}
            try:
                eid = int(d.get("id"))
            except (TypeError, ValueError):
                continue
            if node.get("isSelected") is True:
                sel_eids.append(eid)
            cat = str(d.get("category") or "").strip()
            orig = str(d.get("_orig_category") or "").strip()
            orig_norm = "" if orig == "(unkategorisiert)" else orig
            if cat != orig_norm:
                if (cat == "" and orig_norm != "") or cat in cat_id_by_name:
                    n_pending += 1
        out["n_pending"] = n_pending
        out["n_selected"] = len(sel_eids)
        # can_promote: every selected row has a category to promote
        if sel_eids:
            sel_set = set(sel_eids)
            ok = True
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                d = node.get("data") or {}
                try:
                    eid_n = int(d.get("id"))
                except (TypeError, ValueError):
                    continue
                if eid_n not in sel_set:
                    continue
                orig = str(d.get("_orig_category") or "").strip()
                orig_norm = "" if orig == "(unkategorisiert)" else orig
                cat = str(d.get("category") or "").strip()
                if not (cat or orig_norm):
                    ok = False
                    break
            out["can_promote"] = ok
        out["can_inspect"] = out["n_selected"] == 1
        if out["can_inspect"]:
            out["sel_single_eid"] = sel_eids[0]
        return out

    _top_counts = _counts_from_grid_state(aggrid_key)
    save_top_clicked, revert_top_clicked, auto_label_top_clicked, \
        promote_top_clicked, see_details_top_clicked = \
        _render_action_buttons("top", _top_counts)
    _render_caption(_top_counts)
    response = AgGrid(
        df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=(
            GridUpdateMode.VALUE_CHANGED
            | GridUpdateMode.SELECTION_CHANGED
            | GridUpdateMode.FILTERING_CHANGED
            | GridUpdateMode.SORTING_CHANGED
        ),
        allow_unsafe_jscode=True,
        height=680,
        theme="streamlit",
        reload_data=False,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        key=aggrid_key,
    )
    # Capture the full grid_state (includes filterModel, sortModel, columns)
    # so the NEXT render's initialState can restore it.
    _gs = response.grid_state
    if _gs:
        st.session_state["data_aggrid_grid_state"] = _gs

    # AgGrid response: edited df + selected rows (1.x returns DataFrames).
    edited_df = response.get("data")
    if edited_df is None:
        edited_df = df
    selected_rows = response.get("selected_rows")
    if isinstance(selected_rows, pd.DataFrame):
        sel_ids = [int(x) for x in selected_rows["id"].tolist()] \
            if not selected_rows.empty else []
    elif isinstance(selected_rows, list):
        sel_ids = [int(r["id"]) for r in selected_rows]
    else:
        sel_ids = []

    # Pending edits derived from the grid state. Each entry is
    # (eid, action_dict). action is one of:
    #   * "set_user"  cat_id           -> add user label
    #   * "set_model" cat_id, conf    -> add model label
    #   * "clear"                       -> delete every label for the row
    pending_updates: list[tuple[int, dict]] = []
    pending_user_typed_now: dict[int, str | None] = {}
    if edited_df is not None and not edited_df.empty:
        for i in range(len(edited_df)):
            row = edited_df.iloc[i]
            eid = int(row["id"])
            new_cat = str(row.get("category", "") or "").strip()
            orig_cat = str(row.get("_orig_category", "") or "").strip()
            orig_norm = "" if orig_cat == "(unkategorisiert)" else orig_cat

            # What we passed INTO AgGrid for this row's category (DB value
            # + any user_typed / autolabel overlay). If AgGrid sends back a
            # different value, the user actively edited the cell -- so we
            # MUST treat the save as a user label even if the new value
            # happens to coincide with the staged prediction by chance.
            input_cat = ""
            if i < len(df):
                raw = df.iloc[i].get("category", "")
                input_cat = str(raw or "").strip()
            user_actually_edited = (input_cat != new_cat)

            # Promote stash: always wants a user label of the row's current
            # (model) category. If the user ALSO edited the cell, that wins.
            if eid in promote_stage:
                cat_to_save = new_cat or orig_norm
                cid = cat_id_by_name.get(cat_to_save)
                if cid is not None:
                    pending_updates.append((eid, {"action": "set_user", "cat_id": cid}))
                continue

            if new_cat == orig_norm and not user_actually_edited:
                continue  # no change

            if new_cat == "":
                # Blanking a labeled row -> delete labels.
                if orig_norm:
                    pending_updates.append((eid, {"action": "clear"}))
                    pending_user_typed_now[eid] = None
                continue

            if new_cat not in cat_id_by_name:
                continue  # invalid value (shouldn't happen with valueParser)

            cid = cat_id_by_name[new_cat]

            # If the user ACTIVELY touched this cell, save as user -- even
            # if the new value matches the staged prediction (they're now
            # confirming with intent, not just accepting the cascade).
            if user_actually_edited:
                pending_updates.append((eid, {"action": "set_user", "cat_id": cid}))
                pending_user_typed_now[eid] = new_cat
                continue

            # Untouched cell. If matches staged auto-label prediction, save
            # as model with the recorded confidence.
            stage_item = autolabel_stage.get(eid)
            if stage_item and stage_item.get("cat") == new_cat:
                conf_str = stage_item.get("conf", "")
                try:
                    conf = float(conf_str)
                except (TypeError, ValueError):
                    conf = None
                pending_updates.append((eid, {
                    "action": "set_model",
                    "cat_id": cid,
                    "confidence": conf,
                }))
            else:
                # Fallback: differs from DB without an edit or a stage --
                # treat as user just to be safe.
                pending_updates.append((eid, {"action": "set_user", "cat_id": cid}))
                pending_user_typed_now[eid] = new_cat

    # ---- Bulk Category: pick a value in one selected row -> all selected ---
    # If 2+ rows are selected AND the user edited the Category cell of one
    # of them, propagate the new value to every selected row. Detected by
    # finding the FIRST selected row whose displayed category differs from
    # the value we passed in (i.e. the user changed it this render).
    # Implementation: stash the value as user-typed for every selected eid,
    # bump the grid key, re-render (selection is preserved via the existing
    # pre_select_eids mechanism).
    sel_set_now: set[int] = {int(x) for x in sel_ids}
    if len(sel_set_now) >= 2 and edited_df is not None and not edited_df.empty:
        bulk_cat: str | None = None
        for i in range(len(edited_df)):
            eid = int(edited_df.iloc[i]["id"])
            if eid not in sel_set_now:
                continue
            new_cat = str(edited_df.iloc[i].get("category", "") or "").strip()
            input_cat = ""
            if i < len(df):
                raw = df.iloc[i].get("category", "")
                input_cat = str(raw or "").strip()
            if new_cat == input_cat:
                continue  # not edited this render
            # Edited! Only propagate real-category picks or explicit clears
            # (empty string when the row used to be labeled).
            if new_cat == "" or new_cat in cat_id_by_name:
                bulk_cat = new_cat
                break
        if bulk_cat is not None:
            stash = dict(st.session_state.get("data_user_typed_edits", {}))
            for eid_n in sel_set_now:
                stash[int(eid_n)] = bulk_cat if bulk_cat != "" else None
            st.session_state["data_user_typed_edits"] = stash
            # Keep the selection so the user can keep refining or Save.
            st.session_state["data_aggrid_pre_select_eids"] = sel_set_now
            st.session_state["data_aggrid_seed"] = (
                st.session_state.get("data_aggrid_seed", 0) + 1
            )
            st.toast(
                f"applied {bulk_cat or '(clear)'} to {len(sel_set_now)} selected rows"
            )
            st.rerun()

    n_pending = len(pending_updates)
    n_selected = len(sel_ids)
    n_rows = len(edited_df) if edited_df is not None else 0

    # Promote is only valid when every selected row already has a category
    # to promote (DB original OR a staged value). If even one selected row
    # is uncategorized, disable the button so the user can't accidentally
    # promote nothing.
    sel_set = {int(x) for x in sel_ids}
    n_promotable = 0
    if sel_set and edited_df is not None and not edited_df.empty:
        for i in range(len(edited_df)):
            eid = int(edited_df.iloc[i]["id"])
            if eid not in sel_set:
                continue
            orig_cat = str(edited_df.iloc[i].get("_orig_category", "") or "").strip()
            orig_norm = "" if orig_cat == "(unkategorisiert)" else orig_cat
            new_cat = str(edited_df.iloc[i].get("category", "") or "").strip()
            if new_cat or orig_norm:
                n_promotable += 1
    can_promote = n_selected > 0 and n_promotable == n_selected

    def _capture_user_typed_into_stash() -> None:
        """Merge the current render's manual edits into the user-typed
        stash so they survive a key bump (auto-label / promote / extended)."""
        if not pending_user_typed_now:
            return
        existing = dict(st.session_state.get("data_user_typed_edits", {}))
        existing.update(pending_user_typed_now)
        st.session_state["data_user_typed_edits"] = existing

    # --- Render BOTH action bars with the freshly-computed counts ---------
    can_inspect = n_selected == 1
    sel_single_eid = int(sel_ids[0]) if can_inspect else None
    _current_counts = {
        "n_pending": n_pending,
        "n_selected": n_selected,
        "n_rows": n_rows,
        "can_promote": can_promote,
        "can_inspect": can_inspect,
        "sel_single_eid": sel_single_eid,
    }

    # Bottom: caption ABOVE buttons (closer to the table, per spec).
    _render_caption(_current_counts)
    save_bot_clicked, revert_bot_clicked, auto_label_bot_clicked, \
        promote_bot_clicked, see_details_bot_clicked = \
        _render_action_buttons("bot", _current_counts)

    # Top OR bottom click of the same action.
    auto_label_clicked = auto_label_top_clicked or auto_label_bot_clicked
    promote_clicked = promote_top_clicked or promote_bot_clicked
    see_details_clicked = see_details_top_clicked or see_details_bot_clicked

    # --- Action handlers ---------------------------------------------------
    if auto_label_clicked and sel_ids:
        # Preserve any user-typed cell edits across the upcoming key bump.
        _capture_user_typed_into_stash()
        # Skip rows whose latest label is already 'user' — protect prior
        # confirmations. Show a toast with the skipped count.
        label_source_by_id = dict(zip(
            edited_df["id"].astype(int).tolist(),
            edited_df.get("label_source", pd.Series([""] * len(edited_df))).fillna("").tolist(),
            strict=True,
        ))
        eligible = [int(i) for i in sel_ids
                    if label_source_by_id.get(int(i), "") != "user"]
        skipped = len(sel_ids) - len(eligible)
        if not eligible:
            st.warning(
                f"all {skipped} selected row(s) already have a user label — "
                "nothing to auto-label. Use ⬆️ Promote to user if you want "
                "to re-stamp them, or clear them from Settings."
            )
        else:
            preds = _autolabel_predictions(
                eligible,
                f"auto-labeling {len(eligible)} record(s)"
                + (f" · skipping {skipped} user-labeled" if skipped else "")
                + "…",
            )
            stage = dict(st.session_state.get("data_autolabel_stage", {}))
            # User-typed edits take priority -- don't override them with a
            # prediction (the user-typed stash already wins on render).
            current_user_typed = set(
                st.session_state.get("data_user_typed_edits", {}).keys()
            )
            for p in preds:
                if p.category_id is None:
                    continue
                eid = int(p.expense_id)
                if eid in current_user_typed:
                    continue
                name = cat_name_by_id.get(p.category_id)
                if name:
                    stage[eid] = {
                        "cat": name,
                        "conf": f"{float(p.confidence):.2f}",
                        "stage": p.stage,
                    }
            st.session_state["data_autolabel_stage"] = stage
            st.session_state["data_aggrid_pre_select_eids"] = set(int(x) for x in sel_ids)
            st.session_state["data_aggrid_seed"] = (
                st.session_state.get("data_aggrid_seed", 0) + 1
            )
            st.rerun()

    if promote_clicked and sel_ids:
        _capture_user_typed_into_stash()
        existing = set(st.session_state.get("data_promote_stage", set()) or set())
        # Only meaningful for rows that currently HAVE a category (otherwise
        # there's nothing to re-stamp). Use edited_df / DB info to filter.
        elig_count = 0
        for i in range(len(edited_df)):
            eid = int(edited_df.iloc[i]["id"])
            if eid not in sel_ids:
                continue
            orig_cat = str(edited_df.iloc[i].get("_orig_category", "") or "").strip()
            orig_norm = "" if orig_cat == "(unkategorisiert)" else orig_cat
            new_cat = str(edited_df.iloc[i].get("category", "") or "").strip()
            visible_cat = new_cat or orig_norm
            if not visible_cat:
                continue
            existing.add(eid)
            elig_count += 1
        if elig_count == 0:
            st.warning(
                "selected rows have no category to promote -- pick a "
                "category first (or Auto-Label) then Promote to user."
            )
        else:
            st.session_state["data_promote_stage"] = existing
            st.session_state["data_aggrid_pre_select_eids"] = set(int(x) for x in sel_ids)
            st.session_state["data_aggrid_seed"] = (
                st.session_state.get("data_aggrid_seed", 0) + 1
            )
            st.rerun()

    if see_details_clicked and sel_single_eid is not None:
        st.session_state["data_inspect_open_eid"] = int(sel_single_eid)
        st.rerun()

    if revert_top_clicked or revert_bot_clicked:
        # Drop every pending stash + reset AgGrid so the input df comes
        # straight from the DB with no overlays.
        for k in ("data_autolabel_stage", "data_promote_stage",
                  "data_user_typed_edits"):
            st.session_state.pop(k, None)
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )
        st.toast(f"reverted {n_pending} pending change(s)")
        st.rerun()

    if save_top_clicked or save_bot_clicked:
        from expense_analyzer.storage.admin import clear_labels_for_expense

        n_user = 0
        n_model = 0
        n_cleared = 0
        for eid, act in pending_updates:
            a = act["action"]
            if a == "clear":
                clear_labels_for_expense(conn, int(eid))
                n_cleared += 1
            elif a == "set_user":
                add_label(conn, int(eid), int(act["cat_id"]), "user")
                n_user += 1
            elif a == "set_model":
                add_label(conn, int(eid), int(act["cat_id"]), "model",
                          confidence=act.get("confidence"))
                n_model += 1
        # Drain every stash; the next render reads a clean DB.
        for k in ("data_autolabel_stage", "data_promote_stage",
                  "data_user_typed_edits"):
            st.session_state.pop(k, None)
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )
        parts = []
        if n_user:
            parts.append(f"{n_user} user")
        if n_model:
            parts.append(f"{n_model} model")
        if n_cleared:
            parts.append(f"{n_cleared} cleared")
        st.toast("saved: " + ", ".join(parts) if parts else "nothing to save")
        st.rerun()



# --- Inspect dialog ----------------------------------------------------------
# Module-level so we can call it from inside the `with tab_data:` block via
# session_state without re-decorating each render.

def _on_inspect_dismiss() -> None:
    """Clear the open-eid sentinel when the dialog is dismissed via the X
    button or by clicking outside. Without this the next rerun re-renders
    the dialog because the session_state value is still set."""
    st.session_state.pop("data_inspect_open_eid", None)


@st.dialog("Record details", width="large", on_dismiss=_on_inspect_dismiss)
def _show_inspect_dialog(eid: int) -> None:
    row = conn.execute("SELECT * FROM expenses WHERE id = ?", (eid,)).fetchone()
    if row is None:
        st.warning(f"no record with id {eid}")
        if st.button("Close", key="inspect_close_missing"):
            st.session_state.pop("data_inspect_open_eid", None)
            st.rerun()
        return
    full_dict = dict(row)
    st.markdown(f"##### Record #{eid}")
    head = st.columns(3)
    head[0].write(f"**Date:** {full_dict.get('buchungsdatum')}")
    head[1].write(f"**Amount:** {full_dict.get('betrag_cents', 0) / 100:.2f} €")
    head[2].write(f"**Counterparty:** {full_dict.get('counterparty')}")
    if full_dict.get("verwendungszweck"):
        st.caption(full_dict["verwendungszweck"])
    meta_cols = st.columns(3)
    meta_cols[0].write(f"**Source file:** {full_dict.get('source_file') or '—'}")
    meta_cols[1].write(f"**IBAN:** {full_dict.get('iban') or '—'}")
    meta_cols[2].write(f"**Umsatztyp:** {full_dict.get('umsatztyp') or '—'}")

    note = get_note(conn, eid) or ""
    new_note = st.text_area("Note", value=note, key=f"inspect_note_{eid}")
    btn_cols = st.columns([1, 1, 4])
    if btn_cols[0].button("💾 Save note", key=f"inspect_save_note_{eid}"):
        set_note(conn, eid, new_note)
        st.toast("note saved")
    if btn_cols[1].button("Close", key=f"inspect_close_{eid}"):
        st.session_state.pop("data_inspect_open_eid", None)
        st.rerun()

    with st.expander("All fields", expanded=True):
        st.json({
            k: (v if not isinstance(v, bytes) else f"<{len(v)} bytes>")
            for k, v in full_dict.items()
        })


if st.session_state.get("data_inspect_open_eid"):
    _show_inspect_dialog(int(st.session_state["data_inspect_open_eid"]))


# ---------------------------------------------------------------------------
# Tab 4: Settings
# ---------------------------------------------------------------------------

with tab_settings:
    from expense_analyzer.config import save_user_config
    from expense_analyzer.features.model_registry import (
        EMBEDDING_MODELS,
        ZEROSHOT_MODELS,
        hf_cache_dir,
        is_downloaded,
        trigger_download,
    )

    # -------------------------------------------------------------------
    # H1: Settings
    # -------------------------------------------------------------------
    st.title("Settings")

    # H2 Compute Device
    st.header("Compute Device")
    st.write(f"**Device:** `{cfg.device}`")
    st.caption(f"HF cache: `{hf_cache_dir()}`")

    def _render_model_table(
        role_label: str,
        models,
        current_id: str,
        cfg_key: str,
        explanation: str,
    ) -> None:
        # Caller emits the section heading; we just render the table + picker.
        st.caption(explanation)
        rows = []
        for m in models:
            present, size_gb = is_downloaded(m.model_id)
            rows.append(
                {
                    "model_id": m.model_id,
                    "dim": m.dim if m.dim is not None else "—",
                    "languages": m.languages,
                    "downloaded": "✅" if present else "—",
                    "size_GB": round(size_gb, 2) if present else round(m.approx_size_mb / 1024, 2),
                    "notes": m.notes,
                    "active": "●" if m.model_id == current_id else "",
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "model_id": st.column_config.TextColumn("Model"),
                "downloaded": st.column_config.TextColumn("Cached"),
                "size_GB": st.column_config.NumberColumn(
                    "Size (GB)",
                    help="Actual on-disk if cached, else approximate download size.",
                    format="%.2f",
                ),
                "active": st.column_config.TextColumn("Active"),
            },
        )
        st.caption("Switch active model")
        # Collapse the selectbox label so the dropdown and both buttons share
        # the same top y-coordinate -- no manual st.write('') vertical hacks.
        sel_cols = st.columns([3, 1, 1])
        with sel_cols[0]:
            picked = st.selectbox(
                "model picker",
                [m.model_id for m in models],
                index=next((i for i, m in enumerate(models) if m.model_id == current_id), 0),
                key=f"model_pick_{cfg_key}",
                label_visibility="collapsed",
            )
        with sel_cols[1]:
            present, _ = is_downloaded(picked)
            dl_label = "Download" if not present else "Re-download"
            if st.button(dl_label, key=f"model_dl_{cfg_key}", width="stretch"):
                with st.status(f"Downloading {picked}...", expanded=True) as status:
                    role = "embedding" if cfg_key == "embedding_model" else "zeroshot"
                    try:
                        trigger_download(picked, role=role)
                        status.update(label=f"Downloaded {picked}", state="complete")
                    except Exception as e:
                        status.update(label=f"Download failed: {e}", state="error")
                st.rerun()
        with sel_cols[2]:
            if st.button(
                "Use this", key=f"model_use_{cfg_key}", type="primary",
                disabled=picked == current_id, width="stretch",
            ):
                save_user_config({cfg_key: picked}, data_dir=cfg.data_dir)
                st.cache_resource.clear()
                st.success(
                    f"`{cfg_key}` set to `{picked}`. Restart the UI for it to take effect: "
                    "`expense ui-restart`."
                )

    # H2 Embeddings
    st.header("Embeddings")
    _render_model_table(
        "Embedding model",
        EMBEDDING_MODELS,
        cfg.embedding_model,
        "embedding_model",
        explanation=(
            "Converts each expense's text "
            "(`counterparty_normalized` + ` | ` + `verwendungszweck_normalized`) "
            "into a fixed-dimensional vector. Those vectors power the "
            "**k-NN lookup** (finds the closest already-labeled expenses), the "
            "**supervised classifier** (logistic regression / random forest "
            "trained on your user labels), and the **category-similarity** "
            "stage (cosine match against embedded category names + "
            "descriptions). Pick a German-aware model for best results on DE "
            "bank text; larger models are more accurate but use more disk "
            "and run slower on CPU."
        ),
    )
    # H2 Zero-Shot
    st.header("Zero-Shot")
    _render_model_table(
        "Zero-shot model",
        ZEROSHOT_MODELS,
        cfg.zeroshot_model,
        "zeroshot_model",
        explanation=(
            "**Fallback only** — invoked when every earlier cascade stage "
            "(vendor exact match → k-NN → classifier → category similarity) "
            "comes back with low confidence. It does multilingual "
            "natural-language inference: it asks *\"does this expense text "
            "belong to category X?\"* for every category description and "
            "picks the best fit. Slow per call (especially on CPU) but "
            "rarely needed once you have a few user labels seeded; safe to "
            "leave on the default for most installs."
        ),
    )

    # -------------------------------------------------------------------
    # H1: Privacy
    # -------------------------------------------------------------------
    st.title("Privacy")
    st.write(f"Vendor web lookup enabled: **{cfg.vendor_lookup.enabled}**")
    if cfg.vendor_lookup.enabled:
        st.warning(
            "Vendor lookup is ON. Only `counterparty_normalized` is sent to "
            f"{cfg.vendor_lookup.backend}; never amount/IBAN/Verwendungszweck."
        )
    else:
        st.info("Vendor lookup is OFF. Set `vendor_lookup.enabled: true` in your config to enable.")

    # -------------------------------------------------------------------
    # H1: My Accounts (own IBANs)
    # -------------------------------------------------------------------
    st.title("My Accounts")
    st.caption(
        "Your own IBANs. Rows whose IBAN matches one listed here are "
        "marked **internal** (`iban_is_known_self = 1`) and become a "
        "signal the classifier can use to recognise transfers between "
        "your own accounts. Adding or removing an IBAN here retroactively "
        "re-flags every matching transaction."
    )
    from expense_analyzer.storage.own_ibans import (
        add_own_iban as _add_own_iban,
    )
    from expense_analyzer.storage.own_ibans import (
        list_own_ibans as _list_own_ibans,
    )
    from expense_analyzer.storage.own_ibans import (
        remove_own_iban as _remove_own_iban,
    )
    from expense_analyzer.storage.own_ibans import (
        update_label as _update_own_iban_label,
    )

    own_rows = _list_own_ibans(conn)
    if not own_rows:
        st.info(
            "No own IBANs registered yet. Add one below to start tagging "
            "internal transfers."
        )
    else:
        # Header strip + one row per IBAN with an inline-editable label
        # and a per-row delete.
        own_widths = [4, 3, 0.6]
        h = st.columns(own_widths)
        h[0].markdown("**IBAN**")
        h[1].markdown("**Label**")
        h[2].markdown("")
        for r in own_rows:
            row = st.columns(own_widths)
            row[0].code(r.iban, language=None)
            new_lbl = row[1].text_input(
                "label",
                value=r.label or "",
                key=f"own_iban_lbl_{r.iban}",
                label_visibility="collapsed",
                placeholder="(no label)",
            )
            if new_lbl != (r.label or ""):
                _update_own_iban_label(conn, r.iban, new_lbl)
            if row[2].button("✕", key=f"own_iban_del_{r.iban}",
                             help=f"Remove {r.iban}"):
                rep = _remove_own_iban(conn, r.iban)
                st.toast(
                    f"removed; cleared the flag on {rep.n_was_self} "
                    "transaction(s)."
                )
                st.rerun()

    # --- Add a new own-IBAN row ---------------------------------------
    st.markdown("**Add own IBAN**")
    add_cols = st.columns([4, 3, 1])
    new_iban = add_cols[0].text_input(
        "new iban",
        key="new_own_iban_iban",
        label_visibility="collapsed",
        placeholder="DE89 3704 0044 0532 0130 00",
    )
    new_label = add_cols[1].text_input(
        "new label",
        key="new_own_iban_label",
        label_visibility="collapsed",
        placeholder="Friendly name (optional)",
    )
    if add_cols[2].button("Add", type="primary", key="new_own_iban_add",
                          disabled=not new_iban.strip()):
        try:
            rep = _add_own_iban(conn, new_iban, label=new_label or None)
        except ValueError as e:
            st.error(f"refusing: {e}")
        else:
            st.toast(
                f"added; flagged {rep.n_now_self} existing transaction(s) "
                "as internal."
            )
            # Clear the inputs.
            for k in ("new_own_iban_iban", "new_own_iban_label"):
                st.session_state.pop(k, None)
            st.rerun()

    # -------------------------------------------------------------------
    # H1: Database
    # -------------------------------------------------------------------
    st.title("Database")

    import datetime as _dt
    import tempfile as _tempfile

    from expense_analyzer.storage.backup import (
        export_database,
        restore_database,
        validate_backup,
    )

    # H2 Stats
    st.header("Stats")
    db_path = cfg.db_path
    try:
        db_mtime = _dt.datetime.fromtimestamp(db_path.stat().st_mtime) \
            if db_path.exists() else None
    except OSError:
        db_mtime = None
    stat_cols = st.columns([1.5, 4])
    stat_cols[0].metric(
        "Last modified",
        db_mtime.strftime("%Y-%m-%d %H:%M") if db_mtime else "—",
    )
    stat_cols[1].caption(f"Path: `{db_path}`")

    # H2 Administration (wraps Download / Restore / Danger Zone)
    st.header("Administration")

    st.markdown("**Download Backup**")
    st.caption(
        "Download a complete copy of the SQLite database. Includes every "
        "ingested row, label, embedding, note, vendor-cache entry, and "
        "category. The file is a standard SQLite 3 DB -- open it in any "
        "SQLite browser, or re-import here on another machine."
    )
    # Render the backup bytes lazily into the download_button.  We use
    # SQLite's online backup API so this is safe even with the live UI
    # connection open.  TemporaryDirectory (rather than NamedTemporaryFile)
    # because the latter leaves the file handle locked on Windows even
    # after its `with` block exits -> export_database can't overwrite it.
    # ignore_cleanup_errors: on Windows, SQLite's handle release lags
    # after conn.close(), so the TemporaryDirectory's own rmdir at the end
    # of the `with` block can hit WinError 32. The directory eventually
    # gets cleaned up by the OS; we just don't want that to crash the UI.
    try:
        with _tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _td:
            _bk_tmp = Path(_td) / "backup.sqlite"
            export_database(conn, _bk_tmp)
            _bk_bytes = _bk_tmp.read_bytes()
        st.download_button(
            "⬇️ Download backup",
            data=_bk_bytes,
            file_name=(
                f"expense-analyzer-backup-{_dt.date.today().isoformat()}.sqlite"
            ),
            mime="application/x-sqlite3",
            key="db_backup_download",
            type="primary",
        )
    except Exception as e:
        st.error(f"backup failed: {e}")

    st.markdown("**Restore Backup**")
    st.caption(
        "Replace the **current** database with an uploaded `.sqlite` backup. "
        "A timestamped safety copy of the current DB is saved alongside it "
        "(`db.pre-restore.<ts>.sqlite`) so you can roll back manually if "
        "the restore turns out to be the wrong file."
    )
    upload = st.file_uploader(
        "Pick a backup file (.sqlite)",
        type=["sqlite", "db"],
        key="db_restore_uploader",
    )
    if upload is not None:
        # Persist the upload to a temp dir so validate_backup /
        # restore_database can read from a Path. TemporaryDirectory avoids
        # the Windows NamedTemporaryFile handle-lock pitfall; the cleanup
        # itself is best-effort because SQLite handles linger briefly.
        with _tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _td:
            _upload_path = Path(_td) / "uploaded_backup.sqlite"
            _upload_path.write_bytes(upload.getbuffer())
            result = validate_backup(_upload_path)
            if not result.ok:
                st.error("upload is not a valid backup: " + "; ".join(result.errors))
            else:
                st.success(
                    "valid backup — rows: "
                    + ", ".join(f"{k}={v}" for k, v in result.table_counts.items())
                )
                confirm_restore = st.text_input(
                    "Type `restore` to confirm",
                    key="confirm_db_restore",
                )
                if st.button("Restore from this backup", type="primary",
                             key="db_do_restore"):
                    if confirm_restore.strip().lower() != "restore":
                        st.error("type the confirmation phrase exactly")
                    else:
                        try:
                            # Close the cached live connection before
                            # swapping the file on disk (Windows holds
                            # locks otherwise).
                            try:
                                conn.close()
                            except Exception:
                                pass
                            _connect_cached.clear()
                            report = restore_database(
                                cfg.db_path, _upload_path, keep_safety=True,
                            )
                            st.success(
                                "restored: "
                                + ", ".join(
                                    f"{k}={v}" for k, v in report.table_counts.items()
                                )
                            )
                            if report.safety_copy:
                                st.info(f"safety copy: `{report.safety_copy}`")
                            # Drop the resource cache so next render reopens.
                            st.cache_resource.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"restore failed: {e}")

    # H3 Danger Zone (subheader = h3)
    st.subheader(":red[Danger Zone]")
    with st.expander("Delete User Labels", expanded=False):
        from expense_analyzer.storage.admin import delete_user_labels as _del_user_labels

        n_user_labels = conn.execute(
            "SELECT COUNT(*) AS n FROM labels WHERE source='user'"
        ).fetchone()["n"]
        st.write(
            f"Currently **{n_user_labels}** row(s) in `labels` with `source='user'`. "
            "Deleting them lets you re-run Auto-label across the whole DB without "
            "your previous confirmations dominating the cascade. Model labels stay, "
            "so rows that have both keep their visible category via the remaining "
            "model entry; rows that had **only** a user label become uncategorized."
        )
        confirm_user = st.text_input(
            "Type `delete user labels` to confirm", key="confirm_delete_user_labels"
        )
        if st.button("Delete all user labels", type="secondary"):
            if confirm_user.strip().lower() == "delete user labels":
                n = _del_user_labels(conn)
                st.success(f"deleted {n} user label row(s)")
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")

    with st.expander("Empty Database", expanded=False):
        st.write(
            "Deletes every row in `expenses`, `labels`, `notes`, `embeddings`, "
            "`vendor_cache` and `model_versions`. Categories and own-IBANs are kept."
        )
        confirm_data = st.text_input(
            "Type `clear data` to confirm", key="confirm_reset_data"
        )
        if st.button("Clear ingested data"):
            if confirm_data.strip().lower() == "clear data":
                report = reset_data(conn)
                st.success(
                    f"deleted {report.total} row(s) across {len(report.table_counts)} table(s)"
                )
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")

    with st.expander("Factory Reset (incl. category deletion)", expanded=False):
        st.write(
            "Wipes every table including categories and own-IBANs. The DB schema "
            "stays so you can immediately re-init."
        )
        confirm_all = st.text_input(
            "Type `factory reset` to confirm", key="confirm_reset_all"
        )
        if st.button("Factory reset"):
            if confirm_all.strip().lower() == "factory reset":
                report = reset_all(conn)
                st.success(
                    f"deleted {report.total} row(s) across {len(report.table_counts)} table(s)"
                )
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")
