"""Dashboard tab: headline stats, charts, records table.

Order of sections, top to bottom:
  1. Date-range preset radio (default: "Past 90 days").
  2. Headline tiles -- savings rate / income / expenses / to-savings,
     in a bordered container.
  3. Collapsible per-category charts (Summe Ausgaben + Monatlicher
     Saldo expanded by default).
  4. Records table (read-only; category cells coloured with the
     user-chosen colour from the Categories tab).
  5. All-time helper expanders -- recurring vendors + anomalies.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from expensa.storage.categories import list_categories, savings_category_names
from expensa.ui._components import chart_expander, date_preset_row, de_eur
from expensa.ui._shared import get_conn
from expensa.utils.colors import readable_text_color
from expensa.viz import (
    anomalies,
    bar_spend_by_category,
    categorization_mix,
    category_period_comparison,
    fixed_vs_variable,
    income_vs_expense_chart,
    month_to_date_pace,
    monthly_flow_by_category,
    monthly_income_vs_expense,
    period_totals,
    recurring_subscriptions,
    savings_flow,
    spend_by_category,
    stacked_monthly_by_category,
    stacked_weekly_by_category,
    upcoming_recurring,
    weekly_by_category,
)

# Shared SELECT for the records table at the bottom of the dashboard.
# The `latest_label` view defined in schema.sql is what gets joined --
# SQLite inlines it at plan time.
_DASHBOARD_RECORDS_SELECT = """
    SELECT
        e.id, e.buchungsdatum,
        e.counterparty,
        e.verwendungszweck,
        e.betrag_cents / 100.0 AS "betrag_€",
        COALESCE(c.name, '(unkategorisiert)') AS category,
        e.iban
    FROM expenses e
    LEFT JOIN latest_label ll ON ll.expense_id = e.id
    LEFT JOIN categories c ON c.id = ll.category_id
"""


def render() -> None:
    conn = get_conn()
    st.header("Dashboard")

    n_exp = conn.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
    if n_exp == 0:
        _render_empty_state(conn)
        return

    since, until = date_preset_row(key_prefix="dashboard")
    _maybe_low_data_hint(conn)
    savings = tuple(savings_category_names(conn))
    _render_headline_tiles(conn, since, until, savings)
    _render_insight_tiles(conn, savings)
    _render_top_movers(conn, since, until, savings)
    _render_upcoming_recurring(conn)
    _render_charts(conn, since, until, savings)
    st.divider()
    _render_records_table(conn, since, until)
    _render_alltime_helpers(conn)


def _render_empty_state(conn) -> None:
    """Onboarding shown when there are no expenses yet.

    A blank dashboard with a one-liner left new users guessing. This walks
    them through the three concrete first steps and notes whether default
    categories are already seeded so they know what's left to do.
    """
    n_cats = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
    st.info("**Welcome to Expensa!** No expenses yet — here's how to get started.")
    cat_step = (
        f"✅ **{n_cats} categories** are ready (manage them in the **Categories** tab)."
        if n_cats
        else "Add a few **categories** in the *Categories* tab (or seed the German defaults)."
    )
    st.markdown(
        "#### Quick start\n"
        "1. **Import data** — open the **Data** tab, expand *Import Data*, and "
        "drop one or more German bank-export CSVs (`;`-separated, comma decimal). "
        "On the command line: `expensa ingest path/to/export.csv`.\n"
        f"2. **Categories** — {cat_step}\n"
        "3. **Label & review** — label a handful of examples in the **Review** tab; "
        "the model then auto-categorizes the rest and surfaces low-confidence rows "
        "for you to confirm.\n\n"
        "Charts and stats appear here automatically once expenses are imported."
    )


def _maybe_low_data_hint(conn) -> None:
    """A gentle nudge when there are expenses but essentially no labels yet,
    so the dashboard's categories look empty for an understandable reason."""
    n_user = conn.execute(
        "SELECT COUNT(DISTINCT expense_id) AS n FROM labels WHERE source='user'"
    ).fetchone()["n"]
    if n_user == 0:
        st.caption(
            "ℹ️ No expenses are categorized yet — head to the **Review** tab to "
            "label a few examples and let the model take over."
        )


def _prev_since(since, until):
    """Start of the window immediately preceding ``[since, until]``."""
    from expensa.viz.data import _previous_period

    return _previous_period(since, until)[0]


def _prev_until(since, until):
    """End of the window immediately preceding ``[since, until]``."""
    from expensa.viz.data import _previous_period

    return _previous_period(since, until)[1]


def _render_insight_tiles(conn, savings) -> None:
    """Three always-all-time insight tiles below the headline row:
    month-to-date spend pace, fixed-vs-variable split, and the
    auto-categorization mix. These are deliberately NOT date-range scoped
    -- they answer "where do I stand right now / overall", which is
    independent of the chart date picker."""
    pace = month_to_date_pace(conn, savings_categories=savings)
    fv = fixed_vs_variable(conn)
    mix = categorization_mix(conn)

    with st.container(border=True):
        cols = st.columns(3)

        # ── Month-to-date pace ──
        with cols[0]:
            proj_delta = None
            if pace["baseline"] is not None:
                proj_delta = de_eur(pace["projected"] - pace["baseline"])
            st.metric(
                "This month (projected)",
                de_eur(pace["projected"]),
                delta=proj_delta,
                delta_color="inverse",
                help=(
                    f"Spent {de_eur(pace['spent'])} in the first "
                    f"{pace['days_elapsed']} of {pace['days_in_month']} days; "
                    "linearly extrapolated to month-end. "
                    + (
                        f"Δ vs trailing-6-month average ({de_eur(pace['baseline'])})."
                        if pace["baseline"] is not None
                        else "No prior months yet for a baseline."
                    )
                ),
            )

        # ── Fixed vs variable ──
        with cols[1]:
            if fv["fixed_share"] is not None:
                share_pct = fv["fixed_share"] * 100
                st.metric(
                    "Committed / month",
                    de_eur(fv["fixed_monthly"]),
                    help=(
                        f"Estimated recurring commitments (subscriptions, rent, "
                        f"insurance) — {share_pct:.0f}% of your ~"
                        f"{de_eur(fv['total_monthly'])} average monthly spend. "
                        f"Discretionary: ~{de_eur(fv['variable_monthly'])}/mo."
                    ),
                )
                st.caption(
                    f"🔁 {share_pct:.0f}% fixed · "
                    f"🛒 {100 - share_pct:.0f}% discretionary"
                )
            else:
                st.metric("Committed / month", "—",
                          help="Not enough history to detect recurring vendors yet.")

        # ── Categorization mix ──
        with cols[2]:
            total = mix["total"] or 1
            auto_pct = (mix["user"] + mix["high"]) / total * 100
            st.metric(
                "Auto-categorized",
                f"{auto_pct:.0f}%",
                help=(
                    f"{mix['user']} user-labeled · {mix['high']} high-confidence "
                    f"model · {mix['medium']} to confirm · {mix['low']} low "
                    f"confidence · {mix['uncategorized']} uncategorized "
                    f"(of {mix['total']})."
                ),
            )
            need_review = mix["medium"] + mix["low"] + mix["uncategorized"]
            if need_review:
                st.caption(
                    f"📋 {need_review} need review — see the **Review** tab."
                )
            else:
                st.caption("✅ Everything is categorized.")


def _render_top_movers(conn, since, until, savings) -> None:
    """Biggest per-category spend changes vs the previous same-length
    window. Skipped for All-time (no previous period to compare)."""
    if since is None or until is None:
        return
    movers = category_period_comparison(
        conn, since=since, until=until, savings_categories=savings
    )
    if movers.empty:
        return
    # Only surface categories that actually moved.
    movers = movers[movers["delta"].abs() >= 0.005]
    if movers.empty:
        return
    with st.expander("Top movers vs previous period", expanded=False):
        st.caption(
            "Per-category spend change against the immediately-preceding "
            "window of the same length. ▲ = spending more, ▼ = less."
        )
        top = movers.head(8).copy()
        top["direction"] = top["delta"].map(lambda d: "▲" if d > 0 else "▼")
        top["pct_str"] = top["pct"].map(
            lambda p: f"{p * 100:+.0f}%" if p is not None else "new"
        )
        display = pd.DataFrame({
            "": top["direction"],
            "Category": top["name"],
            "Now": top["current"],
            "Before": top["previous"],
            "Change": top["delta"],
            "%": top["pct_str"],
        })
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "Now": st.column_config.NumberColumn("Now (€)", format="%.2f"),
                "Before": st.column_config.NumberColumn("Before (€)", format="%.2f"),
                "Change": st.column_config.NumberColumn("Change (€)", format="%+.2f"),
            },
        )


def _render_upcoming_recurring(conn) -> None:
    """Forecast of recurring charges expected in the next 30 days, from
    the cadence detector. All-time history; not date-range scoped."""
    upcoming = upcoming_recurring(conn, horizon_days=30)
    if upcoming.empty:
        return
    total = float(upcoming["typical_amount"].sum())
    with st.expander(
        f"Upcoming recurring charges (next 30 days) — ~{de_eur(total)}",
        expanded=False,
    ):
        st.caption(
            "Projected from each recurring vendor's detected cadence "
            "(last seen + typical gap). Amounts are the vendor's typical "
            "charge — actuals may vary for variable-amount vendors."
        )
        st.dataframe(
            upcoming,
            hide_index=True,
            width="stretch",
            column_config={
                "name": st.column_config.TextColumn("Vendor"),
                "cadence": st.column_config.TextColumn("Cadence"),
                "expected_date": st.column_config.DateColumn(
                    "Expected", format="DD.MM.YYYY"
                ),
                "typical_amount": st.column_config.NumberColumn(
                    "Typical (€)", format="%.2f"
                ),
                "days_until": st.column_config.NumberColumn(
                    "In days", format="%d"
                ),
            },
        )


def _render_headline_tiles(conn, since, until, savings) -> None:
    """Savings rate / income / expenses / to-savings tiles + caption.
    Wrapped in a bordered container so it reads as one grouped unit."""
    ivex_df = monthly_income_vs_expense(
        conn, since=since, until=until, savings_categories=savings
    )
    sav_df = savings_flow(conn, since=since, until=until, savings_categories=savings)
    total_income = float(ivex_df["income"].sum()) if not ivex_df.empty else 0.0
    total_exp = float(ivex_df["expenses"].sum()) if not ivex_df.empty else 0.0
    total_to_sav = float(sav_df["to_savings"].sum()) if not sav_df.empty else 0.0
    total_from_sav = float(sav_df["from_savings"].sum()) if not sav_df.empty else 0.0
    net_to_sav = total_to_sav - total_from_sav

    if total_income > 0:
        pct = ((total_income - total_exp) / total_income) * 100
        dot = "🟢" if pct >= 20 else ("🔴" if pct < 0 else "🟡")
        sr_value = f"{dot} {pct:.0f}%"
    else:
        pct = None
        sr_value = "—"

    # Period-over-period deltas: compare this window to the immediately
    # preceding window of the same length. Only meaningful for a bounded
    # range (All-time has no "previous period"), so the deltas are None
    # there and st.metric simply omits them.
    prev = period_totals(
        conn, since=_prev_since(since, until), until=_prev_until(since, until),
        savings_categories=savings,
    ) if since is not None and until is not None else None

    def _money_delta(curr: float, key: str) -> str | None:
        if not prev:
            return None
        base = prev[key]
        return de_eur(curr - base) if base else None

    sr_delta = None
    if prev and prev["savings_rate"] is not None and pct is not None:
        sr_delta = f"{pct - prev['savings_rate'] * 100:+.0f} pp"

    with st.container(border=True):
        sr_cols = st.columns(4)
        # Savings rate: higher is better (default delta colour).
        sr_cols[0].metric("Savings rate", sr_value, delta=sr_delta)
        sr_cols[1].metric("Income", de_eur(total_income),
                          delta=_money_delta(total_income, "income"))
        # Expenses: rising spend is "bad", so invert the delta colour.
        sr_cols[2].metric("Expenses", de_eur(total_exp),
                          delta=_money_delta(total_exp, "expenses"),
                          delta_color="inverse")
        sav_label = ", ".join(savings) if savings else "your savings categories"
        sr_cols[3].metric(
            "💰 To savings (net)",
            de_eur(net_to_sav),
            help=(
                f"Money moved to your own accounts (category: {sav_label}) "
                f"minus what came back. Gross out: {de_eur(total_to_sav)} · "
                f"gross in: {de_eur(total_from_sav)}."
            ),
        )
        if savings:
            savings_note = (
                f"Rows categorised **{sav_label}** and rows matching a "
                "registered own IBAN (`iban_is_known_self`) are treated as "
                "neutral on both the income and expense sides — so the savings "
                "rate captures real income vs real consumption."
            )
        else:
            savings_note = (
                "Tip: mark a category as **Sparen** in the Categories tab to "
                "treat transfers to your own accounts as neutral here."
            )
        st.caption(
            "Reflects the currently selected date range above. "
            "🟢 ≥20% · 🟡 0–20% · 🔴 negative. " + savings_note
        )

    # Stash the ivex_df under session_state so the Income-vs-Expense chart
    # below can reuse the already-computed DataFrame instead of running
    # the SQL twice per page render.
    st.session_state["_dashboard_ivex_df"] = ivex_df


def _render_charts(conn, since, until, savings) -> None:
    """Collapsible per-category charts. Savings categories pre-hidden where
    shown inline (the monthly-saldo chart drops them upstream)."""
    dash_cats = list_categories(conn)
    color_map: dict[str, str] = {c.name: c.color for c in dash_cats}
    color_map["(unkategorisiert)"] = "#bbbbbb"

    chart_expander(
        "Summe Ausgaben nach Kategorie",
        bar_spend_by_category(
            spend_by_category(conn, since=since, until=until),
            color_map=color_map,
            hidden_categories=savings,
        ),
        expanded=True,
        key="dashboard_bar_chart",
    )
    chart_expander(
        "Wöchentliche Ausgaben nach Kategorie",
        stacked_weekly_by_category(
            weekly_by_category(conn, since=since, until=until),
            color_map=color_map,
            hidden_categories=savings,
        ),
        expanded=False,
        key="dashboard_weekly_chart",
    )
    chart_expander(
        "Monatlicher Saldo je Kategorie",
        stacked_monthly_by_category(
            monthly_flow_by_category(
                conn, since=since, until=until, savings_categories=savings
            ),
            color_map=color_map,
        ),
        expanded=True,
        key="dashboard_trend_chart",
    )
    ivex_df = st.session_state.get("_dashboard_ivex_df")
    if ivex_df is None:
        # Defensive: should always be populated by _render_headline_tiles.
        ivex_df = monthly_income_vs_expense(
            conn, since=since, until=until, savings_categories=savings
        )
    chart_expander(
        "Einkommen vs Ausgaben (monatlich)",
        income_vs_expense_chart(ivex_df),
        expanded=False,
        key="dashboard_ivex_chart",
    )


def _render_records_table(conn, since, until) -> None:
    """Read-only records table. Category cells get the category colour
    as background; text colour is auto-picked for legibility."""
    params: list = []
    clauses: list[str] = []
    if since is not None:
        clauses.append("e.buchungsdatum >= ?")
        params.append(since.isoformat())
    if until is not None:
        clauses.append("e.buchungsdatum <= ?")
        params.append(until.isoformat())

    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        _DASHBOARD_RECORDS_SELECT
        + where_sql
        + " ORDER BY e.buchungsdatum DESC, e.id DESC LIMIT 5000"
    )
    full_df = pd.read_sql_query(sql, conn, params=params)
    if not full_df.empty:
        full_df["buchungsdatum"] = pd.to_datetime(full_df["buchungsdatum"])
        full_df["category"] = full_df["category"].fillna("(unkategorisiert)")

    date_label = (
        "all dates" if since is None and until is None
        else f"{since} … {until}"
    )
    # Signed total across the visible rows. The table is unfiltered
    # by direction (income + expenses), so this is the net cashflow
    # for the selected date range.
    total_eur = float(full_df["betrag_€"].sum()) if not full_df.empty else 0.0
    st.caption(
        f"{len(full_df)} record(s) · date range: {date_label} · "
        f"total {de_eur(total_eur)}"
    )

    dash_cats = list_categories(conn)
    color_map: dict[str, str] = {c.name: c.color for c in dash_cats}

    def _style_category(val: str) -> str:
        name = (val or "").strip()
        if not name or name == "(unkategorisiert)":
            return ""
        bg = color_map.get(name, "#bbbbbb")
        fg = readable_text_color(bg)
        return f"background-color: {bg}; color: {fg};"

    styled = full_df if full_df.empty else full_df.style.map(
        _style_category, subset=["category"]
    )
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


def _render_alltime_helpers(conn) -> None:
    """Recurring vendors + Unusual amounts. Both intentionally use the
    full data history and live below the records table because their
    semantics differ from the date-range-scoped section above."""
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
