"""Dashboard tab: headline stats + charts.

The layout is organised into scope-based zones. One rule drives it:
**everything above the date picker is date-independent; everything below
reflects the selected range.**

  Zone A · "Right now" (date-INDEPENDENT) -- month-to-date pace,
           committed/month, auto-categorization mix.
  ── date-range preset radio (default: "Past 90 days") ──
  Zone B · Period summary (scoped) -- savings rate / income / expenses /
           to-savings, in a bordered container.
  Zone C · Trends (scoped) -- per-category charts in an st.tabs switcher.
  Zone D · Detail (scoped) -- "what changed" top movers.

Browsing individual transactions lives on the **Data** tab; the
date-range link under the picker deep-links there with the same window.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from expensa.storage.categories import list_categories, savings_category_names
from expensa.ui._components import date_preset_row, de_eur, render_chart
from expensa.ui._shared import get_conn
from expensa.viz import (
    bar_spend_by_category,
    categorization_mix,
    category_period_comparison,
    fixed_vs_variable,
    income_vs_expense_chart,
    month_to_date_pace,
    monthly_flow_by_category,
    monthly_income_vs_expense,
    period_totals,
    savings_flow,
    spend_by_category,
    stacked_monthly_by_category,
    stacked_weekly_by_category,
    weekly_by_category,
)


def render() -> None:
    conn = get_conn()
    st.header("Dashboard")

    n_exp = conn.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
    if n_exp == 0:
        _render_empty_state(conn)
        return

    savings = tuple(savings_category_names(conn))

    # ── Zone A · Right now (date-independent) ──
    _maybe_low_data_hint(conn)
    _render_status_now(conn, savings)

    st.divider()

    # ── Date picker · scopes everything below ──
    since, until = date_preset_row(key_prefix="dashboard")
    st.caption("📅 The sections below reflect the selected date range.")
    _render_view_in_data_link()

    # ── Zone B · Period summary (scoped) ──
    _render_headline_tiles(conn, since, until, savings)

    # ── Zone C · Trends (scoped) -- per-category charts + top movers,
    #    all under one st.tabs switcher. ──
    _render_charts(conn, since, until, savings)


def _render_view_in_data_link() -> None:
    """Point the user to the Data tab pre-filtered to the dashboard's
    current date range.

    The dashboard no longer carries its own records table -- the Data
    tab is the canonical place to browse individual transactions. We
    mirror the dashboard's date selection onto the Data tab's widget
    state so the user lands on the same window. Streamlit can't switch
    ``st.tabs`` programmatically (same constraint as the header's
    "Show in Data ↗"), so we set the state + point rather than auto-
    jump. The copy happens during the dashboard render, which runs
    *before* the Data tab in the same pass, so writing the Data tab's
    widget keys here is safe.
    """
    if st.button(
        "🔎 View this time range on the Data tab",
        key="dash_view_range_in_data",
        type="tertiary",
        help="Copies the selected date range to the Data tab so you can "
             "browse the individual transactions there.",
    ):
        preset = st.session_state.get("dashboard_date_preset")
        if preset is not None:
            st.session_state["data_date_preset"] = preset
            if preset == "Custom":
                st.session_state["data_from"] = st.session_state.get("dashboard_from")
                st.session_state["data_to"] = st.session_state.get("dashboard_to")
        st.toast(
            "Data tab set to this date range — open the **Data** tab to view.",
            icon="📅",
        )


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


def _render_status_now(conn, savings) -> None:
    """Zone A: three date-independent "right now" tiles — month-to-date
    spend pace, fixed-vs-variable split, and the auto-categorization mix.
    These sit ABOVE the date picker precisely because they answer "where
    do I stand right now / overall", independent of the selected range."""
    pace = month_to_date_pace(conn, savings_categories=savings)
    fv = fixed_vs_variable(conn, savings_categories=savings)
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
    window. Rendered as the "Top movers" tab inside the Trends switcher,
    so it always shows *something* (a caption + an info note when there's
    nothing to compare) rather than vanishing like the old expander did."""
    st.caption(
        "Per-category spend change against the immediately-preceding "
        "window of the same length. ▲ = spending more, ▼ = less."
    )
    if since is None or until is None:
        st.info(
            "Pick a bounded date range (not *All-time*) to compare against "
            "the previous period."
        )
        return
    movers = category_period_comparison(
        conn, since=since, until=until, savings_categories=savings
    )
    # Only surface categories that actually moved.
    if not movers.empty:
        movers = movers[movers["delta"].abs() >= 0.005]
    if movers.empty:
        st.info("No category moved meaningfully versus the previous period.")
        return
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

    # Colour the direction arrow + % cells by the sign of the change:
    # red = spending MORE (bad), green = spending LESS (good). Keyed on
    # the numeric "Change" column so "new" / rounded-to-0% strings still
    # get the right colour; the numeric columns keep their NumberColumn
    # formatting untouched (we don't style them).
    def _sign_style(row: pd.Series) -> list[str]:
        delta = row["Change"]
        css = (
            "color: #d9534f; font-weight: 600" if delta > 0
            else "color: #2e7d32; font-weight: 600" if delta < 0
            else ""
        )
        return [css if col in ("", "%") else "" for col in display.columns]

    styled = display.style.apply(_sign_style, axis=1)
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        column_config={
            "Now": st.column_config.NumberColumn("Now (€)", format="%.2f"),
            "Before": st.column_config.NumberColumn("Before (€)", format="%.2f"),
            "Change": st.column_config.NumberColumn("Change (€)", format="%+.2f"),
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
            "Savings rate: 🟢 ≥20% · 🟡 0–20% · 🔴 negative. " + savings_note
        )

    # Stash the ivex_df under session_state so the Income-vs-Expense chart
    # below can reuse the already-computed DataFrame instead of running
    # the SQL twice per page render.
    st.session_state["_dashboard_ivex_df"] = ivex_df


def _render_charts(conn, since, until, savings) -> None:
    """Zone C: per-category trend charts in an st.tabs switcher (one chart
    visible at a time). Savings categories are pre-hidden where shown
    inline (the monthly-saldo chart drops them upstream)."""
    dash_cats = list_categories(conn)
    color_map: dict[str, str] = {c.name: c.color for c in dash_cats}
    color_map["(unkategorisiert)"] = "#bbbbbb"

    by_cat, monthly, weekly, ivex, movers = st.tabs(
        ["By category", "Monthly balance", "Weekly", "Income vs expenses",
         "Top movers"]
    )

    with by_cat:
        render_chart(
            bar_spend_by_category(
                spend_by_category(conn, since=since, until=until),
                color_map=color_map,
                hidden_categories=savings,
            ),
            key="dashboard_bar_chart",
        )
    with monthly:
        render_chart(
            stacked_monthly_by_category(
                monthly_flow_by_category(
                    conn, since=since, until=until, savings_categories=savings
                ),
                color_map=color_map,
            ),
            key="dashboard_trend_chart",
        )
    with weekly:
        render_chart(
            stacked_weekly_by_category(
                weekly_by_category(conn, since=since, until=until),
                color_map=color_map,
                hidden_categories=savings,
            ),
            key="dashboard_weekly_chart",
        )
    with ivex:
        ivex_df = st.session_state.get("_dashboard_ivex_df")
        if ivex_df is None:
            # Defensive: should always be populated by _render_headline_tiles.
            ivex_df = monthly_income_vs_expense(
                conn, since=since, until=until, savings_categories=savings
            )
        render_chart(
            income_vs_expense_chart(ivex_df),
            key="dashboard_ivex_chart",
        )
    with movers:
        _render_top_movers(conn, since, until, savings)
