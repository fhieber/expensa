"""Plotly chart factories. Each takes a DataFrame and returns a Figure.

Every category-coloured chart accepts an optional ``color_map``
``{category_name -> hex_color}``. The Dashboard builds one global map
from the ``categories`` table and passes it to every chart so the same
category gets the same colour across pie, histogram, trend lines and
stacked daily bars (otherwise Plotly's default colour cycle would assign
a different colour per chart, defeating visual cross-referencing).
"""

from __future__ import annotations

from collections.abc import Iterable

import plotly.express as px
import plotly.graph_objects as go


def _hide_traces(fig: go.Figure, hidden_categories: Iterable[str] | None) -> None:
    """Pre-hide the given category traces by flipping them to ``legendonly``.

    The legend entry stays visible so the user can re-enable a hidden
    category with a single click. Used by the Dashboard to start Sparen
    (and any other configured neutral/transfer category) out-of-view
    without permanently dropping its data."""
    if not hidden_categories:
        return
    hide = {str(c) for c in hidden_categories}
    for tr in fig.data:
        if tr.name in hide:
            tr.visible = "legendonly"


def _resolve_color_map(df, color_map: dict[str, str] | None) -> dict[str, str]:
    """When the caller supplies a ``color_map`` (the canonical one comes
    from the Categories tab), use it AS-IS so the same category gets the
    same user-picked colour everywhere; any name not in the map falls
    back to neutral grey instead of Plotly's auto-cycle. Without a
    supplied map we use whatever's in the DataFrame's ``color`` column
    (data-fn fallback) for backwards compatibility with the CLI."""
    if color_map:
        out = dict(color_map)
        if df is not None and not df.empty and "name" in df.columns:
            for n in df["name"].dropna().unique():
                out.setdefault(str(n), "#bbbbbb")
        return out
    if df is not None and not df.empty and "color" in df.columns:
        return {r["name"]: r["color"] for _, r in df.drop_duplicates("name").iterrows()}
    return {}


def bar_spend_by_category(
    spend_df,
    color_map: dict[str, str] | None = None,
    hidden_categories: Iterable[str] | None = None,
) -> go.Figure:
    """Horizontal bar of spend per category, sorted desc. Same data as the
    pie -- but historically clickable. Streamlit's
    `st.plotly_chart(on_select=...)` only captures selection events from
    traces that expose a ``selectedpoints`` attribute (scatter, bar,
    histogram, box). Pie traces don't, so a bar is the cleanest path to
    an Ausgaben-nach-Kategorie visual.

    ``hidden_categories`` lets the caller pre-hide specific category
    bars (re-enableable by clicking the legend). One trace per category
    is emitted (``color='name'``), so legend toggling is per-category.
    Used by the Dashboard to hide *Sparen* (transfers to own accounts)
    by default so the chart reflects real consumption."""
    if spend_df.empty:
        return go.Figure(layout={"title": "Ausgaben nach Kategorie (keine Daten)"})
    cmap = _resolve_color_map(spend_df, color_map)
    df = spend_df.sort_values("amount", ascending=True)  # ascending => top of chart is biggest
    fig = px.bar(
        df,
        x="amount",
        y="name",
        color="name",
        orientation="h",
        color_discrete_map=cmap,
        title="Summe Ausgaben nach Kategorie",
        labels={"amount": "Summe (€)", "name": "Kategorie"},
    )
    # Keep legend visible so hidden categories (Sparen) can be re-enabled.
    fig.update_layout(showlegend=bool(hidden_categories))
    _hide_traces(fig, hidden_categories)
    return fig


def pie_chart(spend_df, color_map: dict[str, str] | None = None) -> go.Figure:
    if spend_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Ausgaben nach Kategorie (keine Daten)")
        return fig
    cmap = _resolve_color_map(spend_df, color_map)
    fig = px.pie(
        spend_df,
        values="amount",
        names="name",
        color="name",
        color_discrete_map=cmap,
        hole=0.3,
        title="Ausgaben nach Kategorie",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def bar_top_counterparties(top_df) -> go.Figure:
    if top_df.empty:
        return go.Figure(layout={"title": "Top-Empfänger (keine Daten)"})
    fig = px.bar(
        top_df,
        x="amount",
        y="name",
        orientation="h",
        title="Top-Empfänger nach Ausgabe",
        labels={"amount": "Summe (€)", "name": "Empfänger"},
        hover_data=["n_tx"],
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def histogram_amounts(
    dist_df, nbins: int = 30, color_map: dict[str, str] | None = None
) -> go.Figure:
    if dist_df.empty:
        return go.Figure(layout={"title": "Beträge (keine Daten)"})
    cmap = _resolve_color_map(dist_df, color_map)
    fig = px.histogram(
        dist_df,
        x="amount",
        color="name",
        nbins=nbins,
        color_discrete_map=cmap,
        title="Verteilung der Ausgabenhöhe",
        labels={"amount": "Betrag (€)", "name": "Kategorie"},
    )
    return fig


def trend_lines(
    monthly_df, color_map: dict[str, str] | None = None
) -> go.Figure:
    if monthly_df.empty:
        return go.Figure(layout={"title": "Monatlicher Verlauf (keine Daten)"})
    cmap = _resolve_color_map(monthly_df, color_map)
    fig = px.line(
        monthly_df,
        x="ym",
        y="amount",
        color="name",
        markers=True,
        color_discrete_map=cmap,
        title="Monatlicher Saldo je Kategorie",
        labels={"ym": "Monat", "amount": "Betrag (€)", "name": "Kategorie"},
    )
    return fig


def stacked_monthly_by_category(
    monthly_df, color_map: dict[str, str] | None = None
) -> go.Figure:
    """Stacked bar of monthly per-category amount.

    Uses Plotly's ``barmode='relative'`` so **negative amounts stack
    *below* zero** (expenses pulling the bar down) and **positive amounts
    stack *above* zero** (income pushing it up). The net month-over-month
    visual is whatever sticks out from y=0 on each side.

    Cleaner read than `trend_lines` when the user cares about
    composition + net-flow rather than per-category trajectory.
    """
    if monthly_df.empty:
        return go.Figure(
            layout={"title": "Monatlicher Saldo je Kategorie (keine Daten)"}
        )
    cmap = _resolve_color_map(monthly_df, color_map)
    fig = px.bar(
        monthly_df,
        x="ym",
        y="amount",
        color="name",
        color_discrete_map=cmap,
        title="Monatlicher Saldo je Kategorie",
        labels={"ym": "Monat", "amount": "Betrag (€)", "name": "Kategorie"},
    )
    # 'relative' = stack positives above 0 / negatives below 0 (a.k.a.
    # diverging stacked bars). 'stack' would put everything above zero
    # regardless of sign, which loses the income-vs-expense distinction.
    fig.update_layout(barmode="relative", legend_title_text="Kategorie")
    # Zero-line emphasis so the income/expense split is obvious.
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    return fig


def income_vs_expense_chart(df) -> go.Figure:
    """Diverging bar chart of monthly income vs expenses.

    Income (positive total) plots **above** zero in green; expenses
    (positive total too, flipped via ``y = -expenses``) plot **below**
    zero in red. The y=0 line emphasises the split. Net flow per month
    is whatever sticks out on each side relative to the other.

    Different from ``stacked_monthly_by_category`` because it doesn't
    split by category -- two flat traces (income / expense) per month,
    no Plotly auto-stack juggling. Built directly with ``go`` because
    we need the explicit y-sign flip on the expense trace.
    """
    if df.empty:
        return go.Figure(
            layout={"title": "Einkommen vs Ausgaben (keine Daten)"}
        )
    fig = go.Figure()
    fig.add_bar(
        x=df["ym"], y=df["income"],
        name="Einkommen",
        marker_color="#22c55e",
        hovertemplate="<b>%{x}</b><br>Einkommen: %{y:,.2f} €<extra></extra>",
    )
    fig.add_bar(
        x=df["ym"], y=-df["expenses"],
        name="Ausgaben",
        marker_color="#ef4444",
        # Show the original (positive) value in the tooltip; the negated
        # y is just a layout trick.
        customdata=df["expenses"],
        hovertemplate="<b>%{x}</b><br>Ausgaben: %{customdata:,.2f} €<extra></extra>",
    )
    fig.update_layout(
        title="Einkommen vs Ausgaben (monatlich)",
        barmode="relative",
        yaxis_title="Betrag (€)",
        xaxis_title="Monat",
        legend_title_text="",
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    return fig


def calendar_heatmap(cal_df) -> go.Figure:
    """A simple per-date bar series. Real calendar heatmaps need x=week,
    y=weekday — but a date-bar gives a clean overview without extra deps."""
    if cal_df.empty:
        return go.Figure(layout={"title": "Tägliche Ausgaben (keine Daten)"})
    fig = px.bar(
        cal_df,
        x="d",
        y="amount",
        title="Tägliche Ausgaben",
        labels={"d": "Tag", "amount": "Betrag (€)"},
    )
    return fig


def stacked_daily_by_category(
    df, color_map: dict[str, str] | None = None
) -> go.Figure:
    """Stacked bars per day showing how much was spent on each category.
    Useful for spotting e.g. "spent X on Lebensmittel on May 8th"."""
    if df.empty:
        return go.Figure(layout={"title": "Tägliche Ausgaben nach Kategorie (keine Daten)"})
    cmap = _resolve_color_map(df, color_map)
    fig = px.bar(
        df,
        x="d",
        y="amount",
        color="name",
        color_discrete_map=cmap,
        title="Tägliche Ausgaben nach Kategorie",
        labels={"d": "Tag", "amount": "Betrag (€)", "name": "Kategorie"},
    )
    fig.update_layout(barmode="stack", legend_title_text="Kategorie")
    return fig


def stacked_weekly_by_category(
    df,
    color_map: dict[str, str] | None = None,
    hidden_categories: Iterable[str] | None = None,
) -> go.Figure:
    """Stacked bars per week. Less fine-grained than the daily variant
    -- the bars stay readable over multi-month windows. Same colour
    plumbing as the daily/monthly stacked charts.

    ``hidden_categories`` pre-hides specific category stacks (still
    re-enableable via the legend). Used by the Dashboard to start
    *Sparen* hidden so the visible bars match real consumption."""
    if df.empty:
        return go.Figure(layout={"title": "Wöchentliche Ausgaben nach Kategorie (keine Daten)"})
    cmap = _resolve_color_map(df, color_map)
    fig = px.bar(
        df,
        x="w",
        y="amount",
        color="name",
        color_discrete_map=cmap,
        title="Wöchentliche Ausgaben nach Kategorie",
        labels={"w": "Woche", "amount": "Betrag (€)", "name": "Kategorie"},
    )
    fig.update_layout(barmode="stack", legend_title_text="Kategorie")
    _hide_traces(fig, hidden_categories)
    return fig
