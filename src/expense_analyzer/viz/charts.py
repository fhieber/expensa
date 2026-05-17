"""Plotly chart factories. Each takes a DataFrame and returns a Figure.

Every category-coloured chart accepts an optional ``color_map``
``{category_name -> hex_color}``. The Dashboard builds one global map
from the ``categories`` table and passes it to every chart so the same
category gets the same colour across pie, histogram, trend lines and
stacked daily bars (otherwise Plotly's default colour cycle would assign
a different colour per chart, defeating visual cross-referencing).
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


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
    spend_df, color_map: dict[str, str] | None = None
) -> go.Figure:
    """Horizontal bar of spend per category, sorted desc. Same data as the
    pie -- but clickable. Streamlit's `st.plotly_chart(on_select=...)`
    only captures selection events from traces that expose a
    ``selectedpoints`` attribute (scatter, bar, histogram, box). Pie
    traces don't, so a bar is the cleanest path to a clickable Ausgaben-
    nach-Kategorie visual."""
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
        title="Ausgaben nach Kategorie",
        labels={"amount": "Summe (€)", "name": "Kategorie"},
    )
    fig.update_layout(showlegend=False)
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
