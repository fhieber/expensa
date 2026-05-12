"""Plotly chart factories. Each takes a DataFrame and returns a Figure."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


def pie_chart(spend_df) -> go.Figure:
    if spend_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Ausgaben nach Kategorie (keine Daten)")
        return fig
    fig = px.pie(
        spend_df,
        values="amount",
        names="name",
        color="name",
        color_discrete_map={r["name"]: r["color"] for _, r in spend_df.iterrows()},
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


def histogram_amounts(dist_df, nbins: int = 30) -> go.Figure:
    if dist_df.empty:
        return go.Figure(layout={"title": "Beträge (keine Daten)"})
    fig = px.histogram(
        dist_df,
        x="amount",
        color="name",
        nbins=nbins,
        title="Verteilung der Ausgabenhöhe",
        labels={"amount": "Betrag (€)", "name": "Kategorie"},
    )
    return fig


def trend_lines(monthly_df) -> go.Figure:
    if monthly_df.empty:
        return go.Figure(layout={"title": "Monatlicher Verlauf (keine Daten)"})
    fig = px.line(
        monthly_df,
        x="ym",
        y="amount",
        color="name",
        markers=True,
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


def stacked_daily_by_category(df) -> go.Figure:
    """Stacked bars per day showing how much was spent on each category.
    Useful for spotting e.g. "spent X on Lebensmittel on May 8th"."""
    if df.empty:
        return go.Figure(layout={"title": "Tägliche Ausgaben nach Kategorie (keine Daten)"})
    color_map = {r["name"]: r["color"] for _, r in df.drop_duplicates("name").iterrows()}
    fig = px.bar(
        df,
        x="d",
        y="amount",
        color="name",
        color_discrete_map=color_map,
        title="Tägliche Ausgaben nach Kategorie",
        labels={"d": "Tag", "amount": "Betrag (€)", "name": "Kategorie"},
    )
    fig.update_layout(barmode="stack", legend_title_text="Kategorie")
    return fig
