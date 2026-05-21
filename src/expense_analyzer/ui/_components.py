"""Small UI helpers reused across multiple Streamlit tabs.

Keep this module narrow: it should depend only on Streamlit / pandas /
plotly, never on tab-specific state. Tab modules import helpers from
here; this module doesn't import from any tab module.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from expense_analyzer.ui._date_presets import (
    DEFAULT_PRESET,
    PRESETS,
    resolve_range,
)


def date_preset_row(
    *,
    key_prefix: str,
    default: str = DEFAULT_PRESET,
) -> tuple[date | None, date | None]:
    """Render the date-range preset radio + (when ``Custom``) From/To
    inputs on a single row. Returns the resolved ``(since, until)`` pair.

    ``key_prefix`` namespaces the widget keys so the Dashboard and Data
    tabs can render the row side-by-side without colliding on
    session_state.
    """
    preset_col, from_col, to_col = st.columns([6, 1.5, 1.5])
    with preset_col:
        preset = st.radio(
            "Date range",
            PRESETS,
            index=PRESETS.index(default),
            horizontal=True,
            key=f"{key_prefix}_date_preset",
        )
    if preset == "Custom":
        with from_col:
            custom_from = st.date_input(
                "From", value=None, key=f"{key_prefix}_from"
            )
        with to_col:
            custom_to = st.date_input(
                "To", value=None, key=f"{key_prefix}_to"
            )
        return resolve_range(preset, custom_from, custom_to)
    return resolve_range(preset)


def chart_expander(label: str, fig, *, expanded: bool, key: str) -> None:
    """Wrap a Plotly figure in an ``st.expander`` and render it.

    The expander label IS the chart title, so we suppress the in-chart
    title to avoid double-rendering. Trimming the top margin removes the
    reserved gap above the chart. (`title=None` would render as the
    literal string "undefined" in some Plotly/Streamlit version combos.)
    """
    with st.expander(label, expanded=expanded):
        fig.update_layout(
            title_text="",
            margin={"t": 20, "b": 20, "l": 0, "r": 0},
        )
        st.plotly_chart(fig, width="stretch", key=key)


def de_eur(v: float) -> str:
    """Format a euro amount in DE locale: thousands `.`, decimal `,`."""
    return (
        f"{v:,.2f} €"
        .replace(",", "X").replace(".", ",").replace("X", ".")
    )
