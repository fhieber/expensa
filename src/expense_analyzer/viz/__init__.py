"""Visualizations: SQL data views + Plotly chart factories + file export."""

from expense_analyzer.viz.charts import (
    bar_spend_by_category,
    bar_top_counterparties,
    calendar_heatmap,
    histogram_amounts,
    pie_chart,
    stacked_daily_by_category,
    stacked_monthly_by_category,
    trend_lines,
)
from expense_analyzer.viz.data import (
    amount_distribution,
    daily_by_category,
    daily_calendar,
    monthly_flow_by_category,
    spend_by_category,
    top_counterparties,
)
from expense_analyzer.viz.exporter import save_figure

CHART_BUILDERS = {
    "pie": (spend_by_category, pie_chart),
    "histogram": (amount_distribution, histogram_amounts),
    "trend": (monthly_flow_by_category, trend_lines),
    "top": (top_counterparties, bar_top_counterparties),
    "calendar": (daily_calendar, calendar_heatmap),
    "daily-stacked": (daily_by_category, stacked_daily_by_category),
}

__all__ = [
    "amount_distribution",
    "bar_spend_by_category",
    "bar_top_counterparties",
    "calendar_heatmap",
    "CHART_BUILDERS",
    "daily_by_category",
    "daily_calendar",
    "histogram_amounts",
    "monthly_flow_by_category",
    "pie_chart",
    "save_figure",
    "spend_by_category",
    "stacked_daily_by_category",
    "stacked_monthly_by_category",
    "top_counterparties",
    "trend_lines",
]
