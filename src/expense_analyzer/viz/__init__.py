"""Visualizations: SQL data views + Plotly chart factories + file export."""

from expense_analyzer.viz.charts import (
    bar_spend_by_category,
    bar_top_counterparties,
    calendar_heatmap,
    histogram_amounts,
    income_vs_expense_chart,
    pie_chart,
    stacked_daily_by_category,
    trend_lines,
)
from expense_analyzer.viz.data import (
    amount_distribution,
    anomalies,
    daily_by_category,
    daily_calendar,
    monthly_flow_by_category,
    monthly_income_vs_expense,
    recurring_subscriptions,
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
    "anomalies",
    "bar_spend_by_category",
    "bar_top_counterparties",
    "calendar_heatmap",
    "CHART_BUILDERS",
    "daily_by_category",
    "daily_calendar",
    "histogram_amounts",
    "income_vs_expense_chart",
    "monthly_flow_by_category",
    "monthly_income_vs_expense",
    "pie_chart",
    "recurring_subscriptions",
    "save_figure",
    "spend_by_category",
    "stacked_daily_by_category",
    "top_counterparties",
    "trend_lines",
]
