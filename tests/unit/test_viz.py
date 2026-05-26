"""Visualization tests: data builders, chart factories, file export."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import plotly.graph_objects as go

from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.storage.categories import add_label, upsert_category
from expense_analyzer.viz import (
    CHART_BUILDERS,
    amount_distribution,
    bar_top_counterparties,
    daily_calendar,
    histogram_amounts,
    monthly_flow_by_category,
    pie_chart,
    save_figure,
    spend_by_category,
    top_counterparties,
    trend_lines,
)


def _label_some(conn: sqlite3.Connection) -> None:
    food = upsert_category(conn, "Lebensmittel", color="#4caf50")
    rent = upsert_category(conn, "Miete", color="#5e35b1")
    income = upsert_category(conn, "Einkommen", color="#66bb6a")
    rows = conn.execute(
        "SELECT id, counterparty_normalized, is_income FROM expenses"
    ).fetchall()
    for r in rows:
        if r["is_income"]:
            add_label(conn, int(r["id"]), income, "user")
        elif r["counterparty_normalized"] in {"markt alpha", "markt beta", "markt gamma"}:
            add_label(conn, int(r["id"]), food, "user")
        elif r["counterparty_normalized"] == "vermieter":
            add_label(conn, int(r["id"]), rent, "user")


def test_spend_by_category(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    _label_some(tmp_db)
    df = spend_by_category(tmp_db)
    assert "Lebensmittel" in df["name"].tolist()
    assert "Miete" in df["name"].tolist()
    assert (df["amount"] > 0).all()


def test_monthly_flow_two_months(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    _label_some(tmp_db)
    df = monthly_flow_by_category(tmp_db)
    months = set(df["ym"])
    assert months == {"2026-01", "2026-02"}


def test_top_counterparties(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = top_counterparties(tmp_db, n=5)
    assert len(df) == 5
    # Vermieter GmbH has the largest single transaction (rent), but BahnCard 100
    # is also high. Just sanity-check it's sorted descending.
    assert (df["amount"].diff().dropna() <= 0).all()


def test_amount_distribution_excludes_income(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = amount_distribution(tmp_db)
    assert (df["amount"] >= 0).all()
    # 50 total rows, 2 of which are income.
    assert len(df) == 48


def test_daily_calendar(tmp_db: sqlite3.Connection, fixtures_dir: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    df = daily_calendar(tmp_db)
    assert not df.empty
    assert "amount" in df.columns


def test_chart_factories_return_figures(
    tmp_db: sqlite3.Connection, fixtures_dir: Path
) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    _label_some(tmp_db)
    assert isinstance(pie_chart(spend_by_category(tmp_db)), go.Figure)
    assert isinstance(histogram_amounts(amount_distribution(tmp_db)), go.Figure)
    assert isinstance(trend_lines(monthly_flow_by_category(tmp_db)), go.Figure)
    assert isinstance(bar_top_counterparties(top_counterparties(tmp_db)), go.Figure)


def test_chart_builders_dict_complete() -> None:
    assert set(CHART_BUILDERS) == {
        "pie", "histogram", "trend", "top", "calendar", "daily-stacked",
    }


def test_save_figure_html(tmp_db: sqlite3.Connection, fixtures_dir: Path, tmp_path: Path) -> None:
    ingest_csv(tmp_db, fixtures_dir / "sample_de.csv")
    fig = pie_chart(spend_by_category(tmp_db))
    out = save_figure(fig, tmp_path / "out.html")
    assert out.exists()
    assert "<html" in out.read_text(encoding="utf-8").lower()


def test_chart_factories_handle_empty_db(tmp_db: sqlite3.Connection) -> None:
    """No expenses + no labels: the chart factories must not crash."""
    fig = pie_chart(spend_by_category(tmp_db))
    assert isinstance(fig, go.Figure)
    fig = histogram_amounts(amount_distribution(tmp_db))
    assert isinstance(fig, go.Figure)
    fig = trend_lines(monthly_flow_by_category(tmp_db))
    assert isinstance(fig, go.Figure)


def test_evaluation_chart_factories() -> None:
    """The evaluation charts return Figures for synthetic inputs and for
    empty inputs (no crash)."""
    import numpy as np

    from expense_analyzer.ml.evaluation import StageBreakdown
    from expense_analyzer.viz import (
        ablation_cumulative_curve,
        ablation_leave_one_out_bar,
        confusion_matrix_heatmap,
        stage_breakdown_bar,
    )

    cm = np.array([[3, 1], [0, 4]])
    assert isinstance(confusion_matrix_heatmap(cm, ["A", "B"]), go.Figure)
    assert isinstance(confusion_matrix_heatmap(np.zeros((0, 0)), []), go.Figure)

    breakdown = [
        StageBreakdown("vendor_exact_match", 5, 5, 1.0),
        StageBreakdown("knn", 3, 2, 2 / 3),
    ]
    assert isinstance(stage_breakdown_bar(breakdown), go.Figure)
    assert isinstance(stage_breakdown_bar([]), go.Figure)

    cumulative = [("vendor_exact_match", 0.5, 0.4), ("vendor_exact_match+knn", 0.7, 0.6)]
    assert isinstance(ablation_cumulative_curve(cumulative), go.Figure)
    assert isinstance(ablation_cumulative_curve([]), go.Figure)

    loo = [("knn", 0.6, 0.5, -0.1), ("classifier", 0.72, 0.6, 0.02)]
    assert isinstance(ablation_leave_one_out_bar(loo), go.Figure)
    assert isinstance(ablation_leave_one_out_bar([]), go.Figure)
