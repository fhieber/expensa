"""CLI smoke tests using click.testing.CliRunner.

We patch the embedder factory so the CLI doesn't try to download a 1 GB
HF model in unit tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from expense_analyzer.cli import cli
from expense_analyzer.features.embeddings import HashEmbedder


def _runner_env(tmp_path: Path) -> dict:
    return {"EXPENSE_ANALYZER_HOME": str(tmp_path)}


def test_help_lists_subcommands() -> None:
    r = CliRunner().invoke(cli, ["--help"])
    assert r.exit_code == 0
    for cmd in ("init", "ingest", "label", "train", "predict", "viz", "ui", "status"):
        assert cmd in r.output


def test_init_creates_db_and_categories(tmp_path: Path) -> None:
    runner = CliRunner()
    r = runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert (tmp_path / "db.sqlite").exists()
    r2 = runner.invoke(cli, ["categories", "list"], env=_runner_env(tmp_path))
    assert "Lebensmittel" in r2.output


def test_ingest_reports_counts(tmp_path: Path, fixtures_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    r = runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")], env=_runner_env(tmp_path)
    )
    assert r.exit_code == 0, r.output
    assert "new=  50" in r.output
    r2 = runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_overlap.csv")], env=_runner_env(tmp_path)
    )
    assert "new=   7" in r2.output
    assert "duplicate=   6" in r2.output


def test_status_after_ingest(tmp_path: Path, fixtures_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")], env=_runner_env(tmp_path)
    )
    r = runner.invoke(cli, ["status"], env=_runner_env(tmp_path))
    assert r.exit_code == 0
    assert "expenses:     50" in r.output


def test_viz_writes_html(tmp_path: Path, fixtures_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")], env=_runner_env(tmp_path)
    )
    out = tmp_path / "pie.html"
    r = runner.invoke(cli, ["viz", "pie", "--out", str(out)], env=_runner_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert out.exists()


def test_export_csv(tmp_path: Path, fixtures_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")], env=_runner_env(tmp_path)
    )
    out = tmp_path / "out.csv"
    r = runner.invoke(cli, ["export", "--format", "csv", "--out", str(out)], env=_runner_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert out.exists()
    text = out.read_text(encoding="utf-8-sig")
    assert "buchungsdatum" in text


def test_train_with_mocked_embedder(tmp_path: Path, fixtures_dir: Path) -> None:
    """Train should run with the HashEmbedder injected via patch."""
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")], env=_runner_env(tmp_path)
    )
    # Add a couple of labels.
    runner.invoke(cli, ["categories", "add", "TestA"], env=_runner_env(tmp_path))
    runner.invoke(cli, ["categories", "add", "TestB"], env=_runner_env(tmp_path))
    # Easier: poke the DB directly.
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    conn.execute("INSERT INTO labels(expense_id, category_id, source) VALUES (1, 1, 'user')")
    conn.execute("INSERT INTO labels(expense_id, category_id, source) VALUES (2, 2, 'user')")
    conn.commit()
    conn.close()

    with patch("expense_analyzer.cli._embedder", return_value=HashEmbedder(dim=64)):
        r = runner.invoke(cli, ["train"], env=_runner_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "trained" in r.output


def test_vendor_lookup_disabled_message(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    # Write a user config that explicitly disables vendor lookup so the test
    # doesn't depend on the bundled default.
    (tmp_path / "config.yaml").write_text(
        "vendor_lookup:\n  enabled: false\n",
        encoding="utf-8",
    )
    r = runner.invoke(cli, ["vendor-lookup", "REWE"], env=_runner_env(tmp_path))
    assert r.exit_code != 0
    assert "enabled is False" in r.output


def test_categories_remove_unused(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(cli, ["categories", "add", "ToRemove"], env=_runner_env(tmp_path))
    r = runner.invoke(
        cli, ["categories", "remove", "ToRemove", "--yes"], env=_runner_env(tmp_path)
    )
    assert r.exit_code == 0, r.output
    assert "removed ToRemove" in r.output


def test_categories_remove_missing_returns_error(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    r = runner.invoke(
        cli, ["categories", "remove", "Ghost", "--yes"], env=_runner_env(tmp_path)
    )
    assert r.exit_code != 0
    assert "no such category" in r.output


def test_categories_remove_refuses_when_labels_exist(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")], env=_runner_env(tmp_path)
    )
    # Manually attach a label to category #1.
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    conn.execute("INSERT INTO labels(expense_id, category_id, source) VALUES (1, 1, 'user')")
    conn.commit()
    name = conn.execute("SELECT name FROM categories WHERE id=1").fetchone()[0]
    conn.close()

    r = runner.invoke(
        cli, ["categories", "remove", name, "--yes"], env=_runner_env(tmp_path)
    )
    assert r.exit_code != 0
    assert "refusing" in r.output

    r2 = runner.invoke(
        cli,
        ["categories", "remove", name, "--force", "--yes"],
        env=_runner_env(tmp_path),
    )
    assert r2.exit_code == 0, r2.output
    assert "1 label(s) cascaded" in r2.output


def test_reset_clears_data_keeps_categories(tmp_path: Path, fixtures_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")], env=_runner_env(tmp_path)
    )
    r = runner.invoke(cli, ["reset", "--yes"], env=_runner_env(tmp_path))
    assert r.exit_code == 0, r.output
    s = runner.invoke(cli, ["status"], env=_runner_env(tmp_path))
    assert "expenses:     0" in s.output
    assert "categories:   17" in s.output  # default cats kept


def test_reset_all_wipes_categories_too(tmp_path: Path, fixtures_dir: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")], env=_runner_env(tmp_path)
    )
    r = runner.invoke(cli, ["reset", "--all", "--yes"], env=_runner_env(tmp_path))
    assert r.exit_code == 0, r.output
    s = runner.invoke(cli, ["status"], env=_runner_env(tmp_path))
    assert "expenses:     0" in s.output
    assert "categories:   0" in s.output


def test_reset_on_empty_db_says_so(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    # init installed 17 categories — wipe with --all so nothing's left.
    runner.invoke(cli, ["reset", "--all", "--yes"], env=_runner_env(tmp_path))
    r = runner.invoke(cli, ["reset", "--all", "--yes"], env=_runner_env(tmp_path))
    assert r.exit_code == 0
    assert "already empty" in r.output
