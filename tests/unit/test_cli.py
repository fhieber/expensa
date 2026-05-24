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


def test_eval_with_mocked_embedder(tmp_path: Path, fixtures_dir: Path) -> None:
    """`expense eval` runs CV on user labels with the HashEmbedder."""
    runner = CliRunner()
    runner.invoke(cli, ["init"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["ingest", "--no-embed", str(fixtures_dir / "sample_de.csv")],
        env=_runner_env(tmp_path),
    )
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    # Two categories, three labels each -> 2-fold stratifiable.
    for eid in (1, 2, 3):
        conn.execute(
            "INSERT INTO labels(expense_id, category_id, source) VALUES (?, 1, 'user')",
            (eid,),
        )
    for eid in (4, 5, 6):
        conn.execute(
            "INSERT INTO labels(expense_id, category_id, source) VALUES (?, 2, 'user')",
            (eid,),
        )
    conn.commit()
    conn.close()

    with patch("expense_analyzer.cli._embedder", return_value=HashEmbedder(dim=64)):
        r = runner.invoke(
            cli,
            ["eval", "--folds", "2", "--no-zeroshot", "--no-ablation"],
            env=_runner_env(tmp_path),
        )
    assert r.exit_code == 0, r.output
    assert "accuracy=" in r.output
    assert "per-stage contribution" in r.output


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


# ---------------------------------------------------------------------------
# `expense account` subgroup -- multi-account support
# ---------------------------------------------------------------------------


def test_account_list_empty(tmp_path: Path) -> None:
    """Fresh tmp_path has no accounts.yaml -- list should say so cleanly
    instead of crashing."""
    r = CliRunner().invoke(cli, ["account", "list"], env=_runner_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "no accounts registered" in r.output


def test_account_add_creates_dir_and_db(tmp_path: Path) -> None:
    runner = CliRunner()
    r = runner.invoke(
        cli, ["account", "add", "Business"], env=_runner_env(tmp_path)
    )
    assert r.exit_code == 0, r.output
    biz_dir = tmp_path / "accounts" / "business"
    assert biz_dir.is_dir()
    assert (biz_dir / "db.sqlite").is_file()
    assert "added account: business" in r.output
    assert "active account is now: business" in r.output


def test_account_add_seeds_categories_by_default(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Business"], env=_runner_env(tmp_path))
    r = runner.invoke(cli, ["categories", "list"], env=_runner_env(tmp_path))
    assert "Lebensmittel" in r.output  # bundled German default


def test_account_add_no_defaults_skips_categories(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(
        cli, ["account", "add", "Business", "--no-defaults"],
        env=_runner_env(tmp_path),
    )
    r = runner.invoke(cli, ["categories", "list"], env=_runner_env(tmp_path))
    assert "Lebensmittel" not in r.output


def test_account_add_no_use_keeps_prior_active(tmp_path: Path) -> None:
    """--no-use leaves the active account untouched."""
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "First"], env=_runner_env(tmp_path))
    runner.invoke(
        cli, ["account", "add", "Second", "--no-use"], env=_runner_env(tmp_path)
    )
    r = runner.invoke(cli, ["account", "list"], env=_runner_env(tmp_path))
    # active is still First (the * marker is next to its row).
    lines = [line for line in r.output.splitlines() if line.startswith("  *")]
    assert any("first" in line for line in lines)


def test_account_list_shows_active_marker(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Personal"], env=_runner_env(tmp_path))
    runner.invoke(cli, ["account", "add", "Business"], env=_runner_env(tmp_path))
    # Business is active because each `add` auto-uses by default.
    r = runner.invoke(cli, ["account", "list"], env=_runner_env(tmp_path))
    active_lines = [line for line in r.output.splitlines() if line.lstrip().startswith("*")]
    assert any("business" in line for line in active_lines)


def test_account_use_switches_active(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Personal"], env=_runner_env(tmp_path))
    runner.invoke(cli, ["account", "add", "Business"], env=_runner_env(tmp_path))
    r = runner.invoke(
        cli, ["account", "use", "Personal"], env=_runner_env(tmp_path)
    )
    assert r.exit_code == 0
    assert "active account is now: personal" in r.output


def test_account_use_unknown_raises(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Personal"], env=_runner_env(tmp_path))
    r = runner.invoke(cli, ["account", "use", "ghost"], env=_runner_env(tmp_path))
    assert r.exit_code != 0
    assert "no such account" in r.output


def test_account_remove_keeps_files_on_disk(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Business"], env=_runner_env(tmp_path))
    biz_dir = tmp_path / "accounts" / "business"
    assert biz_dir.is_dir()
    r = runner.invoke(
        cli, ["account", "remove", "Business", "--yes"],
        env=_runner_env(tmp_path),
    )
    assert r.exit_code == 0, r.output
    # Registry no longer lists it -- but the on-disk files survive.
    r2 = runner.invoke(cli, ["account", "list"], env=_runner_env(tmp_path))
    assert "business" not in r2.output
    assert biz_dir.is_dir()
    assert (biz_dir / "db.sqlite").is_file()
    assert "data dir still on disk" in r.output


def test_account_rename_keeps_slug_and_data_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Personal"], env=_runner_env(tmp_path))
    r = runner.invoke(
        cli, ["account", "rename", "Personal", "Privatkonto"],
        env=_runner_env(tmp_path),
    )
    assert r.exit_code == 0, r.output
    assert "renamed: personal" in r.output
    # Slug still works.
    r2 = runner.invoke(
        cli, ["--account", "personal", "status"], env=_runner_env(tmp_path)
    )
    assert r2.exit_code == 0
    # Display name updated in list.
    r3 = runner.invoke(cli, ["account", "list"], env=_runner_env(tmp_path))
    assert "Privatkonto" in r3.output


def test_account_add_with_explicit_slug(tmp_path: Path) -> None:
    runner = CliRunner()
    r = runner.invoke(
        cli, ["account", "add", "Business Account", "--id", "biz"],
        env=_runner_env(tmp_path),
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "accounts" / "biz" / "db.sqlite").is_file()


def test_account_add_rejects_duplicate_slug(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Personal"], env=_runner_env(tmp_path))
    r = runner.invoke(
        cli, ["account", "add", "Other", "--id", "personal"],
        env=_runner_env(tmp_path),
    )
    assert r.exit_code != 0
    assert "already in use" in r.output


# --- --account flag targets non-active account ----------------------------


def test_account_flag_targets_named_account(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """Ingest into 'personal', then verify --account business shows
    an empty DB while --account personal shows the 50 rows."""
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Personal"], env=_runner_env(tmp_path))
    runner.invoke(cli, ["account", "add", "Business"], env=_runner_env(tmp_path))
    # Business is active now (most recently added). Ingest into Personal
    # via the --account flag override.
    runner.invoke(
        cli,
        ["--account", "Personal", "ingest", "--no-embed",
         str(fixtures_dir / "sample_de.csv")],
        env=_runner_env(tmp_path),
    )
    r_pers = runner.invoke(
        cli, ["--account", "Personal", "status"], env=_runner_env(tmp_path)
    )
    assert "expenses:     50" in r_pers.output
    r_biz = runner.invoke(
        cli, ["--account", "Business", "status"], env=_runner_env(tmp_path)
    )
    assert "expenses:     0" in r_biz.output


def test_account_flag_unknown_raises(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(cli, ["account", "add", "Personal"], env=_runner_env(tmp_path))
    r = runner.invoke(
        cli, ["--account", "ghost", "status"], env=_runner_env(tmp_path),
    )
    assert r.exit_code != 0
    assert "no such account" in r.output


# --- Legacy single-account install still works ----------------------------


def test_legacy_db_auto_registers_default(tmp_path: Path) -> None:
    """If the user upgrades from a single-account install (db.sqlite at
    the root, no accounts.yaml), the first CLI call must transparently
    register a 'Default' account pointing at the existing data dir."""
    # Hand-build the legacy layout.
    (tmp_path / "db.sqlite").write_bytes(b"not really a db, just present")
    r = CliRunner().invoke(cli, ["account", "list"], env=_runner_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "default" in r.output
    # The default account points at the global home itself -- no data
    # has been moved.
    assert str(tmp_path) in r.output
