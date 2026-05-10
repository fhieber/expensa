"""Click-based CLI entrypoint.

The `main` function is the console-script target declared in pyproject.
Heavy imports (torch, sentence-transformers, transformers) are deferred
to inside the commands that need them, so `expense --help` stays snappy.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from datetime import date
from pathlib import Path

import click

from expense_analyzer import __version__
from expense_analyzer.config import (
    Config,
    load_config,
    packaged_default_categories,
)
from expense_analyzer.utils.logging import configure_logging

_CTX_KEY = "ea_state"


# --- Helpers -----------------------------------------------------------------

def _connect(cfg: Config) -> sqlite3.Connection:
    from expense_analyzer.storage.database import get_or_create_database

    return get_or_create_database(cfg.db_path)


def _embedder(cfg: Config):
    """Build the configured SentenceTransformerEmbedder. Heavy import."""
    from expense_analyzer.features.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        model_name=cfg.embedding_model,
        device=cfg.device,
        batch_size=cfg.embedding_batch_size,
    )


def _parse_date_opt(s: str | None) -> date | None:
    if not s:
        return None
    return date.fromisoformat(s)


# --- Top-level group ---------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None,
              help="Path to a YAML config file.")
@click.option("-v", "--verbose", is_flag=True)
@click.version_option(__version__)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool) -> None:
    """expense-analyzer-de — local German expense analysis."""
    configure_logging(verbose)
    cfg = load_config(config_path)
    ctx.ensure_object(dict)
    ctx.obj[_CTX_KEY] = {"config": cfg}


# --- init --------------------------------------------------------------------

@cli.command()
@click.option("--with-defaults/--no-defaults", default=True,
              help="Install the bundled German default categories.")
@click.pass_context
def init(ctx: click.Context, with_defaults: bool) -> None:
    """Create the data directory, initialize the DB, install categories."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect(cfg)
    try:
        if with_defaults:
            from expense_analyzer.storage.categories import import_categories_from_yaml

            n = import_categories_from_yaml(conn, packaged_default_categories())
            click.echo(f"installed {n} default categories")
        click.echo(f"data dir: {cfg.data_dir}")
        click.echo(f"database: {cfg.db_path}")
    finally:
        conn.close()


# --- status ------------------------------------------------------------------

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show counts: expenses, labeled vs unlabeled, categories."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        n_exp = conn.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
        n_lab = conn.execute(
            "SELECT COUNT(DISTINCT expense_id) AS n FROM labels WHERE source='user'"
        ).fetchone()["n"]
        n_cat = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
        click.echo(f"expenses:     {n_exp}")
        click.echo(f"user-labeled: {n_lab}")
        click.echo(f"categories:   {n_cat}")
        click.echo(f"data dir:     {cfg.data_dir}")
    finally:
        conn.close()


# --- categories --------------------------------------------------------------

@cli.group()
def categories() -> None:
    """Manage expense categories."""


@categories.command("list")
@click.pass_context
def categories_list(ctx: click.Context) -> None:
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.storage.categories import list_categories

        cats = list_categories(conn)
        for c in cats:
            click.echo(f"  [{c.id:>3}] {c.color}  {c.name:<20} {c.description}")
    finally:
        conn.close()


@categories.command("add")
@click.argument("name")
@click.option("--description", default="")
@click.option("--color", default="#888")
@click.pass_context
def categories_add(ctx: click.Context, name: str, description: str, color: str) -> None:
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.storage.categories import upsert_category

        cid = upsert_category(conn, name, description, color)
        click.echo(f"category #{cid}: {name}")
    finally:
        conn.close()


# --- ingest ------------------------------------------------------------------

@cli.command()
@click.argument("csvs", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def ingest(ctx: click.Context, csvs: tuple[Path, ...]) -> None:
    """Ingest one or more German bank-export CSVs (dedup-aware)."""
    if not csvs:
        click.echo("no files given", err=True)
        ctx.exit(2)
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.ingestion import ingest_csv

        for path in csvs:
            r = ingest_csv(conn, path)
            click.echo(
                f"{r.file:<40} parsed={r.parsed:>4}  new={r.inserted:>4}  duplicate={r.duplicates:>4}"
            )
    finally:
        conn.close()


# --- label / predict / train -------------------------------------------------

@cli.command()
@click.option("--n", type=int, default=None, help="How many candidates to ask about.")
@click.option(
    "--strategy",
    type=click.Choice(["uncertainty", "diverse", "outliers", "mixed"]),
    default=None,
)
@click.pass_context
def label(ctx: click.Context, n: int | None, strategy: str | None) -> None:
    """Interactively label active-learning candidates."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.features.embeddings import store_embeddings
        from expense_analyzer.ml.active_learning import pick_candidates
        from expense_analyzer.ml.classifier import CategorizationCascade
        from expense_analyzer.storage.categories import (
            add_label,
            list_categories,
        )

        cats = list_categories(conn)
        if not cats:
            click.echo("No categories yet. Run `expense init` or `expense categories add ...`.", err=True)
            ctx.exit(2)

        emb = _embedder(cfg)
        # Make sure embeddings exist for all expenses (they're needed by k-NN
        # and the diverse strategy).
        rows = conn.execute("SELECT id, combined_text FROM expenses").fetchall()
        store_embeddings(conn, emb, [(r["id"], r["combined_text"]) for r in rows])

        cascade = CategorizationCascade(conn, cfg, emb)
        try:
            cascade.fit()
        except Exception:
            pass  # may not have enough labels yet
        ids = pick_candidates(conn, cfg, emb, cascade, n=n, strategy=strategy)
        if not ids:
            click.echo("Nothing to label — every expense already has a user label.")
            return

        # Pretty menu of categories
        for i, c in enumerate(cats, start=1):
            click.echo(f"  {i:>2}) {c.name}")
        click.echo("   s) skip   q) quit\n")

        for eid in ids:
            row = conn.execute(
                "SELECT buchungsdatum, betrag_cents, counterparty, verwendungszweck "
                "FROM expenses WHERE id = ?",
                (eid,),
            ).fetchone()
            click.echo("─" * 70)
            click.echo(
                f"{row['buchungsdatum']}  {row['betrag_cents'] / 100:>10.2f} €  "
                f"{row['counterparty']}"
            )
            if row["verwendungszweck"]:
                click.echo(f"    {row['verwendungszweck']}")
            ans = click.prompt("category", default="s", show_default=False).strip().lower()
            if ans == "q":
                break
            if ans == "s" or not ans:
                continue
            try:
                idx = int(ans)
            except ValueError:
                click.echo("  ! not a number; skipping")
                continue
            if not (1 <= idx <= len(cats)):
                click.echo("  ! out of range; skipping")
                continue
            add_label(conn, eid, cats[idx - 1].id, "user")
            click.echo(f"  -> {cats[idx - 1].name}")
    finally:
        conn.close()


@cli.command()
@click.pass_context
def train(ctx: click.Context) -> None:
    """Re-train the supervised classifier on user-labeled data."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.ml.classifier import CategorizationCascade

        emb = _embedder(cfg)
        cascade = CategorizationCascade(conn, cfg, emb)
        report = cascade.fit()
        click.echo(
            f"trained {report.classifier_type}: "
            f"n_train={report.n_train} n_classes={report.n_classes} "
            f"score={report.train_score:.3f}"
        )
        if report.notes:
            click.echo(f"  note: {report.notes}")
    finally:
        conn.close()


@cli.command()
@click.option("--threshold", type=float, default=None,
              help="Override classifier confidence threshold.")
@click.option("--dry-run", is_flag=True, help="Print predictions without persisting them.")
@click.pass_context
def predict(ctx: click.Context, threshold: float | None, dry_run: bool) -> None:
    """Auto-categorize unlabeled expenses."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    if threshold is not None:
        cfg.classifier.confidence_threshold = threshold
    conn = _connect(cfg)
    try:
        from expense_analyzer.ml.classifier import CategorizationCascade
        from expense_analyzer.storage.categories import add_label

        emb = _embedder(cfg)
        cascade = CategorizationCascade(conn, cfg, emb)
        cascade.fit()  # tolerant if too few labels
        rows = conn.execute(
            """
            SELECT id FROM expenses
            WHERE id NOT IN (SELECT DISTINCT expense_id FROM labels WHERE source='user')
            """
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        if not ids:
            click.echo("nothing to predict — every expense already has a user label")
            return
        preds = cascade.predict_batch(ids)
        from collections import Counter

        counts = Counter(p.stage for p in preds)
        click.echo(f"predicted {len(preds)} records: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if dry_run:
            for p in preds[:10]:
                click.echo(f"  id={p.expense_id} cat={p.category_id} stage={p.stage} conf={p.confidence:.2f}")
            return
        n_persisted = 0
        for p in preds:
            if p.category_id is not None:
                add_label(conn, p.expense_id, p.category_id, "model", confidence=p.confidence)
                n_persisted += 1
        click.echo(f"persisted {n_persisted} model labels")
    finally:
        conn.close()


# --- cluster -----------------------------------------------------------------

@cli.command()
@click.pass_context
def cluster(ctx: click.Context) -> None:
    """Re-run UMAP+HDBSCAN clustering and persist cluster_id."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.ml.clustering import cluster_all

        emb = _embedder(cfg)
        report = cluster_all(conn, cfg, emb)
        click.echo(
            f"clusters={report.n_clusters} outliers={report.n_outliers} of {report.n_points} points"
        )
    finally:
        conn.close()


# --- viz ---------------------------------------------------------------------

@cli.command()
@click.argument("name", type=click.Choice(["pie", "histogram", "trend", "top", "calendar"]))
@click.option("--out", "out", type=click.Path(path_type=Path), default=None)
@click.option("--from", "since", type=str, default=None, help="ISO date YYYY-MM-DD")
@click.option("--to", "until", type=str, default=None, help="ISO date YYYY-MM-DD")
@click.pass_context
def viz(ctx: click.Context, name: str, out: Path | None, since: str | None, until: str | None) -> None:
    """Render a chart to HTML or PNG."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.viz import CHART_BUILDERS, save_figure

        data_fn, chart_fn = CHART_BUILDERS[name]
        date_kwargs = {"since": _parse_date_opt(since), "until": _parse_date_opt(until)}
        # `top_counterparties` doesn't take include_income; the others ignore it.
        df = data_fn(conn, **date_kwargs)
        fig = chart_fn(df)
        out = out or (cfg.data_dir / "exports" / f"{name}.html")
        path = save_figure(fig, out)
        click.echo(f"wrote {path}")
    finally:
        conn.close()


# --- vendor-lookup -----------------------------------------------------------

@cli.command("vendor-lookup")
@click.argument("counterparty")
@click.pass_context
def vendor_lookup(ctx: click.Context, counterparty: str) -> None:
    """Look up a vendor (only if vendor_lookup.enabled)."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.enrichment.vendor_web import (
            VendorLookupDisabled,
            lookup_vendor,
        )
        from expense_analyzer.ingestion.normalizer import normalize_counterparty

        cp = normalize_counterparty(counterparty)
        try:
            info = lookup_vendor(conn, cp, cfg.vendor_lookup)
        except VendorLookupDisabled as e:
            click.echo(str(e), err=True)
            ctx.exit(2)
        click.echo(f"counterparty: {info.counterparty_normalized}")
        click.echo(f"industry:     {info.industry}")
        click.echo(f"summary:      {info.summary[:300] or '(empty)'}")
    finally:
        conn.close()


# --- export ------------------------------------------------------------------

@cli.command()
@click.option("--format", "fmt", type=click.Choice(["csv", "parquet"]), default="csv")
@click.option("--out", "out", type=click.Path(path_type=Path), default=None)
@click.pass_context
def export(ctx: click.Context, fmt: str, out: Path | None) -> None:
    """Export the expenses table (with latest label) to CSV or Parquet."""
    import pandas as pd

    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        sql = """
            WITH latest_label AS (
                SELECT l.expense_id, l.category_id, l.source AS label_source, l.confidence
                FROM labels l
                JOIN (SELECT expense_id, MAX(id) AS m FROM labels GROUP BY expense_id) x
                  ON l.id = x.m
            )
            SELECT e.*, c.name AS category, ll.label_source, ll.confidence,
                   n.text AS note
            FROM expenses e
            LEFT JOIN latest_label ll ON ll.expense_id = e.id
            LEFT JOIN categories c ON c.id = ll.category_id
            LEFT JOIN notes n ON n.expense_id = e.id
            ORDER BY e.buchungsdatum, e.id
        """
        df = pd.read_sql_query(sql, conn)
        out = out or (cfg.data_dir / "exports" / f"expenses.{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            df.to_csv(out, index=False, encoding="utf-8-sig")
        else:
            df.to_parquet(out, index=False)
        click.echo(f"wrote {out}  ({len(df)} rows)")
    finally:
        conn.close()


# --- ui ----------------------------------------------------------------------

@cli.command()
@click.pass_context
def ui(ctx: click.Context) -> None:
    """Launch the local Streamlit UI (binds to 127.0.0.1)."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    streamlit_cmd = shutil.which("streamlit")
    if streamlit_cmd is None:
        click.echo("streamlit not found in PATH (pip install streamlit)", err=True)
        ctx.exit(2)
    app = Path(__file__).parent / "ui" / "streamlit_app.py"
    env = os.environ.copy()
    env["EXPENSE_ANALYZER_HOME"] = str(cfg.data_dir)
    args = [
        streamlit_cmd, "run", str(app),
        "--server.address", cfg.streamlit.host,
        "--server.port", str(cfg.streamlit.port),
        "--server.headless", "true" if cfg.streamlit.headless else "false",
        "--browser.gatherUsageStats", "false",
    ]
    click.echo(" ".join(args))
    subprocess.run(args, env=env, check=False)


# --- entrypoint --------------------------------------------------------------

def main() -> None:  # console-script target
    cli(obj={})


if __name__ == "__main__":
    main()
