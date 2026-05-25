"""Click-based CLI entrypoint.

The `main` function is the console-script target declared in pyproject.
Heavy imports (torch, sentence-transformers, transformers) are deferred
to inside the commands that need them, so `expense --help` stays snappy.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

import click

from expense_analyzer import __version__
from expense_analyzer.accounts import (
    AccountInfo,
    AccountNotFoundError,
    AccountRegistry,
    init_account_db,
    migrate_legacy_if_needed,
)
from expense_analyzer.config import (
    Config,
    load_config_for_account,
    load_global_config,
    packaged_default_categories,
)
from expense_analyzer.utils.logging import configure_logging

_CTX_KEY = "ea_state"


def _global_home() -> Path:
    """Resolve the global home directory (one per machine).

    Honours ``$EXPENSE_ANALYZER_HOME`` so tests can point us at a
    temp directory. The global home holds ``config.yaml``,
    ``accounts.yaml``, ``active_account``, and the ``accounts/``
    subtree.
    """
    return Path(
        os.environ.get("EXPENSE_ANALYZER_HOME", "~/.expense-analyzer")
    ).expanduser()


def _resolve_account(
    registry: AccountRegistry,
    explicit: str | None,
    global_home: Path,
) -> AccountInfo:
    """Pick the account a command should run against.

    Resolution order:
        1. ``--account`` flag (matched by id OR display name).
        2. ``active_account`` file (if it points at a registered slug).
        3. First registered account (deterministic fallback).
        4. Legacy single-account layout: register a ``Default`` account
           pointing at ``global_home`` so the command can still run on
           a fresh install.

    Raises :class:`click.ClickException` only when ``--account`` is
    explicitly given and doesn't match anything (the user supplied a
    wrong name -- surface it loudly).
    """
    if explicit:
        match = registry.get_by_name_or_id(explicit)
        if match is None:
            raise click.ClickException(
                f"no such account: {explicit!r}. "
                "Run `expense account list` to see what's available."
            )
        return match
    active_id = registry.get_active_id()
    if active_id is not None:
        info = registry.get(active_id)
        if info is not None:
            return info
    rows = registry.all()
    if rows:
        return rows[0]
    # Brand-new install or test environment with neither accounts.yaml
    # nor a legacy db.sqlite. Synthesize a default pointing at
    # global_home so commands like `expense init` keep working.
    return AccountInfo(id="default", name="Default", data_dir=global_home)


# --- Helpers -----------------------------------------------------------------

def _connect(cfg: Config) -> sqlite3.Connection:
    from expense_analyzer.storage import crypto
    from expense_analyzer.storage.database import get_or_create_database

    password: str | None = None
    if crypto.looks_encrypted(cfg.db_path):
        password = os.environ.get("EXPENSE_ANALYZER_DB_PASSWORD")
        if not password:
            if sys.stdin.isatty():
                password = click.prompt("Database password", hide_input=True)
            else:
                raise click.ClickException(
                    "database is encrypted: set EXPENSE_ANALYZER_DB_PASSWORD "
                    "or run the command interactively to be prompted."
                )
    return get_or_create_database(cfg.db_path, password)


def _embedder(cfg: Config, verbose: bool = True):
    """Build the configured SentenceTransformerEmbedder. Heavy import.

    `verbose=True` (the default for CLI use) makes large encode() calls
    show a tqdm progress bar.
    """
    from expense_analyzer.features.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        model_name=cfg.embedding_model,
        device=cfg.device,
        batch_size=cfg.embedding_batch_size,
        verbose=verbose,
    )


def _parse_date_opt(s: str | None) -> date | None:
    if not s:
        return None
    return date.fromisoformat(s)


# --- Top-level group ---------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None,
              help="Path to a YAML config file.")
@click.option("--account", "account_id", default=None,
              help="Target account by name or slug (overrides the active account).")
@click.option("-v", "--verbose", is_flag=True)
@click.version_option(__version__)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Path | None,
    account_id: str | None,
    verbose: bool,
) -> None:
    """expense-analyzer-de — local German expense analysis."""
    configure_logging(verbose)
    global_home = _global_home()
    global_cfg = load_global_config(config_path)
    registry = migrate_legacy_if_needed(global_home)
    account = _resolve_account(registry, account_id, global_home)
    cfg = load_config_for_account(account, global_cfg)
    ctx.ensure_object(dict)
    ctx.obj[_CTX_KEY] = {
        "config": cfg,
        "global_cfg": global_cfg,
        "registry": registry,
        "account": account,
        "global_home": global_home,
    }


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


@categories.command("remove")
@click.argument("name")
@click.option("--force", is_flag=True,
              help="Delete even if labels reference this category (cascades).")
@click.option("--yes", "yes", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def categories_remove(ctx: click.Context, name: str, force: bool, yes: bool) -> None:
    """Remove a category. Refuses if labels reference it unless --force."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.storage.admin import (
            category_removal_impact,
            remove_category,
        )

        impact = category_removal_impact(conn, name)
        if not impact.exists:
            click.echo(f"no such category: {name!r}", err=True)
            ctx.exit(2)
        if impact.n_labels > 0 and not force:
            click.echo(
                f"refusing: {impact.n_labels} label(s) reference {name!r}. "
                "Re-run with --force to cascade-delete those labels.",
                err=True,
            )
            ctx.exit(3)
        if not yes:
            msg = f"Remove category {name!r}"
            if impact.n_labels > 0:
                msg += f" and cascade-delete {impact.n_labels} label(s)"
            msg += "?"
            click.confirm(msg, abort=True)
        result = remove_category(conn, name)
        click.echo(
            f"removed {result.name}; {result.n_labels_deleted} label(s) cascaded"
        )
    finally:
        conn.close()


# --- account -----------------------------------------------------------------

@cli.group("account")
def account() -> None:
    """Manage accounts (separate SQLite DBs)."""


@account.command("list")
@click.pass_context
def account_list(ctx: click.Context) -> None:
    """Show every registered account; the active one is marked ``*``."""
    registry: AccountRegistry = ctx.obj[_CTX_KEY]["registry"]
    if len(registry) == 0:
        click.echo("(no accounts registered yet)")
        click.echo("Hint: `expense account add NAME` to create your first.")
        return
    active_id = registry.get_active_id()
    for a in registry.all():
        marker = "*" if a.id == active_id else " "
        click.echo(f"  {marker} {a.id:<20} {a.name:<24} {a.data_dir}")


@account.command("add")
@click.argument("name")
@click.option("--id", "slug", default=None,
              help="Override the auto-derived slug (sanitised).")
@click.option("--with-defaults/--no-defaults", default=True,
              help="Seed the bundled German default categories.")
@click.option("--use/--no-use", default=True,
              help="Switch the active account to the newly-created one.")
@click.pass_context
def account_add(
    ctx: click.Context,
    name: str,
    slug: str | None,
    with_defaults: bool,
    use: bool,
) -> None:
    """Register a new account, create its directory + DB."""
    registry: AccountRegistry = ctx.obj[_CTX_KEY]["registry"]
    try:
        info = registry.add(name, slug=slug)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    registry.save()
    conn = init_account_db(info, with_defaults=with_defaults)
    conn.close()
    click.echo(f"added account: {info.id}  ({info.name})")
    click.echo(f"  data dir:    {info.data_dir}")
    click.echo(f"  database:    {info.db_path}")
    if with_defaults:
        click.echo("  seeded default German categories.")
    if use:
        registry.set_active_id(info.id)
        click.echo(f"  active account is now: {info.id}")


@account.command("remove")
@click.argument("name")
@click.option("--yes", "yes", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def account_remove(ctx: click.Context, name: str, yes: bool) -> None:
    """Drop an account from the registry. **Does not delete files**;
    the path is printed so you can remove it manually if you want."""
    registry: AccountRegistry = ctx.obj[_CTX_KEY]["registry"]
    info = registry.get_by_name_or_id(name)
    if info is None:
        raise click.ClickException(f"no such account: {name!r}")
    if not yes:
        click.confirm(
            f"Remove account {info.id!r} from the registry? "
            "(The DB on disk will NOT be deleted.)",
            abort=True,
        )
    try:
        registry.remove(info.id)
    except AccountNotFoundError as e:
        raise click.ClickException(str(e)) from e
    registry.save()
    click.echo(f"removed account: {info.id}")
    click.echo(
        f"  data dir still on disk: {info.data_dir}\n"
        "  (delete it manually if you want the files gone too)"
    )


@account.command("rename")
@click.argument("name")
@click.argument("new_name")
@click.pass_context
def account_rename(ctx: click.Context, name: str, new_name: str) -> None:
    """Change the display name. The slug + directory stay put -- this
    is purely cosmetic so existing scripts using ``--account <slug>``
    keep working."""
    registry: AccountRegistry = ctx.obj[_CTX_KEY]["registry"]
    info = registry.get_by_name_or_id(name)
    if info is None:
        raise click.ClickException(f"no such account: {name!r}")
    try:
        updated = registry.rename(info.id, new_name)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    registry.save()
    click.echo(f"renamed: {updated.id}  -> {updated.name}")


@account.command("use")
@click.argument("name")
@click.pass_context
def account_use(ctx: click.Context, name: str) -> None:
    """Make this account active (writes ``active_account``)."""
    registry: AccountRegistry = ctx.obj[_CTX_KEY]["registry"]
    info = registry.get_by_name_or_id(name)
    if info is None:
        raise click.ClickException(f"no such account: {name!r}")
    registry.set_active_id(info.id)
    click.echo(f"active account is now: {info.id}  ({info.name})")


def _resolve_account_for_crypto(ctx: click.Context, name: str | None) -> AccountInfo:
    """Pick the account a crypto command targets: an explicit NAME if given,
    otherwise the resolved active account from the root group."""
    if name:
        registry: AccountRegistry = ctx.obj[_CTX_KEY]["registry"]
        info = registry.get_by_name_or_id(name)
        if info is None:
            raise click.ClickException(f"no such account: {name!r}")
        return info
    return ctx.obj[_CTX_KEY]["account"]


@account.command("encrypt")
@click.argument("name", required=False)
@click.option(
    "--delete-plaintext/--keep-plaintext",
    "delete_plain",
    default=None,
    help="Delete (or keep) the plaintext safety copy without prompting. "
    "If neither is given you're asked interactively.",
)
@click.pass_context
def account_encrypt(
    ctx: click.Context, name: str | None, delete_plain: bool | None
) -> None:
    """Encrypt an account's database at rest (AES-256 via SQLCipher).

    Prompts for a new password (twice), then offers to delete the
    timestamped plaintext safety copy it keeps next to the DB."""
    from expense_analyzer.storage import crypto

    info = _resolve_account_for_crypto(ctx, name)
    if not crypto.encryption_available():
        raise click.ClickException(
            "encryption needs the optional dependency: "
            "pip install -e '.[encryption]'  (from the repo root)"
        )
    if not info.db_path.is_file():
        raise click.ClickException(
            f"no database yet for {info.id!r}; run `expense init` first."
        )
    if crypto.looks_encrypted(info.db_path):
        raise click.ClickException(f"{info.id!r} is already encrypted.")
    pw = click.prompt("New password", hide_input=True, confirmation_prompt=True)
    try:
        safety = crypto.encrypt_file(info.db_path, pw, keep_safety=True)
    except crypto.EncryptionError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"encrypted {info.db_path}")
    if safety is None:
        return
    # Offer to remove the plaintext safety copy. Default to keeping it; when
    # the flag isn't set, ask interactively (but only if there's a TTY, so a
    # piped/cron run keeps the copy rather than aborting on the prompt).
    if delete_plain is None:
        delete_plain = sys.stdin.isatty() and click.confirm(
            f"A plaintext copy was saved at {safety}. Delete it now?",
            default=False,
        )
    if delete_plain:
        try:
            safety.unlink()
            click.echo("  deleted the plaintext safety copy.")
        except OSError as e:
            click.echo(f"  could not delete the plaintext copy: {e}", err=True)
    else:
        click.echo(
            f"  plaintext safety copy kept: {safety}\n"
            "  delete it once you've confirmed the password works."
        )


@account.command("decrypt")
@click.argument("name", required=False)
@click.pass_context
def account_decrypt(ctx: click.Context, name: str | None) -> None:
    """Remove encryption from an account's database.

    Reads the current password from ``EXPENSE_ANALYZER_DB_PASSWORD`` or
    prompts for it. Keeps a timestamped encrypted safety copy."""
    from expense_analyzer.storage import crypto

    info = _resolve_account_for_crypto(ctx, name)
    if not crypto.looks_encrypted(info.db_path):
        raise click.ClickException(f"{info.id!r} is not encrypted.")
    pw = os.environ.get("EXPENSE_ANALYZER_DB_PASSWORD") or click.prompt(
        "Current password", hide_input=True
    )
    try:
        safety = crypto.decrypt_file(info.db_path, pw, keep_safety=True)
    except crypto.WrongPassword as e:
        raise click.ClickException("incorrect password.") from e
    except crypto.EncryptionError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"decrypted {info.db_path}")
    if safety is not None:
        click.echo(f"  encrypted safety copy: {safety}")


@account.command("passwd")
@click.argument("name", required=False)
@click.pass_context
def account_passwd(ctx: click.Context, name: str | None) -> None:
    """Change an encrypted account's password.

    Reads the current password from ``EXPENSE_ANALYZER_DB_PASSWORD`` or
    prompts for it, then prompts for the new password (twice)."""
    from expense_analyzer.storage import crypto

    info = _resolve_account_for_crypto(ctx, name)
    if not crypto.looks_encrypted(info.db_path):
        raise click.ClickException(
            f"{info.id!r} is not encrypted; use `expense account encrypt` first."
        )
    old = os.environ.get("EXPENSE_ANALYZER_DB_PASSWORD") or click.prompt(
        "Current password", hide_input=True
    )
    new = click.prompt("New password", hide_input=True, confirmation_prompt=True)
    try:
        crypto.change_password(info.db_path, old, new)
    except crypto.WrongPassword as e:
        raise click.ClickException("incorrect current password.") from e
    except crypto.EncryptionError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"password changed for {info.id!r}.")


# --- own-iban ----------------------------------------------------------------

@cli.group("own-iban")
def own_iban() -> None:
    """Manage your own IBANs (drives the ``iban_is_known_self`` flag)."""


@own_iban.command("list")
@click.pass_context
def own_iban_list(ctx: click.Context) -> None:
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.storage.own_ibans import list_own_ibans

        rows = list_own_ibans(conn)
        if not rows:
            click.echo("(no own IBANs registered)")
            return
        for r in rows:
            click.echo(f"  {r.iban}  {r.label or ''}")
    finally:
        conn.close()


@own_iban.command("add")
@click.argument("iban")
@click.option("--label", default="", help="Friendly name (e.g. 'Main checking').")
@click.pass_context
def own_iban_add(ctx: click.Context, iban: str, label: str) -> None:
    """Register an IBAN and retroactively flag every matching expense."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.storage.own_ibans import add_own_iban

        try:
            rep = add_own_iban(conn, iban, label=label or None)
        except ValueError as e:
            click.echo(f"refusing: {e}", err=True)
            ctx.exit(2)
        click.echo(
            f"added; flagged {rep.n_now_self} existing expense(s) as internal."
        )
    finally:
        conn.close()


@own_iban.command("remove")
@click.argument("iban")
@click.option("--yes", "yes", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def own_iban_remove(ctx: click.Context, iban: str, yes: bool) -> None:
    """Drop an own IBAN and clear the flag on every matching expense."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.storage.own_ibans import remove_own_iban

        if not yes:
            click.confirm(f"Remove own IBAN {iban}?", abort=True)
        rep = remove_own_iban(conn, iban)
        click.echo(
            f"removed; cleared the flag on {rep.n_was_self} expense(s)."
        )
    finally:
        conn.close()


# --- reset -------------------------------------------------------------------

@cli.command()
@click.option("--all", "wipe_all", is_flag=True,
              help="Also wipe categories and own_ibans (default keeps them).")
@click.option("--yes", "yes", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def reset(ctx: click.Context, wipe_all: bool, yes: bool) -> None:
    """Wipe all ingested expenses and ML state. **Destructive.**

    Default: clears expenses + labels + notes + embeddings + vendor_cache +
    model_versions, but keeps categories and own_ibans.

    With --all: also clears categories and own_ibans (full factory reset).
    """
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.storage.admin import (
            _CONFIG_TABLES,
            _DATA_TABLES,
            _row_counts,
            reset_all,
            reset_data,
        )

        tables = _DATA_TABLES + (_CONFIG_TABLES if wipe_all else ())
        counts = _row_counts(conn, tables)
        if sum(counts.values()) == 0:
            click.echo("nothing to delete; database is already empty.")
            return
        click.echo("Will delete:")
        for t, n in counts.items():
            if n:
                click.echo(f"  {t:<16} {n} row(s)")
        if not yes:
            click.confirm("Proceed?", abort=True)
        report = (reset_all if wipe_all else reset_data)(conn)
        click.echo(f"deleted {report.total} row(s) across {len(report.table_counts)} table(s)")
    finally:
        conn.close()


# --- ingest ------------------------------------------------------------------

@cli.command()
@click.argument("csvs", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--no-embed", is_flag=True,
              help="Skip computing embeddings for new rows. They'll be computed lazily later.")
@click.option("--enrich", "enrich_csvs", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Secondary-source CSV(s) (e.g. a PayPal activity export) to "
                   "match against the ingested rows and enrich. Source auto-detected.")
@click.option("--dry-run", is_flag=True,
              help="Don't touch the DB. With --enrich, show on concrete records "
                   "how the secondary source would enrich them (before vs after).")
@click.pass_context
def ingest(
    ctx: click.Context,
    csvs: tuple[Path, ...],
    no_embed: bool,
    enrich_csvs: tuple[Path, ...],
    dry_run: bool,
) -> None:
    """Ingest one or more German bank-export CSVs (dedup-aware).

    By default also computes sentence-transformer embeddings for every newly
    inserted row, so downstream label/predict commands are fast. Pass
    ``--enrich`` with a secondary CSV (e.g. PayPal) to enrich matched rows in
    the same run, or add ``--dry-run`` to preview the enrichment without
    writing anything.
    """
    if not csvs:
        click.echo("no files given", err=True)
        ctx.exit(2)
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    if dry_run:
        _preview_enrich(cfg, csvs, enrich_csvs)
        return
    conn = _connect(cfg)
    try:
        from expense_analyzer.ingestion import ingest_csv

        emb = None
        if not no_embed:
            click.echo(f"loading embedding model `{cfg.embedding_model}`...")
            emb = _embedder(cfg)

        for path in csvs:
            click.echo(f"ingesting {path.name}...")
            r = ingest_csv(conn, path, embedder=emb)
            click.echo(
                f"{r.file:<40} parsed={r.parsed:>4}  new={r.inserted:>4}  "
                f"duplicate={r.duplicates:>4}  embedded={r.embedded:>4}"
            )

        for path in enrich_csvs:
            _run_enrich(conn, cfg, path, source="auto", embedder=emb)
    finally:
        conn.close()


def _run_enrich(conn, cfg: Config, path: Path, source: str, embedder) -> None:
    """Resolve an adapter for ``path``, parse it, run the enrichment engine
    and echo a one-line report. Shared by ``ingest --enrich`` and ``enrich``."""
    from expense_analyzer.enrichment.secondary import enrich_from_records
    from expense_analyzer.ingestion.sources import detect_adapter, get_adapter

    adapter = get_adapter(source) if source != "auto" else detect_adapter(path)
    records = adapter.parse(path)
    rep = enrich_from_records(
        conn, records, adapter, embedder=embedder,
        date_window_days=cfg.enrichment.date_window_days,
    )
    click.echo(
        f"{path.name:<40} source={rep.source}  parsed={rep.parsed:>4}  "
        f"matched={rep.matched:>4}  ambiguous={rep.ambiguous:>4}  "
        f"unmatched={rep.unmatched_expenses:>4}  reembedded={rep.reembedded:>4}"
    )


def _preview_enrich(
    cfg: Config, csvs: tuple[Path, ...], enrich_csvs: tuple[Path, ...]
) -> None:
    """``ingest --dry-run`` showcase: parse the bank CSV(s) and the secondary
    CSV(s) in memory, match them, and print each enriched record before vs
    after. Writes nothing."""
    from expense_analyzer.enrichment.secondary import preview_enrichment
    from expense_analyzer.ingestion.csv_loader import parse_csv
    from expense_analyzer.ingestion.sources import detect_adapter

    if not enrich_csvs:
        click.echo(
            "--dry-run showcases secondary-source enrichment; "
            "pass --enrich <csv> (e.g. a PayPal export) to see it."
        )
        return

    rows = [row for path in csvs for row in parse_csv(path)]
    click.echo(f"parsed {len(rows)} bank row(s) from {len(csvs)} file(s) (not stored)\n")

    for path in enrich_csvs:
        adapter = detect_adapter(path)
        records = adapter.parse(path)
        rep = preview_enrichment(
            rows, records, adapter,
            date_window_days=cfg.enrichment.date_window_days,
        )
        click.echo(
            f"=== {path.name} (source={rep.source}) — "
            f"matched={rep.matched}, ambiguous={rep.ambiguous}, "
            f"unmatched={rep.unmatched} of {rep.candidate_rows} candidate row(s) ==="
        )
        if not rep.previews:
            click.echo("  (no records matched a bank row)\n")
            continue
        for pv in rep.previews:
            amount = pv.row.betrag_cents / 100
            click.echo(
                f"\n● {pv.row.buchungsdatum}  {amount:>9.2f} €  "
                f"{pv.row.zahlungsempfaenger or pv.row.zahlungspflichtiger}"
            )
            click.echo(f"    bank Verwendungszweck : {pv.row.verwendungszweck or '—'}")
            click.echo(f"    without enrichment    : {pv.combined_before}")
            click.echo(f"    with enrichment       : {pv.combined_after}")
            click.echo(
                f"    └ {rep.source} txn {pv.record.source_ref}: "
                f"{pv.record.counterparty} — {pv.record.description or '—'}"
            )
        click.echo("")


@cli.command()
@click.argument("csvs", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--source", default="auto",
              help="Secondary-source adapter name (e.g. 'paypal') or 'auto' to detect.")
@click.option("--no-embed", is_flag=True,
              help="Skip re-embedding enriched rows. Their combined_text is still updated.")
@click.pass_context
def enrich(ctx: click.Context, csvs: tuple[Path, ...], source: str, no_embed: bool) -> None:
    """Enrich already-ingested expenses from a secondary CSV.

    A secondary source (e.g. a PayPal "Aktivitäten" export) is matched to the
    bank transactions already in the DB by amount + nearby date; the matched
    expense gets the real merchant/item and is re-embedded so categorization
    improves. Re-running is idempotent.
    """
    if not csvs:
        click.echo("no files given", err=True)
        ctx.exit(2)
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        emb = None
        if not no_embed:
            click.echo(f"loading embedding model `{cfg.embedding_model}`...")
            emb = _embedder(cfg)
        for path in csvs:
            _run_enrich(conn, cfg, path, source=source, embedder=emb)
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

        click.echo(f"loading embedding model `{cfg.embedding_model}`...")
        emb = _embedder(cfg)
        # Make sure embeddings exist for all expenses (they're needed by k-NN
        # and the diverse strategy).
        rows = conn.execute("SELECT id, combined_text FROM expenses").fetchall()
        click.echo(f"computing embeddings for {len(rows)} expense(s)...")
        n_added = store_embeddings(conn, emb, [(r["id"], r["combined_text"]) for r in rows])
        click.echo(f"  {n_added} new, {len(rows) - n_added} cached")

        cascade = CategorizationCascade(conn, cfg, emb)
        click.echo("training cascade on existing labels...")
        try:
            cascade.fit()
        except Exception:
            pass  # may not have enough labels yet
        click.echo(f"selecting {n or cfg.active_learning.default_batch_size} candidates...")
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

        click.echo(f"loading embedding model `{cfg.embedding_model}`...")
        emb = _embedder(cfg)
        cascade = CategorizationCascade(conn, cfg, emb)
        click.echo("training on user labels (embeddings will be computed/cached)...")
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

        click.echo(f"loading embedding model `{cfg.embedding_model}`...")
        emb = _embedder(cfg)
        cascade = CategorizationCascade(conn, cfg, emb)
        click.echo("training on user labels...")
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
        click.echo(f"computing embeddings + predicting for {len(ids)} expense(s)...")
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


@cli.command("eval")
@click.option("--folds", type=int, default=5, show_default=True,
              help="Number of cross-validation folds.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--ablation/--no-ablation", "do_ablation", default=True,
              help="Also run cumulative + leave-one-out stage ablation.")
@click.option("--no-zeroshot", is_flag=True,
              help="Skip the slow zero-shot NLI stage during evaluation.")
@click.pass_context
def evaluate(
    ctx: click.Context, folds: int, seed: int, do_ablation: bool, no_zeroshot: bool
) -> None:
    """Cross-validate the cascade on user labels and report quality metrics."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    if no_zeroshot:
        cfg.zeroshot.enabled = False
    conn = _connect(cfg)
    try:
        from expense_analyzer.ml.evaluation import ablation, cross_validate
        from expense_analyzer.storage.categories import list_categories

        click.echo(f"loading embedding model `{cfg.embedding_model}`...")
        emb = _embedder(cfg)
        click.echo(f"cross-validating ({folds} folds)...")
        result = cross_validate(conn, cfg, emb, n_folds=folds, seed=seed)

        if result.n_folds == 0:
            click.echo(result.notes or "not enough labeled data to evaluate")
            return

        id_to_name = {c.id: c.name for c in list_categories(conn)}
        click.echo(
            f"\nlabels evaluated: {result.n_labeled} · folds: {result.n_folds}"
        )
        if result.dropped_singletons:
            click.echo(
                f"  ({result.dropped_singletons} dropped: category had <2 examples)"
            )
        click.echo(
            f"accuracy={result.accuracy:.3f} "
            f"accuracy_covered={result.accuracy_covered:.3f} "
            f"coverage={result.coverage:.3f}"
        )
        click.echo(
            f"macro_f1={result.macro_f1:.3f} weighted_f1={result.weighted_f1:.3f}"
        )

        click.echo("\nper-stage contribution (coverage / accuracy):")
        for s in result.stage_breakdown:
            click.echo(
                f"  {s.stage:<20} fired={s.n_predicted:<4} "
                f"correct={s.n_correct:<4} acc={s.accuracy:.3f}"
            )

        click.echo("\nper-category (precision / recall / f1 / support):")
        for pc in result.per_category:
            name = id_to_name.get(pc.category_id, str(pc.category_id))
            click.echo(
                f"  {name:<24} P={pc.precision:.3f} R={pc.recall:.3f} "
                f"F1={pc.f1:.3f} n={pc.support}"
            )

        if do_ablation:
            click.echo("\nrunning stage ablation...")
            abl = ablation(conn, cfg, emb, n_folds=folds, seed=seed)
            click.echo("cumulative stages (accuracy / macro_f1):")
            for label, acc, f1 in abl.cumulative:
                click.echo(f"  {label:<60} acc={acc:.3f} f1={f1:.3f}")
            click.echo(
                f"leave-one-out (full accuracy={abl.full_accuracy:.3f}):"
            )
            for stage, acc, _f1, delta in abl.leave_one_out:
                click.echo(
                    f"  without {stage:<20} acc={acc:.3f} (Δ={delta:+.3f})"
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
@click.argument("counterparty", required=False)
@click.option("--all", "all_vendors", is_flag=True,
              help="Look up every distinct counterparty in the DB that isn't yet cached.")
@click.pass_context
def vendor_lookup(ctx: click.Context, counterparty: str | None, all_vendors: bool) -> None:
    """Look up one vendor by name, or `--all` to populate vendor_cache for
    every distinct counterparty in the DB. Requires vendor_lookup.enabled."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    if not counterparty and not all_vendors:
        click.echo("provide a counterparty name or pass --all", err=True)
        ctx.exit(2)
    conn = _connect(cfg)
    try:
        from expense_analyzer.enrichment.vendor_web import (
            VendorLookupDisabled,
            lookup_vendor,
        )
        from expense_analyzer.ingestion.normalizer import normalize_counterparty

        try:
            if all_vendors:
                rows = conn.execute(
                    """
                    SELECT DISTINCT e.counterparty_normalized
                    FROM expenses e
                    LEFT JOIN vendor_cache vc
                      ON vc.counterparty_normalized = e.counterparty_normalized
                    WHERE e.counterparty_normalized IS NOT NULL
                      AND e.counterparty_normalized <> ''
                      AND vc.counterparty_normalized IS NULL
                    ORDER BY e.counterparty_normalized
                    """
                ).fetchall()
                if not rows:
                    click.echo("nothing to look up — every counterparty is already cached.")
                    return
                click.echo(f"looking up {len(rows)} vendor(s)...")
                for r in rows:
                    cp = r["counterparty_normalized"]
                    try:
                        info = lookup_vendor(conn, cp, cfg.vendor_lookup)
                    except VendorLookupDisabled as e:
                        click.echo(str(e), err=True)
                        ctx.exit(2)
                    click.echo(f"  {cp:<40} -> {info.industry}")
            else:
                cp = normalize_counterparty(counterparty)
                info = lookup_vendor(conn, cp, cfg.vendor_lookup)
                click.echo(f"counterparty: {info.counterparty_normalized}")
                click.echo(f"industry:     {info.industry}")
                click.echo(f"summary:      {info.summary[:300] or '(empty)'}")
        except VendorLookupDisabled as e:
            click.echo(str(e), err=True)
            ctx.exit(2)
    finally:
        conn.close()


# --- vendor cache inspection -------------------------------------------------

@cli.group("vendor")
def vendor() -> None:
    """Inspect the cached vendor lookup results (industry tags + snippets).

    Read-only. To populate or refresh the cache, use ``expense vendor-lookup``.
    """


@vendor.command("list")
@click.option("--industry", "industry_filter", default=None,
              help="Only show entries whose industry contains this substring (case-insensitive).")
@click.option("--limit", type=int, default=100,
              help="Cap on rows shown. Use 0 for unlimited.")
@click.option("--snippet-chars", type=int, default=80,
              help="Snippet preview width per row. 0 hides snippets entirely.")
@click.pass_context
def vendor_list(
    ctx: click.Context,
    industry_filter: str | None,
    limit: int,
    snippet_chars: int,
) -> None:
    """Show every cached vendor with its industry tag and a snippet preview.

    Rows are ordered by fetched_at DESC so the most recently looked-up
    vendors surface first.
    """
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.enrichment.vendor_web import normalize_industry

        rows = conn.execute(
            """
            SELECT counterparty_normalized, summary, industry, fetched_at
            FROM vendor_cache
            ORDER BY fetched_at DESC, counterparty_normalized ASC
            """
        ).fetchall()
        if not rows:
            click.echo(
                "(vendor cache is empty — run `expense vendor-lookup --all` "
                "to populate it)"
            )
            return

        # Filter / migrate / cap, then render in fixed-width columns so
        # the output is grep-friendly.
        needle = (industry_filter or "").lower()
        out: list[tuple[str, str, str, str]] = []
        for r in rows:
            ind = normalize_industry(r["industry"]) or "(?)"
            if needle and needle not in ind.lower():
                continue
            snippet = (r["summary"] or "").replace("\n", " ").strip()
            if snippet_chars > 0 and len(snippet) > snippet_chars:
                snippet = snippet[:snippet_chars].rstrip() + "…"
            elif snippet_chars == 0:
                snippet = ""
            fetched = str(r["fetched_at"] or "")[:19]
            out.append((r["counterparty_normalized"], ind, snippet, fetched))

        if not out:
            click.echo("(no vendor cache entries matched the filter)")
            return

        if limit > 0:
            out = out[:limit]

        # Compute column widths from the visible rows so short tables
        # don't waste horizontal space.
        cp_w = min(40, max(18, max(len(o[0]) for o in out)))
        ind_w = min(20, max(8, max(len(o[1]) for o in out)))
        click.echo(
            f"{'COUNTERPARTY':<{cp_w}}  {'INDUSTRY':<{ind_w}}  "
            f"{'FETCHED_AT':<19}  SNIPPET"
        )
        click.echo("-" * (cp_w + ind_w + 19 + 12))
        for cp, ind, snippet, fetched in out:
            line = (
                f"{cp[:cp_w]:<{cp_w}}  {ind[:ind_w]:<{ind_w}}  "
                f"{fetched:<19}  {snippet}"
            )
            click.echo(line)

        total = conn.execute("SELECT COUNT(*) AS n FROM vendor_cache").fetchone()["n"]
        click.echo(f"\n{len(out)} of {total} cached vendor(s) shown.")
    finally:
        conn.close()


@vendor.command("show")
@click.argument("counterparty")
@click.pass_context
def vendor_show(ctx: click.Context, counterparty: str) -> None:
    """Print the full cached entry for one counterparty (industry + complete snippet)."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.enrichment.vendor_web import normalize_industry
        from expense_analyzer.ingestion.normalizer import normalize_counterparty

        # Accept either the raw name or the already-normalized form.
        cp = normalize_counterparty(counterparty)
        r = conn.execute(
            """
            SELECT counterparty_normalized, summary, industry, fetched_at
            FROM vendor_cache
            WHERE counterparty_normalized = ?
            """,
            (cp,),
        ).fetchone()
        if r is None:
            click.echo(
                f"no cache entry for {cp!r}. Run "
                f"`expense vendor-lookup {counterparty!r}` to populate it.",
                err=True,
            )
            ctx.exit(1)
        click.echo(f"counterparty: {r['counterparty_normalized']}")
        click.echo(f"industry:     {normalize_industry(r['industry'])}")
        click.echo(f"fetched_at:   {r['fetched_at']}")
        click.echo("snippet:")
        snippet = (r["summary"] or "").strip()
        if not snippet:
            click.echo("  (empty)")
        else:
            # Indent each wrapped line for easy visual scanning.
            for line in snippet.splitlines() or [snippet]:
                click.echo(f"  {line}")
    finally:
        conn.close()


@vendor.command("clear")
@click.option("--counterparty", default=None,
              help="Remove only this counterparty's cache row. If omitted, clears the whole table.")
@click.option("--yes", "confirmed", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_context
def vendor_clear(
    ctx: click.Context, counterparty: str | None, confirmed: bool
) -> None:
    """Drop cached vendor rows. Use to force a refresh on next lookup."""
    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    conn = _connect(cfg)
    try:
        from expense_analyzer.ingestion.normalizer import normalize_counterparty

        if counterparty:
            cp = normalize_counterparty(counterparty)
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM vendor_cache WHERE counterparty_normalized = ?",
                (cp,),
            ).fetchone()["n"]
            if n == 0:
                click.echo(f"no cache entry for {cp!r}.")
                return
            if not confirmed and not click.confirm(
                f"delete cached entry for {cp!r}?"
            ):
                click.echo("aborted.")
                return
            conn.execute(
                "DELETE FROM vendor_cache WHERE counterparty_normalized = ?", (cp,)
            )
            conn.commit()
            click.echo("deleted 1 row.")
        else:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM vendor_cache"
            ).fetchone()["n"]
            if n == 0:
                click.echo("vendor cache already empty.")
                return
            if not confirmed and not click.confirm(
                f"delete ALL {n} cached vendor row(s)?"
            ):
                click.echo("aborted.")
                return
            conn.execute("DELETE FROM vendor_cache")
            conn.commit()
            click.echo(f"deleted {n} row(s).")
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

def _build_streamlit_args(cfg: Config) -> list[str]:
    app = Path(__file__).parent / "ui" / "streamlit_app.py"
    return [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", cfg.streamlit.host,
        "--server.port", str(cfg.streamlit.port),
        "--server.headless", "true",
        "--server.runOnSave", "true",         # auto-reload on file change
        "--server.fileWatcherType", "auto",
        "--browser.gatherUsageStats", "false",
    ]


def _open_browser_when_ready(host: str, port: int, timeout: float = 30.0) -> None:
    import socket
    import threading
    import time
    import webbrowser

    def worker() -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    webbrowser.open(f"http://{host}:{port}")
                    return
            except OSError:
                time.sleep(0.3)

    threading.Thread(target=worker, daemon=True).start()


@cli.command()
@click.option("--foreground", is_flag=True,
              help="Run streamlit attached to this terminal instead of detaching.")
@click.option("--no-browser", is_flag=True,
              help="Don't auto-open the system browser when the UI is ready.")
@click.pass_context
def ui(ctx: click.Context, foreground: bool, no_browser: bool) -> None:
    """Launch the local Streamlit UI (binds to 127.0.0.1).

    Default: spawns streamlit *detached* — your terminal returns
    immediately. Stop it later with `expense ui-stop`. The Streamlit
    server has `--server.runOnSave true`, so editing source files
    triggers an automatic rerun; just refresh the browser.

    With `--foreground`, the server runs attached to this terminal
    and Ctrl+C stops it (clean shutdown with a 5 s force-kill
    fallback for the Tornado/asyncio Windows shutdown bug).
    """
    from expense_analyzer.ui.process import (
        UiProcessInfo,
        is_alive,
        read_pid_file,
        spawn_detached,
        write_pid_file,
    )

    cfg: Config = ctx.obj[_CTX_KEY]["config"]
    global_home: Path = ctx.obj[_CTX_KEY]["global_home"]
    try:
        import streamlit  # noqa: F401
    except ImportError:
        click.echo("streamlit is not installed in this environment (pip install streamlit)", err=True)
        ctx.exit(2)

    # PID file lives under global_home, not the active account's data_dir
    # -- the Streamlit server is one-per-machine and serves every
    # registered account via the in-UI account picker.
    existing = read_pid_file(global_home)
    if existing is not None and is_alive(existing.pid):
        click.echo(
            f"UI already running (pid {existing.pid}, http://{existing.host}:{existing.port}). "
            "Use `expense ui-stop` first.",
            err=True,
        )
        ctx.exit(2)

    args = _build_streamlit_args(cfg)
    env = os.environ.copy()
    env["EXPENSE_ANALYZER_HOME"] = str(global_home)
    url = f"http://{cfg.streamlit.host}:{cfg.streamlit.port}"
    click.echo(f"-> {url}")

    if not no_browser:
        _open_browser_when_ready(cfg.streamlit.host, cfg.streamlit.port)

    if foreground:
        _run_foreground(args, env)
        return

    log_path = global_home / "ui.log"
    pid = spawn_detached(args, env=env, log_path=log_path)
    info = UiProcessInfo(
        pid=pid,
        port=cfg.streamlit.port,
        host=cfg.streamlit.host,
        started_at=__import__("time").time(),
    )
    write_pid_file(global_home, info)
    click.echo(f"streamlit detached as pid {pid} (logs: {log_path})")
    click.echo("stop with: expense ui-stop")


def _run_foreground(args: list[str], env: dict) -> None:
    import signal

    is_windows = sys.platform == "win32"
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if is_windows else 0
    proc = subprocess.Popen(args, env=env, creationflags=creationflags)
    try:
        proc.wait()
    except KeyboardInterrupt:
        click.echo("\nstopping streamlit...")
        try:
            proc.send_signal(
                signal.CTRL_BREAK_EVENT if is_windows else signal.SIGTERM
            )
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            click.echo("streamlit did not exit cleanly in 5s; killing.")
            proc.kill()
            proc.wait()


@cli.command("ui-stop")
@click.pass_context
def ui_stop(ctx: click.Context) -> None:
    """Stop the detached Streamlit UI."""
    from expense_analyzer.ui.process import (
        clear_pid_file,
        graceful_stop,
        is_alive,
        read_pid_file,
    )

    global_home: Path = ctx.obj[_CTX_KEY]["global_home"]
    info = read_pid_file(global_home)
    if info is None:
        click.echo("no UI pid file found.")
        return
    if not is_alive(info.pid):
        click.echo(f"pid {info.pid} not running; clearing stale pid file.")
        clear_pid_file(global_home)
        return
    click.echo(f"stopping UI (pid {info.pid})...")
    if graceful_stop(info.pid):
        click.echo("stopped.")
    else:
        click.echo("could not confirm process exit; check Task Manager.", err=True)
    clear_pid_file(global_home)


@cli.command("ui-status")
@click.pass_context
def ui_status(ctx: click.Context) -> None:
    """Show whether the detached Streamlit UI is running."""
    from expense_analyzer.ui.process import (
        clear_pid_file,
        is_alive,
        read_pid_file,
    )

    global_home: Path = ctx.obj[_CTX_KEY]["global_home"]
    info = read_pid_file(global_home)
    if info is None:
        click.echo("UI is not running.")
        return
    if not is_alive(info.pid):
        click.echo(f"UI is not running (pid {info.pid} is dead; clearing stale pid file).")
        clear_pid_file(global_home)
        return
    click.echo(f"running: pid={info.pid} http://{info.host}:{info.port}")


@cli.command("ui-restart")
@click.option("--no-browser", is_flag=True)
@click.pass_context
def ui_restart(ctx: click.Context, no_browser: bool) -> None:
    """Stop the running UI (if any) and start a fresh one."""
    # Call ui-stop then ui in sequence.
    ctx.invoke(ui_stop)
    ctx.invoke(ui, foreground=False, no_browser=no_browser)


# --- entrypoint --------------------------------------------------------------

def main() -> None:  # console-script target
    cli(obj={})


if __name__ == "__main__":
    main()
