"""Settings tab: compute device, models, privacy, own IBANs, DB admin.

Sections (top to bottom):
  - Compute Device
  - Embeddings (model picker)
  - Zero-Shot (model picker)
  - Privacy (vendor lookup status)
  - My Accounts (own IBANs CRUD)
  - Database (backup / restore / Danger Zone resets)
"""

from __future__ import annotations

import datetime as _dt
import tempfile as _tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from expense_analyzer.config import save_user_config
from expense_analyzer.features.model_registry import (
    EMBEDDING_MODELS,
    ZEROSHOT_MODELS,
    hf_cache_dir,
    is_downloaded,
    trigger_download,
)
from expense_analyzer.storage.admin import (
    delete_user_labels as _del_user_labels,
)
from expense_analyzer.storage.admin import reset_all, reset_data
from expense_analyzer.storage.backup import (
    export_database,
    restore_database,
    validate_backup,
)
from expense_analyzer.storage.own_ibans import (
    add_own_iban,
    list_own_ibans,
    remove_own_iban,
    update_label,
)
from expense_analyzer.ui._shared import (
    get_config,
    get_conn,
    get_global_home,
    invalidate_connection,
    invalidate_global_config,
)


def render() -> None:
    cfg = get_config()
    conn = get_conn()
    st.title("Settings")
    _render_device_section(cfg)
    _render_model_sections(cfg)
    _render_privacy(cfg)
    _render_own_ibans(conn)
    _render_database_section(cfg, conn)


def _render_device_section(cfg) -> None:
    st.header("Compute Device")
    st.caption("Global setting — applies to all accounts.")
    st.write(f"**Device:** `{cfg.device}`")
    st.caption(f"HF cache: `{hf_cache_dir()}`")


def _model_table_and_picker(
    models, current_id: str, cfg_key: str, explanation: str, cfg
) -> None:
    """Caller emits the section heading; we render explanation + table + picker."""
    st.caption(explanation)
    rows = []
    for m in models:
        present, size_gb = is_downloaded(m.model_id)
        rows.append(
            {
                "model_id": m.model_id,
                "dim": m.dim if m.dim is not None else "—",
                "languages": m.languages,
                "downloaded": "✅" if present else "—",
                "size_GB": round(size_gb, 2) if present else round(m.approx_size_mb / 1024, 2),
                "notes": m.notes,
                "active": "●" if m.model_id == current_id else "",
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "model_id": st.column_config.TextColumn("Model"),
            "downloaded": st.column_config.TextColumn("Cached"),
            "size_GB": st.column_config.NumberColumn(
                "Size (GB)",
                help="Actual on-disk if cached, else approximate download size.",
                format="%.2f",
            ),
            "active": st.column_config.TextColumn("Active"),
        },
    )
    st.caption("Switch active model")
    sel_cols = st.columns([3, 1, 1])
    with sel_cols[0]:
        picked = st.selectbox(
            "model picker",
            [m.model_id for m in models],
            index=next((i for i, m in enumerate(models) if m.model_id == current_id), 0),
            key=f"model_pick_{cfg_key}",
            label_visibility="collapsed",
        )
    with sel_cols[1]:
        present, _ = is_downloaded(picked)
        dl_label = "Download" if not present else "Re-download"
        if st.button(dl_label, key=f"model_dl_{cfg_key}", width="stretch"):
            with st.status(f"Downloading {picked}...", expanded=True) as status:
                role = "embedding" if cfg_key == "embedding_model" else "zeroshot"
                try:
                    trigger_download(picked, role=role)
                    status.update(label=f"Downloaded {picked}", state="complete")
                except Exception as e:
                    status.update(label=f"Download failed: {e}", state="error")
            st.rerun()
    with sel_cols[2]:
        if st.button(
            "Use this", key=f"model_use_{cfg_key}", type="primary",
            disabled=picked == current_id, width="stretch",
        ):
            # Model settings are global; write to the global home's
            # config.yaml, not the per-account data_dir. Narrow the
            # cache-clear too: only the GlobalConfig changed, so the
            # DB connection and embedder caches don't need to flip.
            save_user_config({cfg_key: picked}, data_dir=get_global_home())
            invalidate_global_config()
            st.success(
                f"`{cfg_key}` set to `{picked}`. Restart the UI for it to take effect: "
                "`expense ui-restart`."
            )


def _render_model_sections(cfg) -> None:
    st.header("Embeddings")
    st.caption("Global setting — applies to all accounts.")
    _model_table_and_picker(
        EMBEDDING_MODELS, cfg.embedding_model, "embedding_model",
        explanation=(
            "Converts each expense's text "
            "(`counterparty_normalized` + ` | ` + `verwendungszweck_normalized`) "
            "into a fixed-dimensional vector. Those vectors power the "
            "**k-NN lookup** (finds the closest already-labeled expenses), the "
            "**supervised classifier** (logistic regression / random forest "
            "trained on your user labels), and the **category-similarity** "
            "stage (cosine match against embedded category names + "
            "descriptions). Pick a German-aware model for best results on DE "
            "bank text; larger models are more accurate but use more disk "
            "and run slower on CPU."
        ),
        cfg=cfg,
    )
    st.header("Zero-Shot")
    st.caption("Global setting — applies to all accounts.")
    _model_table_and_picker(
        ZEROSHOT_MODELS, cfg.zeroshot_model, "zeroshot_model",
        explanation=(
            "**Fallback only** — invoked when every earlier cascade stage "
            "(vendor exact match → k-NN → classifier → category similarity) "
            "comes back with low confidence. It does multilingual "
            "natural-language inference: it asks *\"does this expense text "
            "belong to category X?\"* for every category description and "
            "picks the best fit. Slow per call (especially on CPU) but "
            "rarely needed once you have a few user labels seeded; safe to "
            "leave on the default for most installs."
        ),
        cfg=cfg,
    )


def _render_privacy(cfg) -> None:
    st.title("Privacy")
    st.caption("Global setting — applies to all accounts.")
    st.write(f"Vendor web lookup enabled: **{cfg.vendor_lookup.enabled}**")
    if cfg.vendor_lookup.enabled:
        st.warning(
            "Vendor lookup is ON. Only `counterparty_normalized` is sent to "
            f"{cfg.vendor_lookup.backend}; never amount/IBAN/Verwendungszweck."
        )
    else:
        st.info(
            "Vendor lookup is OFF. Set `vendor_lookup.enabled: true` in your "
            "config to enable."
        )


def _render_own_ibans(conn) -> None:
    st.title("My Accounts")
    st.caption(
        "Your own IBANs. Rows whose IBAN matches one listed here are "
        "marked **internal** (`iban_is_known_self = 1`) and become a "
        "signal the classifier can use to recognise transfers between "
        "your own accounts. Adding or removing an IBAN here retroactively "
        "re-flags every matching transaction."
    )

    own_rows = list_own_ibans(conn)
    if not own_rows:
        st.info(
            "No own IBANs registered yet. Add one below to start tagging "
            "internal transfers."
        )
    else:
        own_widths = [4, 3, 0.6]
        h = st.columns(own_widths)
        h[0].markdown("**IBAN**")
        h[1].markdown("**Label**")
        h[2].markdown("")
        for r in own_rows:
            row = st.columns(own_widths)
            row[0].code(r.iban, language=None)
            new_lbl = row[1].text_input(
                "label",
                value=r.label or "",
                key=f"own_iban_lbl_{r.iban}",
                label_visibility="collapsed",
                placeholder="(no label)",
            )
            if new_lbl != (r.label or ""):
                update_label(conn, r.iban, new_lbl)
            if row[2].button("✕", key=f"own_iban_del_{r.iban}",
                             help=f"Remove {r.iban}"):
                rep = remove_own_iban(conn, r.iban)
                st.toast(
                    f"removed; cleared the flag on {rep.n_was_self} "
                    "transaction(s)."
                )
                st.rerun()

    st.markdown("**Add own IBAN**")
    add_cols = st.columns([4, 3, 1])
    new_iban = add_cols[0].text_input(
        "new iban",
        key="new_own_iban_iban",
        label_visibility="collapsed",
        placeholder="DE89 3704 0044 0532 0130 00",
    )
    new_label = add_cols[1].text_input(
        "new label",
        key="new_own_iban_label",
        label_visibility="collapsed",
        placeholder="Friendly name (optional)",
    )
    if add_cols[2].button("Add", type="primary", key="new_own_iban_add",
                          disabled=not new_iban.strip()):
        try:
            rep = add_own_iban(conn, new_iban, label=new_label or None)
        except ValueError as e:
            st.error(f"refusing: {e}")
        else:
            st.toast(
                f"added; flagged {rep.n_now_self} existing transaction(s) "
                "as internal."
            )
            for k in ("new_own_iban_iban", "new_own_iban_label"):
                st.session_state.pop(k, None)
            st.rerun()


def _render_database_section(cfg, conn) -> None:
    st.title("Database")
    _render_db_stats(cfg)
    _render_administration(cfg, conn)


def _render_db_stats(cfg) -> None:
    st.header("Stats")
    db_path = cfg.db_path
    try:
        db_mtime = (
            _dt.datetime.fromtimestamp(db_path.stat().st_mtime)
            if db_path.exists() else None
        )
    except OSError:
        db_mtime = None
    stat_cols = st.columns([1.5, 4])
    stat_cols[0].metric(
        "Last modified",
        db_mtime.strftime("%Y-%m-%d %H:%M") if db_mtime else "—",
    )
    stat_cols[1].caption(f"Path: `{db_path}`")


def _render_administration(cfg, conn) -> None:
    st.header("Administration")
    _render_backup(conn)
    _render_restore(cfg)
    _render_danger_zone(conn)


def _render_backup(conn) -> None:
    st.markdown("**Download Backup**")
    st.caption(
        "Download a complete copy of the SQLite database. Includes every "
        "ingested row, label, embedding, note, vendor-cache entry, and "
        "category. The file is a standard SQLite 3 DB -- open it in any "
        "SQLite browser, or re-import here on another machine."
    )
    # Render the backup bytes lazily into the download_button. SQLite's
    # online backup API is safe with the live UI connection open.
    # TemporaryDirectory (rather than NamedTemporaryFile) because the
    # latter leaves the file handle locked on Windows even after its
    # `with` block exits. ignore_cleanup_errors: SQLite's handle release
    # lags after conn.close() on Windows; the OS eventually cleans up.
    try:
        with _tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _td:
            _bk_tmp = Path(_td) / "backup.sqlite"
            export_database(conn, _bk_tmp)
            _bk_bytes = _bk_tmp.read_bytes()
        st.download_button(
            "⬇️ Download backup",
            data=_bk_bytes,
            file_name=(
                f"expense-analyzer-backup-{_dt.date.today().isoformat()}.sqlite"
            ),
            mime="application/x-sqlite3",
            key="db_backup_download",
            type="primary",
        )
    except Exception as e:
        st.error(f"backup failed: {e}")


def _render_restore(cfg) -> None:
    st.markdown("**Restore Backup**")
    st.caption(
        "Replace the **current** database with an uploaded `.sqlite` backup. "
        "A timestamped safety copy of the current DB is saved alongside it "
        "(`db.pre-restore.<ts>.sqlite`) so you can roll back manually if "
        "the restore turns out to be the wrong file."
    )
    upload = st.file_uploader(
        "Pick a backup file (.sqlite)",
        type=["sqlite", "db"],
        key="db_restore_uploader",
    )
    if upload is None:
        return
    # Persist the upload to a temp dir so validate_backup / restore_database
    # can read from a Path. TemporaryDirectory avoids the Windows
    # NamedTemporaryFile handle-lock pitfall.
    with _tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _td:
        _upload_path = Path(_td) / "uploaded_backup.sqlite"
        _upload_path.write_bytes(upload.getbuffer())
        result = validate_backup(_upload_path)
        if not result.ok:
            st.error("upload is not a valid backup: " + "; ".join(result.errors))
            return
        st.success(
            "valid backup — rows: "
            + ", ".join(f"{k}={v}" for k, v in result.table_counts.items())
        )
        confirm_restore = st.text_input(
            "Type `restore` to confirm",
            key="confirm_db_restore",
        )
        if st.button("Restore from this backup", type="primary",
                     key="db_do_restore"):
            if confirm_restore.strip().lower() != "restore":
                st.error("type the confirmation phrase exactly")
                return
            try:
                # Close the cached live connection before swapping the
                # file on disk (Windows holds locks otherwise).
                conn = get_conn()
                try:
                    conn.close()
                except Exception:
                    pass
                # Narrow cache-clear: drop only the DB connection. The
                # embedder cache (the heavy one) stays warm because the
                # model didn't change, and so does the GlobalConfig and
                # account registry.
                invalidate_connection()
                report = restore_database(
                    cfg.db_path, _upload_path, keep_safety=True,
                )
                st.success(
                    "restored: "
                    + ", ".join(
                        f"{k}={v}" for k, v in report.table_counts.items()
                    )
                )
                if report.safety_copy:
                    st.info(f"safety copy: `{report.safety_copy}`")
                st.rerun()
            except Exception as e:
                st.error(f"restore failed: {e}")


def _render_danger_zone(conn) -> None:
    st.subheader(":red[Danger Zone]")
    with st.expander("Delete User Labels", expanded=False):
        n_user_labels = conn.execute(
            "SELECT COUNT(*) AS n FROM labels WHERE source='user'"
        ).fetchone()["n"]
        st.write(
            f"Currently **{n_user_labels}** row(s) in `labels` with `source='user'`. "
            "Deleting them lets you re-run Auto-label across the whole DB without "
            "your previous confirmations dominating the cascade. Model labels stay, "
            "so rows that have both keep their visible category via the remaining "
            "model entry; rows that had **only** a user label become uncategorized."
        )
        confirm_user = st.text_input(
            "Type `delete user labels` to confirm", key="confirm_delete_user_labels"
        )
        if st.button("Delete all user labels", type="secondary"):
            if confirm_user.strip().lower() == "delete user labels":
                n = _del_user_labels(conn)
                st.success(f"deleted {n} user label row(s)")
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")

    with st.expander("Empty Database", expanded=False):
        st.write(
            "Deletes every row in `expenses`, `labels`, `notes`, `embeddings`, "
            "`vendor_cache` and `model_versions`. Categories and own-IBANs are kept."
        )
        confirm_data = st.text_input(
            "Type `clear data` to confirm", key="confirm_reset_data"
        )
        if st.button("Clear ingested data"):
            if confirm_data.strip().lower() == "clear data":
                report = reset_data(conn)
                st.success(
                    f"deleted {report.total} row(s) across "
                    f"{len(report.table_counts)} table(s)"
                )
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")

    with st.expander("Factory Reset (incl. category deletion)", expanded=False):
        st.write(
            "Wipes every table including categories and own-IBANs. The DB schema "
            "stays so you can immediately re-init."
        )
        confirm_all = st.text_input(
            "Type `factory reset` to confirm", key="confirm_reset_all"
        )
        if st.button("Factory reset"):
            if confirm_all.strip().lower() == "factory reset":
                report = reset_all(conn)
                st.success(
                    f"deleted {report.total} row(s) across "
                    f"{len(report.table_counts)} table(s)"
                )
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")
