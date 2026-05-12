"""Streamlit UI — local-only (binds to 127.0.0.1 via the CLI launcher).

Tabs, in order:
  1. Dashboard  — overview stats.
  2. Categories — types table with per-category stats, add/remove.
  3. Import     — upload CSV → inspect new rows → auto-label → accept/review.
  4. Data       — sortable/filterable table; row drawer for full record + notes.
  5. Clusters   — HDBSCAN exploration.
  6. Settings   — model info, privacy, danger zone.

No sidebar by design — everything lives in tabs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from expense_analyzer.config import Config, load_config
from expense_analyzer.enrichment.notes import get_note, set_note
from expense_analyzer.features.embeddings import (
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
)
from expense_analyzer.ingestion import IngestReport, ingest_csv
from expense_analyzer.ml.classifier import CategorizationCascade, Prediction
from expense_analyzer.ml.clustering import cluster_all
from expense_analyzer.storage.admin import (
    category_removal_impact,
    remove_category,
    reset_all,
    reset_data,
)
from expense_analyzer.storage.categories import (
    add_label,
    list_categories,
    upsert_category,
)
from expense_analyzer.storage.database import get_or_create_database
from expense_analyzer.storage.stats import category_stats, uncategorized_stat
from expense_analyzer.viz import (
    amount_distribution,
    bar_top_counterparties,
    daily_calendar,
    histogram_amounts,
    monthly_flow_by_category,
    pie_chart,
    spend_by_category,
    top_counterparties,
    trend_lines,
)
from expense_analyzer.viz.charts import calendar_heatmap

# Higher threshold for one-click "accept all confident" predictions
# (the cfg.classifier.confidence_threshold is for review queues).
ACCEPT_CONFIDENT_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

st.set_page_config(page_title="expense-analyzer-de", layout="wide", page_icon="💶")


@st.cache_resource
def _load_config_cached() -> Config:
    return load_config()


@st.cache_resource
def _connect_cached(db_path_str: str) -> sqlite3.Connection:
    return get_or_create_database(Path(db_path_str))


@st.cache_resource
def _real_embedder(model_name: str, device: str, batch_size: int) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=model_name, device=device, batch_size=batch_size, verbose=False
    )


@st.cache_resource
def _hash_embedder(dim: int = 64) -> Embedder:
    return HashEmbedder(dim=dim)


cfg = _load_config_cached()
conn = _connect_cached(str(cfg.db_path))


def _embedder() -> Embedder:
    """Pick between real and hash based on the toggle in Settings."""
    if st.session_state.get("use_hash_embedder", False):
        return _hash_embedder()
    return _real_embedder(cfg.embedding_model, cfg.device, cfg.embedding_batch_size)


# ---------------------------------------------------------------------------
# Top header bar (replaces the old sidebar)
# ---------------------------------------------------------------------------

def _render_header() -> None:
    try:
        n_exp = conn.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
        n_lab = conn.execute(
            "SELECT COUNT(DISTINCT expense_id) AS n FROM labels WHERE source='user'"
        ).fetchone()["n"]
        n_cat = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
    except sqlite3.OperationalError:
        n_exp = n_lab = n_cat = 0
    bar = st.container()
    with bar:
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 3])
        with c1:
            st.metric("Expenses", n_exp)
        with c2:
            st.metric("User-labeled", n_lab)
        with c3:
            st.metric("Categories", n_cat)
        with c4:
            st.metric("Model", "Hash (dev)" if st.session_state.get("use_hash_embedder") else "HF")
        with c5:
            st.caption(f"DB · `{cfg.db_path}`")
    st.divider()


_render_header()

tab_dash, tab_cats, tab_import, tab_data, tab_clusters, tab_settings = st.tabs(
    ["Dashboard", "Categories", "Import", "Data", "Clusters", "Settings"]
)


# ---------------------------------------------------------------------------
# Helpers used by multiple tabs
# ---------------------------------------------------------------------------

def _category_options(include_unlabeled: bool = True) -> list[tuple[int | None, str]]:
    """[(category_id, name), ...] for dropdowns. `None` represents 'unlabeled'."""
    out: list[tuple[int | None, str]] = []
    for c in list_categories(conn):
        out.append((c.id, c.name))
    if include_unlabeled:
        out.append((None, "(unkategorisiert)"))
    return out


def _set_user_label(expense_id: int, category_id: int) -> None:
    add_label(conn, expense_id, category_id, "user")


def _format_eur(cents: int) -> str:
    return f"{cents / 100:>9,.2f} €"


# ---------------------------------------------------------------------------
# Tab 1: Dashboard
# ---------------------------------------------------------------------------

with tab_dash:
    st.header("Dashboard")
    n_exp = conn.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
    if n_exp == 0:
        st.info("Import a CSV from the **Import** tab to get started.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(pie_chart(spend_by_category(conn)), width="stretch")
        with c2:
            st.plotly_chart(
                bar_top_counterparties(top_counterparties(conn, n=15)),
                width="stretch",
            )
        st.plotly_chart(trend_lines(monthly_flow_by_category(conn)), width="stretch")
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(histogram_amounts(amount_distribution(conn)), width="stretch")
        with c4:
            st.plotly_chart(calendar_heatmap(daily_calendar(conn)), width="stretch")


# ---------------------------------------------------------------------------
# Tab 2: Categories
# ---------------------------------------------------------------------------

def _save_cat_name(cat_id: int) -> None:
    new_name = (st.session_state.get(f"cat_{cat_id}_name") or "").strip()
    if not new_name:
        st.session_state[f"cat_{cat_id}_error"] = "name cannot be empty"
        return
    try:
        conn.execute("UPDATE categories SET name=? WHERE id=?", (new_name, cat_id))
        st.session_state.pop(f"cat_{cat_id}_error", None)
    except sqlite3.IntegrityError as e:
        st.session_state[f"cat_{cat_id}_error"] = f"name conflict: {e}"


def _save_cat_desc(cat_id: int) -> None:
    new_desc = (st.session_state.get(f"cat_{cat_id}_desc") or "").strip()
    conn.execute("UPDATE categories SET description=? WHERE id=?", (new_desc, cat_id))


def _save_cat_color(cat_id: int) -> None:
    new_color = st.session_state.get(f"cat_{cat_id}_color") or "#888888"
    conn.execute("UPDATE categories SET color=? WHERE id=?", (new_color, cat_id))


with tab_cats:
    from expense_analyzer.config import packaged_default_categories
    from expense_analyzer.storage.categories import import_categories_from_yaml
    from expense_analyzer.utils.colors import random_hex_color

    st.header("Categories")

    stats = category_stats(conn)

    # --- Empty-state bootstrap ------------------------------------------
    if not stats:
        st.info(
            "No categories yet. Install the bundled German defaults to get "
            "started, or add your own below."
        )
        if st.button("Install default German categories", type="primary"):
            n = import_categories_from_yaml(conn, packaged_default_categories())
            st.success(f"installed {n} categories")
            st.rerun()

    # --- Existing categories table --------------------------------------
    if stats:
        st.caption(
            "Edit any cell to save immediately. Name and description commit on "
            "blur/Enter; color commits when you pick a new one. Click ✕ to "
            "delete (cascade prompt if labels reference it)."
        )

        # Header row
        widths = [2, 3, 1, 1, 1, 1.2, 0.5]
        h = st.columns(widths)
        h[0].markdown("**Name**")
        h[1].markdown("**Description**")
        h[2].markdown("**Color**")
        h[3].markdown("**# Records**")
        h[4].markdown("**Abs total €**")
        h[5].markdown("**Last seen**")
        h[6].markdown("")

        for s in stats:
            row = st.columns(widths)
            with row[0]:
                st.text_input(
                    "name",
                    value=s.name,
                    key=f"cat_{s.id}_name",
                    label_visibility="collapsed",
                    on_change=_save_cat_name,
                    args=(s.id,),
                )
            with row[1]:
                st.text_input(
                    "desc",
                    value=s.description,
                    key=f"cat_{s.id}_desc",
                    label_visibility="collapsed",
                    on_change=_save_cat_desc,
                    args=(s.id,),
                    placeholder="(used as zero-shot hypothesis)",
                )
            with row[2]:
                st.color_picker(
                    "color",
                    value=s.color,
                    key=f"cat_{s.id}_color",
                    label_visibility="collapsed",
                    on_change=_save_cat_color,
                    args=(s.id,),
                )
            row[3].write(s.n_expenses)
            row[4].write(f"{s.abs_total_eur:.2f}")
            row[5].write(s.last_seen or "—")
            with row[6]:
                if st.button("✕", key=f"cat_{s.id}_del", help=f"Delete {s.name!r}"):
                    impact = category_removal_impact(conn, s.name)
                    if impact.n_labels == 0:
                        remove_category(conn, s.name)
                        st.rerun()
                    else:
                        st.session_state[f"cat_{s.id}_confirm_delete"] = True

            err = st.session_state.get(f"cat_{s.id}_error")
            if err:
                st.error(f"`{s.name}` — {err}")

            if st.session_state.get(f"cat_{s.id}_confirm_delete"):
                impact = category_removal_impact(conn, s.name)
                with st.container(border=True):
                    st.warning(
                        f"Deleting **{s.name}** will cascade-delete "
                        f"{impact.n_labels} label(s). Continue?"
                    )
                    cc = st.columns([1, 1, 6])
                    if cc[0].button("Yes, delete", key=f"cat_{s.id}_del_yes",
                                    type="secondary"):
                        remove_category(conn, s.name)
                        st.session_state.pop(f"cat_{s.id}_confirm_delete", None)
                        st.rerun()
                    if cc[1].button("Cancel", key=f"cat_{s.id}_del_no"):
                        st.session_state.pop(f"cat_{s.id}_confirm_delete", None)
                        st.rerun()

        # Uncategorized note
        uncat = uncategorized_stat(conn)
        if uncat.n_expenses > 0:
            st.info(
                f"**{uncat.n_expenses}** record(s) currently have no category "
                f"(total |€| {uncat.abs_total_eur:.2f})."
            )

    # --- Add a new category --------------------------------------------
    st.divider()
    st.markdown("**Add a new category**")
    if "new_cat_color" not in st.session_state:
        st.session_state.new_cat_color = random_hex_color()

    add_widths = [2, 3, 1, 1]
    add_cols = st.columns(add_widths)
    with add_cols[0]:
        st.text_input(
            "new name", key="new_cat_name", label_visibility="collapsed",
            placeholder="Name (e.g. Lebensmittel)",
        )
    with add_cols[1]:
        st.text_input(
            "new desc", key="new_cat_desc", label_visibility="collapsed",
            placeholder="Description (optional; used as zero-shot hypothesis)",
        )
    with add_cols[2]:
        st.color_picker(
            "new color", key="new_cat_color", label_visibility="collapsed",
        )
    with add_cols[3]:
        if st.button("➕ Add", type="primary"):
            new_name = (st.session_state.get("new_cat_name") or "").strip()
            new_desc = (st.session_state.get("new_cat_desc") or "").strip()
            new_color = st.session_state.get("new_cat_color") or random_hex_color()
            if not new_name:
                st.error("name is required")
            else:
                try:
                    upsert_category(conn, new_name, new_desc, new_color)
                    # Clear inputs for the next addition, refresh suggested color.
                    for k in ("new_cat_name", "new_cat_desc", "new_cat_color"):
                        st.session_state.pop(k, None)
                    st.session_state.new_cat_color = random_hex_color()
                    st.rerun()
                except sqlite3.IntegrityError as e:
                    st.error(f"could not save: {e}")


# ---------------------------------------------------------------------------
# Tab 3: Import
# ---------------------------------------------------------------------------

if "import_step" not in st.session_state:
    st.session_state.import_step = "upload"
    st.session_state.import_new_ids = []
    st.session_state.import_predictions = {}  # expense_id -> Prediction
    st.session_state.import_overrides = {}    # expense_id -> category_id


def _reset_import_state() -> None:
    st.session_state.import_step = "upload"
    st.session_state.import_new_ids = []
    st.session_state.import_predictions = {}
    st.session_state.import_overrides = {}


with tab_import:
    st.header("Import")

    if st.session_state.import_step == "upload":
        st.write("Upload one or more German bank-export CSVs (`;` separator, comma decimal).")
        files = st.file_uploader(
            "CSV file(s)", accept_multiple_files=True, type=["csv"]
        )
        cols = st.columns(2)
        with cols[0]:
            ingest_clicked = st.button(
                "Ingest", type="primary", disabled=not files
            )
        with cols[1]:
            st.caption(
                "Features (text, IBAN, numeric, embedding) are computed at import "
                "time, so subsequent steps are fast."
            )

        if ingest_clicked and files:
            import tempfile

            emb = _embedder()
            reports: list[IngestReport] = []
            new_ids: list[int] = []
            with st.status("Importing…", expanded=True) as status:
                for f in files:
                    status.write(f"parsing {f.name}…")
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        tmp.write(f.read())
                        p = Path(tmp.name)
                    status.write(f"ingesting + embedding {f.name}…")
                    r = ingest_csv(conn, p, embedder=emb)
                    reports.append(r)
                    new_ids.extend(r.new_ids)
                    status.write(
                        f"{f.name}: parsed={r.parsed} new={r.inserted} "
                        f"duplicate={r.duplicates} embedded={r.embedded}"
                    )
                status.update(label=f"Imported {sum(r.inserted for r in reports)} new row(s).", state="complete")
            st.session_state.import_new_ids = new_ids
            if new_ids:
                st.session_state.import_step = "review"
                st.rerun()
            else:
                st.info("Nothing new — all records were duplicates.")

    else:  # review step
        new_ids = st.session_state.import_new_ids
        if not new_ids:
            _reset_import_state()
            st.rerun()

        st.write(
            f"**{len(new_ids)} new record(s)** ingested. Click **Auto-label** to "
            "have the cascade predict, then edit any wrong Category cell inline "
            "and click **Save labels** to commit."
        )

        # --- Action buttons (above the table for visibility) ----------------
        preds_dict: dict[int, Prediction] = st.session_state.import_predictions
        has_preds = bool(preds_dict)
        c1, c2, c3, c4 = st.columns([1, 1.4, 1.4, 1])
        run_auto = c1.button("Auto-label", type="primary")
        accept_confident = c2.button(
            f"Save ≥ {int(ACCEPT_CONFIDENT_THRESHOLD * 100)}% confident as user labels",
            disabled=not has_preds,
        )
        save_all = c3.button("Save all visible categories as user labels", disabled=not has_preds)
        done = c4.button("Done", help="Exit. Predictions stay as `model` labels until accepted later.")

        if run_auto:
            emb = _embedder()
            cascade = CategorizationCascade(conn, cfg, emb)
            with st.status("Predicting…", expanded=True) as status:
                status.write(f"loading embedding model `{cfg.embedding_model}`…")
                try:
                    cascade.fit()
                except Exception as e:
                    status.write(f"  fit skipped: {e}")
                status.write(f"predicting {len(new_ids)} record(s)…")
                preds = cascade.predict_batch(new_ids)
                from collections import Counter

                stages = Counter(p.stage for p in preds)
                st.session_state.import_predictions = {p.expense_id: p for p in preds}
                for p in preds:
                    if p.category_id is not None:
                        add_label(conn, p.expense_id, p.category_id, "model",
                                  confidence=p.confidence)
                status.update(
                    label=", ".join(f"{k}={v}" for k, v in stages.items()),
                    state="complete",
                )
            # No explicit st.rerun(): predictions are already in
            # session_state.import_predictions, so when execution falls
            # through to the table render below it picks them up. Skipping
            # the rerun avoids a second full re-render that resets the
            # data_editor's scroll / focus position.

        # --- Compact review table -------------------------------------------
        cats = list_categories(conn)
        cat_options_for_editor = [""] + [c.name for c in cats]
        cat_id_by_name = {c.name: c.id for c in cats}

        ph = ",".join("?" * len(new_ids))
        rows_df = pd.read_sql_query(
            f"""
            SELECT id, buchungsdatum, counterparty, verwendungszweck,
                   betrag_cents / 100.0 AS "Betrag €",
                   amount_bucket, iban_country, umsatztyp,
                   has_glaeubiger_id, mandatsreferenz_present
            FROM expenses
            WHERE id IN ({ph})
            ORDER BY buchungsdatum, id
            """,
            conn,
            params=new_ids,
        )
        # Keep the date as a real datetime so sorting works numerically;
        # the column_config below formats it as DD.MM.YYYY for display.
        if not rows_df.empty:
            rows_df["buchungsdatum"] = pd.to_datetime(rows_df["buchungsdatum"])

        def _stage_glyph(stage: str | None) -> str:
            return {
                "vendor_exact_match": "🟢",
                "knn": "🟢",
                "classifier": "🟡",
                "category_similarity": "🟡",
                "zeroshot": "🟠",
                "unknown": "⚪",
            }.get(stage or "", "")

        # Resolve current prediction (or override) per row.
        cats_by_id = {c.id: c.name for c in cats}
        predicted_names: list[str] = []
        confidences: list[str] = []
        stages_visible: list[str] = []
        for eid in rows_df["id"].astype(int).tolist():
            p = preds_dict.get(eid)
            if p is None or p.category_id is None:
                predicted_names.append("")
                confidences.append("")
                stages_visible.append("")
            else:
                predicted_names.append(cats_by_id.get(p.category_id, ""))
                confidences.append(f"{p.confidence:.2f}")
                stages_visible.append(f"{_stage_glyph(p.stage)} {p.stage}")
        rows_df["Category"] = predicted_names
        rows_df["Conf"] = confidences
        rows_df["Stage"] = stages_visible

        extended = st.toggle("Extended columns", value=False, key="import_extended")
        if extended:
            show_cols = [
                "id", "buchungsdatum", "counterparty", "verwendungszweck",
                "Betrag €", "Category", "Conf", "Stage",
                "amount_bucket", "umsatztyp", "iban_country",
                "has_glaeubiger_id", "mandatsreferenz_present",
            ]
        else:
            show_cols = [
                "id", "buchungsdatum", "counterparty", "verwendungszweck",
                "Betrag €", "Category", "Conf", "Stage",
            ]

        column_config = {
            "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "buchungsdatum": st.column_config.DateColumn(
                "Date", disabled=True, format="DD.MM.YYYY"
            ),
            "counterparty": st.column_config.TextColumn("Counterparty", disabled=True),
            "verwendungszweck": st.column_config.TextColumn(
                "Verwendungszweck", disabled=True, width="large"
            ),
            "Betrag €": st.column_config.NumberColumn(
                "Amount €", disabled=True, format="%.2f"
            ),
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=cat_options_for_editor,
                required=False,
                help="Override the prediction (or fill one in when the cascade abstained).",
            ),
            "Conf": st.column_config.TextColumn("Conf", disabled=True, width="small"),
            "Stage": st.column_config.TextColumn("Stage", disabled=True),
            "amount_bucket": st.column_config.TextColumn("Bucket", disabled=True),
            "umsatztyp": st.column_config.TextColumn("Umsatztyp", disabled=True),
            "iban_country": st.column_config.TextColumn("IBAN cc", disabled=True),
            "has_glaeubiger_id": st.column_config.NumberColumn("Gläubiger?", disabled=True),
            "mandatsreferenz_present": st.column_config.NumberColumn(
                "Mandat?", disabled=True
            ),
        }
        editor_disabled = [c for c in show_cols if c != "Category"]

        edited = st.data_editor(
            rows_df[show_cols],
            column_config=column_config,
            disabled=editor_disabled,
            hide_index=True,
            width="stretch",
            height=min(420, 60 + 35 * len(rows_df)),
            key="import_editor",
        )

        if accept_confident:
            n_promoted = 0
            for _, r in edited.iterrows():
                eid = int(r["id"])
                p = preds_dict.get(eid)
                if p is None or p.category_id is None:
                    continue
                if float(p.confidence) >= ACCEPT_CONFIDENT_THRESHOLD:
                    chosen_name = r["Category"] or cats_by_id.get(p.category_id, "")
                    chosen_cid = cat_id_by_name.get(chosen_name)
                    if chosen_cid is not None:
                        add_label(conn, eid, chosen_cid, "user")
                        n_promoted += 1
            st.success(f"promoted {n_promoted} prediction(s) to user labels")

        if save_all:
            n_promoted = 0
            for _, r in edited.iterrows():
                chosen_name = r["Category"]
                if not chosen_name:
                    continue
                chosen_cid = cat_id_by_name.get(chosen_name)
                if chosen_cid is None:
                    continue
                add_label(conn, int(r["id"]), chosen_cid, "user")
                n_promoted += 1
            st.success(f"saved {n_promoted} user label(s)")

        if done:
            _reset_import_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 4: Data
# ---------------------------------------------------------------------------

def _build_data_query(
    date_from, date_to, cats: list[str], source: str,
    search: str, amount_min: float, amount_max: float,
    include_income: bool,
) -> tuple[str, list]:
    """Return (SQL, params) for the Data table given filter widgets."""
    parts: list[str] = []
    params: list = []
    if date_from is not None:
        parts.append("e.buchungsdatum >= ?")
        params.append(date_from.isoformat())
    if date_to is not None:
        parts.append("e.buchungsdatum <= ?")
        params.append(date_to.isoformat())
    if not include_income:
        parts.append("e.is_income = 0")
    if amount_min is not None:
        parts.append("ABS(e.betrag_cents) >= ?")
        params.append(int(amount_min * 100))
    if amount_max is not None:
        parts.append("ABS(e.betrag_cents) <= ?")
        params.append(int(amount_max * 100))
    if search:
        like = f"%{search.lower()}%"
        parts.append(
            "(LOWER(e.counterparty) LIKE ? OR LOWER(e.verwendungszweck) LIKE ?)"
        )
        params.extend([like, like])
    if cats:
        unlabeled_picked = "(unkategorisiert)" in cats
        named = [c for c in cats if c != "(unkategorisiert)"]
        cat_conds = []
        if named:
            ph = ",".join("?" * len(named))
            cat_conds.append(f"c.name IN ({ph})")
            params.extend(named)
        if unlabeled_picked:
            cat_conds.append("c.id IS NULL")
        parts.append("(" + " OR ".join(cat_conds) + ")")
    if source == "user":
        parts.append("ll.source = 'user'")
    elif source == "model":
        parts.append("ll.source = 'model'")
    elif source == "unlabeled":
        parts.append("ll.expense_id IS NULL")

    where = (" WHERE " + " AND ".join(parts)) if parts else ""
    sql = (
        """
        WITH latest_label AS (
            SELECT l.expense_id, l.category_id, l.source, l.confidence
            FROM labels l
            JOIN (
                SELECT expense_id, MAX(id) AS max_id
                FROM labels GROUP BY expense_id
            ) m ON l.id = m.max_id
        )
        SELECT
            e.id, e.buchungsdatum, e.counterparty, e.verwendungszweck,
            e.betrag_cents / 100.0 AS "betrag_€",
            c.name AS category, ll.category_id AS category_id,
            ll.source AS label_source, ll.confidence,
            e.umsatztyp, e.iban_country, e.iban_is_foreign,
            e.cluster_id,
            e.has_glaeubiger_id, e.mandatsreferenz_present
        FROM expenses e
        LEFT JOIN latest_label ll ON ll.expense_id = e.id
        LEFT JOIN categories c ON c.id = ll.category_id
        """
        + where
        + " ORDER BY e.buchungsdatum DESC, e.id DESC"
    )
    return sql, params


with tab_data:
    st.header("Data")
    cat_options = [s.name for s in category_stats(conn)] + ["(unkategorisiert)"]

    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            date_from = st.date_input("From", value=None, key="data_from")
            date_to = st.date_input("To", value=None, key="data_to")
            include_income = st.checkbox("Include income (positive amounts)", value=False)
        with fc2:
            picked_cats = st.multiselect(
                "Categories", options=cat_options, default=[], key="data_cats"
            )
            source = st.selectbox(
                "Label source", ["all", "user", "model", "unlabeled"], index=0
            )
        with fc3:
            search = st.text_input("Search counterparty / Verwendungszweck", value="")
            amount_min = st.number_input("Min |amount| €", value=0.0, step=10.0, min_value=0.0)
            amount_max_raw = st.number_input("Max |amount| €", value=0.0, step=10.0, min_value=0.0,
                                             help="0 = no upper limit")
            amount_max = amount_max_raw if amount_max_raw > 0 else None

    extended = st.toggle("Extended columns", value=False, key="data_extended")

    # --- Snapshot the row set when filters change ---------------------------
    # The Data tab supports inline category editing. Without snapshotting,
    # editing a row whose new category no longer matches the active filter
    # (e.g. you filter "unlabeled" and label one) would make it disappear on
    # the next render, reshuffling the table. We therefore fix the row set
    # at the moment the filters change, and only refresh it when the user
    # actually changes a filter (or clicks "Refresh filter" below).
    filter_signature = (
        str(date_from) if date_from else "",
        str(date_to) if date_to else "",
        tuple(picked_cats),
        source,
        search.strip(),
        amount_min if amount_min > 0 else 0.0,
        amount_max if amount_max is not None else -1.0,
        bool(include_income),
    )
    def _fetch_snapshot_df(ids: list[int]) -> pd.DataFrame:
        if not ids:
            return pd.DataFrame(
                columns=[
                    "id", "buchungsdatum", "counterparty", "verwendungszweck",
                    "betrag_€", "category", "category_id", "label_source",
                    "confidence", "umsatztyp", "iban_country", "iban_is_foreign",
                    "cluster_id", "has_glaeubiger_id", "mandatsreferenz_present",
                    "Select",
                ]
            )
        ph_ids = ",".join("?" * len(ids))
        df_ = pd.read_sql_query(
            f"""
            WITH latest_label AS (
                SELECT l.expense_id, l.category_id, l.source, l.confidence
                FROM labels l
                JOIN (
                    SELECT expense_id, MAX(id) AS max_id
                    FROM labels GROUP BY expense_id
                ) m ON l.id = m.max_id
            )
            SELECT
                e.id, e.buchungsdatum, e.counterparty, e.verwendungszweck,
                e.betrag_cents / 100.0 AS "betrag_€",
                c.name AS category, ll.category_id AS category_id,
                ll.source AS label_source, ll.confidence,
                e.umsatztyp, e.iban_country, e.iban_is_foreign,
                e.cluster_id,
                e.has_glaeubiger_id, e.mandatsreferenz_present
            FROM expenses e
            LEFT JOIN latest_label ll ON ll.expense_id = e.id
            LEFT JOIN categories c ON c.id = ll.category_id
            WHERE e.id IN ({ph_ids})
            ORDER BY e.buchungsdatum DESC, e.id DESC
            """,
            conn,
            params=ids,
        )
        if not df_.empty:
            df_["buchungsdatum"] = pd.to_datetime(df_["buchungsdatum"])
            df_["category"] = df_["category"].fillna("(unkategorisiert)")
            df_["confidence"] = df_["confidence"].fillna("")
            df_["label_source"] = df_["label_source"].fillna("")
        df_.insert(0, "Select", False)
        return df_

    if (
        st.session_state.get("data_filter_signature") != filter_signature
        or st.session_state.pop("data_force_refresh", False)
    ):
        # Filters changed (or user clicked refresh): re-snapshot the IDs and
        # cache a fresh display dataframe.
        id_sql, id_params = _build_data_query(
            date_from or None, date_to or None,
            picked_cats, source, search.strip(),
            amount_min if amount_min > 0 else None, amount_max,
            include_income,
        )
        id_df = pd.read_sql_query(id_sql, conn, params=id_params)
        snapshot_ids = id_df["id"].astype(int).tolist() if not id_df.empty else []
        st.session_state.data_snapshot_ids = snapshot_ids
        st.session_state.data_filter_signature = filter_signature
        st.session_state["data_view_df"] = _fetch_snapshot_df(snapshot_ids)
        # Wipe the data_editor cache so stale row-index edits don't apply.
        st.session_state.pop("data_editor", None)

    # Cached display df. We mutate this in place on edits so the data_editor
    # sees an identical input object between renders -- key to keeping scroll
    # position and avoiding visible re-mounts.
    df: pd.DataFrame = st.session_state.get("data_view_df")
    if df is None:
        df = _fetch_snapshot_df(st.session_state.get("data_snapshot_ids", []))
        st.session_state["data_view_df"] = df

    # Counts BEFORE the fillna pass below, so we can use NaN semantics.
    def _to_conf(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    # Counts are computed against the cached snapshot df; "(unkategorisiert)"
    # is the in-cache sentinel for an unlabeled row (set by _fetch_snapshot_df).
    if not df.empty:
        n_unlabeled = int(
            ((df["label_source"] == "") | (df["category"] == "(unkategorisiert)")).sum()
        )
        n_confident_model = int(
            (
                (df["label_source"] == "model")
                & df["confidence"].apply(_to_conf).ge(ACCEPT_CONFIDENT_THRESHOLD)
            ).sum()
        )
    else:
        n_unlabeled = 0
        n_confident_model = 0

    base_cols = ["Select", "id", "buchungsdatum", "counterparty", "verwendungszweck",
                 "betrag_€", "category"]
    ext_cols = base_cols + ["label_source", "confidence", "umsatztyp",
                            "iban_country", "iban_is_foreign", "cluster_id",
                            "has_glaeubiger_id", "mandatsreferenz_present"]
    show_cols = ext_cols if extended else base_cols

    # --- Batch actions (respect current filters) ---------------------------
    if n_unlabeled > 0 or n_confident_model > 0:
        action_cols = st.columns([2, 2, 3])
        with action_cols[0]:
            if n_unlabeled > 0 and st.button(
                f"Auto-label {n_unlabeled} uncategorized",
                type="primary",
                key="data_auto_label_btn",
            ):
                unlabeled_ids = [
                    int(x) for x in df.loc[df["label_source"] == "", "id"].tolist()
                ]
                with st.status(
                    f"auto-labeling {len(unlabeled_ids)} record(s)…", expanded=True
                ) as status:
                    status.write(f"loading embedding model `{cfg.embedding_model}`…")
                    emb = _embedder()
                    cascade = CategorizationCascade(conn, cfg, emb)
                    status.write("fitting cascade on existing user labels…")
                    try:
                        cascade.fit()
                    except Exception as e:
                        status.write(f"  fit skipped: {e}")
                    status.write(f"predicting {len(unlabeled_ids)} record(s)…")
                    preds = cascade.predict_batch(unlabeled_ids)
                    n_persisted = 0
                    for p in preds:
                        if p.category_id is not None:
                            add_label(
                                conn, p.expense_id, p.category_id, "model",
                                confidence=p.confidence,
                            )
                            n_persisted += 1
                    from collections import Counter

                    stages = Counter(p.stage for p in preds)
                    status.update(
                        label=(
                            f"labeled {n_persisted}/{len(preds)} record(s) · "
                            + ", ".join(f"{k}={v}" for k, v in stages.items())
                        ),
                        state="complete",
                    )
                st.rerun()
        with action_cols[1]:
            if n_confident_model > 0 and st.button(
                f"Accept {n_confident_model} ≥ {int(ACCEPT_CONFIDENT_THRESHOLD * 100)}% as user labels",
                key="data_accept_btn",
            ):
                confident_rows = df[
                    (df["label_source"] == "model")
                    & df["confidence"].apply(_to_conf).ge(ACCEPT_CONFIDENT_THRESHOLD)
                ]
                n_promoted = 0
                for _, r in confident_rows.iterrows():
                    cid = r.get("category_id")
                    if pd.isna(cid):
                        continue
                    add_label(conn, int(r["id"]), int(cid), "user")
                    n_promoted += 1
                st.success(f"promoted {n_promoted} prediction(s) to user labels")
                st.rerun()
        with action_cols[2]:
            st.caption(
                "Both actions apply to the rows that match the **current filters**. "
                "Auto-label respects the cascade's confidence thresholds; the accept "
                f"button only promotes predictions at or above {int(ACCEPT_CONFIDENT_THRESHOLD * 100)}%."
            )

    caption_cols = st.columns([5, 1])
    caption_cols[0].caption(
        f"{len(df)} record(s) in current snapshot · "
        "click the **Category** cell to change a label inline — saves immediately. "
        "Rows stay in the table after labeling so the view doesn't shuffle while you work; "
        "click **Refresh filter** to re-evaluate."
    )
    if caption_cols[1].button("Refresh filter", help="Re-run the filter against the latest DB state."):
        st.session_state["data_force_refresh"] = True
        st.rerun()

    # --- Inline-editable table ----------------------------------------------
    all_cat_names = [s.name for s in category_stats(conn)]
    cat_id_by_name = {s.name: s.id for s in category_stats(conn)}

    # `df` is the cached snapshot dataframe (mutated in place on edits below).
    # The editor needs the unlabeled sentinel rendered as "" so the selectbox
    # starts blank, so we project a view-copy for display only.
    editor_view_df = pd.DataFrame(columns=show_cols) if df.empty else df[show_cols]

    category_options = [""] + all_cat_names
    column_config = {
        "Select": st.column_config.CheckboxColumn(
            "Sel", help="Tick to include in bulk operations below.", width="small",
        ),
        "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
        "buchungsdatum": st.column_config.DateColumn(
            "Date", disabled=True, format="DD.MM.YYYY"
        ),
        "counterparty": st.column_config.TextColumn("Counterparty", disabled=True),
        "verwendungszweck": st.column_config.TextColumn(
            "Verwendungszweck", disabled=True, width="large"
        ),
        "betrag_€": st.column_config.NumberColumn(
            "Amount €", disabled=True, format="%.2f"
        ),
        "category": st.column_config.SelectboxColumn(
            "Category",
            options=category_options,
            required=False,
            help="Pick a category right in the cell — saves as a user label immediately.",
        ),
        "label_source": st.column_config.TextColumn("Source", disabled=True),
        "confidence": st.column_config.TextColumn("Conf", disabled=True),
        "umsatztyp": st.column_config.TextColumn("Umsatztyp", disabled=True),
        "iban_country": st.column_config.TextColumn("IBAN cc", disabled=True),
        "iban_is_foreign": st.column_config.NumberColumn("Foreign?", disabled=True),
        "cluster_id": st.column_config.NumberColumn("Cluster", disabled=True),
        "has_glaeubiger_id": st.column_config.NumberColumn("Gläubiger?", disabled=True),
        "mandatsreferenz_present": st.column_config.NumberColumn("Mandat?", disabled=True),
    }
    # Editor is read-only EXCEPT for Category and Select.
    editor_disabled = [c for c in show_cols if c not in ("category", "Select")]

    # Map "(unkategorisiert)" → "" for the editor input so the dropdown is
    # blank for unlabeled rows. Apply this on a temporary copy to keep the
    # cached df stable across renders.
    if not editor_view_df.empty:
        editor_view_for_render = editor_view_df.copy()
        editor_view_for_render.loc[
            editor_view_for_render["category"] == "(unkategorisiert)", "category"
        ] = ""
    else:
        editor_view_for_render = editor_view_df

    edited_df = st.data_editor(
        editor_view_for_render,
        column_config=column_config,
        disabled=editor_disabled,
        hide_index=True,
        width="stretch",
        height=420,
        key="data_editor",
    )

    # Persist inline Category edits AND sync the Select column back to the
    # cached df. Mutating in place keeps the dataframe object identity
    # stable across reruns, which Streamlit needs to preserve scroll/focus.
    if not df.empty:
        changed_count = 0
        for idx in df.index:
            # Sync Select state
            df.at[idx, "Select"] = bool(edited_df.at[idx, "Select"])

            # Detect Category edits
            new_val = str(edited_df.at[idx, "category"] or "").strip()
            old_val_display = str(editor_view_for_render.at[idx, "category"] or "").strip()
            if new_val and new_val != old_val_display:
                cid = cat_id_by_name.get(new_val)
                if cid is not None:
                    add_label(conn, int(df.at[idx, "id"]), cid, "user")
                    df.at[idx, "category"] = new_val
                    df.at[idx, "label_source"] = "user"
                    df.at[idx, "confidence"] = ""
                    df.at[idx, "category_id"] = cid
                    changed_count += 1
        if changed_count:
            st.toast(f"saved {changed_count} label change(s)")

    # --- Bulk-set category for selected rows -------------------------------
    if not df.empty:
        n_selected = int(df["Select"].sum())
        if n_selected > 0:
            bulk_cols = st.columns([2, 3, 1, 1])
            bulk_cols[0].markdown(f"**{n_selected} row(s) selected**")
            bulk_new_cat = bulk_cols[1].selectbox(
                "Bulk-set category",
                [""] + all_cat_names,
                key="data_bulk_cat",
                label_visibility="collapsed",
                placeholder="Set category for selected…",
            )
            apply_bulk = bulk_cols[2].button(
                f"Apply to {n_selected}",
                type="primary",
                disabled=not bulk_new_cat,
                key="data_bulk_apply",
            )
            clear_sel = bulk_cols[3].button("Clear selection", key="data_bulk_clear")
            if apply_bulk and bulk_new_cat:
                cid = cat_id_by_name.get(bulk_new_cat)
                if cid is not None:
                    sel_mask = df["Select"] == True  # noqa: E712
                    for idx in df.index[sel_mask]:
                        add_label(conn, int(df.at[idx, "id"]), cid, "user")
                        df.at[idx, "category"] = bulk_new_cat
                        df.at[idx, "label_source"] = "user"
                        df.at[idx, "confidence"] = ""
                        df.at[idx, "category_id"] = cid
                    df["Select"] = False
                    st.toast(f"set {bulk_new_cat} on {int(sel_mask.sum())} row(s)")
                    # Clear data_editor's session state so the unticked
                    # checkboxes show; this rerun is intentional and only
                    # happens on an explicit Apply click.
                    st.session_state.pop("data_editor", None)
                    st.rerun()
            if clear_sel:
                df["Select"] = False
                st.session_state.pop("data_editor", None)
                st.rerun()

    # --- Optional row drawer (notes + all-fields inspection) ---------------
    sel_id_text = st.text_input(
        "Inspect record by ID (notes + full fields)",
        value="",
        placeholder="enter an expense id from the ID column",
        key="data_inspect_id",
    )
    if sel_id_text.strip().isdigit():
        eid = int(sel_id_text.strip())
        full = conn.execute("SELECT * FROM expenses WHERE id = ?", (eid,)).fetchone()
        if full is None:
            st.warning(f"no record with id {eid}")
        else:
            full_dict = dict(full)
            with st.container(border=True):
                st.subheader(f"Record #{eid}")
                head = st.columns(3)
                head[0].write(f"**Date:** {full_dict.get('buchungsdatum')}")
                head[1].write(f"**Amount:** {full_dict.get('betrag_cents', 0) / 100:.2f} €")
                head[2].write(f"**Counterparty:** {full_dict.get('counterparty')}")
                if full_dict.get("verwendungszweck"):
                    st.caption(full_dict["verwendungszweck"])
                meta_cols = st.columns(3)
                meta_cols[0].write(f"**Cluster:** {full_dict.get('cluster_id')}")
                meta_cols[1].write(f"**Source file:** {full_dict.get('source_file') or '—'}")
                meta_cols[2].write(f"**IBAN:** {full_dict.get('iban') or '—'}")

                note = get_note(conn, eid) or ""
                new_note = st.text_area("Note", value=note, key=f"drawer_note_{eid}")
                if st.button("Save note", key=f"drawer_save_note_{eid}"):
                    set_note(conn, eid, new_note)
                    st.success("note saved")

                with st.expander("All fields"):
                    st.json({
                        k: (v if not isinstance(v, bytes) else f"<{len(v)} bytes>")
                        for k, v in full_dict.items()
                    })


# ---------------------------------------------------------------------------
# Tab 5: Clusters
# ---------------------------------------------------------------------------

with tab_clusters:
    st.header("Clusters")
    st.caption(
        "Unsupervised grouping (UMAP → HDBSCAN) over the same embeddings the "
        "classifier uses. Useful for **outlier surfacing** (`cluster_id = -1` "
        "are records that don't look like anything else) and **vendor-group "
        "discovery** when you have several similar merchants. Less useful "
        "once your categories are stable."
    )
    if st.button("(Re-)compute clusters"):
        with st.status("clustering…", expanded=True) as status:
            emb = _embedder()
            status.write("computing embeddings (cached)…")
            report = cluster_all(conn, cfg, emb)
            status.update(
                label=f"{report.n_clusters} clusters, {report.n_outliers} outliers of {report.n_points} points",
                state="complete",
            )

    df = pd.read_sql_query(
        """
        SELECT cluster_id,
               COUNT(*) AS n,
               GROUP_CONCAT(DISTINCT counterparty_normalized) AS sample_vendors
        FROM expenses
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
        ORDER BY n DESC
        """,
        conn,
    )
    if df.empty:
        st.info("No clusters computed yet.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab 6: Settings
# ---------------------------------------------------------------------------

with tab_settings:
    from expense_analyzer.config import save_user_config
    from expense_analyzer.features.model_registry import (
        EMBEDDING_MODELS,
        ZEROSHOT_MODELS,
        hf_cache_dir,
        is_downloaded,
        trigger_download,
    )

    st.header("Settings")

    # --- Device / dev toggle --------------------------------------------
    st.subheader("Models")
    info_cols = st.columns([2, 1])
    with info_cols[0]:
        st.write(f"**Current embedding model:** `{cfg.embedding_model}`")
        st.write(f"**Current zero-shot model:** `{cfg.zeroshot_model}`")
        st.write(f"**Device:** `{cfg.device}`")
        st.caption(f"HF cache: `{hf_cache_dir()}`")
    with info_cols[1]:
        st.checkbox(
            "Use hash-based dummy embedder (dev / fast iteration)",
            value=st.session_state.get("use_hash_embedder", False),
            key="use_hash_embedder",
            help="Skip the HF model and use a deterministic hash embedder. "
            "Quality drops sharply; only useful for clicking through the UI.",
        )

    def _render_model_table(role_label: str, models, current_id: str, cfg_key: str) -> None:
        st.markdown(f"##### {role_label}")
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
        sel_cols = st.columns([3, 1, 1])
        with sel_cols[0]:
            picked = st.selectbox(
                "Switch to",
                [m.model_id for m in models],
                index=next((i for i, m in enumerate(models) if m.model_id == current_id), 0),
                key=f"model_pick_{cfg_key}",
            )
        with sel_cols[1]:
            present, _ = is_downloaded(picked)
            dl_label = "Download" if not present else "Re-download"
            if st.button(dl_label, key=f"model_dl_{cfg_key}"):
                with st.status(f"Downloading {picked}...", expanded=True) as status:
                    role = "embedding" if cfg_key == "embedding_model" else "zeroshot"
                    try:
                        trigger_download(picked, role=role)
                        status.update(label=f"Downloaded {picked}", state="complete")
                    except Exception as e:
                        status.update(label=f"Download failed: {e}", state="error")
                st.rerun()
        with sel_cols[2]:
            st.write("")
            st.write("")
            if st.button("Use this", key=f"model_use_{cfg_key}", type="primary",
                         disabled=picked == current_id):
                save_user_config({cfg_key: picked}, data_dir=cfg.data_dir)
                st.cache_resource.clear()
                st.success(
                    f"`{cfg_key}` set to `{picked}`. Restart the UI for it to take effect: "
                    "`expense ui-restart`."
                )

    _render_model_table("Embedding model", EMBEDDING_MODELS, cfg.embedding_model, "embedding_model")
    _render_model_table("Zero-shot model", ZEROSHOT_MODELS, cfg.zeroshot_model, "zeroshot_model")

    st.subheader("Privacy")
    st.write(f"Vendor web lookup enabled: **{cfg.vendor_lookup.enabled}**")
    if cfg.vendor_lookup.enabled:
        st.warning(
            "Vendor lookup is ON. Only `counterparty_normalized` is sent to "
            f"{cfg.vendor_lookup.backend}; never amount/IBAN/Verwendungszweck."
        )
    else:
        st.info("Vendor lookup is OFF. Set `vendor_lookup.enabled: true` in your config to enable.")

    st.subheader(":red[Danger zone — clear data]")
    with st.expander("Clear ingested expenses + ML state", expanded=False):
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
                    f"deleted {report.total} row(s) across {len(report.table_counts)} table(s)"
                )
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")

    with st.expander("Factory reset (everything, incl. categories)", expanded=False):
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
                    f"deleted {report.total} row(s) across {len(report.table_counts)} table(s)"
                )
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")
