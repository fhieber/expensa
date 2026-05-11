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

        st.write(f"**{len(new_ids)} new record(s)** ingested. Review and label below.")

        ph = ",".join("?" * len(new_ids))
        df = pd.read_sql_query(
            f"""
            SELECT id, buchungsdatum, counterparty, verwendungszweck,
                   betrag_cents,
                   amount_bucket, iban_country, iban_is_foreign,
                   has_glaeubiger_id, mandatsreferenz_present,
                   umsatztyp
            FROM expenses
            WHERE id IN ({ph})
            ORDER BY buchungsdatum, id
            """,
            conn,
            params=new_ids,
        )
        df["betrag_€"] = df["betrag_cents"] / 100
        extended = st.toggle("Extended columns", value=False, key="import_extended")
        if extended:
            show_cols = ["id", "buchungsdatum", "counterparty", "verwendungszweck",
                         "betrag_€", "amount_bucket", "umsatztyp",
                         "iban_country", "iban_is_foreign",
                         "has_glaeubiger_id", "mandatsreferenz_present"]
        else:
            show_cols = ["id", "buchungsdatum", "counterparty",
                         "verwendungszweck", "betrag_€"]
        st.dataframe(df[show_cols], width="stretch", hide_index=True)

        # --- Auto-label controls --------------------------------------------
        st.subheader("Auto-label")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        run_auto = c1.button("Auto-label", type="primary")
        accept_confident = c2.button(
            f"Accept all ≥ {int(ACCEPT_CONFIDENT_THRESHOLD * 100)}%",
            disabled=not st.session_state.import_predictions,
        )
        accept_all = c3.button("Accept all", disabled=not st.session_state.import_predictions)
        done = c4.button("Done", help="Leave any not-yet-confirmed predictions as model-labels.")

        if run_auto:
            emb = _embedder()
            cascade = CategorizationCascade(conn, cfg, emb)
            with st.status("Predicting…", expanded=False):
                try:
                    cascade.fit()
                except Exception:
                    pass
                preds = cascade.predict_batch(new_ids)
                st.session_state.import_predictions = {p.expense_id: p for p in preds}
                # Persist model labels for the predictions that fired.
                for p in preds:
                    if p.category_id is not None:
                        add_label(conn, p.expense_id, p.category_id, "model",
                                  confidence=p.confidence)
            st.rerun()

        if accept_confident:
            n_promoted = 0
            for eid, p in st.session_state.import_predictions.items():
                override = st.session_state.import_overrides.get(eid)
                final_cat = override if override is not None else p.category_id
                final_conf = (
                    1.0 if override is not None
                    else p.confidence
                )
                if final_cat is not None and final_conf >= ACCEPT_CONFIDENT_THRESHOLD:
                    add_label(conn, eid, final_cat, "user")
                    n_promoted += 1
            st.success(f"promoted {n_promoted} prediction(s) to user labels")

        if accept_all:
            n_promoted = 0
            for eid, p in st.session_state.import_predictions.items():
                override = st.session_state.import_overrides.get(eid)
                final_cat = override if override is not None else p.category_id
                if final_cat is not None:
                    add_label(conn, eid, final_cat, "user")
                    n_promoted += 1
            st.success(f"promoted {n_promoted} prediction(s) to user labels")

        if done:
            _reset_import_state()
            st.rerun()

        # --- Per-row review --------------------------------------------------
        preds: dict[int, Prediction] = st.session_state.import_predictions
        if preds:
            st.subheader("Per-row review")
            cat_opts = _category_options(include_unlabeled=False)
            id_to_idx = {cid: i for i, (cid, _) in enumerate(cat_opts)}
            for eid in new_ids:
                p = preds.get(eid)
                row = conn.execute(
                    "SELECT buchungsdatum, betrag_cents, counterparty, verwendungszweck "
                    "FROM expenses WHERE id = ?",
                    (eid,),
                ).fetchone()
                stage_color = {
                    "vendor_exact_match": "🟢",
                    "knn": "🟢",
                    "classifier": "🟡",
                    "zeroshot": "🟡",
                    "unknown": "⚪",
                }
                with st.container(border=True):
                    head_cols = st.columns([1, 1, 2, 1])
                    head_cols[0].write(f"**{row['buchungsdatum']}**")
                    head_cols[1].write(_format_eur(row['betrag_cents']))
                    head_cols[2].write(f"**{row['counterparty']}**")
                    if p is not None:
                        head_cols[3].write(
                            f"{stage_color.get(p.stage, '⚪')} {p.stage} · "
                            f"{p.confidence:.2f}"
                        )
                    if row["verwendungszweck"]:
                        st.caption(row["verwendungszweck"])
                    default_cid = (
                        st.session_state.import_overrides.get(eid)
                        or (p.category_id if p else None)
                    )
                    default_idx = id_to_idx.get(default_cid, 0) if default_cid is not None else 0
                    pick = st.selectbox(
                        "Category",
                        options=[name for _, name in cat_opts],
                        index=default_idx,
                        key=f"import_pick_{eid}",
                    )
                    chosen_cid = next(cid for cid, name in cat_opts if name == pick)
                    if chosen_cid != default_cid:
                        st.session_state.import_overrides[eid] = chosen_cid
                    if st.button("Save as user label", key=f"import_save_{eid}"):
                        add_label(conn, eid, chosen_cid, "user")
                        st.success(f"saved → {pick}")


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
    sql, params = _build_data_query(
        date_from or None, date_to or None,
        picked_cats, source, search.strip(),
        amount_min if amount_min > 0 else None, amount_max,
        include_income,
    )
    df = pd.read_sql_query(sql, conn, params=params)

    # Counts BEFORE the fillna pass below, so we can use NaN semantics.
    def _to_conf(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    n_unlabeled = int(df["label_source"].isna().sum()) if not df.empty else 0
    if not df.empty:
        n_confident_model = int(
            (
                (df["label_source"] == "model")
                & df["confidence"].apply(_to_conf).ge(ACCEPT_CONFIDENT_THRESHOLD)
            ).sum()
        )
    else:
        n_confident_model = 0

    if not df.empty:
        df["category"] = df["category"].fillna("(unkategorisiert)")
        df["confidence"] = df["confidence"].fillna("")
        df["label_source"] = df["label_source"].fillna("")
    base_cols = ["id", "buchungsdatum", "counterparty", "verwendungszweck",
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

    st.caption(f"{len(df)} record(s) match the filters")
    event = st.dataframe(
        df[show_cols] if not df.empty else pd.DataFrame(columns=show_cols),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=420,
    )

    # Row drawer
    sel_rows = event.selection.rows if event and event.selection else []
    if sel_rows:
        row_idx = sel_rows[0]
        eid = int(df.iloc[row_idx]["id"])
        with st.container(border=True):
            full = conn.execute(
                "SELECT * FROM expenses WHERE id = ?", (eid,)
            ).fetchone()
            full_dict = dict(full) if full else {}
            st.subheader(f"Record #{eid}")
            head = st.columns(3)
            head[0].write(f"**Date:** {full_dict.get('buchungsdatum')}")
            head[1].write(f"**Amount:** {full_dict.get('betrag_cents', 0) / 100:.2f} €")
            head[2].write(f"**Counterparty:** {full_dict.get('counterparty')}")
            if full_dict.get("verwendungszweck"):
                st.caption(full_dict["verwendungszweck"])

            edit_cols = st.columns([2, 1])
            with edit_cols[0]:
                opts = _category_options(include_unlabeled=False)
                current = conn.execute(
                    """
                    SELECT l.category_id FROM labels l
                    WHERE l.expense_id = ?
                    ORDER BY l.id DESC LIMIT 1
                    """,
                    (eid,),
                ).fetchone()
                current_cid = int(current["category_id"]) if current else None
                idx = 0
                for i, (cid, _) in enumerate(opts):
                    if cid == current_cid:
                        idx = i
                        break
                pick = st.selectbox(
                    "Category", [name for _, name in opts], index=idx,
                    key=f"drawer_cat_{eid}",
                )
                if st.button("Save category", key=f"drawer_save_cat_{eid}"):
                    chosen_cid = next(cid for cid, name in opts if name == pick)
                    add_label(conn, eid, chosen_cid, "user")
                    st.success(f"saved → {pick}")
                    st.rerun()
            with edit_cols[1]:
                st.write("**Cluster:**", full_dict.get("cluster_id"))
                st.write("**Source file:**", full_dict.get("source_file") or "—")
                st.write("**IBAN:**", full_dict.get("iban") or "—")

            note = get_note(conn, eid) or ""
            new_note = st.text_area("Note", value=note, key=f"drawer_note_{eid}")
            if st.button("Save note", key=f"drawer_save_note_{eid}"):
                set_note(conn, eid, new_note)
                st.success("note saved")

            with st.expander("All fields"):
                st.json(
                    {k: (v if not isinstance(v, bytes) else f"<{len(v)} bytes>") for k, v in full_dict.items()}
                )


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
