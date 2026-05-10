"""Streamlit UI. Local-only — binds to 127.0.0.1 via the CLI launcher.

Tabs:
  * Dashboard — interactive charts.
  * Label — active-learning queue.
  * Review — low-confidence model labels.
  * Clusters — HDBSCAN cluster browser.
  * Notes — per-expense free-form notes.
  * Settings — categories + vendor-lookup toggle (read-only summary).
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from expense_analyzer.config import Config, load_config
from expense_analyzer.enrichment.notes import get_note, set_note
from expense_analyzer.features.embeddings import (
    HashEmbedder,
    SentenceTransformerEmbedder,
    store_embeddings,
)
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ml.active_learning import pick_candidates
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.ml.clustering import cluster_all
from expense_analyzer.storage.categories import (
    add_label,
    list_categories,
    upsert_category,
)
from expense_analyzer.storage.database import get_or_create_database
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

# --- Boot --------------------------------------------------------------------

st.set_page_config(page_title="expense-analyzer-de", layout="wide")


@st.cache_resource
def _load_config_cached() -> Config:
    return load_config()


@st.cache_resource
def _connect_cached(db_path_str: str) -> sqlite3.Connection:
    return get_or_create_database(__import__("pathlib").Path(db_path_str))


@st.cache_resource
def _embedder_cached(model_name: str, device: str, batch_size: int, use_hash: bool):
    if use_hash:
        return HashEmbedder(dim=64)
    return SentenceTransformerEmbedder(
        model_name=model_name, device=device, batch_size=batch_size
    )


cfg = _load_config_cached()
conn = _connect_cached(str(cfg.db_path))


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("expense-analyzer-de")
    st.caption(f"DB: `{cfg.db_path}`")

    # Quick stats
    try:
        n_exp = conn.execute("SELECT COUNT(*) AS n FROM expenses").fetchone()["n"]
        n_lab = conn.execute(
            "SELECT COUNT(DISTINCT expense_id) AS n FROM labels WHERE source='user'"
        ).fetchone()["n"]
    except sqlite3.OperationalError:
        n_exp = n_lab = 0
    st.metric("Expenses", n_exp)
    st.metric("User-labeled", n_lab)

    st.divider()
    st.subheader("Ingest CSV")
    uploaded = st.file_uploader("Upload one or more bank-export CSVs", accept_multiple_files=True)
    if uploaded:
        import tempfile
        from pathlib import Path

        for f in uploaded:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(f.read())
                p = Path(tmp.name)
            r = ingest_csv(conn, p)
            st.success(f"{f.name}: parsed={r.parsed} new={r.inserted} duplicate={r.duplicates}")
        st.rerun()

    st.divider()
    use_hash_for_dev = st.checkbox(
        "Use hash-based dummy embedder (skip HF download)",
        value=False,
        help="Useful for very fast iteration. Quality will be much lower than the real model.",
    )

embedder = _embedder_cached(
    cfg.embedding_model, cfg.device, cfg.embedding_batch_size, use_hash_for_dev
)


# --- Tabs --------------------------------------------------------------------

tabs = st.tabs(["Dashboard", "Label", "Review", "Clusters", "Notes", "Settings"])


# Dashboard ------------------------------------------------------------------
with tabs[0]:
    st.header("Dashboard")
    if n_exp == 0:
        st.info("Ingest a CSV to see charts.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(pie_chart(spend_by_category(conn)), use_container_width=True)
        with c2:
            st.plotly_chart(
                bar_top_counterparties(top_counterparties(conn, n=15)),
                use_container_width=True,
            )
        st.plotly_chart(
            trend_lines(monthly_flow_by_category(conn)), use_container_width=True
        )
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                histogram_amounts(amount_distribution(conn)), use_container_width=True
            )
        with c4:
            st.plotly_chart(
                __import__("expense_analyzer.viz", fromlist=["calendar_heatmap"]).calendar_heatmap(
                    daily_calendar(conn)
                ),
                use_container_width=True,
            )


# Label ----------------------------------------------------------------------
with tabs[1]:
    st.header("Label")
    cats = list_categories(conn)
    if not cats:
        st.warning("Define some categories first (Settings tab).")
    elif n_exp == 0:
        st.info("No expenses yet.")
    else:
        col_n, col_strat = st.columns(2)
        with col_n:
            n = st.number_input("How many candidates", min_value=1, max_value=50, value=10)
        with col_strat:
            strategy = st.selectbox(
                "Strategy", ["uncertainty", "diverse", "outliers", "mixed"], index=0
            )
        if st.button("Surface candidates"):
            # Embeddings are needed for kNN + diverse.
            rows = conn.execute("SELECT id, combined_text FROM expenses").fetchall()
            store_embeddings(conn, embedder, [(r["id"], r["combined_text"]) for r in rows])
            cascade = CategorizationCascade(conn, cfg, embedder)
            try:
                cascade.fit()
            except Exception:
                pass
            ids = pick_candidates(conn, cfg, embedder, cascade, n=int(n), strategy=strategy)
            st.session_state["label_ids"] = ids

        ids = st.session_state.get("label_ids", [])
        for eid in ids:
            row = conn.execute(
                "SELECT id, buchungsdatum, betrag_cents, counterparty, verwendungszweck "
                "FROM expenses WHERE id = ?",
                (eid,),
            ).fetchone()
            if row is None:
                continue
            with st.container(border=True):
                st.write(
                    f"**{row['buchungsdatum']}** — "
                    f"{row['betrag_cents'] / 100:.2f} € — "
                    f"**{row['counterparty']}**"
                )
                if row["verwendungszweck"]:
                    st.caption(row["verwendungszweck"])
                cols = st.columns([3, 1])
                with cols[0]:
                    pick = st.selectbox(
                        "Category",
                        ["—"] + [c.name for c in cats],
                        key=f"label_pick_{eid}",
                    )
                with cols[1]:
                    if st.button("Save", key=f"label_save_{eid}"):
                        if pick != "—":
                            cid = next(c.id for c in cats if c.name == pick)
                            add_label(conn, int(eid), cid, "user")
                            st.success(f"labeled #{eid} -> {pick}")


# Review ---------------------------------------------------------------------
with tabs[2]:
    st.header("Review low-confidence model labels")
    rows = conn.execute(
        """
        SELECT e.id, e.buchungsdatum, e.betrag_cents, e.counterparty, e.verwendungszweck,
               c.name AS predicted, l.confidence
        FROM expenses e
        JOIN labels l ON l.id = (
            SELECT MAX(id) FROM labels WHERE expense_id = e.id
        )
        JOIN categories c ON c.id = l.category_id
        WHERE l.source = 'model' AND l.confidence < ?
        ORDER BY l.confidence ASC
        LIMIT 100
        """,
        (cfg.classifier.confidence_threshold,),
    ).fetchall()
    if not rows:
        st.info("Nothing flagged for review (run `expense predict` first).")
    else:
        cats = list_categories(conn)
        for r in rows:
            with st.container(border=True):
                st.write(
                    f"**{r['buchungsdatum']}** — {r['betrag_cents']/100:.2f} € — **{r['counterparty']}** — "
                    f"predicted **{r['predicted']}** (conf {r['confidence']:.2f})"
                )
                if r["verwendungszweck"]:
                    st.caption(r["verwendungszweck"])
                pick = st.selectbox(
                    "Confirm/correct",
                    [c.name for c in cats],
                    index=[c.name for c in cats].index(r["predicted"]),
                    key=f"review_{r['id']}",
                )
                if st.button("Save as user label", key=f"review_save_{r['id']}"):
                    cid = next(c.id for c in cats if c.name == pick)
                    add_label(conn, int(r["id"]), cid, "user")
                    st.success("saved")


# Clusters -------------------------------------------------------------------
with tabs[3]:
    st.header("Clusters")
    if st.button("(Re-)compute clusters"):
        report = cluster_all(conn, cfg, embedder)
        st.success(
            f"{report.n_clusters} clusters, {report.n_outliers} outliers of {report.n_points} points"
        )
    df = pd.read_sql_query(
        """
        SELECT cluster_id, COUNT(*) AS n,
               GROUP_CONCAT(DISTINCT counterparty_normalized) AS sample_vendors
        FROM expenses
        GROUP BY cluster_id
        ORDER BY n DESC
        """,
        conn,
    )
    if df.empty:
        st.info("Run clustering first.")
    else:
        st.dataframe(df, use_container_width=True)


# Notes ----------------------------------------------------------------------
with tabs[4]:
    st.header("Notes")
    eid = st.number_input("Expense id", min_value=1, value=1, step=1)
    row = conn.execute(
        "SELECT buchungsdatum, betrag_cents, counterparty, verwendungszweck "
        "FROM expenses WHERE id = ?",
        (int(eid),),
    ).fetchone()
    if row is None:
        st.info("No expense with this id.")
    else:
        st.write(
            f"**{row['buchungsdatum']}** — {row['betrag_cents']/100:.2f} € — "
            f"**{row['counterparty']}**"
        )
        if row["verwendungszweck"]:
            st.caption(row["verwendungszweck"])
        existing = get_note(conn, int(eid)) or ""
        new_text = st.text_area("Note", value=existing, key=f"note_{eid}")
        if st.button("Save note"):
            set_note(conn, int(eid), new_text)
            st.success("saved")


# Settings -------------------------------------------------------------------
with tabs[5]:
    from expense_analyzer.storage.admin import (
        category_removal_impact,
        remove_category,
        reset_all,
        reset_data,
    )

    st.header("Settings")
    st.subheader("Categories")
    cats = list_categories(conn)
    st.dataframe(
        pd.DataFrame([{"id": c.id, "name": c.name, "description": c.description, "color": c.color} for c in cats]),
        use_container_width=True,
    )
    with st.form("add_cat"):
        name = st.text_input("Add category — name")
        desc = st.text_input("Description")
        color = st.text_input("Color (hex)", value="#888")
        if st.form_submit_button("Add / update"):
            if name.strip():
                upsert_category(conn, name.strip(), desc.strip(), color.strip() or "#888")
                st.success(f"saved {name}")
                st.rerun()

    with st.expander("Remove a category", expanded=False):
        if not cats:
            st.caption("no categories yet")
        else:
            to_remove = st.selectbox(
                "Category to remove",
                ["—"] + [c.name for c in cats],
                key="remove_cat_pick",
            )
            cascade = st.checkbox(
                "Force-delete even if labels reference this category",
                value=False,
                key="remove_cat_cascade",
            )
            if to_remove != "—":
                impact = category_removal_impact(conn, to_remove)
                if impact.n_labels > 0:
                    st.warning(
                        f"{impact.n_labels} label(s) reference {to_remove!r}. "
                        f"Removing it will also delete those labels."
                    )
                if st.button(f"Remove {to_remove}", type="secondary"):
                    if impact.n_labels > 0 and not cascade:
                        st.error("Tick the force-delete box first.")
                    else:
                        result = remove_category(conn, to_remove)
                        st.success(
                            f"removed {result.name}; {result.n_labels_deleted} label(s) cascaded"
                        )
                        st.rerun()

    st.subheader("Privacy")
    st.write(f"Vendor web lookup enabled: **{cfg.vendor_lookup.enabled}**")
    if cfg.vendor_lookup.enabled:
        st.warning(
            "Vendor lookup is ON. Only `counterparty_normalized` is sent to "
            f"{cfg.vendor_lookup.backend}; never amount/IBAN/Verwendungszweck."
        )
    else:
        st.info("Vendor lookup is OFF. Set `vendor_lookup.enabled: true` in your config to enable.")

    st.subheader("Models")
    st.write(f"Embedding model: `{cfg.embedding_model}`")
    st.write(f"Zero-shot model: `{cfg.zeroshot_model}`")
    st.write(f"Device: `{cfg.device}`")

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
                st.success(f"deleted {report.total} row(s) across {len(report.table_counts)} table(s)")
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")

    with st.expander("Factory reset (everything, incl. categories)", expanded=False):
        st.write(
            "Wipes every table including categories and own-IBANs. The DB schema "
            "stays so you can immediately re-`init`."
        )
        confirm_all = st.text_input(
            "Type `factory reset` to confirm", key="confirm_reset_all"
        )
        if st.button("Factory reset"):
            if confirm_all.strip().lower() == "factory reset":
                report = reset_all(conn)
                st.success(f"deleted {report.total} row(s) across {len(report.table_counts)} table(s)")
                st.rerun()
            else:
                st.error("type the confirmation phrase exactly")
