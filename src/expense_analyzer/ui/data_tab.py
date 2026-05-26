"""Data tab: CSV import, sortable/filterable table, label-editing flow.

Pending edits to the Category column accumulate into three orthogonal
stashes (see :mod:`._pending_edits`) which overlay onto the SQL-fetched
rows before they reach AgGrid. Save Changes commits them as user / model
label rows; Revert Changes drains all three stashes.
"""

from __future__ import annotations

import json as _json
import re
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from expense_analyzer.enrichment.notes import get_note, set_note
from expense_analyzer.ingestion import ingest_csv
from expense_analyzer.ml.classifier import CategorizationCascade
from expense_analyzer.storage.admin import clear_labels_for_expense
from expense_analyzer.storage.categories import add_label, list_categories
from expense_analyzer.ui._components import date_preset_row, de_eur
from expense_analyzer.ui._pending_edits import (
    PendingEdits,
    merge_user_typed,
)
from expense_analyzer.ui._pending_edits import (
    clear_all as clear_pending_edits,
)
from expense_analyzer.ui._pending_edits import (
    load as load_pending_edits,
)
from expense_analyzer.ui._shared import get_config, get_conn, get_embedder
from expense_analyzer.utils.colors import readable_text_color

# SELECT that joins the most-recent label via the `latest_label` view
# defined in schema.sql. SQLite inlines views at plan time, so this is
# just a list of selected columns.
#
# `last_modified` captures the most recent write to the row from any
# source: any label insert (user edit, auto-label, predict-all), any
# note update, or the original ingestion timestamp as a fallback.
# Picks the greatest of the three via SQLite's scalar `MAX(a, b, c)`
# (NULL-tolerant: ignores NULL args, picks the largest non-NULL).
# Joined via two LEFT JOINs (cheap on idx_labels_expense + notes pk)
# instead of correlated subqueries.
# Canonical visible-column order, applied at the end of dataframe
# assembly so AgGrid renders cells in this sequence. Hidden columns
# (overlay flags, raw label_source, etc.) get appended after these so
# they don't affect the rendered layout.
#
# Bump `_GRID_STATE_KEY` below when this order changes -- otherwise
# AgGrid will keep restoring the previous order from session_state
# columnState. The suffix orphans the prior cached state on the user's
# next visit (they lose any one-off column drag-reorder customisations,
# but the canonical layout becomes the source of truth again).
_VISIBLE_COL_ORDER = [
    "id", "buchungsdatum",
    "counterparty", "zahlungspflichtiger", "verwendungszweck",
    "betrag_€",
    "category", "confidence", "src", "iban",
    "umsatztyp", "iban_is_foreign",
    "has_glaeubiger_id", "mandatsreferenz_present",
    "last_modified",
]
_GRID_STATE_KEY = "data_aggrid_grid_state_v3"

_DATA_RECORDS_SELECT = """
    SELECT
        e.id, e.buchungsdatum,
        e.counterparty,
        e.zahlungspflichtiger, e.verwendungszweck,
        e.betrag_cents / 100.0 AS "betrag_€",
        c.name AS category, ll.category_id AS category_id,
        ll.label_source AS label_source, ll.confidence,
        MAX(last_lbl.last_at, n.updated_at, e.imported_at) AS last_modified,
        e.umsatztyp, e.iban, e.iban_is_foreign,
        e.has_glaeubiger_id, e.mandatsreferenz_present
    FROM expenses e
    LEFT JOIN latest_label ll ON ll.expense_id = e.id
    LEFT JOIN categories c ON c.id = ll.category_id
    LEFT JOIN (
        SELECT expense_id, MAX(created_at) AS last_at
        FROM labels GROUP BY expense_id
    ) last_lbl ON last_lbl.expense_id = e.id
    LEFT JOIN notes n ON n.expense_id = e.id
"""


def _build_data_query(
    date_from, date_to, cats: list[str], source: str,
    search: str, amount_min: float, amount_max: float,
    include_income: bool,
    pinned_ids: list[int] | None = None,
) -> tuple[str, list]:
    """Return (SQL, params) for the Data table given filter widgets.

    If ``pinned_ids`` is provided, ONLY rows with those IDs are shown
    (other filters are ignored). Used right after a CSV ingest to scope
    the Data table to the just-imported records.
    """
    parts: list[str] = []
    params: list = []
    if pinned_ids:
        ph_pin = ",".join("?" * len(pinned_ids))
        parts.append(f"e.id IN ({ph_pin})")
        params.extend(int(x) for x in pinned_ids)
        where = " WHERE " + " AND ".join(parts)
        return (
            _DATA_RECORDS_SELECT + where + " ORDER BY e.buchungsdatum DESC, e.id DESC",
            params,
        )
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
            "(LOWER(e.counterparty) LIKE ? "
            "OR LOWER(e.verwendungszweck) LIKE ?)"
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
        parts.append("ll.label_source = 'user'")
    elif source == "model":
        parts.append("ll.label_source = 'model'")
    elif source == "unlabeled":
        parts.append("ll.expense_id IS NULL")

    where = (" WHERE " + " AND ".join(parts)) if parts else ""
    return (
        _DATA_RECORDS_SELECT + where + " ORDER BY e.buchungsdatum DESC, e.id DESC",
        params,
    )


# ---------------------------------------------------------------------------
# Inspect dialog -- module-level so st.dialog's decorator runs once.
# ---------------------------------------------------------------------------


def _on_inspect_dismiss() -> None:
    """Clear the open-eid sentinel when the dialog is dismissed via the X
    button or by clicking outside. Without this the next rerun re-renders
    the dialog because the session_state value is still set."""
    st.session_state.pop("data_inspect_open_eid", None)


@st.dialog("Record details", width="large", on_dismiss=_on_inspect_dismiss)
def _show_inspect_dialog(eid: int) -> None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM expenses WHERE id = ?", (eid,)).fetchone()
    if row is None:
        st.warning(f"no record with id {eid}")
        if st.button("Close", key="inspect_close_missing"):
            st.session_state.pop("data_inspect_open_eid", None)
            st.rerun()
        return
    full_dict = dict(row)
    st.markdown(f"##### Record #{eid}")
    head = st.columns(3)
    head[0].write(f"**Date:** {full_dict.get('buchungsdatum')}")
    head[1].write(f"**Amount:** {full_dict.get('betrag_cents', 0) / 100:.2f} €")
    head[2].write(f"**Counterparty:** {full_dict.get('counterparty')}")
    if full_dict.get("verwendungszweck"):
        st.caption(full_dict["verwendungszweck"])
    meta_cols = st.columns(3)
    meta_cols[0].write(f"**Source file:** {full_dict.get('source_file') or '—'}")
    meta_cols[1].write(f"**IBAN:** {full_dict.get('iban') or '—'}")
    meta_cols[2].write(f"**Umsatztyp:** {full_dict.get('umsatztyp') or '—'}")

    note = get_note(conn, eid) or ""
    new_note = st.text_area("Note", value=note, key=f"inspect_note_{eid}")
    btn_cols = st.columns([1, 1, 4])
    if btn_cols[0].button("💾 Save note", key=f"inspect_save_note_{eid}"):
        set_note(conn, eid, new_note)
        st.toast("note saved")
    if btn_cols[1].button("Close", key=f"inspect_close_{eid}"):
        st.session_state.pop("data_inspect_open_eid", None)
        st.rerun()

    with st.expander("All fields", expanded=True):
        st.json({
            k: (v if not isinstance(v, bytes) else f"<{len(v)} bytes>")
            for k, v in full_dict.items()
        })


def maybe_show_inspect_dialog() -> None:
    """Called once per render from the orchestrator. Opens the inspect
    dialog if the Data tab parked an expense id in session_state."""
    if st.session_state.get("data_inspect_open_eid"):
        _show_inspect_dialog(int(st.session_state["data_inspect_open_eid"]))


# ---------------------------------------------------------------------------
# Tab entry point.
# ---------------------------------------------------------------------------


def render() -> None:
    conn = get_conn()
    cfg = get_config()

    _render_import_expander(conn)
    pinned_ids = _render_pinned_banner()

    # Date range + search.
    date_from, date_to = date_preset_row(key_prefix="data", default="All-time")
    search_text = st.text_input(
        "Search all fields",
        value="",
        key="data_quick_search",
        placeholder="e.g. food, *aldi*, REWE*Berlin",
        help=(
            "Case-insensitive substring match across Counterparty, "
            "Verwendungszweck, Category, Source, IBAN, Umsatztyp, ID "
            "and Amount. Use `*` as a wildcard (e.g. `rewe*berlin`). "
            "Multiple terms separated by space must all match somewhere "
            "in the row."
        ),
    )
    include_income = True  # always loaded; filter via Amount column header

    extended = bool(st.session_state.get("data_extended", False))
    _bump_aggrid_seed_on_changes(search_text, date_from, date_to, extended,
                                 date_preset=st.session_state.get("data_date_preset"))

    df, edits = _fetch_and_overlay(
        conn, date_from, date_to, include_income, pinned_ids, search_text
    )

    all_cat_objs = list_categories(conn)
    all_cat_names = sorted(c.name for c in all_cat_objs)
    cat_id_by_name = {c.name: c.id for c in all_cat_objs}
    cat_name_by_id = {c.id: c.name for c in all_cat_objs}
    # {category_name: {"bg": "#rrggbb", "fg": "#000000" | "#ffffff"}} -- consumed
    # by the AgGrid Category cell's cellStyle JS so each cell paints in
    # the user-chosen category colour with auto-picked legible text.
    cat_color_map = {
        c.name: {"bg": c.color or "#888888", "fg": readable_text_color(c.color or "#888888")}
        for c in all_cat_objs
    }

    grid_options, aggrid_key = _build_grid_options(
        df, extended, all_cat_names, cat_color_map,
    )
    _render_top_action_bar_and_grid_and_actions(
        conn, cfg, df, edits, grid_options, aggrid_key,
        cat_id_by_name, cat_name_by_id,
    )


# ---------------------------------------------------------------------------
# Section: Import Data expander.
# ---------------------------------------------------------------------------


def _render_import_expander(conn) -> None:
    with st.expander("Import Data", expanded=False):
        st.caption(
            "Drop one or more German bank-export CSVs (`;` separator, comma "
            "decimal). On Ingest each new row's text/IBAN/numeric features "
            "are computed and stored; the table below then pins to those new "
            "rows so you can review and label them with the Auto-Label flow."
        )
        files = st.file_uploader(
            "CSV file(s)", accept_multiple_files=True, type=["csv"],
            key="data_import_files",
        )
        enrich_files = st.file_uploader(
            "Enrichment CSV(s) — optional (e.g. PayPal activity export)",
            accept_multiple_files=True, type=["csv"],
            key="data_enrich_files",
            help="A secondary CSV describing the same transactions with more "
                 "detail. Matched to bank rows by amount + nearby date; the "
                 "real merchant/item is attached and the row re-embedded.",
        )
        _btn_col, _chk_col = st.columns([1, 3], vertical_alignment="center")
        ingest_clicked = _btn_col.button(
            "Ingest", type="primary", disabled=not (files or enrich_files),
            key="data_ingest_btn",
        )
        skip_embed = _chk_col.checkbox(
            "Skip embedding computation",
            value=False,
            key="data_ingest_skip_embed",
            help=(
                "Skip the sentence-transformer embedding step. Useful when "
                "importing a large backlog or when the model isn't downloaded "
                "yet. Missing embeddings are filled automatically the first "
                "time Auto-Label or Predict-all runs."
            ),
        )
        if ingest_clicked and (files or enrich_files):
            emb = None if skip_embed else get_embedder()
            new_ids: list[int] = []
            with st.status("Importing…", expanded=True) as status:
                progress = st.progress(0.0, text="starting…")
                for f in files:
                    status.write(f"parsing {f.name}…")
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        tmp.write(f.read())
                        p = Path(tmp.name)

                    def _cb(phase: str, done: int, total: int, fname: str = f.name) -> None:
                        if total <= 0:
                            return
                        label_map = {
                            "parse": "parsing",
                            "insert": "inserting rows",
                            "embed": "computing embeddings",
                        }
                        phase_label = label_map.get(phase, phase)
                        progress.progress(
                            min(1.0, done / total),
                            text=f"{fname}: {phase_label} {done}/{total}",
                        )

                    r = ingest_csv(conn, p, embedder=emb, progress_callback=_cb)
                    new_ids.extend(r.new_ids)
                    status.write(
                        f"{f.name}: parsed={r.parsed} · new={r.inserted} · "
                        f"duplicate={r.duplicates} · embedded={r.embedded}"
                    )
                _run_ui_enrichment(conn, enrich_files, emb, status)
                progress.empty()
                status.update(
                    label=f"Imported {len(new_ids)} new row(s).",
                    state="complete",
                )
            if new_ids:
                st.session_state["data_pinned_ids"] = new_ids
                st.rerun()
            elif files:
                st.info("Nothing new — all records were duplicates.")


def _run_ui_enrichment(conn, enrich_files, emb, status) -> None:
    """Match each uploaded secondary CSV against existing expenses and
    enrich. Source is auto-detected per file."""
    if not enrich_files:
        return
    from expense_analyzer.enrichment.secondary import enrich_from_records
    from expense_analyzer.ingestion.sources import detect_adapter

    cfg = get_config()
    for f in enrich_files:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(f.read())
            p = Path(tmp.name)
        try:
            adapter = detect_adapter(p)
            records = adapter.parse(p)
            rep = enrich_from_records(
                conn, records, adapter, embedder=emb,
                date_window_days=cfg.enrichment.date_window_days,
            )
        except Exception as e:  # noqa: BLE001 - surface parse/detect errors in UI
            status.write(f"{f.name}: enrichment skipped ({e})")
            continue
        parts = [
            f"source: {rep.source}",
            f"parsed: {rep.parsed}",
            f"matched: {rep.matched}",
        ]
        if rep.already_enriched:
            parts.append(f"already enriched: {rep.already_enriched}")
        if rep.ambiguous:
            parts.append(f"ambiguous: {rep.ambiguous}")
        if rep.unmatched:
            parts.append(f"no match: {rep.unmatched}")
        if rep.reembedded:
            parts.append(f"re-embedded: {rep.reembedded}")
        status.write(f"{f.name}: " + " · ".join(parts))


def _render_pinned_banner() -> list[int]:
    pinned_ids: list[int] = st.session_state.get("data_pinned_ids") or []
    if pinned_ids:
        pin_cols = st.columns([5, 1])
        pin_cols[0].info(
            f"📌 Showing **{len(pinned_ids)} record(s)** from your last import. "
            "Filters below are ignored while pinned."
        )
        if pin_cols[1].button("Unpin", key="data_unpin_btn"):
            st.session_state.pop("data_pinned_ids", None)
            st.rerun()
    return pinned_ids


# ---------------------------------------------------------------------------
# Section: data fetch + overlay pending edits.
# ---------------------------------------------------------------------------


def _bump_aggrid_seed_on_changes(
    search_text: str, date_from, date_to, extended: bool, *, date_preset: str
) -> None:
    """AgGrid's client-wins sync caches its row data; we need to force a
    fresh init when filters change so the grid actually reloads."""
    if st.session_state.get("data_quick_search_prev") != search_text:
        st.session_state["data_quick_search_prev"] = search_text
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )
    date_range_signature = (
        date_preset,
        date_from.isoformat() if date_from else None,
        date_to.isoformat() if date_to else None,
    )
    if st.session_state.get("data_date_range_prev") != date_range_signature:
        st.session_state["data_date_range_prev"] = date_range_signature
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )
    if st.session_state.get("data_extended_prev") != extended:
        st.session_state["data_extended_prev"] = extended
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )


def _fetch_and_overlay(
    conn, date_from, date_to, include_income, pinned_ids, search_text
) -> tuple[pd.DataFrame, PendingEdits]:
    sql, params = _build_data_query(
        date_from or None, date_to or None,
        [],          # picked_cats handled by AgGrid header filter
        "all",       # source handled by AgGrid header filter
        "",          # search handled in pandas below
        None, None,  # amount min/max handled by AgGrid header filter
        include_income,
        pinned_ids=pinned_ids if pinned_ids else None,
    )
    df = pd.read_sql_query(sql, conn, params=params)
    if not df.empty:
        df["buchungsdatum"] = pd.to_datetime(df["buchungsdatum"]).dt.strftime("%Y-%m-%d")
        df["category"] = df["category"].fillna("(unkategorisiert)")
        df["confidence"] = df["confidence"].apply(
            lambda v: f"{float(v):.2f}" if v is not None and v == v else ""
        )
        df["label_source"] = df["label_source"].fillna("")
        df["src"] = df["label_source"].map(
            {"user": "✅ user", "model": "🤖 model"}
        ).fillna("")  # unlabeled rows -> empty Source cell
        # Compact `YYYY-MM-DD HH:MM` for the Modified column. ISO
        # ordering means lexicographic sort in AgGrid matches
        # chronological sort -- no custom comparator needed.
        df["last_modified"] = pd.to_datetime(df["last_modified"]).dt.strftime("%Y-%m-%d %H:%M")
    else:
        df["src"] = ""
        df["label_source"] = ""
    df["_orig_category"] = df["category"]
    df.loc[df["category"] == "(unkategorisiert)", "category"] = ""

    # Free-form wildcard search applied in pandas so the haystack covers
    # every visible column (including the synthetic `src` column).
    if search_text and search_text.strip() and not df.empty:
        df = _apply_search(df, search_text)

    edits = load_pending_edits()
    df = _overlay_pending(df, edits)
    # Pin canonical visible-column order. Hidden / overlay columns
    # tail the visible ones; their position is irrelevant because
    # AgGrid hides them via `configure_column(hide=True)`.
    visible_first = [c for c in _VISIBLE_COL_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in visible_first]
    df = df[visible_first + rest]
    return df, edits


def _apply_search(df: pd.DataFrame, search_text: str) -> pd.DataFrame:
    text_cols = [
        "id", "buchungsdatum", "counterparty", "zahlungspflichtiger",
        "verwendungszweck", "betrag_€", "category", "_orig_category",
        "src", "umsatztyp", "iban", "last_modified",
    ]
    cols_present = [c for c in text_cols if c in df.columns]

    def _safe_str(v) -> str:
        if v is None:
            return ""
        try:
            if isinstance(v, float) and pd.isna(v):
                return ""
        except Exception:
            pass
        return str(v)

    haystack = df[cols_present].apply(
        lambda row: " · ".join(_safe_str(v) for v in row),
        axis=1,
    ).str.lower()
    mask = pd.Series([True] * len(df), index=df.index)
    for term in search_text.strip().lower().split():
        pattern = re.escape(term).replace(r"\*", ".*")
        try:
            term_mask = haystack.str.contains(pattern, regex=True, na=False)
        except re.error:
            term_mask = haystack.str.contains(re.escape(term), regex=True, na=False)
        mask &= term_mask
    return df[mask].reset_index(drop=True)


def _overlay_pending(df: pd.DataFrame, edits: PendingEdits) -> pd.DataFrame:
    """Apply the three pending-edit stashes onto the DataFrame columns the
    AgGrid JS valueGetters / cellStyles read."""
    df["_user_pending"] = False
    df["_stage_cat"] = ""
    df["_stage_conf"] = ""
    df["_stage_stage"] = ""
    df["_promote"] = False

    if df.empty:
        df["_pre_selected"] = False
        return df

    for i in range(len(df)):
        eid = int(df.iloc[i]["id"])
        # Priority: user-typed > auto-label stash. Promote is orthogonal
        # (doesn't change the category cell, only the source tag).
        if edits.has_user_typed(eid):
            val = edits.user_typed[eid]
            df.iat[i, df.columns.get_loc("category")] = val if val is not None else ""
            df.iat[i, df.columns.get_loc("_user_pending")] = True
        elif (item := edits.autolabel_for(eid)) is not None:
            cat_name = item.get("cat", "") or ""
            df.iat[i, df.columns.get_loc("category")] = cat_name
            df.iat[i, df.columns.get_loc("_stage_cat")] = cat_name
            df.iat[i, df.columns.get_loc("_stage_conf")] = item.get("conf", "")
            df.iat[i, df.columns.get_loc("_stage_stage")] = item.get("stage", "")
        if edits.is_promoted(eid):
            df.iat[i, df.columns.get_loc("_promote")] = True

    pre_select_eids: set[int] = set(
        st.session_state.pop("data_aggrid_pre_select_eids", set()) or set()
    )
    df["_pre_selected"] = df["id"].astype(int).isin(pre_select_eids)
    return df


# ---------------------------------------------------------------------------
# Section: AgGrid setup.
# ---------------------------------------------------------------------------


def _build_grid_options(
    df: pd.DataFrame,
    extended: bool,
    all_cat_names: list[str],
    cat_color_map: dict[str, dict[str, str]],
):
    from st_aggrid import GridOptionsBuilder
    from st_aggrid.shared import JsCode

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        sortable=True, filter=True, resizable=True,
        editable=False, floatingFilter=False,
    )
    gb.configure_selection(
        selection_mode="multiple", use_checkbox=True,
        header_checkbox=True, header_checkbox_filtered_only=True,
    )
    gb.configure_grid_options(
        rowHeight=28, headerHeight=34, animateRows=False,
        suppressFieldDotNotation=True, domLayout="normal",
        enableCellTextselection=True, ensureDomOrder=True,
    )

    for hid in ("_orig_category", "category_id", "label_source", "_pre_selected",
                "_user_pending", "_stage_cat", "_stage_conf", "_stage_stage",
                "_promote"):
        if hid in df.columns:
            gb.configure_column(hid, hide=True)

    gb.configure_grid_options(onFirstDataRendered=JsCode("""
        function(params) {
          var nodesToSelect = [];
          params.api.forEachNode(function(node) {
            if (node.data && node.data._pre_selected === true) {
              nodesToSelect.push(node);
            }
          });
          if (nodesToSelect.length) {
            params.api.setNodesSelected({nodes: nodesToSelect, newValue: true});
          }
        }
    """))

    gb.configure_column("id", header_name="ID", width=80, filter="agNumberColumnFilter")
    gb.configure_column("buchungsdatum", header_name="Date", width=110,
                        filter="agDateColumnFilter")
    gb.configure_column("counterparty", header_name="Counterparty", width=200,
                        filter="agTextColumnFilter")
    gb.configure_column("zahlungspflichtiger", header_name="Payer", width=170,
                        filter="agTextColumnFilter")
    gb.configure_column("verwendungszweck", header_name="Verwendungszweck",
                        width=320, filter="agTextColumnFilter")

    amount_cell_style = JsCode("""
        function(p){
          if (p.value == null) return null;
          var v = Number(p.value);
          if (v < 0) return {'color': '#d62728', 'fontWeight': '500'};
          if (v > 0) return {'color': '#157347', 'fontWeight': '500'};
          return null;
        }
    """)
    gb.configure_column(
        "betrag_€", header_name="Amount €", width=110,
        type=["numericColumn"],
        filter="agNumberColumnFilter",
        valueFormatter=JsCode(
            "function(p){"
            "  if (p.value == null) return '';"
            "  return Number(p.value).toLocaleString('de-DE', "
            "    {minimumFractionDigits: 2, maximumFractionDigits: 2});"
            "}"
        ),
        cellStyle=amount_cell_style,
    )
    # Pending-state highlight for Source / Confidence columns (no
    # category-colour overlay there -- those cells aren't categories).
    pending_cell_style = JsCode("""
        function(params) {
          if (!params.data) return null;
          var d = params.data;
          if (d._user_pending === true) return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          if (d._stage_cat && d._stage_cat !== '') return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          if (d._promote === true) return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          return null;
        }
    """)
    _valid_cats_js = _json.dumps([""] + all_cat_names)
    _cat_colors_js = _json.dumps(cat_color_map)
    category_value_parser = JsCode(
        "function(params){"
        f"  var valid = {_valid_cats_js};"
        "  if (valid.indexOf(params.newValue) >= 0) return params.newValue;"
        "  return params.oldValue;"
        "}"
    )
    # Category cell: pending-edit highlight wins; otherwise paint the
    # cell in the user-chosen category colour with auto-picked legible
    # text. Mirrors the Dashboard records table's styling.
    category_cell_style = JsCode(
        "function(params) {"
        "  if (!params.data) return null;"
        "  var d = params.data;"
        "  if (d._user_pending === true) return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};"
        "  if (d._stage_cat && d._stage_cat !== '') return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};"
        "  if (d._promote === true) return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};"
        f"  var colors = {_cat_colors_js};"
        "  var v = params.value;"
        "  if (v && colors[v]) {"
        "    return {'backgroundColor': colors[v].bg, 'color': colors[v].fg};"
        "  }"
        "  return null;"
        "}"
    )
    gb.configure_column(
        "category", header_name="Category", width=170,
        editable=True,
        singleClickEdit=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": [""] + all_cat_names},
        valueParser=category_value_parser,
        cellStyle=category_cell_style,
        filter="agTextColumnFilter",
    )

    src_value_getter = JsCode("""
        function(params) {
          if (!params.data) return '';
          var d = params.data;
          if (d._user_pending === true) {
            var v = d.category == null ? '' : d.category;
            return v === '' ? '📝 → clear' : '📝 → user';
          }
          if (d._stage_cat && d._stage_cat !== '') {
            var cur = d.category == null ? '' : d.category;
            if (cur !== d._stage_cat) {
              return '📝 → user';   // user overrode the prediction
            }
            return '🤖 ' + (d._stage_stage || 'model');
          }
          if (d._promote === true) {
            return '📝 → user';
          }
          return d.src || '';
        }
    """)
    gb.configure_column("src", header_name="Source", width=102,
                        filter="agTextColumnFilter",
                        valueGetter=src_value_getter,
                        cellStyle=pending_cell_style)

    conf_value_getter = JsCode("""
        function(params) {
          if (!params.data) return '';
          var d = params.data;
          if (d._user_pending === true) return '';
          if (d._stage_cat && d._stage_cat !== '') {
            var cur = d.category == null ? '' : d.category;
            if (cur !== d._stage_cat) return '';
            return d._stage_conf || '';
          }
          return d.confidence || '';
        }
    """)
    # Conf cell style: pending-edit highlight wins; otherwise paint
    # the cell by confidence bucket so the user can spot low-quality
    # predictions at a glance. Thresholds match
    # `review_tab._CONF_LOW = 0.40` / `_CONF_MED = 0.70`:
    #   conf < 0.40  -> red    (low; cascade isn't sure)
    #   0.40 <= conf < 0.70 -> yellow ("to confirm" bucket in Review)
    #   conf >= 0.70 -> no bg  (high; auto-coverage)
    conf_cell_style = JsCode("""
        function(params) {
          if (!params.data) return null;
          var d = params.data;
          if (d._user_pending === true) return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          if (d._stage_cat && d._stage_cat !== '') return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          if (d._promote === true) return {'backgroundColor': 'rgba(212, 160, 23, 0.22)'};
          var v = parseFloat(params.value);
          if (isNaN(v)) return null;
          if (v < 0.40) return {'backgroundColor': 'rgba(220, 50, 50, 0.28)'};
          if (v < 0.70) return {'backgroundColor': 'rgba(255, 200, 0, 0.30)'};
          return null;
        }
    """)
    gb.configure_column("confidence", header_name="Conf", width=80,
                        filter="agNumberColumnFilter",
                        valueGetter=conf_value_getter,
                        cellStyle=conf_cell_style)

    # Modified column: latest write timestamp across labels / notes /
    # original ingest (computed in the SELECT). Sortable so the user
    # can pull recent activity to the top with one click. Date filter
    # for "show me everything touched in the last week" workflows.
    gb.configure_column(
        "last_modified", header_name="Last Modified", width=145,
        filter="agDateColumnFilter",
        tooltipField="last_modified",
    )

    gb.configure_column("iban", header_name="IBAN", width=220,
                        filter="agTextColumnFilter")

    ext_columns = ("umsatztyp", "iban_is_foreign",
                   "has_glaeubiger_id", "mandatsreferenz_present")
    if extended:
        gb.configure_column("umsatztyp", header_name="Umsatztyp", width=120)
        gb.configure_column("iban_is_foreign", header_name="Foreign?", width=80)
        gb.configure_column("has_glaeubiger_id", header_name="Gläubiger?", width=90)
        gb.configure_column("mandatsreferenz_present", header_name="Mandat?", width=90)
    else:
        for c in ext_columns:
            if c in df.columns:
                gb.configure_column(c, hide=True)

    # Restore the prior AgGrid state (filter / sort / col widths) IF
    # it was captured under the SAME column set. After adding or
    # removing columns the saved `columnState` would either hide the
    # new column or list ghost columns -- safest is to drop the state
    # and let AgGrid re-derive defaults from the current columnDefs.
    _saved_grid_state = st.session_state.get(_GRID_STATE_KEY)
    if isinstance(_saved_grid_state, dict) and _saved_grid_state:
        col_state = _saved_grid_state.get("columnState") or []
        saved_col_ids = {
            item.get("colId") for item in col_state
            if isinstance(item, dict) and "colId" in item
        }
        if saved_col_ids and not set(df.columns).issubset(saved_col_ids):
            # New columns added since the saved state was captured.
            # Drop the state + bump the AgGrid widget key so it
            # re-mounts cleanly with the new column set.
            _saved_grid_state = None
            st.session_state.pop(_GRID_STATE_KEY, None)
            st.session_state["data_aggrid_seed"] = (
                st.session_state.get("data_aggrid_seed", 0) + 1
            )
    if isinstance(_saved_grid_state, dict) and _saved_grid_state:
        _init_state = {k: v for k, v in _saved_grid_state.items()
                       if k != "rowSelection"}
        if _init_state:
            gb.configure_grid_options(initialState=_init_state)

    aggrid_key = "data_aggrid_v" + str(st.session_state.get("data_aggrid_seed", 0))
    return gb.build(), aggrid_key


# ---------------------------------------------------------------------------
# Section: action bar + grid + click handlers.
# ---------------------------------------------------------------------------


def _sync_extended_from(src_key: str) -> None:
    """Mirror a toggle's value into the master state (data_extended)."""
    st.session_state["data_extended"] = bool(st.session_state.get(src_key, False))


def _render_action_buttons(prefix: str, counts: dict):
    """Render the 6-button action bar (no caption).
    Returns (save, revert, autolabel, promote, see_details) booleans."""
    n_pend = int(counts.get("n_pending", 0))
    n_sel = int(counts.get("n_selected", 0))
    can_prom = bool(counts.get("can_promote", False))
    can_insp = bool(counts.get("can_inspect", False))

    cols = st.columns([1.6, 1.6, 1.5, 1.9, 1.4, 1.4, 0.6])
    save_c = cols[0].button(
        f"💾 Save Changes ({n_pend})" if n_pend else "💾 Save Changes",
        type="tertiary", disabled=n_pend == 0,
        key=f"data_save_{prefix}_btn",
        help="Commit highlighted rows.",
    )
    revert_c = cols[1].button(
        f"↩ Revert Changes ({n_pend})" if n_pend else "↩ Revert Changes",
        type="tertiary", disabled=n_pend == 0,
        key=f"data_revert_{prefix}_btn",
        help="Discard every highlighted pending change without saving.",
    )
    auto_c = cols[2].button(
        f"🤖 Auto Label ({n_sel})" if n_sel else "🤖 Auto Label",
        type="tertiary", disabled=n_sel == 0,
        key=f"data_autolabel_{prefix}_btn",
        help=(
            "Run the cascade on selected rows. Predictions appear "
            "highlighted; user-labeled rows are skipped."
        ),
    )
    promote_c = cols[3].button(
        f"⬆️ Promote to User Label ({n_sel})" if n_sel else "⬆️ Promote to User Label",
        type="tertiary", disabled=not can_prom,
        key=f"data_promote_{prefix}_btn",
        help=(
            "Re-save selected rows as `source='user'`. Disabled if any "
            "selected row is uncategorized."
        ),
    )
    see_c = cols[4].button(
        "👁 See Details",
        type="tertiary", disabled=not can_insp,
        key=f"data_see_details_{prefix}_btn",
        help="Full record popup. Active only when exactly one row is selected.",
    )
    cols[5].toggle(
        "Extended Columns",
        value=bool(st.session_state.get("data_extended", False)),
        key=f"data_extended_{prefix}_toggle",
        on_change=_sync_extended_from,
        args=(f"data_extended_{prefix}_toggle",),
        help="Reveal Umsatztyp / Foreign? / Gläubiger? / Mandat? columns. "
             "(IBAN is always shown.)",
    )
    return save_c, revert_c, auto_c, promote_c, see_c


def _render_caption(counts: dict) -> None:
    n_pend = int(counts.get("n_pending", 0))
    n_sel = int(counts.get("n_selected", 0))
    n_row = int(counts.get("n_rows", 0))
    total = float(counts.get("total_amount", 0.0))
    sel_total = float(counts.get("selected_amount", 0.0))
    # When the user has a selection, surface its subtotal too -- saves
    # the user from grabbing a calculator. Total reflects the SQL-level
    # filter view (date / search / pinned); it does NOT honour AgGrid
    # per-column header filters (those run client-side; the grid state
    # AgGrid posts back doesn't cleanly expose post-filter visibility
    # without extra JS plumbing).
    sel_str = (
        f"  ·  selected total {de_eur(sel_total)}"
        if n_sel > 0 else ""
    )
    st.markdown(
        f"<div style='text-align:center; font-size:0.78rem; "
        f"opacity:0.75; margin: 0.25rem 0 0.6rem 0;'>"
        f"{n_pend} unsaved changes  ·  {n_sel} of {n_row} selected  ·  "
        f"total {de_eur(total)}{sel_str}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _counts_from_grid_state(
    key: str,
    cat_id_by_name: dict[str, int],
    df: pd.DataFrame | None = None,
) -> dict:
    """Read AgGrid's just-written grid_state out of session_state and
    derive the counts used by the top action bar -- before the grid
    itself renders this turn.

    On the first render after a tab switch or page load the grid_state
    isn't in session_state yet (AgGrid posts back after it mounts), so
    `_top_counts` would otherwise read all zeros. Pass the Python-side
    `df` as a fallback so the top bar at least shows total row count +
    total amount immediately.
    """
    out = {"n_pending": 0, "n_selected": 0, "n_rows": 0,
           "total_amount": 0.0, "selected_amount": 0.0,
           "can_promote": False, "can_inspect": False, "sel_single_eid": None}

    def _seed_from_df() -> None:
        # Fallback when grid_state hasn't arrived: take the row count
        # and signed total from the SQL-filter view. Pending and
        # selected stay zero because both require user interaction
        # with AgGrid.
        if df is None or df.empty:
            return
        out["n_rows"] = len(df)
        try:
            out["total_amount"] = float(df["betrag_€"].sum())
        except (KeyError, ValueError):
            pass

    raw = st.session_state.get(key)
    if not isinstance(raw, dict):
        _seed_from_df()
        return out
    nodes = raw.get("nodes") or []
    if not isinstance(nodes, list) or not nodes:
        _seed_from_df()
        return out
    out["n_rows"] = len(nodes)
    sel_eids: list[int] = []
    n_pending = 0
    total_amount = 0.0
    selected_amount = 0.0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        d = node.get("data") or {}
        try:
            eid = int(d.get("id"))
        except (TypeError, ValueError):
            continue
        # Sum the signed amount across every visible row, and again
        # restricted to selected rows so the caption can show a
        # subtotal when the user has a selection. AgGrid column-
        # filters aren't reflected here -- this is the SQL-filter
        # view, same scope as `n_rows`.
        try:
            amt = float(d.get("betrag_€") or 0)
        except (TypeError, ValueError):
            amt = 0.0
        total_amount += amt
        if node.get("isSelected") is True:
            sel_eids.append(eid)
            selected_amount += amt
        cat = str(d.get("category") or "").strip()
        orig = str(d.get("_orig_category") or "").strip()
        orig_norm = "" if orig == "(unkategorisiert)" else orig
        if cat != orig_norm:
            if (cat == "" and orig_norm != "") or cat in cat_id_by_name:
                n_pending += 1
    out["n_pending"] = n_pending
    out["n_selected"] = len(sel_eids)
    out["total_amount"] = total_amount
    out["selected_amount"] = selected_amount
    if sel_eids:
        sel_set = set(sel_eids)
        ok = True
        for node in nodes:
            if not isinstance(node, dict):
                continue
            d = node.get("data") or {}
            try:
                eid_n = int(d.get("id"))
            except (TypeError, ValueError):
                continue
            if eid_n not in sel_set:
                continue
            orig = str(d.get("_orig_category") or "").strip()
            orig_norm = "" if orig == "(unkategorisiert)" else orig
            cat = str(d.get("category") or "").strip()
            if not (cat or orig_norm):
                ok = False
                break
        out["can_promote"] = ok
    out["can_inspect"] = out["n_selected"] == 1
    if out["can_inspect"]:
        out["sel_single_eid"] = sel_eids[0]
    return out


def _render_top_action_bar_and_grid_and_actions(
    conn, cfg, df, edits, grid_options, aggrid_key,
    cat_id_by_name, cat_name_by_id,
) -> None:
    from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode

    # Force both Extended toggles to reflect the master value BEFORE either
    # widget is instantiated this run.
    _master_ext = bool(st.session_state.get("data_extended", False))
    st.session_state["data_extended_top_toggle"] = _master_ext
    st.session_state["data_extended_bot_toggle"] = _master_ext

    _top_counts = _counts_from_grid_state(aggrid_key, cat_id_by_name, df=df)
    save_top_clicked, revert_top_clicked, auto_label_top_clicked, \
        promote_top_clicked, see_details_top_clicked = \
        _render_action_buttons("top", _top_counts)
    _render_caption(_top_counts)

    response = AgGrid(
        df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=(
            GridUpdateMode.VALUE_CHANGED
            | GridUpdateMode.SELECTION_CHANGED
            | GridUpdateMode.FILTERING_CHANGED
            | GridUpdateMode.SORTING_CHANGED
        ),
        allow_unsafe_jscode=True,
        height=680,
        theme="streamlit",
        reload_data=False,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        key=aggrid_key,
    )
    _gs = response.grid_state
    if _gs:
        st.session_state[_GRID_STATE_KEY] = _gs

    edited_df = response.get("data")
    if not isinstance(edited_df, pd.DataFrame):
        edited_df = df
    selected_rows = response.get("selected_rows")
    if isinstance(selected_rows, pd.DataFrame):
        sel_ids = [int(x) for x in selected_rows["id"].tolist()] \
            if not selected_rows.empty else []
    elif isinstance(selected_rows, list):
        sel_ids = [int(r["id"]) for r in selected_rows]
    else:
        sel_ids = []

    pending_updates, pending_user_typed_now = _classify_edits(
        df, edited_df, edits, cat_id_by_name
    )

    # Bulk-edit propagation. Must run BEFORE the bottom action bar so a
    # bulk-edit triggers a rerun without showing a stale state.
    if _maybe_apply_bulk_edit(df, edited_df, sel_ids, cat_id_by_name):
        return

    n_pending = len(pending_updates)
    n_selected = len(sel_ids)
    n_rows = len(edited_df) if isinstance(edited_df, pd.DataFrame) else 0

    # Signed totals over the SQL-filter scope (= every row AgGrid
    # received). Column-filtering at the grid level is invisible from
    # here; see comment in `_render_caption`.
    if isinstance(edited_df, pd.DataFrame) and not edited_df.empty:
        total_amount = float(edited_df["betrag_€"].sum())
        sel_set_for_sum = {int(x) for x in sel_ids}
        if sel_set_for_sum:
            selected_amount = float(
                edited_df.loc[
                    edited_df["id"].astype(int).isin(sel_set_for_sum),
                    "betrag_€",
                ].sum()
            )
        else:
            selected_amount = 0.0
    else:
        total_amount = 0.0
        selected_amount = 0.0

    can_promote = _compute_can_promote(edited_df, sel_ids)
    can_inspect = n_selected == 1
    sel_single_eid = int(sel_ids[0]) if can_inspect else None

    _current_counts = {
        "n_pending": n_pending,
        "n_selected": n_selected,
        "n_rows": n_rows,
        "total_amount": total_amount,
        "selected_amount": selected_amount,
        "can_promote": can_promote,
        "can_inspect": can_inspect,
        "sel_single_eid": sel_single_eid,
    }

    _render_caption(_current_counts)
    save_bot_clicked, revert_bot_clicked, auto_label_bot_clicked, \
        promote_bot_clicked, see_details_bot_clicked = \
        _render_action_buttons("bot", _current_counts)

    auto_label_clicked = auto_label_top_clicked or auto_label_bot_clicked
    promote_clicked = promote_top_clicked or promote_bot_clicked
    see_details_clicked = see_details_top_clicked or see_details_bot_clicked
    revert_clicked = revert_top_clicked or revert_bot_clicked
    save_clicked = save_top_clicked or save_bot_clicked

    if auto_label_clicked and sel_ids:
        _handle_autolabel(conn, cfg, edited_df, sel_ids, pending_user_typed_now,
                          cat_name_by_id)
        return

    if promote_clicked and sel_ids:
        _handle_promote(edited_df, sel_ids, pending_user_typed_now)
        return

    if see_details_clicked and sel_single_eid is not None:
        st.session_state["data_inspect_open_eid"] = int(sel_single_eid)
        st.rerun()

    if revert_clicked:
        clear_pending_edits()
        st.session_state["data_aggrid_seed"] = (
            st.session_state.get("data_aggrid_seed", 0) + 1
        )
        st.toast(f"reverted {n_pending} pending change(s)")
        st.rerun()

    if save_clicked:
        _handle_save(conn, pending_updates)


# ---- Edit-classification helpers ------------------------------------------


def _classify_edits(df, edited_df, edits, cat_id_by_name):
    """Walk the grid's returned rows and classify each diff vs the
    SQL-original DataFrame as one of: set_user / set_model / clear /
    no-op. Also collects user-typed changes that should be merged back
    into the persistent stash on the next action (so they survive a
    grid-key bump)."""
    pending_updates: list[tuple[int, dict]] = []
    pending_user_typed_now: dict[int, str | None] = {}
    if edited_df is None or edited_df.empty:
        return pending_updates, pending_user_typed_now

    for i in range(len(edited_df)):
        row = edited_df.iloc[i]
        eid = int(row["id"])
        new_cat = str(row.get("category", "") or "").strip()
        orig_cat = str(row.get("_orig_category", "") or "").strip()
        orig_norm = "" if orig_cat == "(unkategorisiert)" else orig_cat

        input_cat = ""
        if i < len(df):
            raw = df.iloc[i].get("category", "")
            input_cat = str(raw or "").strip()
        user_actually_edited = (input_cat != new_cat)

        # Promote stash: always wants a user label of the row's current
        # (model) category. If the user ALSO edited the cell, that wins.
        if edits.is_promoted(eid):
            cat_to_save = new_cat or orig_norm
            cid = cat_id_by_name.get(cat_to_save)
            if cid is not None:
                pending_updates.append((eid, {"action": "set_user", "cat_id": cid}))
            continue

        if new_cat == orig_norm and not user_actually_edited:
            continue  # no change

        if new_cat == "":
            if orig_norm:
                pending_updates.append((eid, {"action": "clear"}))
                pending_user_typed_now[eid] = None
            continue

        if new_cat not in cat_id_by_name:
            continue  # invalid value (shouldn't happen with valueParser)

        cid = cat_id_by_name[new_cat]

        if user_actually_edited:
            pending_updates.append((eid, {"action": "set_user", "cat_id": cid}))
            pending_user_typed_now[eid] = new_cat
            continue

        # Untouched cell. If matches staged auto-label prediction, save
        # as model with the recorded confidence.
        stage_item = edits.autolabel_for(eid)
        if stage_item and stage_item.get("cat") == new_cat:
            conf_str = stage_item.get("conf", "")
            try:
                conf = float(conf_str)
            except (TypeError, ValueError):
                conf = None
            pending_updates.append((eid, {
                "action": "set_model",
                "cat_id": cid,
                "confidence": conf,
            }))
        else:
            # Fallback: treat as user just to be safe.
            pending_updates.append((eid, {"action": "set_user", "cat_id": cid}))
            pending_user_typed_now[eid] = new_cat

    return pending_updates, pending_user_typed_now


def _maybe_apply_bulk_edit(df, edited_df, sel_ids, cat_id_by_name) -> bool:
    """If 2+ rows are selected AND the user edited the Category of one,
    propagate the new value to every selected row. Returns True iff the
    edit was applied and the caller should stop (we trigger a rerun).
    """
    sel_set_now: set[int] = {int(x) for x in sel_ids}
    if len(sel_set_now) < 2 or edited_df is None or edited_df.empty:
        return False
    bulk_cat: str | None = None
    for i in range(len(edited_df)):
        eid = int(edited_df.iloc[i]["id"])
        if eid not in sel_set_now:
            continue
        new_cat = str(edited_df.iloc[i].get("category", "") or "").strip()
        input_cat = ""
        if i < len(df):
            raw = df.iloc[i].get("category", "")
            input_cat = str(raw or "").strip()
        if new_cat == input_cat:
            continue  # not edited this render
        if new_cat == "" or new_cat in cat_id_by_name:
            bulk_cat = new_cat
            break
    if bulk_cat is None:
        return False

    merge_user_typed({
        int(eid_n): bulk_cat if bulk_cat != "" else None
        for eid_n in sel_set_now
    })
    st.session_state["data_aggrid_pre_select_eids"] = sel_set_now
    st.session_state["data_aggrid_seed"] = (
        st.session_state.get("data_aggrid_seed", 0) + 1
    )
    st.toast(f"applied {bulk_cat or '(clear)'} to {len(sel_set_now)} selected rows")
    st.rerun()
    return True  # not actually reached


def _compute_can_promote(edited_df, sel_ids) -> bool:
    """Every selected row must already have a category (DB original OR
    a staged value). If even one is uncategorized, disable Promote."""
    sel_set = {int(x) for x in sel_ids}
    if not sel_set or edited_df is None or edited_df.empty:
        return False
    n_promotable = 0
    for i in range(len(edited_df)):
        eid = int(edited_df.iloc[i]["id"])
        if eid not in sel_set:
            continue
        orig_cat = str(edited_df.iloc[i].get("_orig_category", "") or "").strip()
        orig_norm = "" if orig_cat == "(unkategorisiert)" else orig_cat
        new_cat = str(edited_df.iloc[i].get("category", "") or "").strip()
        if new_cat or orig_norm:
            n_promotable += 1
    return n_promotable == len(sel_set)


# ---- Action handlers -----------------------------------------------------


def _autolabel_predictions(conn, cfg, target_ids: list[int], label_text: str):
    """Run cascade on `target_ids`, return predictions. No DB writes."""
    if not target_ids:
        return []
    with st.status(label_text, expanded=True) as status:
        status.write(f"loading embedding model `{cfg.embedding_model}`…")
        emb = get_embedder()
        cascade = CategorizationCascade(conn, cfg, emb)
        status.write("fitting cascade on the latest user labels…")
        try:
            cascade.fit()
        except Exception as e:
            status.write(f"  fit skipped: {e}")
        status.write(f"predicting {len(target_ids)} record(s)…")
        progress = st.progress(0, text=f"0 / {len(target_ids)}")

        def _cb(done: int, total: int) -> None:
            progress.progress(done / total, text=f"{done} / {total}")

        preds = cascade.predict_batch(target_ids, progress_callback=_cb)
        progress.empty()
        stages = Counter(p.stage for p in preds)
        n_with_cat = sum(1 for p in preds if p.category_id is not None)
        status.update(
            label=(
                f"staged {n_with_cat}/{len(preds)} prediction(s) · "
                + ", ".join(f"{k}={v}" for k, v in stages.items())
                + " — review highlighted cells and click Save changes to commit"
            ),
            state="complete",
        )
    return preds


def _handle_autolabel(
    conn, cfg, edited_df, sel_ids: list[int],
    pending_user_typed_now: dict[int, str | None],
    cat_name_by_id: dict[int, str],
) -> None:
    # Preserve any user-typed cell edits across the upcoming key bump.
    merge_user_typed(pending_user_typed_now)

    # Skip rows whose latest label is already 'user' — protect prior
    # confirmations. Show a toast with the skipped count.
    label_source_by_id = dict(zip(
        edited_df["id"].astype(int).tolist(),
        edited_df.get("label_source", pd.Series([""] * len(edited_df))).fillna("").tolist(),
        strict=True,
    ))
    eligible = [int(i) for i in sel_ids
                if label_source_by_id.get(int(i), "") != "user"]
    skipped = len(sel_ids) - len(eligible)
    if not eligible:
        st.warning(
            f"all {skipped} selected row(s) already have a user label — "
            "nothing to auto-label. Use ⬆️ Promote to user if you want "
            "to re-stamp them, or clear them from Settings."
        )
        return

    preds = _autolabel_predictions(
        conn, cfg, eligible,
        f"auto-labeling {len(eligible)} record(s)"
        + (f" · skipping {skipped} user-labeled" if skipped else "")
        + "…",
    )
    stage = dict(st.session_state.get("data_autolabel_stage", {}))
    # User-typed edits take priority -- don't override them with a
    # prediction (the user-typed stash already wins on render).
    current_user_typed = set(
        st.session_state.get("data_user_typed_edits", {}).keys()
    )
    for p in preds:
        if p.category_id is None:
            continue
        eid = int(p.expense_id)
        if eid in current_user_typed:
            continue
        name = cat_name_by_id.get(p.category_id)
        if name:
            stage[eid] = {
                "cat": name,
                "conf": f"{float(p.confidence):.2f}",
                "stage": p.stage,
            }
    st.session_state["data_autolabel_stage"] = stage
    st.session_state["data_aggrid_pre_select_eids"] = set(int(x) for x in sel_ids)
    st.session_state["data_aggrid_seed"] = (
        st.session_state.get("data_aggrid_seed", 0) + 1
    )
    st.rerun()


def _handle_promote(
    edited_df, sel_ids: list[int],
    pending_user_typed_now: dict[int, str | None],
) -> None:
    merge_user_typed(pending_user_typed_now)
    existing = set(st.session_state.get("data_promote_stage", set()) or set())
    elig_count = 0
    for i in range(len(edited_df)):
        eid = int(edited_df.iloc[i]["id"])
        if eid not in sel_ids:
            continue
        orig_cat = str(edited_df.iloc[i].get("_orig_category", "") or "").strip()
        orig_norm = "" if orig_cat == "(unkategorisiert)" else orig_cat
        new_cat = str(edited_df.iloc[i].get("category", "") or "").strip()
        visible_cat = new_cat or orig_norm
        if not visible_cat:
            continue
        existing.add(eid)
        elig_count += 1
    if elig_count == 0:
        st.warning(
            "selected rows have no category to promote -- pick a "
            "category first (or Auto-Label) then Promote to user."
        )
        return
    st.session_state["data_promote_stage"] = existing
    st.session_state["data_aggrid_pre_select_eids"] = set(int(x) for x in sel_ids)
    st.session_state["data_aggrid_seed"] = (
        st.session_state.get("data_aggrid_seed", 0) + 1
    )
    st.rerun()


def _handle_save(conn, pending_updates: list[tuple[int, dict]]) -> None:
    n_user = 0
    n_model = 0
    n_cleared = 0
    for eid, act in pending_updates:
        a = act["action"]
        if a == "clear":
            clear_labels_for_expense(conn, int(eid))
            n_cleared += 1
        elif a == "set_user":
            add_label(conn, int(eid), int(act["cat_id"]), "user")
            n_user += 1
        elif a == "set_model":
            add_label(conn, int(eid), int(act["cat_id"]), "model",
                      confidence=act.get("confidence"))
            n_model += 1
    clear_pending_edits()
    st.session_state["data_aggrid_seed"] = (
        st.session_state.get("data_aggrid_seed", 0) + 1
    )
    parts = []
    if n_user:
        parts.append(f"{n_user} user")
    if n_model:
        parts.append(f"{n_model} model")
    if n_cleared:
        parts.append(f"{n_cleared} cleared")
    st.toast("saved: " + ", ".join(parts) if parts else "nothing to save")
    st.rerun()
