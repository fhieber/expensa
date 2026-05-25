"""Settings tab: compute device, models, privacy, own IBANs, DB admin.

Sections (top to bottom):
  - Compute Device
  - Models (Embeddings + Zero-Shot, collapsible)
  - Privacy (vendor lookup status)
  - My Accounts (own IBANs CRUD)
  - Database (backup / restore / Danger Zone resets)
"""

from __future__ import annotations

import datetime as _dt
import tempfile as _tempfile
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from expense_analyzer.accounts import slugify
from expense_analyzer.config import (
    ActiveLearningConfig,
    CategorySimilarityConfig,
    ClassifierConfig,
    KnnConfig,
    VendorExactMatchConfig,
    VendorLookupConfig,
    ZeroshotConfig,
    save_user_config,
)
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
from expense_analyzer.storage.crypto import (
    EncryptionError,
    change_password,
    cipher_version,
    decrypt_file,
    encrypt_file,
    encryption_available,
    export_encrypted_copy,
    looks_encrypted,
)
from expense_analyzer.storage.own_ibans import (
    add_own_iban,
    list_own_ibans,
    remove_own_iban,
    update_label,
)
from expense_analyzer.storage.stats import database_overview
from expense_analyzer.ui._shared import (
    account_is_encrypted,
    clear_password_for,
    get_active_account,
    get_config,
    get_conn,
    get_global_home,
    get_password_for,
    invalidate_connection,
    invalidate_global_config,
    set_password_for,
)


def render() -> None:
    """Settings page.

    Top-level layout (all collapsed by default so the page is quick
    to scan and the user only opens what they need to change):

      1. My Accounts          -- own-IBAN registry
      2. Active Learning      -- review batch sizing / strategy
      3. Privacy              -- vendor web lookup
      4. Classification       -- compute device + ML models
            ├── Compute Device
            └── Models
                  ├── Embeddings (+ batch size)
                  └── Zero-Shot (+ prompting)
      5. Categorization Cascade -- per-stage tuning, ordered as the
            cascade actually runs (1. vendor_exact_match → 5. zeroshot)

      Database admin lives at the very bottom, outside the numbered
      sections, since it's a destructive surface that should stay
      visible.
    """
    cfg = get_config()
    conn = get_conn()
    st.title("Settings")

    with st.expander("My Accounts", expanded=False):
        _render_own_ibans_body(conn)

    with st.expander("Active Learning", expanded=False):
        _render_active_learning_body(cfg)

    with st.expander("Privacy", expanded=False):
        _render_privacy_body(cfg)

    with st.expander("Classification", expanded=False):
        st.caption("Global setting — applies to all accounts.")
        with st.expander("Compute Device", expanded=False):
            _render_device_body(cfg)
        with st.expander("Models", expanded=False):
            _render_models_body(cfg)

    with st.expander("Categorization Cascade", expanded=False):
        _render_cascade_body(cfg)

    with st.expander("Database", expanded=False):
        _render_database_body(cfg, conn)


def _persist(key: str, values: dict, model_cls) -> None:
    """Validate ``values`` against ``model_cls``, then merge ``{key: values}``
    into the global ``config.yaml`` and drop the cached GlobalConfig so the
    next render reads the new value. Shows an error and skips the write if
    validation fails."""
    try:
        model_cls(**values)
    except ValidationError as e:
        st.error(f"invalid settings — not saved: {e}")
        return
    save_user_config({key: values}, data_dir=get_global_home())
    invalidate_global_config()
    st.success("Saved.")


def _persist_scalar(updates: dict) -> None:
    save_user_config(updates, data_dir=get_global_home())
    invalidate_global_config()
    st.success("Saved.")


_DEVICE_CHOICES = ["auto", "cpu", "cuda", "mps"]


def _render_device_body(cfg) -> None:
    """Compute Device body. The outer expander supplies title + framing."""
    with st.form("settings_device"):
        picked = st.selectbox(
            "Device",
            _DEVICE_CHOICES,
            index=_DEVICE_CHOICES.index(cfg.device)
            if cfg.device in _DEVICE_CHOICES else 0,
            help=(
                "`auto` picks the best available: CUDA on NVIDIA, MPS on "
                "Apple Silicon, otherwise CPU. Override per host if needed."
            ),
        )
        if st.form_submit_button("Save", type="primary"):
            _persist_scalar({"device": picked})
    st.caption(f"HF cache: `{hf_cache_dir()}`")


def _model_table_and_picker(
    models, current_id: str, cfg_key: str, explanation: str, cfg
) -> None:
    """Compact model picker: popover explanation + radio list + action buttons."""
    with st.popover("ℹ️ How this works"):
        st.caption(explanation)

    def _label(m) -> str:
        present, size_gb = is_downloaded(m.model_id)
        cached = "✅" if present else "—"
        size = size_gb if present else m.approx_size_mb / 1024
        short_id = m.model_id.split("/")[-1]
        dim_tag = f"  dim={m.dim}" if m.dim else ""
        return f"{cached}  {short_id}  |  {m.languages}{dim_tag}  |  {size:.1f} GB  —  {m.notes}"

    options = [m.model_id for m in models]
    labels = [_label(m) for m in models]
    current_idx = next((i for i, m in enumerate(models) if m.model_id == current_id), 0)

    picked_idx = st.radio(
        "model picker",
        range(len(options)),
        index=current_idx,
        format_func=lambda i: labels[i],
        key=f"model_pick_{cfg_key}",
        label_visibility="collapsed",
    )
    picked = options[picked_idx]

    act_cols = st.columns([1, 1, 4])
    with act_cols[0]:
        present, _ = is_downloaded(picked)
        dl_label = "Download" if not present else "Re-download"
        if st.button(dl_label, key=f"model_dl_{cfg_key}", use_container_width=True):
            with st.status(f"Downloading {picked}...", expanded=True) as status:
                role = "embedding" if cfg_key == "embedding_model" else "zeroshot"
                try:
                    trigger_download(picked, role=role)
                    status.update(label=f"Downloaded {picked}", state="complete")
                except Exception as e:
                    status.update(label=f"Download failed: {e}", state="error")
            st.rerun()
    with act_cols[1]:
        if st.button(
            "Use this", key=f"model_use_{cfg_key}", type="primary",
            disabled=picked == current_id, use_container_width=True,
        ):
            save_user_config({cfg_key: picked}, data_dir=get_global_home())
            invalidate_global_config()
            st.success(
                f"`{cfg_key}` set to `{picked}`. Restart the UI for it to take effect: "
                "`expense ui-restart`."
            )


def _render_models_body(cfg) -> None:
    """Models sub-section body: Embeddings (+ batch size) and Zero-Shot
    (+ prompting). The outer expander supplies the section title."""
    # Embeddings + batch size live together: the batch size controls
    # the embedder's forward-pass throughput, so it's a sibling knob.
    with st.expander("Embeddings", expanded=False):
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
        st.markdown("---")
        with st.form("settings_embedding_batch"):
            batch = st.number_input(
                "Embedding batch size",
                min_value=1, max_value=1024, step=1,
                value=int(cfg.embedding_batch_size),
                help=(
                    "How many expense texts are embedded per forward pass. "
                    "Higher uses more memory but is faster on a GPU."
                ),
            )
            if st.form_submit_button("Save", type="primary"):
                _persist_scalar({"embedding_batch_size": int(batch)})
    with st.expander("Zero-Shot", expanded=False):
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
        # Per-call prompting knobs live with the model picker rather than
        # under cascade tuning -- users come here to *configure* zero-shot.
        _render_zeroshot_prompting_form(cfg)


def _render_privacy_body(cfg) -> None:
    """Privacy / vendor-lookup body."""
    vl = cfg.vendor_lookup
    st.caption("Global setting — applies to all accounts.")
    st.warning(
        "Only `counterparty_normalized` is ever sent to the search backend "
        "— never amount, IBAN or Verwendungszweck. This whitelist is "
        "enforced in code and cannot be widened here."
    )
    backends = ["duckduckgo", "searxng"]
    with st.form("settings_vendor_lookup"):
        enabled = st.toggle("Enable vendor web lookup", value=vl.enabled)
        backend = st.selectbox(
            "Backend", backends,
            index=backends.index(vl.backend) if vl.backend in backends else 0,
        )
        searxng_url = st.text_input(
            "SearXNG URL (only used when backend = searxng)",
            value=vl.searxng_url,
            placeholder="https://searxng.example.org",
        )
        cache_ttl_days = st.number_input(
            "Cache TTL (days)", min_value=0, max_value=3650, step=1,
            value=int(vl.cache_ttl_days),
            help="How long a fetched vendor summary stays cached before refetch.",
        )
        if st.form_submit_button("Save", type="primary"):
            _persist(
                "vendor_lookup",
                {
                    "enabled": bool(enabled),
                    "backend": backend,
                    "searxng_url": searxng_url.strip(),
                    "cache_ttl_days": int(cache_ttl_days),
                },
                VendorLookupConfig,
            )


def _render_active_learning_body(cfg) -> None:
    """Active-learning body: defaults for the Review tab batch picker."""
    al = cfg.active_learning
    strategies = ["uncertainty", "low-confidence-first", "diverse", "mixed"]
    st.caption("Global setting — applies to all accounts. Defaults for the Review tab.")
    with st.form("settings_active_learning"):
        batch = st.number_input(
            "Default review batch size", min_value=1, max_value=500, step=1,
            value=int(al.default_batch_size),
        )
        strat = st.selectbox(
            "Default sampling strategy", strategies,
            index=strategies.index(al.default_strategy)
            if al.default_strategy in strategies else 0,
        )
        if st.form_submit_button("Save", type="primary"):
            _persist(
                "active_learning",
                {"default_batch_size": int(batch), "default_strategy": strat},
                ActiveLearningConfig,
            )


def _render_cascade_body(cfg) -> None:
    """Per-stage cascade tuning. Stages are listed in pipeline-execution
    order so the numbering matches the order rows flow through the
    cascade -- read top to bottom to follow what happens to one row.

    The defaults are well-tuned for German bank text; change a stage
    only if the Quality tab's leave-one-out ablation tells you the
    stage is net-negative on your data.
    """
    st.caption("Global setting — applies to all accounts.")
    st.caption(
        "Pipeline order: each stage gets a chance to predict; the first "
        "one that meets its confidence threshold wins. Re-run the "
        "**Quality** tab after any change to measure the impact."
    )
    _render_vendor_exact_form(cfg)
    _render_knn_form(cfg)
    _render_classifier_form(cfg)
    _render_category_similarity_form(cfg)
    _render_zeroshot_form(cfg)


def _render_classifier_form(cfg) -> None:
    c = cfg.classifier
    types = ["logistic_regression", "random_forest"]
    st.markdown("### 3. Classifier")
    with st.form("settings_classifier"):
        ctype = st.selectbox(
            "Type", types,
            index=types.index(c.type) if c.type in types else 0,
        )
        conf = st.number_input(
            "Confidence threshold (below → flagged for review)",
            min_value=0.0, max_value=1.0, step=0.01, value=float(c.confidence_threshold),
        )
        rf_switch = st.number_input(
            "Switch to random forest after N labels",
            min_value=1, step=1, value=int(c.rf_switch_threshold),
        )
        retrain = st.number_input(
            "Retrain after N new labels",
            min_value=1, step=1, value=int(c.retrain_after_n_new_labels),
        )
        if st.form_submit_button("Save classifier", type="primary"):
            _persist(
                "classifier",
                {
                    "type": ctype,
                    "confidence_threshold": float(conf),
                    "rf_switch_threshold": int(rf_switch),
                    "retrain_after_n_new_labels": int(retrain),
                },
                ClassifierConfig,
            )


def _render_vendor_exact_form(cfg) -> None:
    v = cfg.vendor_exact_match
    st.markdown("### 1. Vendor exact match")
    with st.form("settings_vendor_exact"):
        enabled = st.toggle("Enabled", value=v.enabled, key="vem_enabled")
        agree = st.number_input(
            "Agreement min (fraction of past labels for the vendor that must agree)",
            min_value=0.0, max_value=1.0, step=0.01, value=float(v.agreement_min),
        )
        if st.form_submit_button("Save vendor exact match", type="primary"):
            _persist(
                "vendor_exact_match",
                {"enabled": bool(enabled), "agreement_min": float(agree)},
                VendorExactMatchConfig,
            )


def _render_knn_form(cfg) -> None:
    k = cfg.knn
    st.markdown("### 2. k-NN (embedding neighbours)")
    with st.form("settings_knn"):
        enabled = st.toggle("Enabled", value=k.enabled, key="knn_enabled")
        kk = st.number_input(
            "k (neighbours)", min_value=1, max_value=100, step=1, value=int(k.k),
        )
        agree = st.number_input(
            "Agreement min (neighbours that must share the winning category)",
            min_value=1, max_value=100, step=1, value=int(k.agreement_min),
        )
        if st.form_submit_button("Save k-NN", type="primary"):
            _persist(
                "knn",
                {"enabled": bool(enabled), "k": int(kk), "agreement_min": int(agree)},
                KnnConfig,
            )


def _render_category_similarity_form(cfg) -> None:
    cs = cfg.category_similarity
    st.markdown("### 4. Category similarity")
    with st.form("settings_category_similarity"):
        enabled = st.toggle("Enabled", value=cs.enabled, key="catsim_enabled")
        min_top1 = st.number_input(
            "Min top-1 cosine", min_value=0.0, max_value=1.0, step=0.01,
            value=float(cs.min_top1),
        )
        min_margin = st.number_input(
            "Min margin (top1 − top2)", min_value=0.0, max_value=1.0, step=0.01,
            value=float(cs.min_margin),
        )
        use_industry = st.toggle(
            "Use vendor industry tag", value=cs.use_vendor_industry,
            key="catsim_industry",
        )
        if st.form_submit_button("Save category similarity", type="primary"):
            _persist(
                "category_similarity",
                {
                    "enabled": bool(enabled),
                    "min_top1": float(min_top1),
                    "min_margin": float(min_margin),
                    "use_vendor_industry": bool(use_industry),
                },
                CategorySimilarityConfig,
            )


def _render_zeroshot_form(cfg) -> None:
    """Pipeline-level knobs for the zero-shot stage (on/off, when to
    invoke). Per-call prompting (template, vendor enrichment) lives
    in :func:`_render_zeroshot_prompting_form` under Models → Zero-Shot
    where users land when they want to configure zero-shot."""
    z = cfg.zeroshot
    st.markdown("### 5. Zero-shot NLI (fallback)")
    st.caption(
        "Template + vendor-context enrichment live under "
        "**Classification → Models → Zero-Shot** above."
    )
    with st.form("settings_zeroshot"):
        enabled = st.toggle("Enabled", value=z.enabled, key="zs_enabled")
        below = st.number_input(
            "Use when confidence below", min_value=0.0, max_value=1.0, step=0.01,
            value=float(z.use_when_confidence_below),
        )
        if st.form_submit_button("Save zero-shot", type="primary"):
            # Preserve the prompting fields that live in the other form.
            _persist(
                "zeroshot",
                {
                    "enabled": bool(enabled),
                    "use_when_confidence_below": float(below),
                    "hypothesis_template": z.hypothesis_template,
                    "use_vendor_context": z.use_vendor_context,
                    "vendor_summary_max_chars": z.vendor_summary_max_chars,
                    "batch_size": z.batch_size,
                },
                ZeroshotConfig,
            )


def _render_zeroshot_prompting_form(cfg) -> None:
    """Hypothesis template + vendor-context enrichment.

    Rendered inside the Models → Zero-Shot expander so users
    configuring zero-shot find every per-call knob in one place.
    The pipeline-level on/off + confidence threshold stay in the
    cascade tuning section (those are pipeline composition, not
    prompting).
    """
    z = cfg.zeroshot
    st.markdown("---")
    st.markdown("**Prompting**")
    with st.form("settings_zeroshot_prompting"):
        template = st.text_input(
            "Hypothesis template",
            value=z.hypothesis_template,
            help=(
                "NLI hypothesis format. The `{}` placeholder is replaced "
                "with each category label (name + description) at call "
                "time. The German default `\"In diesem Text geht es um "
                "{}.\"` performs notably better than English templates "
                "on German bank text with multilingual mDeBERTa. Switch "
                "back to `\"This text is about {}.\"` for English data. "
                "A/B-test impact via the **Quality** tab."
            ),
        )
        use_vendor_context = st.toggle(
            "Enrich premise with vendor lookup context",
            value=z.use_vendor_context,
            key="zs_vendor_ctx",
            help=(
                "When ON, the NLI premise is augmented with the cached "
                "vendor industry tag and a short slice of the cached web "
                "summary (requires Vendor web lookup to be enabled and "
                "the cache to be populated). Privacy: nothing new leaves "
                "the machine — the snippet was already cached locally."
            ),
        )
        summary_max_chars = st.number_input(
            "Vendor summary cap (chars)",
            min_value=0, max_value=2000, step=20,
            value=int(z.vendor_summary_max_chars),
            help=(
                "Maximum characters of the cached vendor summary appended "
                "to the premise. Only used when the toggle above is on."
            ),
        )
        batch_size = st.number_input(
            "NLI pipeline batch size",
            min_value=1, max_value=512, step=1,
            value=int(z.batch_size),
            help=(
                "How many (text × candidate-label) pairs the zero-shot "
                "pipeline processes per GPU forward pass. **16** is a "
                "safe CPU default; bump to **32–64** on a GPU to amortise "
                "kernel-launch overhead — typically a 5–20× speedup on "
                "Quality-tab runs that exercise zero-shot. Higher values "
                "use more VRAM."
            ),
        )
        if st.form_submit_button("Save prompting", type="primary"):
            # Preserve the pipeline knobs from the other form.
            _persist(
                "zeroshot",
                {
                    "enabled": z.enabled,
                    "use_when_confidence_below": z.use_when_confidence_below,
                    "hypothesis_template": template,
                    "use_vendor_context": bool(use_vendor_context),
                    "vendor_summary_max_chars": int(summary_max_chars),
                    "batch_size": int(batch_size),
                },
                ZeroshotConfig,
            )


def _render_own_ibans_body(conn) -> None:
    """My Accounts body (own-IBAN registry + add form)."""
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
        # One width tuple shared by header / per-IBAN rows / Add
        # row so every column lines up vertically.
        own_widths = [2, 2, 0.6]
        h = st.columns(own_widths)
        h[0].markdown("**IBAN**")
        h[1].markdown("**Label**")
        h[2].markdown("")
        for r in own_rows:
            row = st.columns(own_widths)
            # Disabled text_input rather than st.code so the IBAN
            # box renders at the same height as the editable Label
            # input next to it -- st.code's <pre> block was
            # systematically shorter and rows looked misaligned.
            row[0].text_input(
                "iban",
                value=r.iban,
                key=f"own_iban_display_{r.iban}",
                label_visibility="collapsed",
                disabled=True,
            )
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
                             help=f"Remove {r.iban}",
                             use_container_width=True):
                rep = remove_own_iban(conn, r.iban)
                st.toast(
                    f"removed; cleared the flag on {rep.n_was_self} "
                    "transaction(s)."
                )
                st.rerun()

    st.markdown("**Add own IBAN**")
    add_cols = st.columns([2, 2, 0.6])
    new_iban = add_cols[0].text_input(
        "new iban",
        key="new_own_iban_iban",
        label_visibility="collapsed",
        placeholder="IBAN",
    )
    new_label = add_cols[1].text_input(
        "new label",
        key="new_own_iban_label",
        label_visibility="collapsed",
        placeholder="Name",
    )
    if add_cols[2].button("Add", type="primary", key="new_own_iban_add",
                          disabled=not new_iban.strip(),
                          use_container_width=True):
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


# Keys of buttons whose action would destroy data. Used by
# :func:`_inject_destructive_button_css` to paint them red so they
# stand out from the benign Backup / Restore buttons sharing the
# Database section.
_DESTRUCTIVE_BUTTON_KEYS: tuple[str, ...] = (
    "db_delete_user_labels",
    "db_reset_data",
    "db_factory_reset",
    "db_remove_encryption",
)


def _inject_destructive_button_css() -> None:
    """Paint the destructive buttons red.

    Streamlit (1.41) has no native "danger" button type. The standard
    workaround is to target the ``st-key-<key>`` class Streamlit
    assigns to each widget's wrapper and override the button colours
    via a single ``st.html`` injection. Same trick as the chip-colour
    rendering in :mod:`review_tab`.
    """
    rules: list[str] = []
    for k in _DESTRUCTIVE_BUTTON_KEYS:
        # Match the wrapper class + the underlying button so the
        # hover/focus states stay coherent with the danger framing.
        rules.append(
            f".st-key-{k} button {{"
            "  background-color: #d9534f !important;"
            "  border-color: #d43f3a !important;"
            "  color: #ffffff !important;"
            "}}"
        )
        rules.append(
            f".st-key-{k} button:hover {{"
            "  background-color: #c9302c !important;"
            "  border-color: #ac2925 !important;"
            "}}"
        )
    st.html("<style>" + " ".join(rules) + "</style>")


def _render_database_body(cfg, conn) -> None:
    """Database section body: stats + encryption + backup/restore + resets."""
    _inject_destructive_button_css()
    _render_db_stats(cfg, conn)
    with st.expander("Encryption", expanded=False):
        _render_encryption(cfg)
    _render_administration(cfg, conn)


def _render_db_stats(cfg, conn) -> None:
    """Detailed database statistics: file metadata, encryption status, and a
    structure overview (tables, row + column counts, views, indexes)."""
    db_path = cfg.db_path
    try:
        st_info = db_path.stat() if db_path.exists() else None
    except OSError:
        st_info = None
    db_mtime = (
        _dt.datetime.fromtimestamp(st_info.st_mtime) if st_info else None
    )
    size_mb = (st_info.st_size / (1024 * 1024)) if st_info else 0.0
    encrypted = account_is_encrypted()

    try:
        overview = database_overview(conn)
    except Exception as e:  # never let a stats query break the page
        st.caption(f"(could not read structure: {e})")
        overview = None

    m = st.columns(5)
    m[0].metric("Last modified", db_mtime.strftime("%Y-%m-%d %H:%M") if db_mtime else "—")
    m[1].metric("File size", f"{size_mb:.2f} MB")
    m[2].metric(
        "Encryption",
        "🔒 Encrypted" if encrypted else "🔓 Plaintext",
        help=(
            f"SQLCipher {cipher_version()}." if encrypted
            else "Stored as a plain SQLite file. Set a password under "
            "Encryption below to protect it at rest."
        ),
    )
    if overview is not None:
        m[3].metric(
            "Tables",
            overview.n_tables,
            help=f"{overview.n_columns_total} columns across all tables; "
                 f"{len(overview.views)} view(s), {len(overview.indexes)} index(es).",
        )
        m[4].metric("Total rows", f"{overview.n_rows_total:,}")
        sv = overview.schema_version
        st.caption(
            f"Path: `{db_path}`"
            + (f"  ·  schema version {sv}" if sv is not None else "")
        )
        with st.expander("Structure overview", expanded=False):
            _render_structure_overview(overview)
    else:
        st.caption(f"Path: `{db_path}`")


def _render_structure_overview(overview) -> None:
    """Table-by-table breakdown: per-table row/column summary plus the
    column list (name, type, flags) for each table."""
    summary = [
        {"Table": t.name, "Rows": t.n_rows, "Columns": t.n_columns}
        for t in overview.tables
    ]
    st.dataframe(summary, hide_index=True, use_container_width=True)
    if overview.views:
        st.caption("Views: " + ", ".join(f"`{v}`" for v in overview.views))
    if overview.indexes:
        st.caption("Indexes: " + ", ".join(f"`{i}`" for i in overview.indexes))

    st.markdown("**Columns per table**")
    for t in overview.tables:
        with st.expander(f"{t.name}  ({t.n_columns} columns · {t.n_rows:,} rows)",
                         expanded=False):
            cols = [
                {
                    "Column": c.name,
                    "Type": c.type or "—",
                    "Not null": "✓" if c.notnull else "",
                    "PK": "✓" if c.pk else "",
                }
                for c in t.columns
            ]
            st.dataframe(cols, hide_index=True, use_container_width=True)


def _render_encryption(cfg) -> None:
    """Set / change / remove the active account's database password.

    Encryption is per-account and backed by SQLCipher (optional
    ``[encryption]`` extra). When the dependency is missing and the DB is
    plaintext, we explain how to enable it; an already-encrypted account is
    always manageable because reaching this screen means it was unlocked.
    """
    active = get_active_account()
    encrypted = account_is_encrypted(active)
    st.caption(
        "Per-account setting — encrypts **this account's** database file at "
        "rest with AES-256 (SQLCipher). The password is never written to "
        "disk; you'll be asked for it when you switch to this account."
    )

    if not encryption_available() and not encrypted:
        st.info(
            "Encryption needs the optional SQLCipher dependency. Install it "
            "with `pip install expense-analyzer-de[encryption]` and restart "
            "the UI (`expense ui-restart`)."
        )
        return

    _render_plaintext_safety_cleanup(cfg)

    if encrypted:
        _render_change_password(cfg, active)
        st.markdown("---")
        _render_remove_encryption(cfg, active)
    else:
        _render_set_password(cfg, active)


def _plaintext_safety_copies(cfg) -> list[Path]:
    """Leftover plaintext ``*.pre-encrypt.*.sqlite`` snapshots next to the DB.

    These are written when an account is encrypted (UI or CLI) so a
    forgotten password isn't fatal -- but they're unencrypted, so we
    surface them for deletion once the password is confirmed working."""
    db_path = Path(cfg.db_path)
    return sorted(db_path.parent.glob(f"{db_path.stem}.pre-encrypt.*.sqlite"))


def _render_plaintext_safety_cleanup(cfg) -> None:
    leftovers = _plaintext_safety_copies(cfg)
    if not leftovers:
        return
    st.warning(
        "A **plaintext** safety copy from encrypting this account is still "
        "on disk — your data is readable from it regardless of the password. "
        "Delete it once you've confirmed the password works."
    )
    for p in leftovers:
        cols = st.columns([5, 1])
        cols[0].code(str(p), language=None)
        if cols[1].button("Delete", key=f"del_safety_{p.name}",
                          use_container_width=True):
            try:
                p.unlink()
                st.toast(f"deleted {p.name}")
            except OSError as e:
                st.error(f"could not delete: {e}")
            st.rerun()


def _close_live_connection() -> None:
    """Close + drop the cached DB connection so the file can be rewritten.

    SQLCipher's encrypt/decrypt/rekey rewrite or replace the file on disk;
    a live handle would block that on Windows and risk a partial read
    elsewhere. The next ``get_conn()`` reopens with the new key."""
    try:
        get_conn().close()
    except Exception:
        pass
    invalidate_connection()


def _render_set_password(cfg, active) -> None:
    with st.form("db_encrypt_form"):
        st.markdown("**Encrypt this database**")
        pw1 = st.text_input("New password", type="password", key="db_enc_pw1")
        pw2 = st.text_input("Confirm password", type="password", key="db_enc_pw2")
        if st.form_submit_button("Encrypt database", type="primary"):
            if not pw1:
                st.error("password cannot be empty")
                return
            if pw1 != pw2:
                st.error("passwords don't match")
                return
            try:
                _close_live_connection()
                safety = encrypt_file(cfg.db_path, pw1, keep_safety=True)
            except EncryptionError as e:
                st.error(f"encryption failed: {e}")
                invalidate_connection()
                return
            set_password_for(active.id, pw1)
            invalidate_connection()
            st.success("Database encrypted.")
            if safety is not None:
                st.warning(
                    f"A **plaintext** safety copy was kept at `{safety}` in "
                    "case you forget the password. Delete it once you've "
                    "confirmed the new password works — until then your data "
                    "is still readable from that copy."
                )
            st.rerun()


def _render_change_password(cfg, active) -> None:
    with st.form("db_change_pw_form"):
        st.markdown("**Change password**")
        old = st.text_input("Current password", type="password", key="db_old_pw")
        new1 = st.text_input("New password", type="password", key="db_new_pw1")
        new2 = st.text_input("Confirm new password", type="password", key="db_new_pw2")
        if st.form_submit_button("Change password", type="primary"):
            if not new1:
                st.error("new password cannot be empty")
                return
            if new1 != new2:
                st.error("new passwords don't match")
                return
            try:
                _close_live_connection()
                change_password(cfg.db_path, old, new1)
            except EncryptionError as e:
                st.error(f"could not change password: {e}")
                invalidate_connection()
                return
            set_password_for(active.id, new1)
            invalidate_connection()
            st.success("Password changed.")
            st.rerun()


def _render_remove_encryption(cfg, active) -> None:
    st.markdown("**Remove encryption**")
    st.write(
        "Decrypts the database back to a plain SQLite file. A timestamped "
        "copy of the still-encrypted DB is kept alongside as a safety net."
    )
    confirm = st.text_input("Type `decrypt` to confirm", key="db_decrypt_confirm")
    if st.button("🔓 Remove encryption", key="db_remove_encryption"):
        if confirm.strip().lower() != "decrypt":
            st.error("type the confirmation phrase exactly")
            return
        pw = get_password_for(active.id)
        try:
            _close_live_connection()
            safety = decrypt_file(cfg.db_path, pw, keep_safety=True)
        except EncryptionError as e:
            st.error(f"decryption failed: {e}")
            invalidate_connection()
            return
        clear_password_for(active.id)
        invalidate_connection()
        st.success("Encryption removed; database is now plaintext.")
        if safety is not None:
            st.info(f"Encrypted safety copy: `{safety}`")
        st.rerun()


def _render_administration(cfg, conn) -> None:
    """Backup / Restore + the three destructive ops.

    Previously the destructive ops sat under a separate "Danger Zone"
    subheader inside another bordered container. With the whole
    Database section now in its own collapsed expander, that extra
    framing was redundant -- the buttons are painted red instead so
    the visual cue lives on the button itself, not the section header.
    """
    with st.expander("Backup", expanded=False):
        _render_backup(cfg, conn)
    with st.expander("Restore", expanded=False):
        _render_restore(cfg)
    with st.expander("Delete user labels", expanded=False):
        _render_delete_user_labels(conn)
    with st.expander("Empty database", expanded=False):
        _render_reset_data(conn)
    with st.expander("Factory reset (incl. category deletion)", expanded=False):
        _render_factory_reset(conn)


def _render_backup(cfg, conn) -> None:
    encrypted = account_is_encrypted()
    st.caption(
        "Download a complete copy of the database. Includes every "
        "ingested row, label, embedding, note, vendor-cache entry, and "
        "category."
    )
    if encrypted:
        st.info(
            "This account is encrypted, so the backup is **also encrypted** "
            "under this account's current password. You'll need that same "
            "password to restore it."
        )
    else:
        st.caption(
            "The file is a standard SQLite 3 DB -- open it in any SQLite "
            "browser, or re-import here on another machine."
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
            if encrypted:
                # Export a self-contained SQLCipher copy under the same key
                # (conn.backup() can't cross the sqlcipher↔sqlite3 boundary).
                export_encrypted_copy(
                    cfg.db_path, get_password_for(get_active_account().id), _bk_tmp
                )
            else:
                export_database(conn, _bk_tmp)
            _bk_bytes = _bk_tmp.read_bytes()
        # Embed the active account's slug in the filename so backups
        # from multiple accounts don't collide in the user's Downloads
        # folder. Re-slugify defensively in case the registry slug ever
        # picks up something funky from a hand-edit.
        active = get_active_account()
        slug = slugify(active.id) or "account"
        # `.enc.sqlite` hints the file is encrypted; restore detects it by
        # header regardless of name.
        ext = "enc.sqlite" if encrypted else "sqlite"
        st.download_button(
            "⬇️ Download backup",
            data=_bk_bytes,
            file_name=(
                f"expense-analyzer-{slug}-backup-"
                f"{_dt.date.today().isoformat()}.{ext}"
            ),
            mime="application/octet-stream" if encrypted else "application/x-sqlite3",
            key="db_backup_download",
            type="primary",
        )
    except Exception as e:
        st.error(f"backup failed: {e}")


def _render_restore(cfg) -> None:
    st.caption(
        "Replace the **current** database with an uploaded backup. "
        "A timestamped safety copy of the current DB is saved alongside it "
        "(`db.pre-restore.<ts>.sqlite`) so you can roll back manually if "
        "the restore turns out to be the wrong file."
    )
    upload = st.file_uploader(
        "Pick a backup file (.sqlite / .enc.sqlite)",
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
        encrypted_upload = looks_encrypted(_upload_path)

        backup_pw: str | None = None
        if encrypted_upload:
            st.info(
                "This backup is **encrypted**. Enter the password it was "
                "created with — the restored account will use that password."
            )
            backup_pw = st.text_input(
                "Backup password", type="password", key="db_restore_pw"
            )
            if not backup_pw:
                return  # wait for the password before validating

        result = validate_backup(_upload_path, password=backup_pw)
        if not result.ok:
            st.error("upload is not a valid backup: " + "; ".join(result.errors))
            return
        st.success(
            "valid backup — rows: "
            + ", ".join(f"{k}={v}" for k, v in result.table_counts.items())
        )
        if account_is_encrypted() and not encrypted_upload:
            st.warning(
                "Heads up: this account is encrypted but the backup is "
                "plaintext — restoring leaves the database **unencrypted**. "
                "Re-encrypt it afterwards under Encryption above."
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
                    password=backup_pw,
                )
                # Sync the session password to whatever the restored file
                # is: the backup's password if encrypted, else cleared so
                # the account is treated as plaintext.
                active_id = get_active_account().id
                if encrypted_upload:
                    set_password_for(active_id, backup_pw)
                else:
                    clear_password_for(active_id)
                invalidate_connection()
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


def _render_delete_user_labels(conn) -> None:
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
    if st.button("🗑 Delete all user labels", key="db_delete_user_labels"):
        if confirm_user.strip().lower() == "delete user labels":
            n = _del_user_labels(conn)
            st.success(f"deleted {n} user label row(s)")
            st.rerun()
        else:
            st.error("type the confirmation phrase exactly")


def _render_reset_data(conn) -> None:
    st.write(
        "Deletes every row in `expenses`, `labels`, `notes`, `embeddings`, "
        "`vendor_cache` and `model_versions`. Categories and own-IBANs are kept."
    )
    confirm_data = st.text_input(
        "Type `clear data` to confirm", key="confirm_reset_data"
    )
    if st.button("🗑 Clear ingested data", key="db_reset_data"):
        if confirm_data.strip().lower() == "clear data":
            report = reset_data(conn)
            st.success(
                f"deleted {report.total} row(s) across "
                f"{len(report.table_counts)} table(s)"
            )
            st.rerun()
        else:
            st.error("type the confirmation phrase exactly")


def _render_factory_reset(conn) -> None:
    st.write(
        "Wipes every table including categories and own-IBANs. The DB schema "
        "stays so you can immediately re-init."
    )
    confirm_all = st.text_input(
        "Type `factory reset` to confirm", key="confirm_reset_all"
    )
    if st.button("🗑 Factory reset", key="db_factory_reset"):
        if confirm_all.strip().lower() == "factory reset":
            report = reset_all(conn)
            st.success(
                f"deleted {report.total} row(s) across "
                f"{len(report.table_counts)} table(s)"
            )
            st.rerun()
        else:
            st.error("type the confirmation phrase exactly")
