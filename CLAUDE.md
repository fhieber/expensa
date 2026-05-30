# CLAUDE.md — guidance for the Claude assistant working on this repo

This file is read by Claude Code at the start of each session. Keep it concise and current.

## Original user spec (verbatim, 2026-05-10)

> Build a Python solution to analyze, cluster and categorize expenses provided incrementally over time as csv files. Columns of the CSV file are (in German): `"Buchungsdatum";"Wertstellung";"Status";"Zahlungspflichtige*r";"Zahlungsempfänger*in";"Verwendungszweck";"Umsatztyp";"IBAN";"Betrag (€)";"Gläubiger-ID";"Mandatsreferenz";"Kundenreferenz"`
>
> I want the csv data NEVER to be exposed to any cloud-based LLM. Analysis, clustering and categorization should happen entirely on this PC via locally available models (e.g. Huggingface download). The tool should allow the user to define an initial set of expense categories, label a few example cases for the system to then increasingly auto-categorize new expenses (coming from new CSV exports the user provides). The system should deduplicate incoming new CSVs so that the user doesn't have to worry about what they provide to the system (e.g. on overlapping data).
>
> For analysis the system should build visualizations like pie charts, histograms, trend lines etc.
>
> Organize all of this in a complete Python package, with requirements.txt, setup.py etc.
>
> The package should be a git repository (with no remote location for now). You should commit changes each time to allow easy revert if necessary and to keep track of changes.
>
> You'll need to build unit and maybe even integration tests (e.g. on toy/fictituous expense data).
>
> IMPORTANT: Expense data will be in German (take that into account for any HF model choice).
>
> Feature ideas to start from:
> - string similarity of vendor, recipient or "Verwendungszweck"
> - vendor search via the web (maybe integrate with some search API) to get more context on what it is
> - user should be able to store descriptions/notes to a record
> - some time series aspect (if a similar expense was done before it is likely to be of the same category)

## Confirmed design decisions

- **Default embedding model:** `T-Systems-onsite/cross-en-de-roberta-sentence-transformer` (DE/EN, 768-d). Configurable in `config/default_config.yaml`.
- **Zero-shot fallback:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.
- **UI:** Both `click` CLI and a local-only Streamlit app (binds to `127.0.0.1`).
- **Vendor web lookup:** Off by default. When enabled, **only the normalized counterparty name** is sent — never amount, IBAN, or Verwendungszweck.
- **Storage:** SQLite single-file under `~/.expensa/db.sqlite`.
- **Min Python:** 3.10.

## Privacy invariants (must not break)

1. No expense field ever sent to a cloud LLM.
2. Streamlit must bind to `127.0.0.1`, never `0.0.0.0`.
3. Vendor lookup module must whitelist exactly one field (`counterparty_normalized`); reject any code path that would forward `verwendungszweck`, `iban`, `betrag`, etc.
4. No telemetry, no auto-update checks.

## Commit policy

- Remote: <https://github.com/fhieber/expensa> (`origin`). The original
  spec said "no remote"; the user provisioned this GitHub repo later.
- **Work via pull requests** going forward: branch off `main`, push the
  branch, open a PR with `gh pr create`, never push directly to `main`.
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`.
- Commit at logical milestones (scaffolding, ingestion, features, ML, viz, UI, tests). Small, frequent commits so the user can revert cleanly.
- Never `--amend` published commits; never `--no-verify`.
- License is Apache 2.0 (GitHub repo initialised with it; supersedes the
  original MIT spec).

## Glossary of computed features

See the "Proposed feature set per expense" section of `../../.claude/plans/build-a-python-solution-binary-dijkstra.md` for the canonical list. Notable shorthand:
- `combined_text` — `counterparty_normalized + " | " + verwendungszweck_normalized`; the single string fed to the embedding model.
- `dedup_hash` — `sha256(buchungsdatum | wertstellung | betrag_cents | iban | counterparty_normalized | first_120_chars(verwendungszweck_normalized))`.
- Cascade stages: `vendor_exact_match` → `knn_embedding` → `classifier` → `zeroshot_nli`.

## Future guidance from the user

(Append new instructions here verbatim with date so context is preserved.)

### 2026-05-30 — Feature-engineering batch

Added a batch of new computed features (all unit-tested; full unit suite
green: 405 passed). Lands on top of the merged ML/UI/dashboard work.

- **Restored `week`** to `_NUMERIC_COLS` — it was computed by
  `basic_calendar_features` but dropped before the model (same bug class
  as the earlier `day_of_month` fix).
- **Cyclical calendar encodings** — `month`/`day_of_week`/`day_of_month`
  now also enter as `(sin, cos)` pairs so linear models see true circular
  distance. Raw integers stay for trees.
- **`umsatztyp` one-hot** — the German transaction-type field (previously
  stored/displayed but never modelled) is folded to a fixed vocabulary
  (`umsatztyp_bucket`, `UMSATZTYP_BUCKETS`) and one-hot encoded. Strong
  signal: Dauerauftrag→rent/subs, Gehalt→income, Bargeld→cash.
- **Gläubiger-ID (SEPA creditor id) identity** — added as a stage-1
  `vendor_exact_match` fallback (tried *before* IBAN) via
  `glaeubiger_label_distribution`, plus a leak-free
  `glaeubiger_count_before` frequency feature and an
  `is_recurring_stable_key` recurrence flag keyed on creditor-id/IBAN
  (catches subscriptions whose display name drifts). Config:
  `vendor_exact_match.use_glaeubiger`.
- **Global amount z-score** (`amount_zscore_global`) — backstop for
  `amount_zscore_within_cp`, which is NULL until a vendor has ≥2 prior
  charges; the global z exists from the 3rd row. Leak-free (prior rows
  only).
- **Amount-pattern flags** — `has_cents`, `is_small_verification`
  (≈€0–1 card-probe), `amount_ends_99` (retail pricing).
- **Text-shape features** — `vz_length`, `vz_token_count`,
  `vz_digit_ratio` over the normalised purpose text.

The classifier numeric block grew 23 → 48 columns. All new label-keyed
helpers honour the `restrict_ids` CV-leak guard.

### 2026-05-29 — Robustness / ML / UI review pass

A review for robustness, ML feature-set, and UI opportunities landed a
focused first batch (all unit-tested; full suite green: 455 passed) plus a
backlog doc at `docs/improvement-proposals.md`.

Shipped:
- Fixed a pre-existing test-isolation failure: a demo test in
  `test_enrich_secondary.py` reassigned `clf._build_zeroshot_premise`
  directly (never restored), leaking into and failing
  `test_zeroshot_premise_includes_industry_and_summary` in full-suite
  runs. Now patched via `monkeypatch.setattr`.
- `parse_german_amount` tolerates a currency symbol, whitespace thousands
  separators (incl. NBSP / thin space), and a leading `+`.
- `day_of_month` is now an actual classifier feature (it was computed in
  `basic_calendar_features` but dropped before `_NUMERIC_COLS`).
- Category-similarity lexical bonus is configurable via
  `category_similarity.lexical_weight` / `lexical_max` (defaults reproduce
  the old hard-coded 0.10 / 0.30).
- Active-learning uncertainty sampling breaks ties by runner-up margin.

Top backlog items (see the doc for the full list): IBAN-based merchant
identity feature, classifier probability calibration, sign-consistency
guardrail, richer recurrence signals; empty-state onboarding + confidence
badges + bulk labelling in the UI; and `--json` / `--dry-run` CLI parity.

### 2026-05-25 — Per-account database encryption (SQLCipher)

Accounts can be encrypted at rest with AES-256 via SQLCipher.
Encryption is **opt-in per account** and the driver is an **optional
extra** (`pip install expensa[encryption]`, package
`sqlcipher3-wheels` — ships Linux/macOS/**Windows** wheels, imports as
`sqlcipher3`). Plaintext accounts keep using stdlib `sqlite3`, so the
dependency stays optional.

- **Source of truth:** a file is encrypted iff its header is *not* the
  plaintext `b"SQLite format 3\x00"` magic (`crypto.looks_encrypted`).
  No flag in `accounts.yaml` to drift.
- **Passwords are never persisted.** In the UI they live only in
  `st.session_state["_account_passwords"]` (per slug). The CLI reads
  `EXPENSA_DB_PASSWORD` or prompts interactively.
- **UI flow:** switching to an encrypted account hits the unlock gate in
  `streamlit_app.py` (`_render_unlock_gate`), which `st.stop()`s the page
  until the right password is entered. Set / change / remove the password
  under **Settings → Database → Encryption**.
- **CLI parity:** `expense account encrypt|decrypt|passwd [NAME]`
  (defaults to the active account). encrypt prompts for a new password,
  then asks whether to delete the plaintext safety copy
  (`--delete-plaintext/--keep-plaintext` to skip the prompt; non-TTY
  keeps it). decrypt/passwd read `EXPENSA_DB_PASSWORD` or
  prompt. Read-only commands open encrypted DBs via the same env var /
  interactive prompt.
- **Set-password migration:** `crypto.encrypt_file` exports the plain DB
  into a fresh SQLCipher file and keeps a timestamped **plaintext**
  `*.pre-encrypt.*.sqlite` safety copy. The Settings → Encryption section
  globs for leftover `*.pre-encrypt.*.sqlite` copies (from UI *or* CLI)
  and offers a per-file Delete button. `decrypt_file` /
  `change_password` (PRAGMA rekey) mirror the export approach.
- **Backups follow the account:** an encrypted account exports an
  **encrypted** SQLCipher backup under its current key
  (`crypto.export_encrypted_copy`); a plaintext account exports plaintext
  (`backup.export_database`). `validate_backup` / `restore_database` take
  an optional `password=`; encrypted uploads require it (the UI prompts),
  and the restored DB stays encrypted under that key (session password is
  synced). Restoring a plaintext backup into an encrypted account leaves
  it plaintext (password cleared). A non-SQLite/SQLCipher upload is
  rejected by header + page-size sanity check.

The same Settings → Database section gained a **detailed structure
overview** (`stats.database_overview`): file size, encryption status +
cipher version, schema version, table count, total rows, and a per-table
breakdown of row/column counts plus each table's columns
(type / not-null / PK), views and indexes.

Key files: `storage/crypto.py` (all encryption logic, Streamlit-free),
`storage/database.py` (`connect(..., password=)`), `storage/backup.py`
(password-aware validate/restore + encrypted export), `storage/stats.py`,
`ui/_shared.py` (unlock/password helpers + password-keyed connection
cache), `ui/streamlit_app.py` (unlock gate), `ui/settings.py`,
`cli.py` (`account encrypt|decrypt|passwd`).

### 2026-05-22 — Multi-account support

The package now supports multiple accounts (e.g. Personal vs Business)
each backed by its own SQLite DB. Layout under `$EXPENSA_HOME`
(default `~/.expensa/`):

    config.yaml              # global: ML models, vendor_lookup, streamlit
    accounts.yaml            # registry: [{id, name, data_dir}]
    active_account           # plain text: slug of the active account
    accounts/
      personal/db.sqlite
      business/db.sqlite

Per-account: `expenses`, `categories`, `labels`, `notes`, `embeddings`,
`vendor_cache`, `own_ibans`, `model_versions`. Global: ML settings,
device, vendor_lookup, streamlit binding.

**Migration:** transparent. On first launch with the new code, if a
legacy `db.sqlite` exists at the root but no `accounts.yaml`, a
`Default` account is auto-registered pointing at the global home
itself (zero file movement). Rollback = delete `accounts.yaml` +
`active_account`.

**CLI:** `expense account list/add/remove/rename/use`. Root group
takes `--account NAME_OR_SLUG` to target a non-active account for one
command. The PID file (`expense ui` and friends) lives under the
global home so there's one Streamlit server per machine -- account
switching happens in-UI.

**UI:** account picker above the header metrics (`Dashboard | …`).
Add / Rename / Remove buttons open `@st.dialog` flows. Switching
accounts wipes tab-scoped session_state and re-renders against the
new DB. Settings sections that affect global config are flagged
"Global setting — applies to all accounts." and write through to
`<global_home>/config.yaml`.

Key files:
- `src/expensa/accounts.py` — registry + slugify + migration.
- `src/expensa/config.py` — `GlobalConfig` / `Config` /
  `load_config_for_account()`.
- `src/expensa/cli.py` — `expense account ...` subgroup.
- `src/expensa/ui/_shared.py` — cached per-session state.
- `src/expensa/ui/streamlit_app.py` — `_render_account_picker()`
  + Add/Rename/Remove dialogs.

### 2026-05-21 — Clustering removed; deps trimmed

The HDBSCAN/UMAP clustering module from the original plan was never
implemented and the `cluster_id` column went unused. As part of the
big-cleanup refactor:

- Dropped deps from `pyproject.toml` + `requirements.txt`:
  `umap-learn`, `hdbscan`, `matplotlib`, `sqlalchemy`. Made `kaleido`
  an optional `[png-export]` extra (only needed for PNG/SVG/PDF chart
  export; HTML export needs nothing extra).
- Dropped `cluster_id` column + `idx_expenses_cluster` index from
  `schema.sql`. Schema bumped to version 2.
- If clustering is reintroduced later it should land as an opt-in
  `[clustering]` extra with its own module under `ml/`.

### 2026-05-10 — GPU acceleration deferred to follow-up

Initial build ships CPU-friendly defaults so the package is portable and
tests stay fast. The `device: auto` config auto-picks `cuda` (NVIDIA),
`mps` (Apple Silicon) or `cpu`. Follow-up items for users with a real
GPU:

- Add `requirements-cuda.txt` extras pinning a `torch` build with a CUDA
  version matching the user's hardware (e.g. CUDA 12.8+ for Blackwell).
- Promote `aari1995/German_Semantic_V3` (1024-d, German-specialized, 8K
  context) to default once the GPU path is wired up — it's the strongest
  German-aware embedder but heavy on CPU.
- Add opt-in `enrichment/local_llm.py` for a quantized 7–8B local LLM
  (Llama 3.1 / Qwen 2.5 in 4-bit via `bitsandbytes` or `llama-cpp-python`)
  for vendor description and category suggestion on ambiguous records.
  Off by default, fully offline, never touches cloud APIs.
- Document GPU detection / CPU fallback in README.

## Where to extend

- **Add a new computed feature:** add column to `storage/schema.sql`, populate in `features/pipeline.py`, expose in `viz/` if useful for charts.
- **Swap the embedding model:** edit `embedding_model` in `config/default_config.yaml`. Re-run `expense train` afterward — embeddings of existing records are recomputed lazily.
- **Add a new visualization:** create a function in `viz/` returning a Plotly Figure, register it in `cli.py viz` and in the Streamlit dashboard.
- **Add a new CLI command:** add a `@cli.command` in `src/expensa/cli.py` and a corresponding test in `tests/unit/test_cli.py`.
