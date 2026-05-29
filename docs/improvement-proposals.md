# Robustness, ML & UI improvement proposals

_Last updated: 2026-05-29_

This document captures a review of the codebase for **robustness / edge
cases**, **ML feature-set** opportunities, and **UI/UX + CLI** gaps. Items
marked ✅ were implemented in the same change set; the rest are a prioritised
backlog with enough detail to pick up.

---

## 1. Implemented in this change set ✅

| Area | Change | Files |
| --- | --- | --- |
| Test isolation | Fixed a pre-existing suite failure: `test_zeroshot_premise_monkeypatch_demo` reassigned `clf._build_zeroshot_premise` directly (never restored), leaking a templated premise into `test_zeroshot_premise_includes_industry_and_summary` so it failed only in the full run. Now patched via `monkeypatch.setattr`. | `tests/unit/test_enrich_secondary.py` |
| Robustness | `parse_german_amount` tolerates a currency symbol (`12,34 €`), whitespace thousands separators incl. non-breaking / thin spaces (`1 234,56`), and an explicit leading `+`. | `ingestion/_parsing.py` |
| ML features | `day_of_month` is now fed to the classifier. It was computed by `basic_calendar_features` but silently dropped before the model — a free signal for fixed-date recurring payments (rent on the 1st, subscriptions on the 15th). | `ml/classifier.py` |
| ML tuning | Category-similarity lexical-overlap bonus is now configurable (`lexical_weight`, `lexical_max`) instead of hard-coded `0.10` / `0.30`. Defaults reproduce prior behaviour. | `config.py`, `ml/classifier.py`, `config/default_config.yaml` |
| Active learning | Uncertainty sampling now breaks ties by the **margin** to the runner-up (a 0.55-vs-0.50 call is more informative to label than 0.55-vs-0.05). Degrades gracefully to plain confidence for non-classifier stages. | `ml/active_learning.py` |

All ship with unit tests; the full suite is green (455 passed, 2 skipped, 2 xfailed).

---

## 2. ML feature-set backlog (prioritised)

### P0 — IBAN-based merchant identity
The same merchant often files under several name variants (`REWE`, `REWE
MARKT`, `REWE-BONUS`) but a stable IBAN. Add an `iban_seen_before` count
(prior **user-labelled** rows with the same IBAN) computed in the
`temporal.py` bulk SQL pass and exposed via `_NUMERIC_COLS`. Cheap, highly
predictive, and bridges the cold-start gap for variable-name vendors that
`vendor_exact_match` and kNN both miss.

### P0 — Classifier probability calibration
A RandomForest / LogReg on imbalanced label sets emits over-confident scores
for rare classes (e.g. 0.95 for a category with two training examples), which
the `confidence_threshold` then trusts blindly. Wrap the estimator in
`sklearn.calibration.CalibratedClassifierCV` (config flag
`classifier.calibrate_probas`) so the cascade's thresholds map to real
accuracy. Persist a calibration summary in `model_versions`.

### P1 — Transaction-sign consistency
`is_income` exists but isn't used as a guardrail. Compute per-category
`expected_sign` + `sign_consistency` from the training labels; demote a
prediction that violates a category's near-100%-consistent sign (a refund
predicted as `Lebensmittel`). Also fold "Einnahme/Ausgabe" into the zero-shot
premise.

### P1 — Richer recurrence signals
`is_likely_recurring` is a single boolean (≥3 prior months at ±10%). Add
`recurring_months_count` (of last 12), `recurring_is_exact_amount`, and
`day_of_month_stdev_within_cp` so the model can distinguish a fixed monthly
subscription from a noisy variable charge, and surface
cancellation/anomaly detection later.

### P2 — Embedding model-swap safety
Embeddings are keyed by `(expense_id, model_name)`, so swapping
`embedding_model` silently leaves old vectors orphaned and only re-embeds new
rows. `load_embeddings` also assumes a uniform `dim` from `rows[0]`. Add a
startup check that warns when the active model differs from the one with the
most stored vectors, plus a `retrain --force-reembed` flag that purges stale
model rows.

### P2 — kNN tie / runner-up surfacing
`_knn_vote_from_sims` returns only the winner; with `agreement_min=4` a 3-2
split is discarded even though it carries signal. Return the runner-up so the
cascade and the review queue can show "knn: Groceries (3/5), runner-up
Household (2/5)" — and so active learning's new margin tiebreak applies to kNN
rows too.

### P2 — Active-learning: stratified diversity & feedback loop
`select_diverse` ignores label balance and can pick 8 diverse-but-all-grocery
rows. Constrain diversity to under-represented categories first. Separately,
measure held-out accuracy before/after a labelling batch so the UI can report
"your last 10 labels improved accuracy by N%".

---

## 3. UI / UX + CLI backlog

Most of this section shipped (see ✅). Three items were intentionally
deferred by the maintainer and remain open at the bottom.

### ✅ Shipped
- **Empty-state onboarding** (`ui/dashboard.py`) — a fresh DB now shows a
  three-step quick-start (import → categories → label/review) noting whether
  default categories are already seeded, plus a low-data nudge once expenses
  exist but nothing is labelled yet.
- **Ingest error guidance** (`ui/data_tab.py`) — a failing file no longer
  dumps a traceback or aborts the batch; `_ingest_error_hint` maps the common
  failures (missing column, bad encoding, bad amount/date) to an actionable
  remedy and the import continues with the other files.
- **Model-download feedback** (`ui/settings.py`) — the download status panel
  spells out that first-download has no byte-progress and maps failures
  (network / disk / gated-model) to concrete remedies.
- **`--json` output** for `status`, `account list`, `predict`, `eval`,
  `vendor list`; chatter is routed to stderr so stdout stays pipe-clean. CLI
  module docstring now documents the exit-code convention (0/1/2/3).
- **`--dry-run` / impact preview** for `account remove` (shows the on-disk DB
  size + expense count that stays behind); `categories remove` already had
  `--force` + confirm.
- **Progress bars** for `predict` and `eval` via `click.progressbar`
  (suppressed under `--json`).
- **Deep-link from "To review"** — a *Show in Data ↗* button in the header
  pins the review-queue rows in the Data tab (`review_tab.review_queue_ids`).
  Streamlit can't switch `st.tabs` programmatically, so it pins + points
  rather than auto-jumping.
- **`vendor-lookup --all` skip count** — reports "looked up N new, skipped M
  already-cached".
- **`expensa restore <backup>`** CLI mirroring the Settings restore flow
  (validates first, handles encrypted backups via `EXPENSA_DB_PASSWORD` /
  prompt, keeps a pre-restore safety copy).
- **`export --labels-only`** for a clean `(expense_id, category_id, category,
  source, confidence)` hand-off; parquet export now raises a friendly hint
  when no engine is installed.
- **Confidence on review cards** — already present (the review tab shows
  `🤖 <cat> — NN% via <stage>` plus a runner-up); left as-is.

### Deferred (out of scope for this pass)
- **Bulk labelling** — multi-select + "assign category to all selected" /
  "label all like this vendor" in the Data/Review tabs.
- **Keyboard shortcuts** beyond what the review tab already has.
- **Settings search/filter** for the dense settings page.

---

## 4. Robustness / edge-case notes (backlog)

These were found during the review and are worth a follow-up; none are fixed
in this change set.

- **Large-amount overflow.** `ParsedRow.betrag_cents`
  (`ingestion/csv_loader.py`) does `int(betrag * 100)` with no range check; a
  pathological amount could exceed SQLite's signed-64-bit column. A guard with
  a clear `CsvParseError` is cheap insurance.
- **Duplicate / over-long CSV columns.** `parse_csv` pads short rows but
  `dict(zip(headers, row))` silently keeps only the last value when headers
  repeat, and truncates extra columns. Detect duplicate headers and warn on
  over-long rows so malformed exports are visible.
- **`detect_encoding` last resort.** The `latin-1` attempt decodes *any* byte
  sequence, so the trailing `CsvParseError` is effectively unreachable and a
  mis-decoded file can slip through. Acceptable as a fallback, but log when the
  winning encoding isn't UTF-8.
- **Notes upsert is not transactional.** `enrichment/notes.py` issues
  delete+insert in autocommit; wrap in the existing `transaction()` context.
- **`account.data_dir` path trust.** The registry consumes `data_dir` from
  `accounts.yaml` verbatim. Fine for a local single-user tool, but a resolved-
  path sanity check would harden against a hand-edited/corrupted registry.
- **`base_dataframe([])` still runs the bulk window query.** Short-circuit
  `add_temporal_recurrence` when `df` is empty to avoid a needless heavy SQL
  pass on zero-row filters.

### Things that are already solid
- Account `slugify` collapses to `[a-z0-9-]` and falls back to `account`, so
  path traversal via account *names* is not possible.
- `store_embeddings` uses `INSERT OR REPLACE` on the `(expense_id,
  model_name)` PK, so concurrent lazy embedding is idempotent.
- `accounts.yaml` is written atomically (tmp file + `os.replace`).
