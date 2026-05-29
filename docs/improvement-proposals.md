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

## 2. ML feature-set backlog — ✅ shipped

The full ML backlog landed as a batch (all unit-tested). Summary:

### ✅ IBAN-based merchant identity
The vendor-exact-match stage now falls back to the label distribution for the
expense's **IBAN** when the counterparty-name match abstains
(`vendor_exact_match.use_iban`), bridging merchants that vary their display
name but keep a stable IBAN. A leak-free `iban_count_before` feature (prior
rows sharing the IBAN, by date) was also added to the temporal bulk SQL and
`_NUMERIC_COLS`. Helpers: `categories.iban_label_distribution`.

### ✅ Classifier probability calibration
`fit()` wraps the estimator in `CalibratedClassifierCV` (isotonic) once there's
enough data to cross-validate the calibrator — at least
`classifier.calibrate_min_train` rows **and** every class ≥ `calibrate_cv`
members; tiny sets keep the raw estimator. The `FitReport.classifier_type`
gains a `+calibrated` suffix when engaged.

### ✅ Transaction-sign consistency guardrail
`sign_guardrail` (config section) demotes a prediction to abstention when the
expense's income/expense sign contradicts the chosen category's dominant,
well-supported training sign (≥ `min_support` labels, ≥ `min_consistency`
agreement). Helper: `categories.category_sign_consistency`. User labels are
never policed.

### ✅ Richer recurrence signals
Added `recurring_months_12` (months of the trailing 12 with a similar-amount
charge) and `recurring_is_exact_amount` (every prior similar charge identical)
to the temporal bulk SQL + `_NUMERIC_COLS`, alongside the existing
`is_likely_recurring`. Lets the model separate a fixed subscription from a
noisy variable charge.

### ✅ Embedding model-swap safety
`embedding_model_inventory()` and `purge_embeddings_except()` plus a
`train --force-reembed` flag: `train` now warns when the store holds vectors
from a model other than the configured one and, with the flag, purges the
stale rows and recomputes every row. `load_embeddings` now skips
mismatched-dimension rows instead of building a ragged matrix.

### ✅ kNN tie / runner-up surfacing
`_knn_tally_from_sims` exposes the top **and** runner-up vote; the kNN stage
now populates `Prediction.runner_up` / `runner_up_confidence`, so the
active-learning margin tiebreak applies to kNN rows too.

### ✅ Active-learning: stratified diversity & feedback loop
`select_diverse` biases toward rows whose nearest labelled neighbour is an
under-covered category (`active_learning.stratified_diversity`,
`diversity_min_label_per_category`), falling back to plain geometric diversity
at cold start. `evaluate_label_batch_impact()` re-runs leak-free CV with a
freshly-labelled batch masked vs. included, reporting the accuracy delta so the
UI can show "your last N labels moved accuracy by X".

> **Incidental fix:** `predict_batch` passed numpy-`int64` ids to
> `load_embeddings`, which this sqlite3 build doesn't match in an `IN (...)`
> clause — the kNN train-vector load silently returned empty, disabling the
> kNN stage. Now passes plain ints. (Latent on `main`; surfaced while testing
> the runner-up work.)

---

## 3. UI / UX + CLI backlog (prioritised)

### High
- **Empty-state onboarding.** A fresh DB shows blank metrics and charts with
  no next step. Add a dashboard banner pointing to the Data tab / `expense
  ingest` when `expenses` is empty.
- **Surface confidence on review cards.** The review tab uses hidden 0.40 /
  0.70 thresholds but never shows the score; add a colour-coded confidence
  badge (and the producing stage) per card so users build trust intuition.
- **Bulk labelling.** Both UI and CLI are one-record-at-a-time. Add
  multi-select + "assign category to all selected" in the Data/Review tabs and
  a "label all like this vendor" action.
- **Ingest error guidance.** Wrong encoding / missing `Betrag` column should
  produce an actionable message (and a column-preview confirm step in the UI)
  rather than a terse parse error.

### Medium
- **`--json` output** for `predict`, `eval`, `status`, `account list`,
  `vendor list` to make the CLI scriptable; document exit codes (0 success, 1
  expected error, 2 usage).
- **`--dry-run` / impact preview parity** for destructive ops (`categories
  remove`, `account remove`) matching the existing `reset --dry-run`.
- **Progress bars** for long CLI ops (`label`, `eval`, `predict`,
  embedding warm-up) via `click.progressbar` — the cascade already accepts a
  `progress_callback`.
- **Clickable "To review" metric** that deep-links the Data tab filtered to
  unlabelled + low-confidence rows.

### Low
- Keyboard shortcuts in the review tab (Enter = confirm, ← / → = prev / next).
- Settings search/filter; the page is dense.
- `expense restore <backup>` CLI to match the Settings restore flow.
- `export --labels-only` for a clean `(expense_id, category_id, confidence)`
  hand-off.

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
