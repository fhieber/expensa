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

---

## 5. Dashboard statistics & forecasts — ✅ shipped

New `viz/data.py` helpers + dashboard wiring, all unit-tested:

- **Period-over-period deltas** on the headline tiles — income / expenses /
  savings-rate vs the immediately-preceding same-length window
  (`period_totals`, `_previous_period`). Expense delta colour is inverted
  (rising spend is "bad").
- **Top movers** expander — biggest per-category spend changes vs the previous
  period (`category_period_comparison`).
- **Month-to-date pace tile** — spend so far this month, linearly projected to
  month-end, with a Δ vs the trailing-6-month average (`month_to_date_pace`).
- **Fixed vs. variable tile** — estimated committed (recurring) monthly spend
  and its share, built on the existing cadence detector (`fixed_vs_variable`).
- **Upcoming recurring charges** expander — next-30-day forecast projected from
  each recurring vendor's cadence (`upcoming_recurring`).
- **Auto-categorization mix tile** — user / high / medium / low / uncategorized
  breakdown, thresholds aligned with the Review queue (`categorization_mix`).

Backlog ideas not taken in this pass (still open): cash-flow / running-balance
trend line, weekday & day-of-month spending profiles, new-merchant list,
subscription-cancellation (lapse) detection, and unusually-LOW / duplicate-
charge anomaly variants.
