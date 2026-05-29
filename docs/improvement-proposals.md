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
