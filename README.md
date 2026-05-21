# expense-analyzer-de

Local German bank-statement analyzer. Ingest CSV exports, deduplicate, build features, and categorize — all on-device using locally hosted Hugging Face models. **No expense data ever leaves your machine for cloud LLMs.**

## Features

- Incremental ingestion of German-format CSVs (`;` separator, `,` decimal, cp1252/utf-8) with content-hash deduplication.
- Rich per-record feature engineering (text, embeddings, numeric, temporal, IBAN, similarity, behavior).
- Cascaded categorization: vendor exact-match → k-NN on embeddings → supervised classifier → category-similarity → zero-shot NLI fallback.
- Active-learning loop — label a few examples; the system surfaces the next most informative records to label.
- Visualizations: bar, pie, histogram, monthly stacked / weekly stacked / daily stacked, calendar heatmap, recurring-vendor + anomaly tables.
- Two interfaces: a `click` CLI and a local-only Streamlit app.
- Optional, opt-in vendor web lookup that sends **only the merchant name** (never amount/IBAN/Verwendungszweck) to a search engine.

## Privacy guarantees

- All ML inference runs locally via `sentence-transformers` / `transformers`.
- Streamlit binds to `127.0.0.1`; no telemetry.
- Vendor web lookup is **off by default**; enabling it never sends transaction details — only the normalized counterparty name.
- SQLite database stored under `~/.expense-analyzer/` (or `$EXPENSE_ANALYZER_HOME`).

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

First run downloads the German sentence-transformer model (~1 GB) into `~/.cache/huggingface`.

## Quick start

```bash
expense init                                    # creates data dir, DB, default config
expense categories edit                         # edit categories in $EDITOR
expense ingest path/to/export1.csv              # de-duplicating import
expense ingest path/to/export2.csv              # second ingest reports new vs. duplicate
expense label --n 20                            # label 20 active-learning candidates
expense train                                   # fit classifier
expense predict                                 # auto-categorize unlabeled
expense viz pie --out spend_by_category.html
expense ui                                      # Streamlit at http://127.0.0.1:8501
```

## Tests

```bash
pytest -q               # fast unit tests (no model download — embedder is mocked)
pytest -q -m slow       # full pipeline with the real embedder
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
