# expensa

Local German bank-statement analyzer. Ingest CSV exports, deduplicate, build features, and categorize — all on-device using locally hosted Hugging Face models. **No expense data ever leaves your machine.**

## Features

- Incremental ingestion of German-format CSVs (`;` separator, `,` decimal, cp1252/utf-8) with content-hash deduplication.
- Rich per-record feature engineering (text, embeddings, numeric, temporal, IBAN, similarity, behavior).
- Cascaded categorization: vendor exact-match → k-NN on embeddings → supervised classifier → category-similarity → zero-shot NLI fallback.
- Active-learning loop — label a few examples; the system surfaces the next most informative records to label.
- Visualizations: bar, pie, histogram, monthly/weekly/daily stacked, calendar heatmap, recurring-vendor + anomaly tables.
- Two interfaces: `click` CLI and a local-only Streamlit app that opens in your browser automatically.
- Multi-account support — separate SQLite databases per account (personal, business, etc.).
- Optional, opt-in vendor web lookup that sends **only the merchant name** (never amount, IBAN, or Verwendungszweck).

## Privacy guarantees

- All ML inference runs locally via `sentence-transformers` / `transformers`.
- Streamlit binds to `127.0.0.1` only; no telemetry, no auto-update checks.
- Vendor web lookup is **off by default**; when enabled it sends only the normalized counterparty name.
- SQLite database stored under `~/.expensa/` (or `$EXPENSA_HOME`).

---

## Installation

**Prerequisites:** Python 3.10 or newer, `pip`, and `git`.

```bash
# 1. Clone the repository
git clone https://github.com/fhieber/expensa.git
cd expensa

# 2. Create and activate a virtual environment
python -m venv .venv

# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install the package with all core dependencies
pip install -e .

# Or install with development tools (pytest, ruff, mypy, …):
pip install -e ".[dev]"
```

The first run downloads the default German sentence-transformer model (~1 GB) into `~/.cache/huggingface/`.

### Optional extras

| Extra | What it adds | Install |
|---|---|---|
| `vendor-lookup` | DuckDuckGo merchant lookup | `pip install -e ".[vendor-lookup]"` |
| `png-export` | PNG/SVG/PDF chart export via Kaleido | `pip install -e ".[png-export]"` |
| `report-export` | PDF quality reports via ReportLab | `pip install -e ".[report-export]"` |
| `dev` | pytest, ruff, black, mypy, pre-commit | `pip install -e ".[dev]"` |

Combine extras: `pip install -e ".[vendor-lookup,png-export,dev]"`

---

## Quick start

```bash
expensa init                            # create data dir, DB, and default config
expensa categories edit                 # open category list in $EDITOR

expensa ingest path/to/export1.csv      # de-duplicating import
expensa ingest path/to/export2.csv      # second ingest reports new vs. duplicate

expensa label --n 20                    # interactively label 20 active-learning candidates
expensa train                           # fit the classifier on your labels
expensa predict                         # auto-categorize unlabeled expenses

expensa viz pie                         # spend-by-category pie chart (opens as HTML)
expensa viz trend                       # monthly trend line

expensa ui                              # launch Streamlit UI — opens in browser automatically
```

---

## Command reference

Run `expensa --help` or `expensa <command> --help` for full option documentation.

### Core commands

| Command | Description |
|---|---|
| `expensa init [--with-defaults]` | Create data directory, SQLite DB, and config file |
| `expensa status` | Show DB stats, account info, and model status |
| `expensa ingest <file> [<file>…]` | Import CSV(s); duplicates are silently skipped |
| `expensa label [--n N] [--strategy uncertainty\|diverse\|mixed]` | Interactive labeling session |
| `expensa train` | Fit classifier on current labels |
| `expensa predict [--threshold F] [--dry-run]` | Auto-categorize unlabeled expenses |
| `expensa eval` | Evaluate classifier accuracy (cross-validation) |
| `expensa export [--fmt csv\|json] [--out PATH]` | Export categorized expenses |
| `expensa reset [--wipe-all]` | Clear predictions or wipe everything |

### Visualization

```bash
expensa viz pie       [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--out FILE]
expensa viz histogram [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--out FILE]
expensa viz trend     [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--out FILE]
expensa viz top       [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--out FILE]
expensa viz calendar  [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--out FILE]
```

Default output is `~/.expensa/exports/<name>.html`. Add `--out chart.png` to write PNG (requires the `png-export` extra).

### Streamlit UI

```bash
expensa ui                  # detached (background) — browser opens automatically
expensa ui --foreground     # attached to terminal; Ctrl+C to stop
expensa ui --no-browser     # suppress automatic browser tab
expensa ui-stop             # stop the background server
expensa ui-restart          # stop + start (picks up config changes)
expensa ui-status           # show whether the server is running
```

The UI is a single Streamlit server that serves all accounts via an in-app account picker. It reloads automatically when source files change (`--server.runOnSave true`).

### Categories

```bash
expensa categories list
expensa categories add "Travel" [--description "…"] [--color "#4CAF50"]
expensa categories remove "Travel" [--yes]
expensa categories edit      # open in $EDITOR (YAML)
```

### Multi-account management

```bash
expensa account list                        # list accounts (* = active)
expensa account add "Business"              # create a new account
expensa account use "Business"              # switch active account
expensa account rename "Business" "Work"   # rename
expensa account remove "Work" [--yes]      # delete account and its DB
```

Pass `--account NAME` to any command to target a non-active account without switching:
```bash
expensa --account Business ingest export.csv
```

### Own IBANs

Register your own bank account IBANs so the tool can distinguish incoming from outgoing transfers:

```bash
expensa own-iban list
expensa own-iban add DE89370400440532013000 [--label "Girokonto"]
expensa own-iban remove DE89370400440532013000
```

### Vendor lookup (optional)

```bash
# Requires: pip install -e ".[vendor-lookup]"  AND  vendor_lookup.enabled: true in config
expensa vendor-lookup "Amazon"              # look up one merchant
expensa vendor-lookup --all                 # populate cache for every distinct counterparty
expensa vendor list [--min-count N]         # browse the vendor cache
expensa vendor show "Amazon"               # full detail for one vendor
expensa vendor clear [--yes]               # wipe the cache
```

---

## Configuration

`expensa init` writes `~/.expensa/config.yaml` from the built-in defaults. Edit it to tune models and thresholds:

```yaml
# ML models — all run locally
embedding_model: T-Systems-onsite/cross-en-de-roberta-sentence-transformer
zeroshot_model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
device: auto          # auto | cpu | cuda | mps

# Cascade thresholds
classifier:
  confidence_threshold: 0.7   # below this → flagged for manual review
knn:
  k: 5
  agreement_min: 4            # 4 of 5 neighbors must agree

# Vendor web lookup (off by default)
vendor_lookup:
  enabled: false
  backend: duckduckgo         # duckduckgo | searxng
```

Set `$EXPENSA_HOME` to override the default data directory (`~/.expensa/`).

---

## Tests

```bash
pytest -q               # fast unit tests (embedder is mocked — no model download)
pytest -q -m slow       # full pipeline with the real embedding model
pytest --cov=expensa -q   # with coverage report
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
