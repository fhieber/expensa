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
- **Storage:** SQLite single-file under `~/.expense-analyzer/db.sqlite`.
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
- **Add a new CLI command:** add a `@cli.command` in `src/expense_analyzer/cli.py` and a corresponding test in `tests/unit/test_cli.py`.
