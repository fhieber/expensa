# Multi-Account Support — Design Plan

## Context

The expense analyzer is currently single-account: one SQLite DB, one set of categories, one
set of own IBANs, all stored under `~/.expensa/`. Users want to track multiple
separate accounts (e.g. personal, business, spouse), each with their own expense history,
categories, and IBAN allowlists. ML model settings and vendor lookup are shared concerns that
need no per-account configuration.

### Design decisions

| Topic | Decision |
|-------|----------|
| Vendor lookup cache | **Per-account** — stays in each account's DB; no schema change needed |
| Migration from single-account | **Auto on first run** — transparent, zero user action |
| Account deletion | **Keep files, warn user** — remove from registry only; print `data_dir` |

---

## New Data Directory Layout

```
~/.expensa/
  config.yaml              # Global config: ML models, vendor_lookup, streamlit
  accounts.yaml            # Account registry: list of {id, name, data_dir}
  active_account           # Plain-text file: slug of the currently active account
  accounts/
    personal/
      db.sqlite            # expenses, categories, labels, own_ibans, vendor_cache …
    business/
      db.sqlite
    …
  models/                  # HuggingFace model cache — shared, unchanged
```

**Migration (zero-data-movement):** On first run with new code, if `db.sqlite` exists at the
root but `accounts.yaml` does not, auto-create a `"Default"` account whose `data_dir` points at
`~/.expensa/` itself. No files are moved. The old `config.yaml` becomes the global
config automatically (its ML keys are already there). New accounts created afterwards go under
`accounts/<slug>/`.

---

## Config Split

`Config` (Pydantic) remains the single resolved config object for a given account. Loading is
restructured into two layers.

### Global config — shared across all accounts

```
embedding_model, zeroshot_model, embedding_batch_size, device
classifier, vendor_exact_match, knn, zeroshot, category_similarity, active_learning
vendor_lookup
streamlit
```

### Account config — per-account

```
data_dir   →  the account's subdirectory
db_filename → "db.sqlite" (not user-facing)
```

### New loading functions (`config.py`)

```
load_global_config(config_path?) → GlobalConfig
  merges: packaged defaults ← ~/.expensa/config.yaml ← env vars

load_config_for_account(account_info, global_cfg?) → Config
  GlobalConfig + data_dir=account.data_dir → full Config
```

`load_config()` is kept unchanged for backwards compat — it now resolves the active account
internally. Zero impact on callers.

---

## New Module: `src/expensa/accounts.py`

Dependency-free (stdlib + PyYAML only).

```python
@dataclass
class AccountInfo:
    id: str        # slug, e.g. "personal"
    name: str      # display name, e.g. "Personal"
    data_dir: Path

class AccountRegistry:
    """Loaded from ~/.expensa/accounts.yaml."""
    def load(global_home: Path) -> AccountRegistry
    def save() -> None                              # atomic write (tmp + rename)
    def add(name: str, data_dir: Path | None) -> AccountInfo
        # slugify name, ensure uniqueness, create data_dir
    def remove(account_id: str) -> None             # registry only — no file deletion
    def rename(account_id: str, new_name: str) -> AccountInfo
    def get(account_id: str) -> AccountInfo | None
    def all() -> list[AccountInfo]
    def get_active_id() -> str | None
    def set_active_id(account_id: str) -> None      # writes active_account file

def migrate_legacy_if_needed(global_home: Path) -> AccountRegistry:
    """
    Idempotent. If accounts.yaml missing but db.sqlite present at root,
    create a 'default' account pointing at global_home (zero file movement).
    """

def init_account_db(account: AccountInfo, with_defaults: bool = True) -> None:
    """Create data_dir, open/init DB, optionally seed default categories."""
```

**Slug rules:** lowercase, spaces → hyphens, strip non-alphanumeric-hyphen, max 40 chars,
collision suffix (`-2`, `-3`, …).

---

## Streamlit UI Changes

### Account picker (above the existing header metrics)

```
[Account:  Personal ▼ ]  [ + Add ]  [ ✏ Rename ]  [ 🗑 Remove ]
──────────────────────────────────────────────────────────────────
[Expenses][User-labeled][Categorized][Categories][DB size]   ← existing metrics
```

- Selectbox populates from `AccountRegistry.load()`.
- Session state key: `st.session_state["active_account_id"]`.
- On change: clear all tab-scoped state keys (prefixes `dashboard_*`, `cat_*`, `data_*`,
  `own_iban_*`, `new_own_iban_*`, `confirm_*`), persist via `set_active_id()`, `st.rerun()`.
- **Add Account** button: `@st.dialog` → name input → `registry.add()` → `init_account_db()` →
  switch session state → `st.rerun()`.

### DB connection (minimal change)

`_connect_cached(db_path_str)` is already keyed by path string — no functional change. Only
the call site changes:

```python
# Before
conn = _connect_cached(str(cfg.db_path))

# After
active = _get_active_account()          # reads session state + registry
conn = _connect_cached(str(active.data_dir / "db.sqlite"))
```

The ML embedder cache (`_real_embedder`) is global and unchanged.

### Boot sequence refactor

```python
global_cfg = _load_global_config_cached()        # new cached loader
registry   = _load_registry_cached(global_home)  # new cached loader
active_account, cfg = _resolve_active_account(registry, global_cfg, global_home)
conn = _connect_cached(str(cfg.db_path))
```

### Settings tab adjustments

- Add `st.caption("Global setting — applies to all accounts.")` under ML/Device/Vendor headers.
- `save_user_config(...)` call for model changes: pass `global_home` instead of `cfg.data_dir`;
  clear only `_load_global_config_cached` (not DB connections).
- DB restore: narrow `st.cache_resource.clear()` → `_connect_cached.clear()` only (embedder
  doesn't need clearing on DB restore).
- Own IBANs section: no change — `conn` is already account-scoped.

---

## CLI Changes

### New `expense account` subgroup

```
expense account list
expense account add NAME [--id SLUG] [--with-defaults/--no-defaults]
expense account remove NAME [--yes]     # registry only; prints data_dir path
expense account rename NAME NEW_NAME
expense account use NAME                # writes slug to active_account file
```

`expense account add NAME` creates the directory, initialises the DB, and optionally seeds
default categories — equivalent to current `expense init`, now scoped to the named account.

### `--account` flag on main group

```python
@click.group()
@click.option("--account", "account_id", default=None,
              help="Target account by name or slug (overrides active account).")
def cli(ctx, config_path, account_id, verbose):
    global_cfg = load_global_config(config_path)
    registry   = migrate_legacy_if_needed(global_home)
    account    = _resolve_account(registry, account_id, global_home)
    cfg        = load_config_for_account(account, global_cfg)
    ctx.obj[_CTX_KEY] = {"config": cfg, "global_cfg": global_cfg,
                          "registry": registry, "account": account,
                          "global_home": global_home}
```

Resolution order: `--account` flag → `active_account` file → first in registry → error with
helpful message ("run `expense account add` first").

All existing commands (`ingest`, `categories`, `own-iban`, `predict`, `train`, `label`, `viz`,
`export`, `reset`) are unchanged — they read `cfg` from context.

### `expense ui` / `ui-stop` / `ui-status` / `ui-restart`

- PID file moves from `cfg.data_dir` to `global_home` (one server, many accounts).
- `expense ui` passes `EXPENSA_HOME=global_home` env var (already done); account
  switching now happens in-UI via the account picker.

### `expense init`

Keep existing behaviour (bootstraps the active account). Document `expense account add` as the
preferred path for adding a second account. No deprecation warning yet.

---

## Files to Change

| File | Nature |
|------|--------|
| `src/expensa/accounts.py` | **NEW** — AccountInfo, AccountRegistry, migrate_legacy_if_needed, init_account_db |
| `src/expensa/config.py` | Add `GlobalConfig`, `load_global_config()`, `load_config_for_account()`; keep `load_config()` |
| `src/expensa/cli.py` | Add `account` subgroup; `--account` flag; fix PID dir; pass global_home to UI subprocess |
| `src/expensa/ui/streamlit_app.py` | Refactor boot sequence; add account picker to `_render_header()`; fix Settings model-save path; narrow cache-clear on DB restore |
| `tests/unit/test_accounts.py` | **NEW** — slugify, AccountRegistry CRUD, save/load roundtrip, migration, active account I/O |
| `tests/unit/test_config_accounts.py` | **NEW** — GlobalConfig field assertions, load_config_for_account |
| `tests/unit/test_cli.py` | Extend — `expense account` commands, `--account` flag targeting |

---

## Key Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| `_connect_cached` is process-global | Already keyed by `db_path_str`; multiple accounts get separate connections automatically |
| DB restore clears all caches (including embedder) | Narrow `st.cache_resource.clear()` → `_connect_cached.clear()` only |
| Settings model-save writes to account dir instead of global | Pass `global_home` to `save_user_config()`; clear only `_load_global_config_cached` |
| PID file becomes ambiguous with multiple accounts | Fix `ui/ui-stop/ui-status/ui-restart` to use `global_home` — one-line change per command |
| Existing tests pass `EXPENSA_HOME=tmp_path` | Migration sees no `accounts.yaml` and no `db.sqlite` → empty registry; `expense init` sets up default account as before |

---

## Verification Plan

1. **Unit tests** (`test_accounts.py`): slugify edge cases, AccountRegistry add/get/remove/rename,
   save→load roundtrip, migration with existing `db.sqlite`, migration idempotency, active
   account set/get, missing-file fallback.
2. **Config tests** (`test_config_accounts.py`): `GlobalConfig` has no `data_dir` field;
   `load_config_for_account` inherits ML settings and sets correct `data_dir`.
3. **CLI smoke test:**
   ```
   expense account list                   # shows "* default  Default"
   expense account add Business           # creates accounts/business/db.sqlite
   expense account use Business
   expense status                         # empty Business DB
   expense --account default status       # original data intact
   expense account list                   # shows two accounts, Business active (*)
   ```
4. **UI smoke test:** account picker visible above metrics; switching accounts resets staged
   edits and shows correct categories/expenses; adding an account via dialog works.
5. **Migration regression:** existing `db.sqlite` accessible as "Default" after upgrade with
   zero data loss.
6. **Privacy check:** vendor lookup still forwards only `counterparty_normalized`, regardless
   of active account.
