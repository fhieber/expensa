-- expense-analyzer-de schema. SQLite >= 3.35.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS expenses (
    id                          INTEGER PRIMARY KEY,
    buchungsdatum               DATE NOT NULL,
    wertstellung                DATE,
    status                      TEXT,
    zahlungspflichtiger         TEXT,
    zahlungsempfaenger          TEXT,
    verwendungszweck            TEXT,
    umsatztyp                   TEXT,
    iban                        TEXT,
    betrag_cents                INTEGER NOT NULL,
    glaeubiger_id               TEXT,
    mandatsreferenz             TEXT,
    kundenreferenz              TEXT,

    counterparty                TEXT,
    counterparty_normalized     TEXT,
    verwendungszweck_normalized TEXT,
    combined_text               TEXT,

    is_income                   INTEGER NOT NULL DEFAULT 0,
    is_round                    INTEGER NOT NULL DEFAULT 0,
    amount_bucket               TEXT,

    iban_country                TEXT,
    iban_blz                    TEXT,
    iban_is_foreign             INTEGER,
    iban_is_known_self          INTEGER,

    has_glaeubiger_id           INTEGER,
    mandatsreferenz_present     INTEGER,
    is_likely_recurring         INTEGER,

    source_file                 TEXT,
    imported_at                 TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    dedup_hash                  TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_expenses_buchungsdatum ON expenses(buchungsdatum);
CREATE INDEX IF NOT EXISTS idx_expenses_counterparty_norm ON expenses(counterparty_normalized);
CREATE INDEX IF NOT EXISTS idx_expenses_iban ON expenses(iban);

CREATE TABLE IF NOT EXISTS embeddings (
    expense_id   INTEGER PRIMARY KEY REFERENCES expenses(id) ON DELETE CASCADE,
    model_name   TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    color       TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    id          INTEGER PRIMARY KEY,
    expense_id  INTEGER NOT NULL REFERENCES expenses(id) ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    source      TEXT NOT NULL CHECK (source IN ('user', 'model')),
    confidence  REAL,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_labels_expense ON labels(expense_id);
CREATE INDEX IF NOT EXISTS idx_labels_category ON labels(category_id);
CREATE INDEX IF NOT EXISTS idx_labels_source ON labels(source);

-- Most-recent label per expense. SQLite optimises views by inlining the
-- subquery, so this is free at query time and removes the duplicated CTE
-- from every consumer. Includes the "fat" set of columns (source +
-- confidence) so callers that need them just SELECT what they want.
CREATE VIEW IF NOT EXISTS latest_label AS
SELECT l.expense_id,
       l.category_id,
       l.source AS label_source,
       l.confidence
FROM labels l
JOIN (
    SELECT expense_id, MAX(id) AS max_id
    FROM labels
    GROUP BY expense_id
) m ON l.id = m.max_id;

CREATE TABLE IF NOT EXISTS notes (
    expense_id INTEGER PRIMARY KEY REFERENCES expenses(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vendor_cache (
    counterparty_normalized TEXT PRIMARY KEY,
    summary                 TEXT,
    industry                TEXT,
    fetched_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS own_ibans (
    iban  TEXT PRIMARY KEY,
    label TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    id              INTEGER PRIMARY KEY,
    classifier_type TEXT,
    n_train_labels  INTEGER,
    feature_dim     INTEGER,
    metrics_json    TEXT,
    blob            BLOB,
    trained_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '2');
