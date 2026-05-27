"""Ingestion: parse, normalize, dedup, persist.

Heavy feature work happens at ingest time: text normalization, IBAN
classification, numeric flags, dedup hash, and (if an embedder is
passed) the sentence-transformer embedding. After ingest_csv returns,
querying / labeling / predicting against the new rows is cheap because
everything is precomputed.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from expensa.features.embeddings import Embedder, store_embeddings
from expensa.features.iban import classify_iban
from expensa.features.numeric import (
    amount_bucket,
    is_income,
    is_round_amount,
)
from expensa.ingestion.csv_loader import ParsedRow, parse_csv
from expensa.ingestion.dedup import compute_dedup_hash
from expensa.ingestion.normalizer import (
    combined_text,
    normalize_counterparty,
    normalize_verwendungszweck,
)

# Detect PayPal bank records by counterparty field.
_PAYPAL_RE = re.compile(r"paypal", re.IGNORECASE)
# Extract the merchant from "Ihr Einkauf bei <merchant>" in the Verwendungszweck.
_IHR_EINKAUF_BEI = re.compile(r"Ihr\s+Einkauf\s+bei\s+(.+)$", re.IGNORECASE | re.DOTALL)


def _simplify_paypal_vz(zahlungsempfaenger: str, zahlungspflichtiger: str, vz: str) -> str:
    """Return a simplified Verwendungszweck for PayPal bank records when the
    merchant is already present in the VZ (e.g. 'Ihr Einkauf bei El Purica ...').

    Returns the original ``vz`` unchanged when:
    * the record is not a PayPal row, or
    * the VZ has no parseable 'Ihr Einkauf bei <merchant>' pattern, or
    * the extracted merchant is empty (unknown — needs PayPal CSV enrichment).
    """
    if not _PAYPAL_RE.search(zahlungsempfaenger or "") and not _PAYPAL_RE.search(zahlungspflichtiger or ""):
        return vz
    m = _IHR_EINKAUF_BEI.search(vz)
    if not m:
        return vz
    merchant = m.group(1).strip()
    if not merchant:
        return vz
    return merchant


@dataclass
class IngestReport:
    """Summary of one CSV ingest."""

    file: str
    parsed: int
    inserted: int
    duplicates: int
    embedded: int = 0
    errors: int = 0
    new_ids: list[int] = field(default_factory=list)


_INSERT_SQL = """
INSERT OR IGNORE INTO expenses (
    buchungsdatum, wertstellung, status,
    zahlungspflichtiger, zahlungsempfaenger, verwendungszweck,
    umsatztyp, iban, betrag_cents,
    glaeubiger_id, mandatsreferenz, kundenreferenz,
    counterparty, counterparty_normalized, verwendungszweck_normalized, combined_text,
    is_income, is_round, amount_bucket,
    iban_country, iban_blz, iban_is_foreign, iban_is_known_self,
    has_glaeubiger_id, mandatsreferenz_present,
    source_file, dedup_hash
) VALUES (
    :buchungsdatum, :wertstellung, :status,
    :zahlungspflichtiger, :zahlungsempfaenger, :verwendungszweck,
    :umsatztyp, :iban, :betrag_cents,
    :glaeubiger_id, :mandatsreferenz, :kundenreferenz,
    :counterparty, :counterparty_normalized, :verwendungszweck_normalized, :combined_text,
    :is_income, :is_round, :amount_bucket,
    :iban_country, :iban_blz, :iban_is_foreign, :iban_is_known_self,
    :has_glaeubiger_id, :mandatsreferenz_present,
    :source_file, :dedup_hash
)
"""


def _row_to_params(row: ParsedRow, own_ibans: set[str]) -> dict:
    counterparty = row.zahlungsempfaenger or row.zahlungspflichtiger
    cp_norm = normalize_counterparty(counterparty)
    # Simplify PayPal VZ at ingest time when merchant is already present.
    # The dedup_hash is computed from row.verwendungszweck (original CSV value),
    # so modifying the stored VZ here does not break deduplication.
    vz = _simplify_paypal_vz(row.zahlungsempfaenger, row.zahlungspflichtiger, row.verwendungszweck)
    vz_norm = normalize_verwendungszweck(vz)
    iban_info = classify_iban(row.iban, own_ibans=own_ibans)
    return {
        "buchungsdatum": row.buchungsdatum,
        "wertstellung": row.wertstellung,
        "status": row.status,
        "zahlungspflichtiger": row.zahlungspflichtiger,
        "zahlungsempfaenger": row.zahlungsempfaenger,
        "verwendungszweck": vz,
        "umsatztyp": row.umsatztyp,
        "iban": row.iban,
        "betrag_cents": row.betrag_cents,
        "glaeubiger_id": row.glaeubiger_id,
        "mandatsreferenz": row.mandatsreferenz,
        "kundenreferenz": row.kundenreferenz,
        "counterparty": counterparty,
        "counterparty_normalized": cp_norm,
        "verwendungszweck_normalized": vz_norm,
        "combined_text": combined_text(cp_norm, vz_norm),
        "is_income": int(is_income(row.betrag)),
        "is_round": int(is_round_amount(row.betrag)),
        "amount_bucket": amount_bucket(row.betrag),
        "iban_country": iban_info.country,
        "iban_blz": iban_info.blz,
        "iban_is_foreign": int(iban_info.is_foreign) if iban_info.country else None,
        "iban_is_known_self": int(iban_info.is_known_self),
        "has_glaeubiger_id": int(bool(row.glaeubiger_id)),
        "mandatsreferenz_present": int(bool(row.mandatsreferenz)),
        "source_file": row.source_file,
        "dedup_hash": compute_dedup_hash(row),
    }


def _load_own_ibans(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT iban FROM own_ibans").fetchall()
    return {r["iban"] for r in rows}


def ingest_csv(
    conn: sqlite3.Connection,
    path: Path,
    embedder: Embedder | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> IngestReport:
    """Parse one CSV, insert rows, and (if an embedder is passed) immediately
    compute and persist their sentence-transformer embeddings.

    If ``progress_callback`` is provided it's invoked as
    ``cb(phase, done, total)`` where phase is one of
    ``"parse" | "insert" | "embed"``. The UI uses this to drive a
    ``st.progress`` bar inside the ``st.status`` panel.

    Returns counts of new vs. duplicate, plus the list of new expense IDs so
    callers can show "rows just imported" tables without re-querying.
    """
    cb = progress_callback or (lambda *_: None)
    cb("parse", 0, 1)
    parsed = parse_csv(path)
    cb("parse", 1, 1)
    own_ibans = _load_own_ibans(conn)
    new_ids: list[int] = []
    total_rows = len(parsed)
    for i, row in enumerate(parsed):
        params = _row_to_params(row, own_ibans)
        cur = conn.execute(_INSERT_SQL, params)
        if cur.rowcount > 0:
            new_ids.append(int(cur.lastrowid))
        # Cheap: emit every row, the UI throttles render rate itself.
        cb("insert", i + 1, total_rows)
    inserted = len(new_ids)
    duplicates = total_rows - inserted

    embedded = 0
    if embedder is not None and new_ids:
        # Look up combined_text just for the rows we inserted -- we don't want
        # to embed pre-existing rows again.
        ph = ",".join("?" * len(new_ids))
        rows = conn.execute(
            f"SELECT id, combined_text FROM expenses WHERE id IN ({ph})", new_ids
        ).fetchall()
        cb("embed", 0, len(rows))
        embedded = store_embeddings(
            conn, embedder, [(r["id"], r["combined_text"] or "") for r in rows]
        )
        cb("embed", len(rows), len(rows))

    return IngestReport(
        file=path.name,
        parsed=total_rows,
        inserted=inserted,
        duplicates=duplicates,
        embedded=embedded,
        new_ids=new_ids,
    )


def ingest_many(
    conn: sqlite3.Connection,
    paths: list[Path],
    embedder: Embedder | None = None,
) -> list[IngestReport]:
    return [ingest_csv(conn, p, embedder=embedder) for p in paths]
