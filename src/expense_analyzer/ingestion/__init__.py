"""Ingestion: parse, normalize, dedup, persist."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from expense_analyzer.features.iban import classify_iban
from expense_analyzer.features.numeric import (
    amount_bucket,
    is_income,
    is_round_amount,
)
from expense_analyzer.ingestion.csv_loader import ParsedRow, parse_csv
from expense_analyzer.ingestion.dedup import compute_dedup_hash
from expense_analyzer.ingestion.normalizer import (
    combined_text,
    normalize_counterparty,
    normalize_verwendungszweck,
)


@dataclass
class IngestReport:
    """Summary of one CSV ingest."""

    file: str
    parsed: int
    inserted: int
    duplicates: int
    errors: int = 0


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
    vz_norm = normalize_verwendungszweck(row.verwendungszweck)
    iban_info = classify_iban(row.iban, own_ibans=own_ibans)
    return {
        "buchungsdatum": row.buchungsdatum,
        "wertstellung": row.wertstellung,
        "status": row.status,
        "zahlungspflichtiger": row.zahlungspflichtiger,
        "zahlungsempfaenger": row.zahlungsempfaenger,
        "verwendungszweck": row.verwendungszweck,
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


def ingest_csv(conn: sqlite3.Connection, path: Path) -> IngestReport:
    """Parse one CSV and insert rows. Returns counts of new vs. duplicate."""
    parsed = parse_csv(path)
    own_ibans = _load_own_ibans(conn)
    inserted = 0
    for row in parsed:
        params = _row_to_params(row, own_ibans)
        cur = conn.execute(_INSERT_SQL, params)
        if cur.rowcount > 0:
            inserted += 1
    duplicates = len(parsed) - inserted
    return IngestReport(
        file=path.name, parsed=len(parsed), inserted=inserted, duplicates=duplicates
    )


def ingest_many(conn: sqlite3.Connection, paths: list[Path]) -> list[IngestReport]:
    return [ingest_csv(conn, p) for p in paths]
