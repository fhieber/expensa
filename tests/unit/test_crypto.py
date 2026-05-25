"""Tests for storage/crypto.py — SQLCipher encrypt / decrypt / rekey.

The whole module is skipped when the optional ``sqlcipher3`` driver isn't
installed, mirroring how the feature degrades at runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlcipher3", reason="SQLCipher driver not installed")

from expense_analyzer.storage import crypto  # noqa: E402
from expense_analyzer.storage.categories import upsert_category  # noqa: E402
from expense_analyzer.storage.database import get_or_create_database  # noqa: E402


def _make_plaintext_db(path: Path, marker: str = "Groceries") -> None:
    conn = get_or_create_database(path)
    try:
        upsert_category(conn, marker, "desc", "#abc")
    finally:
        conn.close()


def _category_names(conn) -> set[str]:
    return {r["name"] for r in conn.execute("SELECT name FROM categories")}


def test_looks_encrypted_distinguishes_files(tmp_path: Path) -> None:
    plain = tmp_path / "plain.sqlite"
    _make_plaintext_db(plain)
    assert crypto.looks_encrypted(plain) is False
    # Nonexistent / empty files aren't "encrypted".
    assert crypto.looks_encrypted(tmp_path / "missing.sqlite") is False
    (tmp_path / "empty.sqlite").write_bytes(b"")
    assert crypto.looks_encrypted(tmp_path / "empty.sqlite") is False

    crypto.encrypt_file(plain, "pw", keep_safety=False)
    assert crypto.looks_encrypted(plain) is True


def test_encrypt_then_open_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _make_plaintext_db(db, "Miete")
    safety = crypto.encrypt_file(db, "s3cret", keep_safety=True)

    # Safety copy is a plaintext readable DB.
    assert safety is not None and safety.exists()
    assert crypto.looks_encrypted(safety) is False

    # The data survives behind the right password.
    conn = get_or_create_database(db, "s3cret")
    try:
        assert "Miete" in _category_names(conn)
    finally:
        conn.close()


def test_wrong_password_rejected(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _make_plaintext_db(db)
    crypto.encrypt_file(db, "right", keep_safety=False)

    assert crypto.verify_password(db, "right") is True
    assert crypto.verify_password(db, "wrong") is False
    with pytest.raises(crypto.WrongPassword):
        crypto.open_connection(db, "wrong")
    with pytest.raises(crypto.PasswordRequired):
        crypto.open_connection(db, "")


def test_change_password(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _make_plaintext_db(db, "Reisen")
    crypto.encrypt_file(db, "old", keep_safety=False)
    crypto.change_password(db, "old", "new")

    assert crypto.verify_password(db, "old") is False
    conn = get_or_create_database(db, "new")
    try:
        assert "Reisen" in _category_names(conn)
    finally:
        conn.close()


def test_decrypt_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _make_plaintext_db(db, "Versicherung")
    crypto.encrypt_file(db, "pw", keep_safety=False)
    safety = crypto.decrypt_file(db, "pw", keep_safety=True)

    assert crypto.looks_encrypted(db) is False
    assert safety is not None and crypto.looks_encrypted(safety) is True
    # Now openable as plaintext again.
    conn = get_or_create_database(db)
    try:
        assert "Versicherung" in _category_names(conn)
    finally:
        conn.close()


def test_export_decrypted_copy_is_plaintext(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _make_plaintext_db(db, "Auto")
    crypto.encrypt_file(db, "pw", keep_safety=False)
    out = tmp_path / "backup.sqlite"
    crypto.export_decrypted_copy(db, "pw", out)

    assert crypto.looks_encrypted(out) is False
    import sqlite3

    conn = sqlite3.connect(str(out))
    conn.row_factory = sqlite3.Row
    try:
        assert "Auto" in {r["name"] for r in conn.execute("SELECT name FROM categories")}
    finally:
        conn.close()


def test_safety_copy_matches_encrypted_contents(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    conn = get_or_create_database(db)
    try:
        for i in range(25):
            upsert_category(conn, f"Cat{i}")
    finally:
        conn.close()

    safety = crypto.encrypt_file(db, "pw", keep_safety=True)
    assert safety is not None

    import sqlite3

    plain = sqlite3.connect(str(safety))
    enc = get_or_create_database(db, "pw")
    try:
        n_plain = plain.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        n_enc = enc.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    finally:
        plain.close()
        enc.close()
    # The plaintext safety copy is a complete snapshot, not a stale main file.
    assert n_plain == n_enc >= 25


def test_encrypt_rejects_already_encrypted(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _make_plaintext_db(db)
    crypto.encrypt_file(db, "pw", keep_safety=False)
    with pytest.raises(crypto.EncryptionError):
        crypto.encrypt_file(db, "pw2", keep_safety=False)
