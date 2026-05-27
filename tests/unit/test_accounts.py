"""AccountRegistry + slugify + legacy-migration tests.

No DB or ML deps needed; the account module is stdlib + PyYAML only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from expensa.accounts import (
    ACCOUNTS_FILENAME,
    ACTIVE_ACCOUNT_FILENAME,
    LEGACY_DB_FILENAME,
    AccountInfo,
    AccountNotFoundError,
    AccountRegistry,
    init_account_db,
    migrate_legacy_if_needed,
    slugify,
)

# --- slugify ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Personal", "personal"),
        ("Business Account", "business-account"),
        ("  Personal  ", "personal"),
        ("Mein Konto (Hauptkonto)", "mein-konto-hauptkonto"),
        ("ÄÖÜß umlauts", "umlauts"),               # non-ASCII collapsed
        ("---weird-name---", "weird-name"),
        ("a/b\\c.d", "a-b-c-d"),
        ("", ""),
        ("---", ""),
        ("123 numbers", "123-numbers"),
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_slugify_truncates_at_40_chars() -> None:
    long_name = "a" * 80
    out = slugify(long_name)
    assert len(out) == 40
    assert out == "a" * 40


# --- AccountRegistry: add / get / remove / rename --------------------------


def test_empty_registry(tmp_path: Path) -> None:
    reg = AccountRegistry.load(tmp_path)
    assert reg.all() == []
    assert reg.get("anything") is None
    assert reg.get_active_id() is None


def test_add_assigns_slug_and_default_data_dir(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    info = reg.add("Personal")
    assert info.id == "personal"
    assert info.name == "Personal"
    assert info.data_dir == tmp_path / "accounts" / "personal"
    assert reg.all() == [info]


def test_add_resolves_slug_collisions(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    a = reg.add("Personal")
    b = reg.add("personal")  # different name, same slug -> -2 suffix
    c = reg.add("PERSONAL")  # again -> -3 suffix
    assert a.id == "personal"
    assert b.id == "personal-2"
    assert c.id == "personal-3"


def test_add_rejects_empty_name(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    with pytest.raises(ValueError):
        reg.add("")
    with pytest.raises(ValueError):
        reg.add("   ")


def test_add_rejects_punctuation_only_name(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    with pytest.raises(ValueError):
        reg.add("---")


def test_add_honours_explicit_data_dir(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    custom = tmp_path / "elsewhere" / "biz"
    info = reg.add("Business", data_dir=custom)
    assert info.data_dir == custom


def test_remove_pops_row(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    reg.add("Personal")
    reg.add("Business")
    reg.remove("personal")
    assert [a.id for a in reg.all()] == ["business"]


def test_remove_missing_raises(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    with pytest.raises(AccountNotFoundError):
        reg.remove("ghost")


def test_rename_updates_only_name(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    info = reg.add("Personal")
    renamed = reg.rename(info.id, "Privatkonto")
    assert renamed.id == "personal"            # slug stays
    assert renamed.data_dir == info.data_dir   # directory stays
    assert renamed.name == "Privatkonto"


def test_rename_missing_raises(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    with pytest.raises(AccountNotFoundError):
        reg.rename("ghost", "irrelevant")


def test_rename_rejects_empty_name(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    info = reg.add("Personal")
    with pytest.raises(ValueError):
        reg.rename(info.id, "  ")


def test_get_by_name_or_id_is_case_insensitive(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    info = reg.add("Business")
    assert reg.get_by_name_or_id("business") is info       # slug
    assert reg.get_by_name_or_id("BUSINESS") is info       # name, upper
    assert reg.get_by_name_or_id("Business") is info       # name, exact
    assert reg.get_by_name_or_id("missing") is None


# --- Save / load roundtrip --------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    a = reg.add("Personal")
    b = reg.add("Business", data_dir=tmp_path / "custom_biz")
    reg.save()

    reloaded = AccountRegistry.load(tmp_path)
    assert [r.id for r in reloaded.all()] == [a.id, b.id]
    assert reloaded.get("business").data_dir == tmp_path / "custom_biz"


def test_load_skips_malformed_rows(tmp_path: Path) -> None:
    """A registry written by hand (or by an older version) with missing
    fields should load the valid rows and silently drop the bad ones."""
    (tmp_path / ACCOUNTS_FILENAME).write_text(
        "accounts:\n"
        "  - id: personal\n"
        "    name: Personal\n"
        "    data_dir: /tmp/p\n"
        "  - id: no_name\n"        # missing required field
        "    data_dir: /tmp/n\n"
        "  - just_garbage\n"        # not a mapping at all
        "  - id: business\n"
        "    name: Business\n"
        "    data_dir: /tmp/b\n",
        encoding="utf-8",
    )
    reg = AccountRegistry.load(tmp_path)
    assert [a.id for a in reg.all()] == ["personal", "business"]


def test_save_atomic_writes_via_tmp_then_rename(tmp_path: Path) -> None:
    """Smoke test that we don't leave a stale .tmp file behind on
    success."""
    reg = AccountRegistry(tmp_path, accounts=[])
    reg.add("Personal")
    reg.save()
    leftover_tmps = list(tmp_path.glob(f"{ACCOUNTS_FILENAME}.*.tmp"))
    assert leftover_tmps == []


# --- Active-account I/O ----------------------------------------------------


def test_set_and_get_active_id(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    reg.add("Personal")
    reg.add("Business")
    reg.set_active_id("business")
    assert reg.get_active_id() == "business"
    # The active file is plain text -- spot check.
    assert (tmp_path / ACTIVE_ACCOUNT_FILENAME).read_text(
        encoding="utf-8"
    ).strip() == "business"


def test_set_active_unknown_raises(tmp_path: Path) -> None:
    reg = AccountRegistry(tmp_path, accounts=[])
    reg.add("Personal")
    with pytest.raises(AccountNotFoundError):
        reg.set_active_id("ghost")


def test_get_active_with_stale_slug_returns_none(tmp_path: Path) -> None:
    """If active_account points at a slug that's no longer registered
    (e.g. the user removed it), the registry should fall back to None
    rather than crashing every read."""
    reg = AccountRegistry(tmp_path, accounts=[])
    reg.add("Personal")
    reg.set_active_id("personal")
    reg.remove("personal")
    # File still on disk, slug no longer valid.
    assert (tmp_path / ACTIVE_ACCOUNT_FILENAME).is_file()
    assert reg.get_active_id() is None


# --- migrate_legacy_if_needed ----------------------------------------------


def test_migrate_no_legacy_returns_empty_registry(tmp_path: Path) -> None:
    """Brand-new install: no db.sqlite, no accounts.yaml -> empty."""
    reg = migrate_legacy_if_needed(tmp_path)
    assert reg.all() == []
    assert not (tmp_path / ACCOUNTS_FILENAME).exists()


def test_migrate_legacy_creates_default_pointing_at_global_home(
    tmp_path: Path,
) -> None:
    """Existing single-account install: db.sqlite present, no
    accounts.yaml -> auto-register 'Default' pointing at global home."""
    (tmp_path / LEGACY_DB_FILENAME).write_bytes(b"fake-but-present")
    reg = migrate_legacy_if_needed(tmp_path)
    assert [a.id for a in reg.all()] == ["default"]
    default = reg.get("default")
    assert default is not None
    assert default.name == "Default"
    assert default.data_dir == tmp_path
    assert (tmp_path / ACCOUNTS_FILENAME).is_file()
    # The default account becomes active automatically -- so the user
    # boots straight into their existing data.
    assert reg.get_active_id() == "default"


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / LEGACY_DB_FILENAME).write_bytes(b"x")
    first = migrate_legacy_if_needed(tmp_path)
    second = migrate_legacy_if_needed(tmp_path)
    assert [a.id for a in first.all()] == [a.id for a in second.all()]


def test_migrate_doesnt_touch_existing_registry(tmp_path: Path) -> None:
    """If accounts.yaml already exists, migrate is a strict read."""
    reg = AccountRegistry(tmp_path, accounts=[])
    reg.add("Pre-existing")
    reg.save()
    again = migrate_legacy_if_needed(tmp_path)
    assert [a.id for a in again.all()] == ["pre-existing"]


# --- init_account_db -------------------------------------------------------


def test_init_account_db_creates_dir_and_seeds_categories(tmp_path: Path) -> None:
    info = AccountInfo(
        id="biz", name="Business", data_dir=tmp_path / "biz",
    )
    conn = init_account_db(info, with_defaults=True)
    try:
        assert info.data_dir.is_dir()
        assert info.db_path.is_file()
        n = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
        assert n > 0  # default categories were seeded
    finally:
        conn.close()


def test_init_account_db_without_defaults(tmp_path: Path) -> None:
    info = AccountInfo(
        id="biz", name="Business", data_dir=tmp_path / "biz",
    )
    conn = init_account_db(info, with_defaults=False)
    try:
        n = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
        assert n == 0
    finally:
        conn.close()


def test_init_account_db_is_idempotent(tmp_path: Path) -> None:
    """Re-bootstrapping an existing account shouldn't double-seed."""
    info = AccountInfo(
        id="biz", name="Business", data_dir=tmp_path / "biz",
    )
    conn1 = init_account_db(info, with_defaults=True)
    n1 = conn1.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
    conn1.close()
    conn2 = init_account_db(info, with_defaults=True)
    try:
        n2 = conn2.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
        assert n1 == n2  # upserted, not duplicated
    finally:
        conn2.close()
