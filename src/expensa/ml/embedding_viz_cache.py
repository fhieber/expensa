"""Per-account disk cache for the category embedding scatter plot.

Mirrors :mod:`eval_cache` (pickle + atomic write + schema-version
gate) so the user's last "Generate" run survives UI restarts. Lives
under ``<account.data_dir>/cache/embedding_viz_latest.pkl``.

Persistence policy is identical to the eval cache:
no auto-invalidation, overwritten on the next Generate click.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from expensa.ml.embedding_viz import ProjectionResult

_SCHEMA_VERSION = 1


@dataclass
class CachedProjection:
    projection: ProjectionResult
    meta: dict[str, Any]
    saved_at: datetime


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / "cache" / "embedding_viz_latest.pkl"


def save(
    data_dir: Path,
    projection: ProjectionResult,
    meta: dict[str, Any],
) -> Path:
    """Persist the latest projection. Caller picks the meta keys
    (typical: ``{"method": "pca", "model_name": "...", "seed": 0}``)."""
    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "projection": projection,
        "meta": dict(meta),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    return p


def load(data_dir: Path) -> CachedProjection | None:
    """Return the cached projection or None. Any deserialization failure
    (missing file, schema-version mismatch, refactored dataclass) is
    treated as "no cache" -- the Categories tab never crashes over a
    stale pickle."""
    p = _path(data_dir)
    if not p.is_file():
        return None
    try:
        with p.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        return None
    try:
        saved_at = datetime.fromisoformat(payload["saved_at"])
    except (KeyError, ValueError, TypeError):
        return None
    proj = payload.get("projection")
    if not isinstance(proj, ProjectionResult):
        return None
    return CachedProjection(
        projection=proj,
        meta=dict(payload.get("meta") or {}),
        saved_at=saved_at,
    )


def clear(data_dir: Path) -> bool:
    """Delete the cache file. Returns True iff a file was removed."""
    p = _path(data_dir)
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False


__all__ = ("CachedProjection", "save", "load", "clear")
