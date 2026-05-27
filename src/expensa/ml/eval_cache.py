"""Disk cache for the Quality tab's latest cross-validation result.

A single pickle per account at ``<data_dir>/cache/eval_latest.pkl``
holding the most recent :class:`EvalResult` + :class:`AblationResult`
+ a meta dict + a timestamp. The Quality tab loads it on render when
session state is empty so the user keeps their last run across UI
restarts.

Design choices:

* **No auto-invalidation.** The user said: "no auto-invalidate, user
  can always run from the quality tab". The cache is overwritten by
  the next successful run; that's the only signal we need.
* **Pickle, not JSON.** :class:`EvalResult.confusion` is a numpy
  ``ndarray``; JSON would need a custom encoder. Pickle stays internal
  to this machine, so the usual "don't load untrusted pickle" caveat
  doesn't apply.
* **Schema-version int.** Bumped if :class:`EvalResult` / :class:`AblationResult`
  ever change shape. Old caches load-fail silently and the user sees an
  empty Quality tab until they re-run.
* **Atomic writes.** Same tmp-then-rename pattern as ``accounts.yaml``,
  so a crashed write never half-poisons the cache.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from expensa.ml.evaluation import AblationResult, EvalResult

# Bump if EvalResult / AblationResult layouts change in a way that
# would break old pickles. Failed loads are non-fatal -- the tab just
# starts empty.
_SCHEMA_VERSION = 1


@dataclass
class CachedEval:
    """In-memory shape returned by :func:`load`."""

    result: EvalResult
    ablation: AblationResult | None
    meta: dict[str, Any]
    saved_at: datetime


def _cache_path(data_dir: Path) -> Path:
    return Path(data_dir) / "cache" / "eval_latest.pkl"


def save(
    data_dir: Path,
    result: EvalResult,
    ablation: AblationResult | None,
    meta: dict[str, Any],
) -> Path:
    """Atomically persist the latest eval run for an account.

    Returns the file path so the caller can surface it for debugging.
    Caller is responsible for not passing PII into ``meta``; the
    standard meta dict is ``{"seed": int, "include_zeroshot": bool}``.
    """
    path = _cache_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "result": result,
        "ablation": ablation,
        "meta": dict(meta),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load(data_dir: Path) -> CachedEval | None:
    """Return the cached eval for an account, or None if absent/unreadable.

    Any deserialization failure (missing file, schema-version mismatch,
    pickle corruption, refactored dataclass) returns None silently --
    the Quality tab treats that as "no cached run yet". This is
    deliberate: we never crash the tab over a stale cache.
    """
    path = _cache_path(data_dir)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
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
    result = payload.get("result")
    if not isinstance(result, EvalResult):
        return None
    ablation = payload.get("ablation")
    if ablation is not None and not isinstance(ablation, AblationResult):
        return None
    return CachedEval(
        result=result,
        ablation=ablation,
        meta=dict(payload.get("meta") or {}),
        saved_at=saved_at,
    )


def clear(data_dir: Path) -> bool:
    """Remove the cache file. Returns True if a file was deleted."""
    path = _cache_path(data_dir)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


__all__ = ("CachedEval", "save", "load", "clear")
