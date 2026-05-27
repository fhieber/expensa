"""Logging configuration. Console-only by default; quiet for noisy libs."""

from __future__ import annotations

import logging
import os


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose or os.environ.get("EXPENSE_DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Tame the chatty ones.
    for noisy in ("urllib3", "matplotlib", "PIL", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
