"""Logging utilities for the trading daemon."""

from __future__ import annotations

import logging

_LOGGER_INITIALIZED = False


def _configure_logging() -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    _LOGGER_INITIALIZED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module-level logger with default configuration."""
    _configure_logging()
    return logging.getLogger(name)
