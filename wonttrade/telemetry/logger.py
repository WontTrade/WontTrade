"""Logging utilities for the trading daemon backed by Logfire."""

from __future__ import annotations

import threading
from typing import Any, Protocol

import logfire

_LOGGER_INITIALIZED = False
_LOCK = threading.Lock()

_DEFAULT_LOGGER = logfire.DEFAULT_LOGFIRE_INSTANCE.with_tags("wonttrade")

class _SupportsLog(Protocol):
    def debug(self, message: str, *args: Any, **values: Any) -> None: ...

    def info(self, message: str, *args: Any, **values: Any) -> None: ...

    def warning(self, message: str, *args: Any, **values: Any) -> None: ...

    def error(self, message: str, *args: Any, **values: Any) -> None: ...

    def exception(self, message: str, *args: Any, **values: Any) -> None: ...


class LogfireLogger(_SupportsLog):
    """Adapter that mimics the standard logging interface."""

    __slots__ = ("_logger",)

    def __init__(self, name: str | None) -> None:
        logger_name = name or "wonttrade"
        self._logger = _DEFAULT_LOGGER.with_tags(logger_name)

    def debug(self, message: str, *args: Any, **values: Any) -> None:
        self._emit("debug", message, args, values)

    def info(self, message: str, *args: Any, **values: Any) -> None:
        self._emit("info", message, args, values)

    def warning(self, message: str, *args: Any, **values: Any) -> None:
        self._emit("warning", message, args, values)

    def error(self, message: str, *args: Any, **values: Any) -> None:
        self._emit("error", message, args, values)

    def exception(self, message: str, *args: Any, **values: Any) -> None:
        self._emit("exception", message, args, values)

    def _emit(self, level: str, message: str, args: tuple[Any, ...], values: dict[str, Any]) -> None:
        formatted = _format(message, args)
        attributes = dict(values)
        exc_info = attributes.pop("exc_info", None)
        log_method = getattr(self._logger, level)
        if exc_info is None:
            log_method(formatted, **attributes)
        else:
            log_method(formatted, _exc_info=exc_info, **attributes)


def _format(message: str, args: tuple[Any, ...]) -> str:
    if not args:
        return message
    try:
        return message % args
    except Exception:  # pragma: no cover - defensive formatting fallback
        rendered = " ".join(str(arg) for arg in args)
        return f"{message} {rendered}".strip()


def _configure_logging() -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    with _LOCK:
        if _LOGGER_INITIALIZED:
            return
        logfire.configure(
            service="wonttrade",
            send_to_logfire=False,
        )
        logfire.instrument_pydantic_ai(include_content=True)
        _LOGGER_INITIALIZED = True


def get_logger(name: str | None = None) -> LogfireLogger:
    """Return a module-level logger backed by Logfire."""
    _configure_logging()
    return LogfireLogger(name)
