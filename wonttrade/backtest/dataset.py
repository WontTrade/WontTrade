"""Backtest dataset ingestion utilities."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..models import Candle


def _parse_timestamp(value: str) -> datetime:
    normalized = value.rstrip("Z")
    if normalized != value:
        return datetime.fromisoformat(f"{normalized}+00:00").astimezone(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_candle(payload: dict[str, object]) -> Candle:
    return Candle(
        open=float(payload["open"]),
        high=float(payload["high"]),
        low=float(payload["low"]),
        close=float(payload["close"]),
        volume=float(payload.get("volume", 0.0) or 0.0),
        timestamp=_parse_timestamp(str(payload["timestamp"])),
    )


@dataclass(slots=True)
class BacktestSymbolRecord:
    """Historical inputs for a single symbol at a given timestep."""

    symbol: str
    candles_3m: list[Candle]
    candles_4h: list[Candle]
    open_interest: float | None
    funding_rate: float | None
    current_volume: float | None
    average_volume: float | None

    @property
    def latest_candle(self) -> Candle | None:
        return self.candles_3m[-1] if self.candles_3m else None


@dataclass(slots=True)
class BacktestRecord:
    """Single chronological record in the backtest dataset."""

    captured_at: datetime
    symbols: dict[str, BacktestSymbolRecord]


def iter_backtest_records(path: Path) -> Iterator[BacktestRecord]:
    """Yield dataset records in chronological order."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            captured_at = _parse_timestamp(payload["captured_at"])
            symbols_section = payload.get("symbols", {})
            symbols: dict[str, BacktestSymbolRecord] = {}
            for symbol, symbol_payload in symbols_section.items():
                candles_3m = [
                    _parse_candle(entry) for entry in symbol_payload.get("candles_3m", [])
                ]
                candles_4h = [
                    _parse_candle(entry) for entry in symbol_payload.get("candles_4h", [])
                ]
                symbols[symbol] = BacktestSymbolRecord(
                    symbol=symbol,
                    candles_3m=candles_3m,
                    candles_4h=candles_4h,
                    open_interest=_to_optional_float(symbol_payload.get("open_interest")),
                    funding_rate=_to_optional_float(symbol_payload.get("funding_rate")),
                    current_volume=_to_optional_float(symbol_payload.get("current_volume")),
                    average_volume=_to_optional_float(symbol_payload.get("average_volume")),
                )
            yield BacktestRecord(captured_at=captured_at, symbols=symbols)


def _to_optional_float(value: object) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)
