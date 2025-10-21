"""Historical data acquisition from Hyperliquid for backtesting."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from hyperliquid.info import Info

from ..models import Candle

THREE_MINUTES = timedelta(minutes=3)
FOUR_HOURS = timedelta(hours=4)
THREE_MINUTES_MS = int(THREE_MINUTES.total_seconds() * 1000)
FOUR_HOURS_MS = int(FOUR_HOURS.total_seconds() * 1000)
MAX_CANDLES_PER_REQUEST = 500
LOOKBACK_3M = 10
LOOKBACK_4H = 10
VOLUME_WINDOW = 480  # 24 hours of 3-minute candles
FUNDING_PADDING = timedelta(hours=12)


@dataclass(slots=True)
class FundingPoint:
    """Single funding observation."""

    timestamp: datetime
    rate: float


@dataclass(slots=True)
class SymbolHistory:
    """Historical slices required for backtesting a single symbol."""

    symbol: str
    candles_3m: list[Candle]
    candles_4h: list[Candle]
    funding: list[FundingPoint]
    _timestamps_3m: list[datetime] = field(init=False, repr=False)
    _timestamps_4h: list[datetime] = field(init=False, repr=False)
    _funding_times: list[datetime] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._timestamps_3m = [candle.timestamp for candle in self.candles_3m]
        self._timestamps_4h = [candle.timestamp for candle in self.candles_4h]
        self._funding_times = [point.timestamp for point in self.funding]

    def index_for_timestamp(self, timestamp: datetime) -> int:
        """Return the index of the candle at or immediately preceding timestamp."""
        idx = bisect_right(self._timestamps_3m, timestamp) - 1
        if idx < 0:
            raise ValueError(f"No 3m candle found before {timestamp} for {self.symbol}.")
        return idx

    def window_3m(self, index: int, length: int) -> list[Candle]:
        start = max(0, index - length + 1)
        return self.candles_3m[start : index + 1]

    def window_4h(self, timestamp: datetime, length: int) -> list[Candle]:
        idx = bisect_right(self._timestamps_4h, timestamp)
        start = max(0, idx - length)
        return self.candles_4h[start:idx]

    def funding_rate_at(self, timestamp: datetime) -> float:
        idx = bisect_right(self._funding_times, timestamp) - 1
        if idx < 0:
            return 0.0
        return self.funding[idx].rate

    def volume_metrics(self, index: int) -> tuple[float, float]:
        current_volume = self.candles_3m[index].volume
        start = max(0, index - VOLUME_WINDOW + 1)
        window = self.candles_3m[start : index + 1]
        average_volume = sum(candle.volume for candle in window) / len(window) if window else 0.0
        return current_volume, average_volume


class HyperliquidHistoricalFetcher:
    """Fetch historical candles and funding rates via the Hyperliquid Info API."""

    def __init__(
        self,
        info: Info,
        *,
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
    ):
        self._info = info
        self._symbols = symbols
        self._start_time = start_time.astimezone(UTC)
        self._end_time = end_time.astimezone(UTC)
        self._start_ms = int(self._start_time.timestamp() * 1000)
        self._end_ms = int(self._end_time.timestamp() * 1000)

    def load(self) -> dict[str, SymbolHistory]:
        histories: dict[str, SymbolHistory] = {}
        for symbol in self._symbols:
            candles_3m = self._fetch_candles(
                symbol,
                interval="3m",
                interval_ms=THREE_MINUTES_MS,
                padding_ms=max(VOLUME_WINDOW, LOOKBACK_3M) * THREE_MINUTES_MS,
            )
            candles_4h = self._fetch_candles(
                symbol,
                interval="4h",
                interval_ms=FOUR_HOURS_MS,
                padding_ms=LOOKBACK_4H * FOUR_HOURS_MS,
            )
            funding = self._fetch_funding(symbol)
            histories[symbol] = SymbolHistory(
                symbol=symbol,
                candles_3m=candles_3m,
                candles_4h=candles_4h,
                funding=funding,
            )
        return histories

    def _fetch_candles(
        self,
        symbol: str,
        *,
        interval: str,
        interval_ms: int,
        padding_ms: int,
    ) -> list[Candle]:
        start_ms = max(0, self._start_ms - padding_ms)
        end_ms = self._end_ms
        candles: dict[int, Candle] = {}
        cursor = start_ms
        while cursor <= end_ms:
            batch_end = min(end_ms, cursor + interval_ms * MAX_CANDLES_PER_REQUEST)
            raw = self._info.candles_snapshot(symbol, interval, cursor, batch_end)
            if not raw:
                cursor = batch_end + interval_ms
                continue
            for entry in raw:
                ts = int(entry["t"])
                if ts > end_ms or ts < start_ms:
                    continue
                candle = Candle(
                    open=float(entry["o"]),
                    high=float(entry["h"]),
                    low=float(entry["l"]),
                    close=float(entry["c"]),
                    volume=float(entry.get("v", 0.0)),
                    timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
                )
                candles[ts] = candle
            cursor = int(raw[-1]["t"]) + interval_ms
        ordered = [candles[key] for key in sorted(candles)]
        if not ordered:
            raise ValueError(f"No candles fetched for {symbol} ({interval}).")
        return ordered

    def _fetch_funding(self, symbol: str) -> list[FundingPoint]:
        start_ms = max(0, self._start_ms - int(FUNDING_PADDING.total_seconds() * 1000))
        raw = self._info.funding_history(symbol, start_ms, self._end_ms)
        points: list[FundingPoint] = []
        for entry in raw:
            timestamp = datetime.fromtimestamp(int(entry["time"]) / 1000, tz=UTC)
            if timestamp > self._end_time:
                continue
            rate = float(entry.get("fundingRate", 0.0))
            points.append(FundingPoint(timestamp=timestamp, rate=rate))
        points.sort(key=lambda item: item.timestamp)
        return points
