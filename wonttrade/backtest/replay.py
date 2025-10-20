"""Backtest state loader that replays historical data."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime

from hyperliquid.info import Info

from ..config import AppConfig
from ..core.indicator_engine import IndicatorEngine
from ..core.state_loader import MarketStateBundle
from ..models import MarketSnapshot, RawSymbolWindow
from ..telemetry.logger import get_logger
from .historical import LOOKBACK_3M, LOOKBACK_4H, HyperliquidHistoricalFetcher
from .simulation import SimulatedExchange


@dataclass(slots=True)
class BacktestReplayProvider:
    """State loader implementation backed by historical data."""

    config: AppConfig
    simulation: SimulatedExchange
    info_client: Info
    indicator_engine: IndicatorEngine | None = None

    def __post_init__(self) -> None:
        if self.config.backtest is None:
            raise ValueError("Backtest settings are required for BacktestReplayProvider.")
        self._indicator_engine = self.indicator_engine or IndicatorEngine()
        self._log = get_logger(__name__)
        self._history = HyperliquidHistoricalFetcher(
            self.info_client,
            symbols=self.config.symbols,
            start_time=self.config.backtest.start_time,
            end_time=self.config.backtest.end_time,
        ).load()
        base_symbol = self.config.symbols[0]
        base_history = self._history[base_symbol]
        timestamps = [candle.timestamp for candle in base_history.candles_3m]
        start_idx = bisect_left(timestamps, self.config.backtest.start_time)
        end_idx = bisect_right(timestamps, self.config.backtest.end_time)
        self._timeline = timestamps[start_idx:end_idx]
        if not self._timeline:
            raise ValueError("No 3m candles available within the requested backtest window.")
        max_steps = self.config.backtest.max_steps
        if max_steps is not None:
            self._timeline = self._timeline[:max_steps]
        self._step = 0

    def close(self) -> None:  # pragma: no cover - generator cleanup
        try:
            self.info_client.session.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            self._log.debug("Failed to close Hyperliquid session: %s", exc)
        self._log.debug("BacktestReplayProvider closed.")

    def load(
        self,
        *,
        invocation: int,
        uptime_minutes: float,
        sharpe_ratio: float,
    ) -> MarketStateBundle:
        if self._step >= len(self._timeline):
            raise StopIteration
        current_timestamp = self._timeline[self._step]
        self._step += 1

        raw_windows = [
            self._build_window(symbol, current_timestamp) for symbol in self.config.symbols
        ]
        market_symbols = {
            window.symbol: self._indicator_engine.enrich(window) for window in raw_windows
        }
        market_snapshot = MarketSnapshot(
            captured_at=current_timestamp,
            uptime_minutes=uptime_minutes,
            invocation_count=invocation,
            sharpe_ratio=sharpe_ratio,
            symbols=market_symbols,
        )

        self.simulation.ingest_prices(market_snapshot)
        account_snapshot = self.simulation.snapshot_account(as_of=current_timestamp)
        return MarketStateBundle(market=market_snapshot, account=account_snapshot)

    def _build_window(self, symbol: str, timestamp: datetime) -> RawSymbolWindow:
        history = self._history[symbol]
        index = history.index_for_timestamp(timestamp)
        candles_3m = history.window_3m(index, LOOKBACK_3M)
        candles_4h = history.window_4h(timestamp, LOOKBACK_4H)
        current_volume, average_volume = history.volume_metrics(index)
        latest = history.candles_3m[index]
        self.simulation.evaluate_protections(
            symbol=symbol,
            price_low=latest.low,
            price_high=latest.high,
            timestamp=timestamp,
        )
        return RawSymbolWindow(
            symbol=symbol,
            candles_3m=candles_3m,
            candles_4h=candles_4h,
            open_interest=0.0,
            funding_rate=history.funding_rate_at(timestamp),
            current_volume=current_volume,
            average_volume=average_volume,
        )
