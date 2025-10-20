"""Backtest state loader that replays historical data."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from ..core.indicator_engine import IndicatorEngine
from ..core.state_loader import MarketStateBundle
from ..models import MarketSnapshot, RawSymbolWindow
from ..telemetry.logger import get_logger
from .dataset import BacktestRecord, iter_backtest_records
from .simulation import SimulatedExchange


@dataclass(slots=True)
class BacktestReplayProvider:
    """State loader implementation backed by historical data."""

    config: AppConfig
    simulation: SimulatedExchange
    indicator_engine: IndicatorEngine | None = None

    def __post_init__(self) -> None:
        if self.config.backtest is None:
            raise ValueError("Backtest settings are required for BacktestReplayProvider.")
        self._indicator_engine = self.indicator_engine or IndicatorEngine()
        self._records = iter_backtest_records(self.config.backtest.dataset_path)
        self._max_steps = self.config.backtest.max_steps
        self._step = 0
        self._log = get_logger(__name__)

    def close(self) -> None:  # pragma: no cover - generator cleanup
        self._log.debug("BacktestReplayProvider closed.")

    def load(
        self,
        *,
        invocation: int,
        uptime_minutes: float,
        sharpe_ratio: float,
    ) -> MarketStateBundle:
        if self._max_steps is not None and self._step >= self._max_steps:
            raise StopIteration
        try:
            record = next(self._records)
        except StopIteration:
            raise
        self._step += 1

        raw_windows = [self._build_window(symbol, record) for symbol in self.config.symbols]
        market_symbols = {
            window.symbol: self._indicator_engine.enrich(window) for window in raw_windows
        }
        market_snapshot = MarketSnapshot(
            captured_at=record.captured_at,
            uptime_minutes=uptime_minutes,
            invocation_count=invocation,
            sharpe_ratio=sharpe_ratio,
            symbols=market_symbols,
        )

        self.simulation.ingest_prices(market_snapshot)
        account_snapshot = self.simulation.snapshot_account(as_of=record.captured_at)
        return MarketStateBundle(market=market_snapshot, account=account_snapshot)

    def _build_window(self, symbol: str, record: BacktestRecord) -> RawSymbolWindow:
        symbol_record = record.symbols.get(symbol)
        if symbol_record is None:
            raise ValueError(f"Backtest dataset missing symbol '{symbol}'.")
        latest = symbol_record.latest_candle
        if latest:
            self.simulation.evaluate_protections(
                symbol=symbol,
                price_low=latest.low,
                price_high=latest.high,
                timestamp=record.captured_at,
            )
        return RawSymbolWindow(
            symbol=symbol,
            candles_3m=symbol_record.candles_3m,
            candles_4h=symbol_record.candles_4h,
            open_interest=symbol_record.open_interest,
            funding_rate=symbol_record.funding_rate,
            current_volume=symbol_record.current_volume,
            average_volume=symbol_record.average_volume,
        )
