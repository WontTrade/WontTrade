"""Domain models shared across the trading loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PositionSide(str, Enum):
    """Enumerate long or short exposure."""

    LONG = "long"
    SHORT = "short"


class Candle:
    """Single OHLCV entry."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


@dataclass(slots=True)
class IndicatorWindow:
    """Rolling technical indicators for a single symbol."""

    timeframe: str
    mid_prices: Sequence[float]
    ema: Sequence[float]
    macd: Sequence[float]
    rsi_short: Sequence[float]
    rsi_long: Sequence[float]
    atr_short: Sequence[float] | None = None
    atr_long: Sequence[float] | None = None


@dataclass(slots=True)
class SymbolMarketData:
    """Aggregated market slice for one symbol."""

    symbol: str
    current_price: float
    ema20: float
    macd: float
    rsi_short: float
    open_interest: float
    funding_rate: float
    volume: float
    indicator_window: IndicatorWindow
    long_term_ema20: float | None = None
    long_term_ema50: float | None = None
    long_term_macd: float | None = None
    long_term_rsi: float | None = None
    atr_short: float | None = None
    atr_long: float | None = None


@dataclass(slots=True)
class RawSymbolWindow:
    """Raw market inputs prior to indicator enrichment."""

    symbol: str
    candles_3m: list[Candle]
    candles_4h: list[Candle]
    open_interest: float | None
    funding_rate: float | None
    current_volume: float | None
    average_volume: float | None


@dataclass(slots=True)
class MarketSnapshot:
    """Full market snapshot consumed by the decision engine."""

    captured_at: datetime
    uptime_minutes: float
    invocation_count: int
    sharpe_ratio: float
    symbols: dict[str, SymbolMarketData]


@dataclass(slots=True)
class ProtectionPlan:
    """Simple stop-loss and take-profit container."""

    stop_loss: float | None
    take_profit: float | None


@dataclass(slots=True)
class PositionSnapshot:
    """Current position state for a symbol."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    liquidation_price: float | None
    unrealized_pnl: float
    leverage: float
    side: PositionSide
    confidence: float | None = None
    protection: ProtectionPlan = field(default_factory=lambda: ProtectionPlan(None, None))


@dataclass(slots=True)
class AccountSnapshot:
    """Account balances and open positions."""

    captured_at: datetime
    total_return_percent: float
    available_cash: float
    account_value: float
    positions: list[PositionSnapshot]


@dataclass(slots=True)
class TargetPosition:
    """Desired terminal exposure produced by the LLM."""

    symbol: str
    target_size: float
    stop_loss: float | None
    take_profit: float | None
    confidence: float
    rationale: str | None = None


class ActionType(str, Enum):
    """Supported execution action types."""

    UPSIZE = "upsize"
    DOWNSIZE = "downsize"
    CLOSE = "close"
    ADJUST_PROTECTION = "adjust_protection"


@dataclass(slots=True)
class ExecutionAction:
    """Represents a single execution step to reach the target state."""

    symbol: str
    action: ActionType
    quantity_delta: float
    target_size: float
    stop_loss: float | None
    take_profit: float | None


@dataclass(slots=True)
class ExecutionPlan:
    """Plan comprised of execution actions."""

    actions: list[ExecutionAction] = field(default_factory=list)

    def is_noop(self) -> bool:
        """Return True when no actions are required."""
        return not self.actions
