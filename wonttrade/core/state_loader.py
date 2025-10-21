"""State loading and market snapshot aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from hyperliquid.info import Info

from ..config import AppConfig
from ..models import (
    AccountSnapshot,
    Candle,
    MarketSnapshot,
    PositionSide,
    PositionSnapshot,
    ProtectionPlan,
    RawSymbolWindow,
    SymbolMarketData,
)
from ..telemetry.logger import get_logger
from .indicator_engine import IndicatorEngine

THREE_MINUTES_MS = 3 * 60 * 1000
FOUR_HOURS_MS = 4 * 60 * 60 * 1000


@dataclass(slots=True)
class MarketStateBundle:
    """Container bundling market and account snapshots."""

    market: MarketSnapshot
    account: AccountSnapshot


class StateLoader:
    """Load market and account state from Hyperliquid."""

    def __init__(
        self, config: AppConfig, info_client: Info, indicator_engine: IndicatorEngine | None = None
    ):
        self._config = config
        self._info = info_client
        self._log = get_logger(__name__)
        self._indicator_engine = indicator_engine or IndicatorEngine()

    def close(self) -> None:
        """Clean up any background resources."""
        try:
            self._info.disconnect_websocket()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            self._log.debug("断开 Hyperliquid WebSocket 失败：%s", exc)

    def load(
        self, *, invocation: int, uptime_minutes: float, sharpe_ratio: float
    ) -> MarketStateBundle:
        """Fetch market and account snapshots."""
        now = datetime.now(tz=UTC)
        raw_windows = self._collect_symbol_windows(now)
        market_symbols: dict[str, SymbolMarketData] = {}
        for window in raw_windows:
            try:
                enriched = self._build_symbol_data(window)
                market_symbols[window.symbol] = enriched
            except Exception as exc:
                self._log.exception("计算 %s 指标失败", window.symbol)
                raise RuntimeError(f"Indicator enrichment failed for {window.symbol}") from exc

        market_snapshot = MarketSnapshot(
            captured_at=now,
            uptime_minutes=uptime_minutes,
            invocation_count=invocation,
            sharpe_ratio=sharpe_ratio,
            symbols=market_symbols,
        )

        account_snapshot = self._load_account_snapshot(now)

        return MarketStateBundle(market=market_snapshot, account=account_snapshot)

    def _collect_symbol_windows(self, now: datetime) -> list[RawSymbolWindow]:
        meta, asset_ctxs = self._info.meta_and_asset_ctxs()
        asset_lookup = {
            asset_meta["name"]: ctx
            for asset_meta, ctx in zip(meta["universe"], asset_ctxs, strict=True)
        }
        windows: list[RawSymbolWindow] = []
        for symbol in self._config.symbols:
            ctx = asset_lookup.get(symbol)
            candles_3m = self._fetch_candles(symbol, interval="3m", count=10, now=now)
            candles_4h = self._fetch_candles(symbol, interval="4h", count=10, now=now)
            windows.append(
                RawSymbolWindow(
                    symbol=symbol,
                    candles_3m=candles_3m,
                    candles_4h=candles_4h,
                    open_interest=float(ctx["openInterest"])
                    if ctx and ctx.get("openInterest")
                    else None,
                    funding_rate=float(ctx["funding"]) if ctx and ctx.get("funding") else None,
                    current_volume=float(ctx["dayBaseVlm"])
                    if ctx and ctx.get("dayBaseVlm")
                    else None,
                    average_volume=float(ctx["dayNtlVlm"])
                    if ctx and ctx.get("dayNtlVlm")
                    else None,
                )
            )
        return windows

    def _fetch_candles(
        self, symbol: str, *, interval: str, count: int, now: datetime
    ) -> list[Candle]:
        end_ms = int(now.timestamp() * 1000)
        if interval == "3m":
            delta_ms = THREE_MINUTES_MS
        elif interval == "4h":
            delta_ms = FOUR_HOURS_MS
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        start_ms = end_ms - delta_ms * count
        candles_raw = self._info.candles_snapshot(symbol, interval, start_ms, end_ms)
        return [self._convert_candle(entry) for entry in candles_raw[-count:]]

    @staticmethod
    def _convert_candle(entry: dict[str, str]) -> Candle:
        return Candle(
            open=float(entry["o"]),
            high=float(entry["h"]),
            low=float(entry["l"]),
            close=float(entry["c"]),
            volume=float(entry.get("v", 0.0)),
            timestamp=datetime.fromtimestamp(entry["t"] / 1000, tz=UTC),
        )

    def _build_symbol_data(self, window: RawSymbolWindow) -> SymbolMarketData:
        return self._indicator_engine.enrich(window)

    def _load_account_snapshot(self, now: datetime) -> AccountSnapshot:
        state = self._info.user_state(self._config.account_address)
        positions = [self._convert_position(entry) for entry in state.get("assetPositions", [])]
        margin_summary = state.get("marginSummary", {})
        free_collateral = margin_summary.get("freeCollateral")
        available_cash = float(
            free_collateral if free_collateral is not None else state.get("withdrawable", 0.0)
        )
        account_value = float(margin_summary.get("accountValue", state.get("equity", 0.0)))
        total_return_percent = 0.0  # Placeholder until performance tracking is wired
        return AccountSnapshot(
            captured_at=now,
            total_return_percent=total_return_percent,
            available_cash=available_cash,
            account_value=account_value,
            positions=positions,
        )

    def _convert_position(self, entry: dict[str, object]) -> PositionSnapshot:
        position = entry.get("position", {})
        quantity = float(position.get("szi", 0.0) or 0.0)
        position_value = float(position.get("positionValue", 0.0) or 0.0)
        current_price = (
            position_value / quantity if quantity else float(position.get("oraclePx", 0.0) or 0.0)
        )
        leverage_info = position.get("leverage", {})
        leverage_value = (
            float(leverage_info.get("value", 0.0) or 0.0)
            if isinstance(leverage_info, dict)
            else 0.0
        )
        liquidation_raw = position.get("liquidationPx")
        liquidation_price = float(liquidation_raw) if liquidation_raw not in (None, "") else None
        entry_px_raw = position.get("entryPx")
        unrealized_raw = position.get("unrealizedPnl")
        margin_raw = position.get("marginUsed")
        margin_value = float(margin_raw) if margin_raw not in (None, "") else None

        return PositionSnapshot(
            symbol=str(position.get("coin", "")),
            quantity=quantity,
            entry_price=float(entry_px_raw) if entry_px_raw not in (None, "") else 0.0,
            current_price=current_price,
            liquidation_price=liquidation_price,
            unrealized_pnl=float(unrealized_raw) if unrealized_raw not in (None, "") else 0.0,
            leverage=leverage_value,
            side=PositionSide.LONG if quantity >= 0 else PositionSide.SHORT,
            protection=ProtectionPlan(stop_loss=None, take_profit=None),
            margin=margin_value,
        )
