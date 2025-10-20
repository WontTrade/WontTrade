"""Simulation primitives used during backtests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..core.execution import ExecutionError, ExecutionReport
from ..models import (
    AccountSnapshot,
    ActionType,
    ExecutionAction,
    ExecutionPlan,
    FillRecord,
    MarketSnapshot,
    PositionSide,
    PositionSnapshot,
    ProtectionPlan,
)
from ..telemetry.logger import get_logger

EPSILON = 1e-9


@dataclass(slots=True)
class SimulatedPosition:
    """Internal representation of an open position."""

    symbol: str
    quantity: float
    avg_entry_price: float
    protection: ProtectionPlan = field(default_factory=lambda: ProtectionPlan(None, None))


class SimulatedExchange:
    """Tracks portfolio state and fills during backtesting."""

    def __init__(
        self,
        *,
        initial_cash: float,
        fee_bps: float,
        slippage_bps: float,
        results_path: Path,
    ):
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._fee_rate = fee_bps / 10_000
        self._slippage_rate = slippage_bps / 10_000
        self._positions: dict[str, SimulatedPosition] = {}
        self._marks: dict[str, float] = {}
        self._realized_pnl = 0.0
        self._results_path = results_path
        self._results_path.parent.mkdir(parents=True, exist_ok=True)
        self._forced_fills: list[FillRecord] = []
        self._log = get_logger(__name__)

    def ingest_prices(self, market: MarketSnapshot) -> None:
        """Update mark prices using the latest market snapshot."""
        for symbol, data in market.symbols.items():
            self._marks[symbol] = data.current_price

    def evaluate_protections(
        self, *, symbol: str, price_low: float, price_high: float, timestamp: datetime
    ) -> None:
        """Trigger protective exits when price ranges touch configured levels."""
        position = self._positions.get(symbol)
        if position is None:
            return
        if abs(position.quantity) <= EPSILON:
            return

        if position.quantity > 0:
            stop = position.protection.stop_loss
            take = position.protection.take_profit
            if stop is not None and price_low <= stop:
                fill = self._close_position(
                    symbol,
                    abs(position.quantity),
                    stop,
                    "protection:stop_loss",
                    timestamp,
                    apply_slippage=False,
                )
                self._forced_fills.append(fill)
                return
            if take is not None and price_high >= take:
                fill = self._close_position(
                    symbol,
                    abs(position.quantity),
                    take,
                    "protection:take_profit",
                    timestamp,
                    apply_slippage=False,
                )
                self._forced_fills.append(fill)
                return
        else:
            stop = position.protection.stop_loss
            take = position.protection.take_profit
            if stop is not None and price_high >= stop:
                fill = self._close_position(
                    symbol,
                    abs(position.quantity),
                    stop,
                    "protection:stop_loss",
                    timestamp,
                    apply_slippage=False,
                )
                self._forced_fills.append(fill)
                return
            if take is not None and price_low <= take:
                fill = self._close_position(
                    symbol,
                    abs(position.quantity),
                    take,
                    "protection:take_profit",
                    timestamp,
                    apply_slippage=False,
                )
                self._forced_fills.append(fill)

    def snapshot_account(self, *, as_of: datetime) -> AccountSnapshot:
        """Return the current account snapshot."""
        positions: list[PositionSnapshot] = []
        equity = self._cash
        for symbol, position in self._positions.items():
            mark = self._marks.get(symbol, position.avg_entry_price)
            unrealized = (mark - position.avg_entry_price) * position.quantity
            equity += mark * position.quantity
            positions.append(
                PositionSnapshot(
                    symbol=symbol,
                    quantity=position.quantity,
                    entry_price=position.avg_entry_price,
                    current_price=mark,
                    liquidation_price=None,
                    unrealized_pnl=unrealized,
                    leverage=0.0,
                    side=PositionSide.LONG if position.quantity >= 0 else PositionSide.SHORT,
                    protection=position.protection,
                )
            )
        total_return = 0.0
        if self._initial_cash:
            total_return = ((equity - self._initial_cash) / self._initial_cash) * 100
        return AccountSnapshot(
            captured_at=as_of,
            total_return_percent=total_return,
            available_cash=self._cash,
            account_value=equity,
            positions=positions,
        )

    def consume_forced_fills(self) -> list[FillRecord]:
        """Return and clear protection-triggered fills."""
        fills = list(self._forced_fills)
        self._forced_fills.clear()
        return fills

    def apply_plan(
        self, plan: ExecutionPlan, market: MarketSnapshot
    ) -> tuple[list[FillRecord], list[ExecutionError]]:
        """Execute the supplied plan against the simulated portfolio."""
        fills = []
        errors: list[ExecutionError] = []
        for action in plan.actions:
            try:
                fills.extend(self._apply_action(action, market))
            except Exception as exc:  # pragma: no cover - deterministic execution expected
                errors.append(ExecutionError(symbol=action.symbol, message=str(exc)))
                self._log.error("Simulation failed for %s: %s", action.symbol, exc)
        return fills, errors

    def _apply_action(self, action: ExecutionAction, market: MarketSnapshot) -> list[FillRecord]:
        data = market.symbols.get(action.symbol)
        if data is None:
            raise ValueError(f"Missing market data for {action.symbol}")
        timestamp = market.captured_at
        if action.action is ActionType.ADJUST_PROTECTION:
            position = self._positions.get(action.symbol)
            if position:
                if action.stop_loss is not None or action.take_profit is not None:
                    stop = (
                        action.stop_loss
                        if action.stop_loss is not None
                        else position.protection.stop_loss
                    )
                    take = (
                        action.take_profit
                        if action.take_profit is not None
                        else position.protection.take_profit
                    )
                    position.protection = ProtectionPlan(stop, take)
            return []
        if action.action is ActionType.CLOSE:
            return [
                self._close_position(
                    action.symbol,
                    abs(action.quantity_delta),
                    data.current_price,
                    "manual:close",
                    timestamp,
                )
            ]
        if action.action is ActionType.UPSIZE:
            return self._buy(
                symbol=action.symbol,
                quantity=action.quantity_delta,
                mark_price=data.current_price,
                stop_loss=action.stop_loss,
                take_profit=action.take_profit,
                timestamp=timestamp,
                apply_slippage=True,
            )
        if action.action is ActionType.DOWNSIZE:
            return self._sell(
                symbol=action.symbol,
                quantity=action.quantity_delta,
                mark_price=data.current_price,
                stop_loss=action.stop_loss,
                take_profit=action.take_profit,
                timestamp=timestamp,
                apply_slippage=True,
            )
        raise ValueError(f"Unsupported execution action {action.action}")

    def _close_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        reason: str,
        timestamp: datetime,
        apply_slippage: bool,
    ) -> FillRecord:
        position = self._positions.get(symbol)
        if position is None or abs(position.quantity) <= EPSILON:
            return FillRecord(
                symbol=symbol,
                action=ActionType.CLOSE,
                quantity=0.0,
                price=price,
                fee=0.0,
                pnl=0.0,
                reason=reason,
            )

        actual_quantity = min(quantity, abs(position.quantity))
        action = ActionType.CLOSE
        if position.quantity > 0:
            fill = self._sell(
                symbol=symbol,
                quantity=actual_quantity,
                mark_price=price,
                stop_loss=None,
                take_profit=None,
                timestamp=timestamp,
                apply_slippage=apply_slippage,
                reason_override=reason,
            )
        else:
            fill = self._buy(
                symbol=symbol,
                quantity=actual_quantity,
                mark_price=price,
                stop_loss=None,
                take_profit=None,
                timestamp=timestamp,
                apply_slippage=apply_slippage,
                reason_override=reason,
            )
        # _sell/_buy return list; when called for closure, take first element.
        return (
            fill[0]
            if fill
            else FillRecord(
                symbol=symbol,
                action=action,
                quantity=0.0,
                price=price,
                fee=0.0,
                pnl=0.0,
                reason=reason,
            )
        )

    def _buy(
        self,
        *,
        symbol: str,
        quantity: float,
        mark_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        timestamp: datetime,
        reason_override: str | None = None,
        apply_slippage: bool = True,
    ) -> list[FillRecord]:
        if quantity <= 0:
            return []
        if apply_slippage:
            fill_price = mark_price * (1 + self._slippage_rate)
        else:
            fill_price = mark_price
        total_value = fill_price * quantity
        total_fee = total_value * self._fee_rate
        self._cash -= total_value + total_fee

        position = self._positions.get(symbol)
        fills: list[FillRecord] = []
        remaining = quantity

        if position and position.quantity < 0:
            closing = min(remaining, abs(position.quantity))
            if closing > 0:
                realized = (position.avg_entry_price - fill_price) * closing
                position.quantity += closing
                self._realized_pnl += realized
                allocated_fee = total_fee * (closing / quantity)
                fills.append(
                    self._log_fill(
                        timestamp,
                        FillRecord(
                            symbol=symbol,
                            action=ActionType.UPSIZE,
                            quantity=closing,
                            price=fill_price,
                            fee=allocated_fee,
                            pnl=realized,
                            reason=reason_override or "manual:buy_cover",
                        ),
                    )
                )
                remaining -= closing
                if abs(position.quantity) <= EPSILON:
                    del self._positions[symbol]
                    position = None

        if remaining > 0:
            allocated_fee = total_fee * (remaining / quantity)
            if position is None:
                position = SimulatedPosition(
                    symbol=symbol,
                    quantity=remaining,
                    avg_entry_price=fill_price,
                    protection=self._build_protection(stop_loss, take_profit),
                )
                self._positions[symbol] = position
            else:
                new_quantity = position.quantity + remaining
                position.avg_entry_price = (
                    position.avg_entry_price * position.quantity + fill_price * remaining
                ) / new_quantity
                position.quantity = new_quantity
                position.protection = self._merge_protection(
                    position.protection,
                    stop_loss,
                    take_profit,
                )
            fills.append(
                self._log_fill(
                    timestamp,
                    FillRecord(
                        symbol=symbol,
                        action=ActionType.UPSIZE,
                        quantity=remaining,
                        price=fill_price,
                        fee=allocated_fee,
                        pnl=0.0,
                        reason=reason_override or "manual:buy_open",
                    ),
                )
            )
        return fills

    def _sell(
        self,
        *,
        symbol: str,
        quantity: float,
        mark_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        timestamp: datetime,
        reason_override: str | None = None,
        apply_slippage: bool = True,
    ) -> list[FillRecord]:
        if quantity <= 0:
            return []
        if apply_slippage:
            fill_price = mark_price * (1 - self._slippage_rate)
        else:
            fill_price = mark_price
        total_value = fill_price * quantity
        total_fee = total_value * self._fee_rate
        self._cash += total_value - total_fee

        position = self._positions.get(symbol)
        fills: list[FillRecord] = []
        remaining = quantity

        if position and position.quantity > 0:
            closing = min(remaining, position.quantity)
            if closing > 0:
                realized = (fill_price - position.avg_entry_price) * closing
                position.quantity -= closing
                self._realized_pnl += realized
                allocated_fee = total_fee * (closing / quantity)
                fills.append(
                    self._log_fill(
                        timestamp,
                        FillRecord(
                            symbol=symbol,
                            action=ActionType.DOWNSIZE,
                            quantity=closing,
                            price=fill_price,
                            fee=allocated_fee,
                            pnl=realized,
                            reason=reason_override or "manual:sell_close",
                        ),
                    )
                )
                remaining -= closing
                if position.quantity <= EPSILON:
                    del self._positions[symbol]
                    position = None

        if remaining > 0:
            allocated_fee = total_fee * (remaining / quantity)
            if position is None:
                position = SimulatedPosition(
                    symbol=symbol,
                    quantity=-remaining,
                    avg_entry_price=fill_price,
                    protection=self._build_protection(stop_loss, take_profit),
                )
                self._positions[symbol] = position
            else:
                new_quantity = position.quantity - remaining
                position.avg_entry_price = (
                    position.avg_entry_price * position.quantity - fill_price * remaining
                ) / new_quantity
                position.quantity = new_quantity
                position.protection = self._merge_protection(
                    position.protection,
                    stop_loss,
                    take_profit,
                )
            fills.append(
                self._log_fill(
                    timestamp,
                    FillRecord(
                        symbol=symbol,
                        action=ActionType.DOWNSIZE,
                        quantity=remaining,
                        price=fill_price,
                        fee=allocated_fee,
                        pnl=0.0,
                        reason=reason_override or "manual:sell_open",
                    ),
                )
            )
        return fills

    def _build_protection(
        self,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> ProtectionPlan:
        return ProtectionPlan(
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    @staticmethod
    def _merge_protection(
        existing: ProtectionPlan,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> ProtectionPlan:
        return ProtectionPlan(
            stop_loss=stop_loss if stop_loss is not None else existing.stop_loss,
            take_profit=take_profit if take_profit is not None else existing.take_profit,
        )

    def _log_fill(self, timestamp: datetime, fill: FillRecord) -> FillRecord:
        record = {
            "timestamp": timestamp.astimezone(UTC).isoformat(),
            "symbol": fill.symbol,
            "action": fill.action.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "fee": fill.fee,
            "pnl": fill.pnl,
            "reason": fill.reason,
            "cash": self._cash,
            "realized_pnl": self._realized_pnl,
        }
        try:
            with self._results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record))
                handle.write("\n")
        except Exception as exc:  # pragma: no cover - telemetry best effort
            self._log.error("Failed to write backtest result: %s", exc)
        return fill


@dataclass(slots=True)
class SimulatedExecutor:
    """Execution implementation that routes orders to the simulated exchange."""

    simulation: SimulatedExchange

    def execute(self, plan: ExecutionPlan, market: MarketSnapshot) -> ExecutionReport:
        forced_fills = self.simulation.consume_forced_fills()
        if plan.is_noop():
            return ExecutionReport(attempted=0, errors=[], fills=forced_fills)

        fills, errors = self.simulation.apply_plan(plan, market)
        all_fills = forced_fills + fills
        return ExecutionReport(
            attempted=len(plan.actions),
            errors=errors,
            fills=all_fills,
        )
