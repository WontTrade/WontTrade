"""Reconcile desired targets with current positions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..models import (
    AccountSnapshot,
    ActionType,
    ExecutionAction,
    ExecutionPlan,
    PositionSnapshot,
    TargetPosition,
)

EPSILON = 1e-6


@dataclass(slots=True)
class ReconciliationResult:
    """Outcome of reconciliation before execution."""

    plan: ExecutionPlan
    untouched_symbols: list[str]


class PositionReconciler:
    """Translate target positions into actionable execution steps."""

    def build_plan(
        self,
        account: AccountSnapshot,
        targets: Iterable[TargetPosition],
    ) -> ReconciliationResult:
        current_positions: dict[str, PositionSnapshot] = {
            pos.symbol: pos for pos in account.positions
        }
        actions: list[ExecutionAction] = []
        untouched: list[str] = []

        processed_symbols: set[str] = set()

        for target in targets:
            processed_symbols.add(target.symbol)
            current = current_positions.get(target.symbol)
            current_size = current.quantity if current else 0.0
            delta = target.target_size - current_size

            if abs(delta) <= EPSILON:
                if self._requires_protection_update(current, target):
                    actions.append(
                        ExecutionAction(
                            symbol=target.symbol,
                            action=ActionType.ADJUST_PROTECTION,
                            quantity_delta=0.0,
                            target_size=target.target_size,
                            stop_loss=target.stop_loss,
                            take_profit=target.take_profit,
                        )
                    )
                else:
                    untouched.append(target.symbol)
                continue

            if target.target_size == 0 and current_size != 0:
                actions.append(
                    ExecutionAction(
                        symbol=target.symbol,
                        action=ActionType.CLOSE,
                        quantity_delta=abs(current_size),
                        target_size=0.0,
                        stop_loss=None,
                        take_profit=None,
                    )
                )
                continue

            if delta > 0:
                actions.append(
                    ExecutionAction(
                        symbol=target.symbol,
                        action=ActionType.UPSIZE,
                        quantity_delta=delta,
                        target_size=target.target_size,
                        stop_loss=target.stop_loss,
                        take_profit=target.take_profit,
                    )
                )
            else:
                actions.append(
                    ExecutionAction(
                        symbol=target.symbol,
                        action=ActionType.DOWNSIZE,
                        quantity_delta=abs(delta),
                        target_size=target.target_size,
                        stop_loss=target.stop_loss,
                        take_profit=target.take_profit,
                    )
                )

        for symbol, position in current_positions.items():
            if symbol not in processed_symbols and abs(position.quantity) > EPSILON:
                actions.append(
                    ExecutionAction(
                        symbol=symbol,
                        action=ActionType.CLOSE,
                        quantity_delta=abs(position.quantity),
                        target_size=0.0,
                        stop_loss=None,
                        take_profit=None,
                    )
                )

        return ReconciliationResult(
            plan=ExecutionPlan(actions=actions), untouched_symbols=untouched
        )

    @staticmethod
    def _requires_protection_update(
        current: PositionSnapshot | None, target: TargetPosition
    ) -> bool:
        if current is None:
            return target.stop_loss is not None or target.take_profit is not None
        current_stop = current.protection.stop_loss
        current_take = current.protection.take_profit
        return (target.stop_loss is not None and target.stop_loss != current_stop) or (
            target.take_profit is not None and target.take_profit != current_take
        )
