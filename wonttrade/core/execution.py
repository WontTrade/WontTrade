"""Execution service translating plans into Hyperliquid API calls."""

from __future__ import annotations

from dataclasses import dataclass

from hyperliquid.exchange import Exchange

from ..models import ActionType, ExecutionAction, ExecutionPlan
from ..telemetry.logger import get_logger


@dataclass(slots=True)
class ExecutionError:
    """Represents a failure to execute a specific action."""

    symbol: str
    message: str


@dataclass(slots=True)
class ExecutionReport:
    """Aggregate execution outcome."""

    attempted: int
    errors: list[ExecutionError]

    @property
    def success(self) -> bool:
        return not self.errors


class ExecutionService:
    """Dispatch execution actions. Real order submission is TODO."""

    def __init__(self, exchange: Exchange):
        self._exchange = exchange
        self._log = get_logger(__name__)

    def execute(self, plan: ExecutionPlan) -> ExecutionReport:
        if plan.is_noop():
            self._log.info("No execution required; plan is empty.")
            return ExecutionReport(attempted=0, errors=[])

        errors: list[ExecutionError] = []
        for action in plan.actions:
            try:
                self._dispatch_stub(action)
            except Exception as exc:  # pragma: no cover - surface runtime issues
                self._log.error("Execution failed for %s: %s", action.symbol, exc)
                errors.append(ExecutionError(symbol=action.symbol, message=str(exc)))
        return ExecutionReport(attempted=len(plan.actions), errors=errors)

    def _dispatch_stub(self, action: ExecutionAction) -> None:
        """Placeholder dispatcher until full execution logic is implemented."""
        if action.action == ActionType.UPSIZE:
            self._log.info(
                "Would upsize %s by %.6f contracts towards target %.6f.",
                action.symbol,
                action.quantity_delta,
                action.target_size,
            )
        elif action.action == ActionType.DOWNSIZE:
            self._log.info(
                "Would downsize %s by %.6f contracts towards target %.6f.",
                action.symbol,
                action.quantity_delta,
                action.target_size,
            )
        elif action.action == ActionType.CLOSE:
            self._log.info(
                "Would close %s position size %.6f.",
                action.symbol,
                action.quantity_delta,
            )
        elif action.action == ActionType.ADJUST_PROTECTION:
            self._log.info(
                "Would adjust protection for %s: stop=%s take=%s.",
                action.symbol,
                action.stop_loss,
                action.take_profit,
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported action {action.action}")
