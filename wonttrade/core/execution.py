"""Execution service translating plans into Hyperliquid API calls."""

from __future__ import annotations

from dataclasses import dataclass, field

from hyperliquid.exchange import Exchange

from ..models import ActionType, ExecutionAction, ExecutionPlan, FillRecord, MarketSnapshot
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
    fills: list[FillRecord] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


class ExecutionService:
    """Dispatch execution actions. Real order submission is TODO."""

    def __init__(self, exchange: Exchange):
        self._exchange = exchange
        self._log = get_logger(__name__)

    def execute(self, plan: ExecutionPlan, market: MarketSnapshot | None = None) -> ExecutionReport:
        if plan.is_noop():
            self._log.info("本轮无需执行，计划为空。")
            return ExecutionReport(attempted=0, errors=[], fills=[])

        errors: list[ExecutionError] = []
        for action in plan.actions:
            try:
                self._dispatch_stub(action)
            except Exception as exc:  # pragma: no cover - surface runtime issues
                self._log.error("执行 %s 时失败：%s", action.symbol, exc)
                errors.append(ExecutionError(symbol=action.symbol, message=str(exc)))
        return ExecutionReport(attempted=len(plan.actions), errors=errors, fills=[])

    def _dispatch_stub(self, action: ExecutionAction) -> None:
        """Placeholder dispatcher until full execution logic is implemented."""
        if action.action == ActionType.UPSIZE:
            self._log.info(
                "计划加仓 %s，增量 %.6f，目标仓位 %.6f。",
                action.symbol,
                action.quantity_delta,
                action.target_size,
            )
        elif action.action == ActionType.DOWNSIZE:
            self._log.info(
                "计划减仓 %s，减量 %.6f，目标仓位 %.6f。",
                action.symbol,
                action.quantity_delta,
                action.target_size,
            )
        elif action.action == ActionType.CLOSE:
            self._log.info(
                "计划平仓 %s，数量 %.6f。",
                action.symbol,
                action.quantity_delta,
            )
        elif action.action == ActionType.ADJUST_PROTECTION:
            self._log.info(
                "计划调整 %s 保护单：止损=%s，止盈=%s。",
                action.symbol,
                action.stop_loss,
                action.take_profit,
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported action {action.action}")
