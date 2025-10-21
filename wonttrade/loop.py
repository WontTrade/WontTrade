"""Main trading loop orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import AppConfig
from .core.execution import ExecutionService
from .core.reconciler import PositionReconciler
from .core.state_loader import MarketStateBundle, StateLoader
from .llm.decision_engine import LLMDecisionEngine
from .models import AccountSnapshot, DecisionResult
from .telemetry.audit import AuditSink
from .telemetry.logger import get_logger


@dataclass(slots=True)
class TradingLoop:
    """Coordinates the periodic trading workflow."""

    config: AppConfig
    state_loader: StateLoader
    decision_engine: LLMDecisionEngine
    reconciler: PositionReconciler
    executor: ExecutionService
    audit_sink: AuditSink
    _start_time: datetime = field(init=False, repr=False)
    _invocation: int = field(init=False, repr=False)
    _sharpe_ratio: float = field(init=False, repr=False)
    _log: Any = field(init=False, repr=False)
    _last_decision: DecisionResult | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._start_time = datetime.now(tz=UTC)
        self._invocation = 0
        self._sharpe_ratio = 0.0
        self._log = get_logger(__name__)
        self._last_decision = None

    def run_forever(self) -> None:
        try:
            while True:
                try:
                    self._run_once()
                except StopIteration:
                    self._log.info("状态加载器已无更多数据，事件循环结束。")
                    break
                except Exception as exc:  # pragma: no cover - production resilience
                    self._log.exception("单次循环执行失败：%s", exc)
                    self.audit_sink.write_heartbeat(
                        "error",
                        details={
                            "invocation": self._invocation,
                            "reason": f"iteration_error: {exc}",
                        },
                    )
                time.sleep(self.config.loop_interval_seconds)
        finally:
            self.state_loader.close()

    def run_backtest(self) -> None:
        try:
            while True:
                try:
                    self._run_once()
                except StopIteration:
                    self._log.info("回测结束，共执行 %s 次循环。", self._invocation)
                    uptime_delta = datetime.now(tz=UTC) - self._start_time
                    self.audit_sink.write_heartbeat(
                        "completed",
                        details={
                            "invocation": self._invocation,
                            "uptime_minutes": uptime_delta.total_seconds() / 60,
                        },
                    )
                    break
                except Exception as exc:  # pragma: no cover - resilience
                    self._log.exception("回测循环失败：%s", exc)
                    self.audit_sink.write_heartbeat(
                        "error",
                        details={
                            "invocation": self._invocation,
                            "reason": f"iteration_error: {exc}",
                        },
                    )
        finally:
            self.state_loader.close()

    def _run_once(self) -> None:
        next_invocation = self._invocation + 1
        uptime_minutes = (datetime.now(tz=UTC) - self._start_time).total_seconds() / 60
        bundle = self.state_loader.load(
            invocation=next_invocation,
            uptime_minutes=uptime_minutes,
            sharpe_ratio=self._sharpe_ratio,
        )
        self._invocation = next_invocation
        self._log_account_snapshot(bundle.account, uptime_minutes)
        decision = self._generate_decision(bundle)
        self._last_decision = decision
        targets = decision.targets
        if self.config.risk.max_leverage > 0:
            for target in targets:
                if target.margin is None:
                    market_data = bundle.market.symbols.get(target.symbol)
                    if market_data is not None:
                        target.margin = (
                            abs(target.target_size)
                            * market_data.current_price
                            / self.config.risk.max_leverage
                        )
        self._log.info("模型解释：%s", decision.explanation)
        self._log.info("无效条件：%s", decision.invalidation_condition)
        for summary in (target.to_chinese_summary() for target in targets):
            self._log.info("持仓指令：%s", summary)
        reconciliation = self.reconciler.build_plan(bundle.account, targets)
        self.audit_sink.record_decision(
            {
                "type": "decision",
                "invocation": self._invocation,
                "uptime_minutes": uptime_minutes,
                "analysis_cn": decision.explanation,
                "invalidation_condition_cn": decision.invalidation_condition,
                "targets": [target.to_dict() for target in targets],
                "targets_cn": [target.to_chinese_summary() for target in targets],
                "untouched_symbols": reconciliation.untouched_symbols,
            }
        )

        execution_report = self.executor.execute(reconciliation.plan, bundle.market)
        self.audit_sink.record_execution(
            {
                "type": "execution",
                "invocation": self._invocation,
                "attempted": execution_report.attempted,
                "success": execution_report.success,
                "fills": [
                    {
                        "symbol": fill.symbol,
                        "action": fill.action.value,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "fee": fill.fee,
                        "pnl": fill.pnl,
                        "reason": fill.reason,
                    }
                    for fill in execution_report.fills
                ],
                "errors": [
                    {
                        "symbol": error.symbol,
                        "message": error.message,
                    }
                    for error in execution_report.errors
                ],
            }
        )

        status = "ok" if execution_report.success else "degraded"
        self.audit_sink.write_heartbeat(
            status,
            details={
                "invocation": self._invocation,
                "uptime_minutes": uptime_minutes,
                "success": execution_report.success,
                "errors": [error.message for error in execution_report.errors],
            },
        )

    def _generate_decision(self, bundle: MarketStateBundle) -> DecisionResult:
        try:
            return self.decision_engine.generate_decision(
                bundle.market,
                bundle.account,
                previous_decision=self._last_decision,
            )
        except Exception as exc:
            self._log.exception("LLM 决策生成失败")
            self.audit_sink.write_heartbeat(
                "error",
                details={
                    "invocation": self._invocation,
                    "reason": f"decision_engine_error: {exc}",
                },
            )
            raise

    def _log_account_snapshot(self, account: AccountSnapshot, uptime_minutes: float) -> None:
        self._log.info(
            "账户概览：净值 %.2f 美元，可用现金 %.2f 美元，累计收益 %.2f%%，运行时长 %.2f 分钟。",
            account.account_value,
            account.available_cash,
            account.total_return_percent,
            uptime_minutes,
        )
        if not account.positions:
            self._log.info("当前没有任何持仓。")
            return
        total_notional = 0.0
        total_margin = 0.0
        for position in account.positions:
            notional = abs(position.quantity) * position.current_price
            total_notional += notional
            if position.margin is not None:
                margin_value = position.margin
            elif self.config.risk.max_leverage > 0:
                margin_value = notional / self.config.risk.max_leverage
            else:
                margin_value = None
            if margin_value is not None:
                total_margin += margin_value
            self._log.info(
                "持仓详情：%s 数量 %.6f，开仓价 %.2f，现价 %.2f，未实现盈亏 %.2f，杠杆 %.2f，"
                "占用保证金 %s。",
                position.symbol,
                position.quantity,
                position.entry_price,
                position.current_price,
                position.unrealized_pnl,
                position.leverage,
                f"{margin_value:.2f}" if margin_value is not None else "未知",
            )
            if (
                position.protection.stop_loss is not None
                or position.protection.take_profit is not None
            ):
                self._log.info(
                    "保护设置：%s 止损 %s，止盈 %s。",
                    position.symbol,
                    position.protection.stop_loss
                    if position.protection.stop_loss is not None
                    else "未设置",
                    position.protection.take_profit
                    if position.protection.take_profit is not None
                    else "未设置",
                )
        if account.positions:
            self._log.info(
                "整体敞口：名义价值 %.2f，美金保证金占用 %.2f。",
                total_notional,
                total_margin,
            )
