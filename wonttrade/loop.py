"""Main trading loop orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from .config import AppConfig
from .core.execution import ExecutionService
from .core.guardrails import GuardrailService
from .core.reconciler import PositionReconciler
from .core.state_loader import MarketStateBundle, StateLoader
from .llm.decision_engine import LLMDecisionEngine
from .models import TargetPosition
from .telemetry.audit import AuditSink
from .telemetry.logger import get_logger


@dataclass(slots=True)
class TradingLoop:
    """Coordinates the periodic trading workflow."""

    config: AppConfig
    state_loader: StateLoader
    decision_engine: LLMDecisionEngine
    guardrails: GuardrailService
    reconciler: PositionReconciler
    executor: ExecutionService
    audit_sink: AuditSink

    def __post_init__(self) -> None:
        self._start_time = datetime.now(tz=UTC)
        self._invocation = 0
        self._sharpe_ratio = 0.0
        self._log = get_logger(__name__)

    def run_forever(self) -> None:
        try:
            while True:
                try:
                    self._run_once()
                except Exception as exc:  # pragma: no cover - production resilience
                    self._log.exception("Iteration failed: %s", exc)
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

    def _run_once(self) -> None:
        self._invocation += 1
        uptime_minutes = (datetime.now(tz=UTC) - self._start_time).total_seconds() / 60
        bundle = self.state_loader.load(
            invocation=self._invocation,
            uptime_minutes=uptime_minutes,
            sharpe_ratio=self._sharpe_ratio,
        )
        targets = self._generate_targets(bundle)
        guard_report = self.guardrails.validate(bundle.market, bundle.account, targets)
        if not guard_report.approved:
            self._audit_guardrail_failure(bundle, guard_report.messages)
            return

        reconciliation = self.reconciler.build_plan(bundle.account, targets)
        self.audit_sink.record_decision(
            {
                "type": "decision",
                "invocation": self._invocation,
                "uptime_minutes": uptime_minutes,
                "targets": [target.__dict__ for target in targets],
                "untouched_symbols": reconciliation.untouched_symbols,
            }
        )

        execution_report = self.executor.execute(reconciliation.plan)
        self.audit_sink.record_execution(
            {
                "type": "execution",
                "invocation": self._invocation,
                "attempted": execution_report.attempted,
                "success": execution_report.success,
                "errors": [error.__dict__ for error in execution_report.errors],
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

    def _generate_targets(self, bundle: MarketStateBundle) -> list[TargetPosition]:
        try:
            return self.decision_engine.generate_targets(bundle.market, bundle.account)
        except Exception as exc:
            self._log.exception("LLM decision generation failed")
            self.audit_sink.write_heartbeat(
                "error",
                details={
                    "invocation": self._invocation,
                    "reason": f"decision_engine_error: {exc}",
                },
            )
            raise

    def _audit_guardrail_failure(self, bundle: MarketStateBundle, messages: list[str]) -> None:
        self._log.warning("Guardrails blocked execution: %s", messages)
        self.audit_sink.record_decision(
            {
                "type": "guardrail_rejection",
                "invocation": self._invocation,
                "messages": messages,
            }
        )
        self.audit_sink.write_heartbeat(
            "guardrail_rejection",
            details={
                "invocation": self._invocation,
                "messages": messages,
            },
        )
