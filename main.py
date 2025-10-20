"""Entry point for the WontTrade daemon."""

from __future__ import annotations

from wonttrade.config import AppConfig
from wonttrade.context import LLMContextBuilder
from wonttrade.core.execution import ExecutionService
from wonttrade.core.guardrails import GuardrailService
from wonttrade.core.reconciler import PositionReconciler
from wonttrade.core.state_loader import StateLoader
from wonttrade.hyperliquid_client import HyperliquidClientFactory
from wonttrade.llm.decision_engine import LLMDecisionEngine
from wonttrade.loop import TradingLoop
from wonttrade.telemetry.audit import AuditSink
from wonttrade.telemetry.logger import get_logger


def main() -> None:
    logger = get_logger(__name__)
    try:
        config = AppConfig.load()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise

    clients = HyperliquidClientFactory(config).create()
    state_loader = StateLoader(config=config, info_client=clients.info)
    decision_engine = LLMDecisionEngine(
        config=config,
        context_builder=LLMContextBuilder(),
    )
    guardrails = GuardrailService(config)
    reconciler = PositionReconciler()
    executor = ExecutionService(clients.exchange)
    audit_sink = AuditSink(
        decision_log_path=config.telemetry.decision_log_path,
        heartbeat_path=config.telemetry.heartbeat_path,
    )

    loop = TradingLoop(
        config=config,
        state_loader=state_loader,
        decision_engine=decision_engine,
        guardrails=guardrails,
        reconciler=reconciler,
        executor=executor,
        audit_sink=audit_sink,
    )

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("WontTrade daemon interrupted by user.")


if __name__ == "__main__":
    main()
