"""Entry point for the WontTrade daemon."""

from __future__ import annotations

import argparse
from pathlib import Path

from hyperliquid.info import Info

from wonttrade.backtest.replay import BacktestReplayProvider
from wonttrade.backtest.simulation import SimulatedExchange, SimulatedExecutor
from wonttrade.config import AppConfig, RuntimeMode
from wonttrade.context import LLMContextBuilder
from wonttrade.core.execution import ExecutionService
from wonttrade.core.reconciler import PositionReconciler
from wonttrade.core.state_loader import StateLoader
from wonttrade.hyperliquid_client import HyperliquidClientFactory
from wonttrade.llm.decision_engine import LLMDecisionEngine
from wonttrade.loop import TradingLoop
from wonttrade.telemetry.audit import AuditSink
from wonttrade.telemetry.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="WontTrade autonomous trading daemon.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("wonttrade.toml"),
        help="Path to the TOML configuration file.",
    )
    args = parser.parse_args()

    logger = get_logger(__name__)
    try:
        config = AppConfig.load(args.config)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise

    decision_engine = LLMDecisionEngine(
        config=config,
        context_builder=LLMContextBuilder(),
    )
    reconciler = PositionReconciler()
    audit_sink = AuditSink(
        decision_log_path=config.telemetry.decision_log_path,
        heartbeat_path=config.telemetry.heartbeat_path,
    )

    if config.runtime_mode is RuntimeMode.BACKTEST:
        if config.backtest is None:
            raise ValueError("Backtest mode requires backtest settings.")
        simulation = SimulatedExchange(
            initial_cash=config.backtest.initial_cash,
            fee_bps=config.backtest.fee_bps,
            slippage_bps=config.backtest.slippage_bps,
            results_path=config.backtest.results_path,
            initial_leverage=config.risk.max_leverage if config.risk.max_leverage > 0 else 10.0,
        )
        info_client = Info(base_url=config.hyperliquid_base_url, skip_ws=True)
        state_loader = BacktestReplayProvider(
            config=config,
            simulation=simulation,
            info_client=info_client,
        )
        executor = SimulatedExecutor(simulation=simulation)
        loop = TradingLoop(
            config=config,
            state_loader=state_loader,
            decision_engine=decision_engine,
            reconciler=reconciler,
            executor=executor,
            audit_sink=audit_sink,
        )
        try:
            loop.run_backtest()
        except KeyboardInterrupt:
            logger.info("Backtest interrupted by user.")
    else:
        clients = HyperliquidClientFactory(config).create()
        state_loader = StateLoader(config=config, info_client=clients.info)
        executor = ExecutionService(clients.exchange)
        loop = TradingLoop(
            config=config,
            state_loader=state_loader,
            decision_engine=decision_engine,
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
