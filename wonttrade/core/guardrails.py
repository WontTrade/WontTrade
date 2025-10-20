"""Guardrail validation for model-generated positions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..config import AppConfig
from ..models import AccountSnapshot, MarketSnapshot, PositionSnapshot, TargetPosition
from ..telemetry.logger import get_logger


@dataclass(slots=True)
class GuardrailReport:
    """Describes the outcome of guardrail evaluation."""

    approved: bool
    messages: list[str]

    def raise_for_failure(self) -> None:
        if not self.approved:
            raise RuntimeError("Guardrails rejected the decision: " + "; ".join(self.messages))


class GuardrailService:
    """Validate LLM decisions against deterministic risk limits."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._log = get_logger(__name__)

    def validate(
        self,
        market: MarketSnapshot,
        account: AccountSnapshot,
        targets: Iterable[TargetPosition],
    ) -> GuardrailReport:
        messages: list[str] = []
        market_symbols = market.symbols
        positions: dict[str, PositionSnapshot] = {pos.symbol: pos for pos in account.positions}
        total_notional = 0.0

        for target in targets:
            symbol = target.symbol
            if symbol not in market_symbols:
                messages.append(f"Symbol {symbol} not available in market snapshot.")
                continue

            market_data = market_symbols[symbol]
            price = market_data.current_price or 0.0
            target_notional = abs(target.target_size) * price
            total_notional += target_notional

            if target_notional > self._config.risk.max_notional_per_symbol:
                messages.append(
                    f"{symbol} target notional {target_notional:.2f} exceeds"
                    f" limit {self._config.risk.max_notional_per_symbol:.2f}."
                )

            current_position = positions.get(symbol)
            current_size = abs(current_position.quantity) if current_position else 0.0
            if (
                abs(market_data.funding_rate) > self._config.risk.funding_rate_limit
                and abs(target.target_size) > current_size
            ):
                messages.append(
                    f"Funding rate {market_data.funding_rate:.6f} for {symbol} exceeds"
                    f" limit {self._config.risk.funding_rate_limit:.6f} during size increase."
                )

        if account.available_cash - self._config.risk.cash_buffer_usd < 0:
            messages.append(
                "Available cash below configured buffer; no automatic deleveraging performed."
            )

        if total_notional > account.account_value * self._config.risk.max_leverage:
            limit = self._config.risk.max_leverage * account.account_value
            messages.append(
                f"Aggregate exposure {total_notional:.2f} exceeds leverage limit {limit:.2f}."
            )

        if not messages:
            self._log.debug("Guardrails approved decision set.")
        else:
            self._log.warning("Guardrails rejected decision set: %s", messages)

        return GuardrailReport(approved=not messages, messages=messages)
