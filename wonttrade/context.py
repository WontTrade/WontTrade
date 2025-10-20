"""Prompt context generation for the LLM decision engine."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from .models import AccountSnapshot, MarketSnapshot, PositionSnapshot, SymbolMarketData


def _format_series(series: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in series) + "]"


def _format_position(position: PositionSnapshot) -> str:
    payload = {
        "symbol": position.symbol,
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "current_price": position.current_price,
        "liquidation_price": position.liquidation_price,
        "unrealized_pnl": position.unrealized_pnl,
        "leverage": position.leverage,
        "confidence": position.confidence,
    }
    return json.dumps(payload, separators=(", ", ": "))


@dataclass(slots=True)
class LLMContextBuilder:
    """Render market and account state into a deterministic prompt."""

    def build(self, market: MarketSnapshot, account: AccountSnapshot) -> str:
        lines: list[str] = []
        lines.append(f"It has been {int(market.uptime_minutes)} minutes since you started trading.")
        lines.append(
            "The current time is "
            f"{market.captured_at.isoformat()} and you've been invoked "
            f"{market.invocation_count} times."
        )
        lines.append(
            "Below, we are providing you with a variety of state data, price data, and predictive "
            "signals so you can discover alpha. Below that is your current account information, "
            "value, performance, positions, etc."
        )
        lines.append("")
        lines.append("ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST")
        lines.append("")
        lines.append("CURRENT MARKET STATE FOR ALL COINS")

        for symbol, data in market.symbols.items():
            lines.append(self._format_symbol_block(symbol, data))

        lines.append("HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE")
        lines.append(f"Current Total Return (percent): {account.total_return_percent:.2f}%")
        lines.append(f"Available Cash: {account.available_cash:.2f}")
        lines.append(f"Current Account Value: {account.account_value:.2f}")
        if account.positions:
            positions_str = " ".join(_format_position(pos) for pos in account.positions)
            lines.append(f"Current live positions & performance: {positions_str}")
        else:
            lines.append("Current live positions & performance: None")
        return "\n".join(lines)

    def _format_symbol_block(self, symbol: str, data: SymbolMarketData) -> str:
        block: list[str] = []
        block.append("")
        block.append(f"ALL {symbol} DATA")
        block.append(
            "current_price = "
            f"{data.current_price:.6f}, current_ema20 = {data.ema20:.6f}, "
            f"current_macd = {data.macd:.6f}, current_rsi (short) = {data.rsi_short:.6f}"
        )
        block.append(
            "open_interest = "
            f"{data.open_interest:.6f}, funding_rate = {data.funding_rate:.6g}, "
            f"current_volume = {data.volume:.6f}"
        )

        window = data.indicator_window
        ema_series = list(window.ema)
        macd_series = list(window.macd)
        rsi_short_series = list(window.rsi_short)
        rsi_long_series = list(window.rsi_long)
        mid_prices = list(window.mid_prices)
        block.append("")
        block.append("Intraday series (3-minute intervals, oldest → latest):")
        block.append(f"Mid prices: {_format_series(mid_prices)}")
        block.append(f"EMA indicators (period={len(ema_series)}): {_format_series(ema_series)}")
        block.append(f"MACD indicators: {_format_series(macd_series)}")
        block.append(f"RSI indicators (short): {_format_series(rsi_short_series)}")
        block.append(f"RSI indicators (long): {_format_series(rsi_long_series)}")

        block.append("")
        block.append("Longer-term context (4-hour timeframe):")
        block.append(
            "20-Period EMA: "
            f"{(data.long_term_ema20 or 0.0):.6f} vs. 50-Period EMA: "
            f"{(data.long_term_ema50 or 0.0):.6f}"
        )
        block.append(f"MACD indicators: {_format_series([data.long_term_macd or 0.0])}")
        block.append(f"RSI indicators (long): {_format_series([data.long_term_rsi or 0.0])}")
        block.append(
            f"ATR (short vs. long): {(data.atr_short or 0.0):.6f} vs. {(data.atr_long or 0.0):.6f}"
        )
        return "\n".join(block)
