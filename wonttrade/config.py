"""Application configuration models."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

from hyperliquid.utils.constants import (
    LOCAL_API_URL,
    MAINNET_API_URL,
    TESTNET_API_URL,
)

DEFAULT_SYMBOLS: Final[list[str]] = ["BTC", "ETH"]


class HyperliquidNetwork(str, Enum):
    """Supported Hyperliquid deployments."""

    MAINNET = "mainnet"
    TESTNET = "testnet"
    LOCAL = "local"


@dataclass(slots=True)
class RiskLimits:
    """Risk guardrails enforced before executing trades."""

    max_leverage: float = 10.0
    max_notional_per_symbol: float = 50_000.0
    funding_rate_limit: float = 0.0005
    cash_buffer_usd: float = 1_000.0


@dataclass(slots=True)
class LLMSettings:
    """Settings for the LLM decision engine."""

    model: str = "gpt-4.1"
    temperature: float = 0.1
    max_output_tokens: int = 1_000
    request_timeout_seconds: float = 30.0
    retry_attempts: int = 3


@dataclass(slots=True)
class TelemetrySettings:
    """Telemetry output configuration."""

    decision_log_path: Path = Path("decision-log.ndjson")
    heartbeat_path: Path = Path("heartbeat.json")


@dataclass(slots=True)
class AppConfig:
    """Top-level immutable configuration."""

    wallet_private_key: str
    account_address: str
    network: HyperliquidNetwork = HyperliquidNetwork.MAINNET
    hyperliquid_base_url: str = "https://api.hyperliquid.xyz"
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    loop_interval_seconds: float = 15.0
    llm: LLMSettings = field(default_factory=LLMSettings)
    risk: RiskLimits = field(default_factory=RiskLimits)
    telemetry: TelemetrySettings = field(default_factory=TelemetrySettings)

    @classmethod
    def load(cls) -> AppConfig:
        """Construct configuration from environment variables."""

        def require_env(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise ValueError(f"Environment variable '{name}' is required.")
            return value

        private_key = require_env("HYPERLIQUID_PRIVATE_KEY")
        openai_key = require_env("OPENAI_API_KEY")

        symbols_raw = os.getenv("REASONTRADE_SYMBOLS")
        symbols = (
            [sym.strip() for sym in symbols_raw.split(",") if sym.strip()]
            if symbols_raw
            else list(DEFAULT_SYMBOLS)
        )

        loop_interval_seconds = float(os.getenv("REASONTRADE_LOOP_INTERVAL", "15"))

        llm = LLMSettings(
            model=os.getenv("REASONTRADE_LLM_MODEL", "gpt-4.1"),
            temperature=float(os.getenv("REASONTRADE_LLM_TEMPERATURE", "0.1")),
            max_output_tokens=int(os.getenv("REASONTRADE_LLM_MAX_TOKENS", "1000")),
            request_timeout_seconds=float(os.getenv("REASONTRADE_LLM_TIMEOUT_SECONDS", "30")),
            retry_attempts=int(os.getenv("REASONTRADE_LLM_RETRY_ATTEMPTS", "3")),
        )

        risk = RiskLimits(
            max_leverage=float(os.getenv("REASONTRADE_MAX_LEVERAGE", "10")),
            max_notional_per_symbol=float(
                os.getenv("REASONTRADE_MAX_NOTIONAL_PER_SYMBOL", "50000")
            ),
            funding_rate_limit=float(os.getenv("REASONTRADE_FUNDING_LIMIT", "0.0005")),
            cash_buffer_usd=float(os.getenv("REASONTRADE_CASH_BUFFER_USD", "1000")),
        )

        telemetry = TelemetrySettings(
            decision_log_path=Path(
                os.getenv("REASONTRADE_DECISION_LOG_PATH", "decision-log.ndjson")
            ),
            heartbeat_path=Path(os.getenv("REASONTRADE_HEARTBEAT_PATH", "heartbeat.json")),
        )

        network_raw = os.getenv("HYPERLIQUID_NETWORK", HyperliquidNetwork.MAINNET.value).lower()
        try:
            network = HyperliquidNetwork(network_raw)
        except ValueError as exc:
            raise ValueError(
                "HYPERLIQUID_NETWORK must be one of "
                f"{', '.join(n.value for n in HyperliquidNetwork)}"
            ) from exc

        base_url_override = os.getenv("HYPERLIQUID_API_BASE_URL")
        base_url = base_url_override or {
            HyperliquidNetwork.MAINNET: MAINNET_API_URL,
            HyperliquidNetwork.TESTNET: TESTNET_API_URL,
            HyperliquidNetwork.LOCAL: LOCAL_API_URL,
        }[network]

        from eth_account import Account  # local import to avoid hard dependency during docs

        # Store OpenAI key in environment for SDK consumption.
        os.environ.setdefault("OPENAI_API_KEY", openai_key)

        wallet = Account.from_key(private_key)
        account_address = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS", wallet.address)

        return cls(
            wallet_private_key=private_key,
            account_address=account_address,
            network=network,
            hyperliquid_base_url=base_url,
            symbols=symbols,
            loop_interval_seconds=loop_interval_seconds,
            llm=llm,
            risk=risk,
            telemetry=telemetry,
        )
