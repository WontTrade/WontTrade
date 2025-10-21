"""Application configuration models."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final

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


class RuntimeMode(str, Enum):
    """Operating modes for the trading engine."""

    LIVE = "live"
    BACKTEST = "backtest"


@dataclass(slots=True)
class RiskLimits:
    """Risk configuration hints available to the strategy."""

    max_leverage: float = 10.0
    max_notional_per_symbol: float = 50_000.0
    funding_rate_limit: float = 0.0005
    cash_buffer_usd: float = 1_000.0


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure"


@dataclass(slots=True)
class AzureOpenAISettings:
    """Azure OpenAI specific configuration."""

    endpoint: str
    deployment: str
    api_version: str


@dataclass(slots=True)
class LLMSettings:
    """Settings for the LLM decision engine."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4.1"
    endpoint: str | None = None
    temperature: float = 0.1
    max_output_tokens: int = 1_000
    request_timeout_seconds: float = 30.0
    retry_attempts: int = 3
    api_key: str | None = None
    azure: AzureOpenAISettings | None = None


@dataclass(slots=True)
class TelemetrySettings:
    """Telemetry output configuration."""

    decision_log_path: Path = Path("decision-log.ndjson")
    heartbeat_path: Path = Path("heartbeat.json")


@dataclass(slots=True)
class BacktestSettings:
    """Configuration controlling backtest execution."""

    start_time: datetime
    end_time: datetime
    results_path: Path = Path("backtest-results.ndjson")
    initial_cash: float = 10_000.0
    fee_bps: float = 1.0
    slippage_bps: float = 0.5
    max_steps: int | None = None


@dataclass(slots=True)
class AppConfig:
    """Top-level immutable configuration."""

    wallet_private_key: str
    account_address: str
    runtime_mode: RuntimeMode = RuntimeMode.LIVE
    network: HyperliquidNetwork = HyperliquidNetwork.MAINNET
    hyperliquid_base_url: str = "https://api.hyperliquid.xyz"
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    loop_interval_seconds: float = 15.0
    llm: LLMSettings = field(default_factory=LLMSettings)
    risk: RiskLimits = field(default_factory=RiskLimits)
    telemetry: TelemetrySettings = field(default_factory=TelemetrySettings)
    backtest: BacktestSettings | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        """Construct configuration from a TOML document."""

        config_path = path or Path("wonttrade.toml")
        if not config_path.exists():
            raise ValueError(f"Configuration file '{config_path}' does not exist.")

        with config_path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)

        base_dir = config_path.parent
        credentials = _expect_table(raw, "credentials")
        openai_key = _optional_string(credentials, "openai_api_key")
        azure_key = _optional_string(credentials, "azure_openai_api_key")
        hyper_private_key = _optional_string(credentials, "hyperliquid_private_key")
        account_address_raw = _optional_string(credentials, "account_address")

        runtime_section = _expect_table(raw, "runtime", optional=True)
        runtime_mode_value = (
            runtime_section.get("mode", RuntimeMode.LIVE.value)
            if runtime_section
            else RuntimeMode.LIVE.value
        )
        runtime_mode = _parse_enum(RuntimeMode, runtime_mode_value, "runtime.mode")
        loop_interval_seconds = float(
            runtime_section.get("loop_interval_seconds", 15.0) if runtime_section else 15.0
        )

        symbols_section = _expect_table(raw, "symbols", optional=True)
        symbols_raw = (
            symbols_section.get("tracked", list(DEFAULT_SYMBOLS))
            if symbols_section
            else list(DEFAULT_SYMBOLS)
        )
        symbols = _parse_symbol_list(symbols_raw)

        hyper_section = _expect_table(raw, "hyperliquid", optional=True)
        network_value = (
            hyper_section.get("network", HyperliquidNetwork.MAINNET.value)
            if hyper_section
            else HyperliquidNetwork.MAINNET.value
        )
        network = _parse_enum(HyperliquidNetwork, network_value, "hyperliquid.network")
        base_url_override = hyper_section.get("base_url") if hyper_section else None
        hyperliquid_base_url = (
            base_url_override
            or {
                HyperliquidNetwork.MAINNET: MAINNET_API_URL,
                HyperliquidNetwork.TESTNET: TESTNET_API_URL,
                HyperliquidNetwork.LOCAL: LOCAL_API_URL,
            }[network]
        )

        llm_section = _expect_table(raw, "llm", optional=True)
        provider_value = _optional_string(
            llm_section,
            "provider",
            default=LLMProvider.OPENAI.value,
        )
        provider = _parse_enum(LLMProvider, provider_value, "llm.provider")
        model_value = _optional_string(llm_section, "model", default="gpt-4.1")
        temperature_value = float(llm_section.get("temperature", 0.1) if llm_section else 0.1)
        max_output_tokens_value = int(
            llm_section.get("max_output_tokens", 1_000) if llm_section else 1_000
        )
        timeout_value = float(
            llm_section.get("request_timeout_seconds", 30.0) if llm_section else 30.0
        )
        retry_attempts_value = int(llm_section.get("retry_attempts", 3) if llm_section else 3)
        endpoint_value = _optional_string(llm_section, "endpoint")

        api_key = None
        azure_settings: AzureOpenAISettings | None = None
        if provider is LLMProvider.OPENAI:
            if not openai_key:
                raise ValueError("credentials.openai_api_key is required for OpenAI provider.")
            api_key = openai_key
            os.environ.setdefault("OPENAI_API_KEY", openai_key)
        elif provider is LLMProvider.AZURE_OPENAI:
            if not azure_key:
                raise ValueError("credentials.azure_openai_api_key is required for Azure provider.")
            if llm_section is None:
                raise ValueError("llm.azure configuration is required for Azure provider.")
            azure_section = _expect_table(llm_section, "azure")
            azure_settings = AzureOpenAISettings(
                endpoint=_expect_string(azure_section, "endpoint"),
                deployment=_expect_string(azure_section, "deployment"),
                api_version=_expect_string(azure_section, "api_version"),
            )
            api_key = azure_key
            os.environ.setdefault("AZURE_OPENAI_API_KEY", azure_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        llm_settings = LLMSettings(
            provider=provider,
            model=model_value or "",
            endpoint=endpoint_value,
            temperature=temperature_value,
            max_output_tokens=max_output_tokens_value,
            request_timeout_seconds=timeout_value,
            retry_attempts=retry_attempts_value,
            api_key=api_key,
            azure=azure_settings,
        )

        risk_section = _expect_table(raw, "risk", optional=True)
        risk_limits = RiskLimits(
            max_leverage=float(risk_section.get("max_leverage", 10.0) if risk_section else 10.0),
            max_notional_per_symbol=float(
                risk_section.get("max_notional_per_symbol", 50_000.0) if risk_section else 50_000.0
            ),
            funding_rate_limit=float(
                risk_section.get("funding_rate_limit", 0.0005) if risk_section else 0.0005
            ),
            cash_buffer_usd=float(
                risk_section.get("cash_buffer_usd", 1_000.0) if risk_section else 1_000.0
            ),
        )

        telemetry_section = _expect_table(raw, "telemetry", optional=True)
        telemetry_settings = TelemetrySettings(
            decision_log_path=_resolve_path(
                base_dir,
                telemetry_section.get("decision_log_path") if telemetry_section else None,
                "decision-log.ndjson",
            ),
            heartbeat_path=_resolve_path(
                base_dir,
                telemetry_section.get("heartbeat_path") if telemetry_section else None,
                "heartbeat.json",
            ),
        )

        if runtime_mode is RuntimeMode.LIVE:
            if not hyper_private_key:
                raise ValueError("Live runtime requires credentials.hyperliquid_private_key.")
            from eth_account import Account  # local import to avoid hard dependency during docs

            default_address = Account.from_key(hyper_private_key).address
            account_address = account_address_raw or default_address
        else:
            account_address = account_address_raw or "WontTradeBacktest"
            hyper_private_key = hyper_private_key or ""

        backtest_settings: BacktestSettings | None = None
        backtest_section = _expect_table(raw, "backtest", optional=True)
        if runtime_mode is RuntimeMode.BACKTEST:
            if backtest_section is None:
                raise ValueError("backtest configuration section is required for backtest runtime.")
            start_raw = _expect_string(backtest_section, "start_time")
            end_raw = _expect_string(backtest_section, "end_time")
            start_time = _parse_datetime(start_raw)
            end_time = _parse_datetime(end_raw)
            if end_time <= start_time:
                raise ValueError("backtest.end_time must be greater than backtest.start_time.")
            backtest_settings = BacktestSettings(
                start_time=start_time,
                end_time=end_time,
                results_path=_resolve_path(
                    base_dir,
                    backtest_section.get("results_path"),
                    "backtest-results.ndjson",
                ),
                initial_cash=float(backtest_section.get("initial_cash", 10_000.0)),
                fee_bps=float(backtest_section.get("fee_bps", 1.0)),
                slippage_bps=float(backtest_section.get("slippage_bps", 0.5)),
                max_steps=_optional_int(backtest_section, "max_steps"),
            )
        elif (
            backtest_section is not None
            and "start_time" in backtest_section
            and "end_time" in backtest_section
        ):
            start_time = _parse_datetime(_expect_string(backtest_section, "start_time"))
            end_time = _parse_datetime(_expect_string(backtest_section, "end_time"))
            if end_time <= start_time:
                raise ValueError("backtest.end_time must be greater than backtest.start_time.")
            backtest_settings = BacktestSettings(
                start_time=start_time,
                end_time=end_time,
                results_path=_resolve_path(
                    base_dir,
                    backtest_section.get("results_path"),
                    "backtest-results.ndjson",
                ),
                initial_cash=float(backtest_section.get("initial_cash", 10_000.0)),
                fee_bps=float(backtest_section.get("fee_bps", 1.0)),
                slippage_bps=float(backtest_section.get("slippage_bps", 0.5)),
                max_steps=_optional_int(backtest_section, "max_steps"),
            )

        return cls(
            wallet_private_key=hyper_private_key or "",
            account_address=account_address,
            runtime_mode=runtime_mode,
            network=network,
            hyperliquid_base_url=hyperliquid_base_url,
            symbols=symbols,
            loop_interval_seconds=loop_interval_seconds,
            llm=llm_settings,
            risk=risk_limits,
            telemetry=telemetry_settings,
            backtest=backtest_settings,
        )


def _expect_table(
    payload: dict[str, Any],
    key: str,
    *,
    optional: bool = False,
) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        if optional:
            return None
        raise ValueError(f"Configuration section '{key}' is required.")
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a table.")
    return value


def _expect_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration key '{key}' must be a non-empty string.")
    return value


def _optional_string(
    payload: dict[str, Any] | None,
    key: str,
    *,
    default: str | None = None,
) -> str | None:
    if payload is None:
        return default
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"Configuration key '{key}' must be a string.")
    return value


def _parse_enum(enum_cls: type[Enum], raw: Any, key: str) -> Enum:
    if not isinstance(raw, str):
        valid = ", ".join(member.value for member in enum_cls)
        raise ValueError(f"{key} must be one of {valid}.")
    try:
        return enum_cls(raw.lower())
    except Exception as exc:
        valid = ", ".join(member.value for member in enum_cls)
        raise ValueError(f"{key} must be one of {valid}.") from exc


def _parse_symbol_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError("symbols.tracked must be an array of strings.")
    symbols: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise ValueError("symbols.tracked must contain strings only.")
        trimmed = entry.strip()
        if trimmed:
            symbols.append(trimmed)
    return symbols or list(DEFAULT_SYMBOLS)


def _resolve_path(base: Path, raw: str | None, default: str) -> Path:
    target = Path(raw) if raw else Path(default)
    if not target.is_absolute():
        target = (base / target).resolve()
    return target


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError(f"Configuration key '{key}' must be an integer when provided.")


def _parse_datetime(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid datetime format: '{raw}'. Expected ISO 8601.") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
