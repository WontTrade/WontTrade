"""Application configuration modeled with pydantic settings."""

from __future__ import annotations

import os
import tomllib
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final

from hyperliquid.utils.constants import LOCAL_API_URL, MAINNET_API_URL, TESTNET_API_URL
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure"


class RiskLimits(BaseModel):
    """Risk configuration hints available to the strategy."""

    model_config = ConfigDict(extra="forbid")

    max_leverage: float = 10.0
    max_notional_per_symbol: float = 50_000.0
    funding_rate_limit: float = 0.0005
    cash_buffer_usd: float = 1_000.0


class AzureOpenAISettings(BaseModel):
    """Azure OpenAI specific configuration."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str
    deployment: str
    api_version: str


class LLMSettings(BaseModel):
    """Settings for the LLM decision engine."""

    model_config = ConfigDict(extra="forbid")

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4.1"
    endpoint: str | None = None
    temperature: float = Field(default=0.1, ge=0)
    max_output_tokens: int = Field(default=1_000, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    retry_attempts: int = Field(default=3, gt=0)
    api_key: str | None = None
    azure: AzureOpenAISettings | None = None


class TelemetrySettings(BaseModel):
    """Telemetry output configuration."""

    model_config = ConfigDict(extra="forbid")

    decision_log_path: Path = Field(default_factory=lambda: Path("decision-log.ndjson"))
    heartbeat_path: Path = Field(default_factory=lambda: Path("heartbeat.json"))


class BacktestSettings(BaseModel):
    """Configuration controlling backtest execution."""

    model_config = ConfigDict(extra="forbid")

    start_time: datetime
    end_time: datetime
    results_path: Path = Field(default_factory=lambda: Path("backtest-results.ndjson"))
    initial_cash: float = Field(default=10_000.0, gt=0)
    fee_bps: float = Field(default=1.0, ge=0)
    slippage_bps: float = Field(default=0.5, ge=0)
    max_steps: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_time_range(self) -> BacktestSettings:
        self.start_time = _ensure_utc(self.start_time)
        self.end_time = _ensure_utc(self.end_time)
        if self.end_time <= self.start_time:
            raise ValueError("backtest.end_time must be greater than backtest.start_time.")
        return self


class AppConfig(BaseSettings):
    """Top-level configuration exposed to the application."""

    model_config = SettingsConfigDict(extra="forbid", env_nested_delimiter="__")

    wallet_private_key: str
    account_address: str
    runtime_mode: RuntimeMode = RuntimeMode.LIVE
    network: HyperliquidNetwork = HyperliquidNetwork.MAINNET
    hyperliquid_base_url: str = MAINNET_API_URL
    symbols: list[str] = Field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    loop_interval_seconds: float = Field(default=15.0, ge=0)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    risk: RiskLimits = Field(default_factory=RiskLimits)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    backtest: BacktestSettings | None = None

    @field_validator("symbols")
    @classmethod
    def _normalize_symbols(cls, values: list[str]) -> list[str]:
        cleaned = [symbol.strip() for symbol in values if symbol.strip()]
        if not cleaned:
            raise ValueError("symbols must contain at least one non-empty symbol.")
        return cleaned

    @model_validator(mode="after")
    def _validate_loop_interval(self) -> AppConfig:
        if self.runtime_mode is RuntimeMode.BACKTEST:
            if self.loop_interval_seconds < 0:
                raise ValueError("loop_interval_seconds must be non-negative in backtest mode.")
        else:
            if self.loop_interval_seconds <= 0:
                raise ValueError("loop_interval_seconds must be greater than zero outside backtest mode.")
        return self

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        """Construct configuration from a TOML document."""

        config_path = path or Path("wonttrade.toml")
        if not config_path.exists():
            raise ValueError(f"Configuration file '{config_path}' does not exist.")

        document = _TomlConfig.from_toml(config_path)

        runtime = document.runtime
        symbols = document.symbols.tracked
        hyperliquid = document.hyperliquid
        credentials = document.credentials
        llm_section = document.llm

        network = hyperliquid.network
        hyperliquid_base_url = hyperliquid.base_url or {
            HyperliquidNetwork.MAINNET: MAINNET_API_URL,
            HyperliquidNetwork.TESTNET: TESTNET_API_URL,
            HyperliquidNetwork.LOCAL: LOCAL_API_URL,
        }[network]

        llm_settings = llm_section.llm_settings(credentials)

        telemetry_paths = _create_telemetry(document.telemetry, config_path.parent)
        backtest_settings = _create_backtest(runtime.mode, document.backtest, config_path.parent)

        wallet_private_key, account_address = _resolve_wallet_credentials(
            runtime.mode,
            credentials,
        )

        return cls.model_validate(
            {
                "wallet_private_key": wallet_private_key,
                "account_address": account_address,
                "runtime_mode": runtime.mode,
                "network": network,
                "hyperliquid_base_url": hyperliquid_base_url,
                "symbols": symbols,
                "loop_interval_seconds": runtime.loop_interval_seconds,
                "llm": llm_settings.model_dump(),
                "risk": document.risk.model_dump(),
                "telemetry": telemetry_paths.model_dump(),
                "backtest": backtest_settings.model_dump() if backtest_settings else None,
            }
        )


class _Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    openai_api_key: str | None = None
    azure_openai_api_key: str | None = None
    hyperliquid_private_key: str | None = None
    account_address: str | None = None


class _Runtime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: RuntimeMode = RuntimeMode.LIVE
    loop_interval_seconds: float = Field(default=15.0, ge=0)


class _Symbols(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracked: list[str] = Field(default_factory=lambda: list(DEFAULT_SYMBOLS))

    @field_validator("tracked")
    @classmethod
    def _ensure_symbols(cls, values: list[str]) -> list[str]:
        cleaned = [symbol.strip() for symbol in values if symbol.strip()]
        if not cleaned:
            raise ValueError("symbols.tracked must contain at least one symbol.")
        return cleaned


class _Hyperliquid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    network: HyperliquidNetwork = HyperliquidNetwork.MAINNET
    base_url: str | None = None


class _Telemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_log_path: Path | None = None
    heartbeat_path: Path | None = None


class _LLMSection(LLMSettings):
    """Extends LLMSettings with TOML-specific helpers."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None  # allow override from file if ever desired

    def llm_settings(self, credentials: _Credentials) -> LLMSettings:
        api_key: str | None = None
        if self.provider is LLMProvider.OPENAI:
            api_key = credentials.openai_api_key or self.api_key
            if not api_key:
                raise ValueError("credentials.openai_api_key is required for OpenAI provider.")
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        elif self.provider is LLMProvider.AZURE_OPENAI:
            api_key = credentials.azure_openai_api_key or self.api_key
            if not api_key:
                raise ValueError("credentials.azure_openai_api_key is required for Azure provider.")
            if self.azure is None:
                raise ValueError("llm.azure configuration is required for Azure provider.")
            os.environ.setdefault("AZURE_OPENAI_API_KEY", api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        return self.model_copy(update={"api_key": api_key})


class _TomlConfig(BaseModel):
    """Representation of the raw TOML structure."""

    model_config = ConfigDict(extra="forbid")

    credentials: _Credentials
    runtime: _Runtime = Field(default_factory=_Runtime)
    symbols: _Symbols = Field(default_factory=_Symbols)
    hyperliquid: _Hyperliquid = Field(default_factory=_Hyperliquid)
    llm: _LLMSection = Field(default_factory=_LLMSection)
    risk: RiskLimits = Field(default_factory=RiskLimits)
    telemetry: _Telemetry = Field(default_factory=_Telemetry)
    backtest: BacktestSettings | None = None

    @classmethod
    def from_toml(cls, path: Path) -> _TomlConfig:
        content = path.read_text(encoding="utf-8")
        data: dict[str, Any] = tomllib.loads(content)
        return cls.model_validate(data)


def _resolve_wallet_credentials(
    mode: RuntimeMode,
    credentials: _Credentials,
) -> tuple[str, str]:
    if mode is RuntimeMode.LIVE:
        private_key = credentials.hyperliquid_private_key
        if not private_key:
            raise ValueError("Live runtime requires credentials.hyperliquid_private_key.")
        from eth_account import Account  # imported lazily to avoid hard dependency for docs/tests

        account_address = credentials.account_address or Account.from_key(private_key).address
        return private_key, account_address

    # Backtest/Testnet-style: optional private key, default account name.
    private_key = credentials.hyperliquid_private_key or ""
    account_address = credentials.account_address or "WontTradeBacktest"
    return private_key, account_address


def _create_telemetry(section: _Telemetry, base_dir: Path) -> TelemetrySettings:
    return TelemetrySettings(
        decision_log_path=_resolve_path(base_dir, section.decision_log_path, "decision-log.ndjson"),
        heartbeat_path=_resolve_path(base_dir, section.heartbeat_path, "heartbeat.json"),
    )


def _create_backtest(mode: RuntimeMode, section: BacktestSettings | None, base_dir: Path) -> BacktestSettings | None:
    if section is None:
        if mode is RuntimeMode.BACKTEST:
            raise ValueError("backtest configuration section is required for backtest runtime.")
        return None
    if mode is not RuntimeMode.BACKTEST:
        return section.model_copy(
            update={
                "results_path": _resolve_path(base_dir, section.results_path, "backtest-results.ndjson"),
            }
        )
    # Normalize paths when backtesting to ensure relative paths resolve.
    return BacktestSettings(
        start_time=section.start_time,
        end_time=section.end_time,
        results_path=_resolve_path(base_dir, section.results_path, "backtest-results.ndjson"),
        initial_cash=section.initial_cash,
        fee_bps=section.fee_bps,
        slippage_bps=section.slippage_bps,
        max_steps=section.max_steps,
    )


def _resolve_path(base_dir: Path, candidate: Path | None, default_name: str) -> Path:
    target = candidate or Path(default_name)
    if not target.is_absolute():
        target = (base_dir / target).resolve()
    return target


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)
