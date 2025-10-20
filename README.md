# WontTrade

Autonomous trading daemon that ingests Hyperliquid market data, prompts a large language model for target positions, applies guardrails, and reconciles execution via the official Hyperliquid Python SDK.

## Prerequisites

- `uv` with a Python `3.14t` toolchain (e.g. `uv python install 3.14t`).
- Hyperliquid credentials (private key and optional explicit account address) for mainnet or testnet.
- API access to either OpenAI or Azure OpenAI for decision generation.

## Configuration

WontTrade is configured exclusively through a TOML file (defaults to `wonttrade.toml`). The configuration contains:

- `[credentials]` – Hyperliquid signing key, optional account address, and the LLM API keys.
- `[runtime]` – Trading mode (`live` or `backtest`) and loop cadence.
- `[hyperliquid]` – Target network (`mainnet`, `testnet`, `local`) and optional base URL override.
- `[symbols]` – Asset universe tracked by the trading loop.
- `[llm]` – Model selection plus OpenAI or Azure-specific settings (`[llm.azure]`).
- `[risk]` – Leverage, notional, and funding guardrails.
- `[telemetry]` – Paths for decision logs and heartbeat output.
- `[backtest]` – Window, cash, and execution parameters for historical simulations.

Copy `examples/backtest-example.toml` and adjust the values to suit your deployment. The loader will automatically export the supplied API keys so the respective SDKs can authenticate.

## Installation

```bash
uv sync
```

## Running

### Live Trading

```bash
uv run python main.py --config wonttrade.toml
```

The loop connects to Hyperliquid using the official SDK, requests decisions from the configured LLM provider, enforces guardrails, and reconciles target positions.

### Backtesting

```bash
uv run python main.py --config examples/backtest-example.toml
```

Backtests stream historical candles and funding data directly from Hyperliquid, replay order book states through the simulator, and still invoke the live LLM provider for each decision tick.

## Linting and Formatting

```bash
uv run ruff check
uv run ruff format
```

## Project Structure

- `wonttrade/config.py` – TOML-backed configuration models.
- `wonttrade/loop.py` – Core orchestration for live and backtest loops.
- `wonttrade/core/` – Guardrails, reconciliation, execution, and state utilities.
- `wonttrade/backtest/` – Historical data fetch, replay, and simulation plumbing.
- `wonttrade/llm/decision_engine.py` – Prompt construction and response parsing for OpenAI/Azure.
- `wonttrade/hyperliquid_client.py` – Factory for live Hyperliquid Info and Exchange clients.
- `wonttrade/context.py` – Prompt context rendering.
- `wonttrade/telemetry/` – Logging and audit sinks.
- `SPEC.md` – High-level system specification.
