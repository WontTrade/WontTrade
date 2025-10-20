# WontTrade

Autonomous trading daemon that ingests Hyperliquid market data, prompts a large language model for target positions, applies guardrails, and reconciles execution via the official Hyperliquid Python SDK.

## Prerequisites

- `uv` package manager with access to Python `3.14t`.
- Hyperliquid credentials (private key and optional explicit account address).
- OpenAI API key with access to the configured model.

## Environment Variables

| Variable | Description |
| --- | --- |
| `HYPERLIQUID_PRIVATE_KEY` | Hex-encoded private key used to sign orders. |
| `HYPERLIQUID_ACCOUNT_ADDRESS` | Optional; defaults to the wallet address derived from the private key. |
| `HYPERLIQUID_NETWORK` | `mainnet` (default), `testnet`, or `local`; selects the target cluster. |
| `HYPERLIQUID_API_BASE_URL` | Optional explicit API endpoint; overrides the network selection. |
| `OPENAI_API_KEY` | OpenAI credential for the decision engine. |
| `REASONTRADE_SYMBOLS` | Comma-separated list of symbols (default `BTC,ETH`). |
| `REASONTRADE_LOOP_INTERVAL` | Seconds between loop iterations (default `15`). |
| `REASONTRADE_LLM_MODEL` | OpenAI model name (default `gpt-4.1`). |
| `REASONTRADE_MAX_LEVERAGE` | Maximum aggregate leverage multiple. |
| `REASONTRADE_MAX_NOTIONAL_PER_SYMBOL` | Per-symbol notional cap in USD. |
| `REASONTRADE_CASH_BUFFER_USD` | Minimum reserve cash to maintain. |
| `REASONTRADE_FUNDING_LIMIT` | Absolute funding rate limit before scaling into bigger positions. |

## Installation

```bash
uv sync
```

This command resolves dependencies for the default environment and the `dev` group.

## Running the Daemon

```bash
uv run python main.py
```

Logs, decision records (`decision-log.ndjson`), and heartbeat (`heartbeat.json`) are written in the project root by default and can be overridden via environment variables.

## Linting and Formatting

```bash
uv run ruff check
uv run ruff format
```

## Project Structure

- `wonttrade/config.py` – Environment-backed configuration models.
- `wonttrade/core/` – State loading, indicator enrichment, guardrails, reconciliation, and execution stubs.
- `wonttrade/llm/decision_engine.py` – Prompting and parsing of the OpenAI decision engine.
- `wonttrade/hyperliquid_client.py` – Hyperliquid client factory for info/exchange access.
- `wonttrade/context.py` – Prompt context rendering.
- `wonttrade/telemetry/` – Logging and audit utilities.
- `SPEC.md` – High-level system specification.
