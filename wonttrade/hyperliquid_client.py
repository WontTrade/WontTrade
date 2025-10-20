"""Hyperliquid client factory utilities."""

from __future__ import annotations

from dataclasses import dataclass

from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

from .config import AppConfig


@dataclass(slots=True)
class HyperliquidClients:
    """Aggregates Hyperliquid SDK clients."""

    info: Info
    exchange: Exchange
    wallet: LocalAccount


class HyperliquidClientFactory:
    """Builds authenticated client instances."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._wallet: LocalAccount | None = None

    def create(self) -> HyperliquidClients:
        """Create fully configured Hyperliquid clients."""
        wallet = self._wallet or Account.from_key(self._config.wallet_private_key)
        self._wallet = wallet

        info_client = Info(base_url=self._config.hyperliquid_base_url)
        exchange_client = Exchange(
            wallet=wallet,
            base_url=self._config.hyperliquid_base_url,
            account_address=self._config.account_address,
        )

        return HyperliquidClients(
            info=info_client,
            exchange=exchange_client,
            wallet=wallet,
        )
