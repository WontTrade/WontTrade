"""LLM-based decision engine."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import AppConfig, LLMProvider
from ..context import LLMContextBuilder
from ..models import AccountSnapshot, MarketSnapshot, TargetPosition
from ..telemetry.logger import get_logger

_SYSTEM_PROMPT = (
    "You are an autonomous trading strategy responsible for managing perpetual futures on "
    'Hyperliquid. Always respond with a JSON object of the form {"targets": [...]} where each '
    "target has fields symbol, target_size, stop_loss, take_profit, confidence, and rationale. "
    "target_size represents the absolute position size (positive for long, negative for short) "
    "that should exist after execution. Do not describe trade sequences; only specify the desired "
    "terminal state."
)


class TargetSchema(BaseModel):
    symbol: str
    target_size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float
    rationale: str | None = None

    def to_domain(self) -> TargetPosition:
        return TargetPosition(
            symbol=self.symbol,
            target_size=self.target_size,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            confidence=self.confidence,
            rationale=self.rationale,
        )


@dataclass(slots=True)
class LLMDecisionEngine:
    """Encapsulates prompt generation and response parsing."""

    config: AppConfig
    context_builder: LLMContextBuilder

    def __post_init__(self) -> None:
        if self.config.llm.provider is LLMProvider.OPENAI:
            self._client = OpenAI(api_key=self.config.llm.api_key)
        elif self.config.llm.provider is LLMProvider.AZURE_OPENAI:
            azure = self.config.llm.azure
            if azure is None:
                raise ValueError("Azure OpenAI configuration is missing.")
            self._client = AzureOpenAI(
                api_key=self.config.llm.api_key,
                api_version=azure.api_version,
                azure_endpoint=azure.endpoint,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")
        self._log = get_logger(__name__)

    def generate_targets(
        self, market: MarketSnapshot, account: AccountSnapshot
    ) -> list[TargetPosition]:
        prompt = self.context_builder.build(market, account)
        raw_text = self._call_model(prompt)
        return self._parse_targets(raw_text)

    def _call_model(self, prompt: str) -> str:
        retryer = Retrying(
            stop=stop_after_attempt(self.config.llm.retry_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                model_name = (
                    self.config.llm.model
                    if self.config.llm.provider is LLMProvider.OPENAI
                    else self._azure_deployment()
                )
                response = self._client.responses.create(
                    model=model_name,
                    input=[
                        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    ],
                    temperature=self.config.llm.temperature,
                    max_output_tokens=self.config.llm.max_output_tokens,
                    response_format={"type": "json_object"},
                )
                segments = []
                for item in response.output:
                    for content in item.content:
                        if content.type == "output_text":
                            segments.append(content.text)
                raw = "".join(segments).strip()
                if not raw:
                    raise ValueError("LLM response was empty")
                self._log.debug("LLM raw output: %s", raw)
                return raw
        raise RuntimeError("LLM retry loop exited without returning output")

    def _azure_deployment(self) -> str:
        azure = self.config.llm.azure
        if azure is None:
            raise ValueError("Azure deployment configuration missing.")
        return azure.deployment

    def _parse_targets(self, raw_text: str) -> list[TargetPosition]:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response could not be decoded as JSON") from exc

        targets_raw = payload.get("targets")
        if not isinstance(targets_raw, list):
            raise ValueError("LLM response missing 'targets' array")

        targets: list[TargetPosition] = []
        errors: list[str] = []
        for item in targets_raw:
            try:
                schema = TargetSchema.model_validate(item)
                targets.append(schema.to_domain())
            except ValidationError as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError("Failed to parse targets: " + "; ".join(errors))
        # Enforce symbol coverage by ensuring the model outputs every tracked symbol.
        missing = set(self.config.symbols) - {target.symbol for target in targets}
        if missing:
            raise ValueError(f"LLM response missing symbols: {sorted(missing)}")
        return targets
