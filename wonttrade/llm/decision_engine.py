"""LLM-based decision engine."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import AppConfig, LLMProvider
from ..context import LLMContextBuilder
from ..models import AccountSnapshot, DecisionResult, MarketSnapshot, TargetPosition
from ..telemetry.logger import get_logger

_SYSTEM_PROMPT = (
    "You are an autonomous trading strategy for Hyperliquid perpetual futures. "
    "Review the previous decision summary and the latest market/account data. "
    "First state in Simplified Chinese whether the prior plan remains valid and why. "
    "Then describe your updated plan. Always respond with a JSON object containing: "
    "(1) 'explanation' (Chinese string); (2) 'invalidation_condition' (Chinese string describing "
    "the concrete situation that invalidates the plan); (3) 'targets' array. "
    "Each target must include symbol, target_size, stop_loss, take_profit, confidence, "
    "rationale, and optionally margin. "
    "Stop-loss and take-profit must be numeric levels aligned with the trade direction. "
    "If the previous plan remains valid, keep target_size unchanged. "
    "Only describe the desired terminal state; do not outline order sequences."
)

_TARGET_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "target_size": {"type": "number"},
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "margin": {"type": "number"},
    },
    "required": [
        "symbol",
        "target_size",
        "stop_loss",
        "take_profit",
        "confidence",
        "rationale",
    ],
}

_RESPONSE_JSON_SCHEMA = {
    "name": "wonttrade_targets",
    "strict": "true",
    "schema": {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "中文解释，说明为什么这样调整仓位",
            },
            "invalidation_condition": {
                "type": "string",
                "description": "如果满足该条件则必须放弃本次分析",
            },
            "targets": {
                "type": "array",
                "items": _TARGET_JSON_SCHEMA,
                "minItems": 1,
            },
        },
        "required": ["explanation", "invalidation_condition", "targets"],
    },
}

_LOG_PREVIEW_LENGTH = 800


class TargetSchema(BaseModel):
    symbol: str
    target_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    rationale: str
    margin: float | None = None

    def to_domain(self) -> TargetPosition:
        return TargetPosition(
            symbol=self.symbol,
            target_size=self.target_size,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            confidence=self.confidence,
            rationale=self.rationale,
            margin=self.margin,
        )


@dataclass(slots=True)
class LLMDecisionEngine:
    """Encapsulates prompt generation and response parsing."""

    config: AppConfig
    context_builder: LLMContextBuilder
    _client: Any = field(init=False, repr=False)
    _log: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.llm.provider is LLMProvider.OPENAI:
            client_kwargs: dict[str, Any] = {"api_key": self.config.llm.api_key}
            if self.config.llm.endpoint:
                client_kwargs["base_url"] = self.config.llm.endpoint
            self._client = OpenAI(**client_kwargs)
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

    def generate_decision(
        self,
        market: MarketSnapshot,
        account: AccountSnapshot,
        *,
        previous_decision: DecisionResult | None = None,
    ) -> DecisionResult:
        prompt = self.context_builder.build(
            market,
            account,
            previous_decision=previous_decision,
        )
        raw_text = self._call_model(prompt)
        explanation, invalidation_condition, targets = self._parse_decision(raw_text)
        return DecisionResult(
            explanation=explanation,
            invalidation_condition=invalidation_condition,
            targets=targets,
        )

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
                self._log.debug(
                    "LLM 请求参数：模型=%s，温度=%.3f，请求预览=%s",
                    model_name,
                    self.config.llm.temperature,
                    _truncate(prompt),
                )
                response = self._client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_output_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": _RESPONSE_JSON_SCHEMA,
                    },
                )
                raw = self._extract_content(response)
                if not raw:
                    raise ValueError("LLM response was empty")
                self._log.debug("LLM 原始输出：%s", _truncate(raw))
                return raw
        raise RuntimeError("LLM retry loop exited without returning output")

    def _azure_deployment(self) -> str:
        azure = self.config.llm.azure
        if azure is None:
            raise ValueError("Azure deployment configuration missing.")
        return azure.deployment

    def _parse_decision(self, raw_text: str) -> tuple[str, str, list[TargetPosition]]:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response could not be decoded as JSON") from exc

        explanation_raw = payload.get("explanation")
        if not isinstance(explanation_raw, str) or not explanation_raw.strip():
            raise ValueError("LLM response missing 'explanation' string")
        explanation = explanation_raw.strip()

        invalidation_raw = payload.get("invalidation_condition")
        if not isinstance(invalidation_raw, str) or not invalidation_raw.strip():
            raise ValueError("LLM response missing 'invalidation_condition' string")
        invalidation_condition = invalidation_raw.strip()

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
        return explanation, invalidation_condition, targets

    def _extract_content(self, response: Any) -> str:
        """Normalize the client response and return textual content."""

        choices = getattr(response, "choices", None)
        payload: dict[str, Any] | None = None
        if choices is None:
            if hasattr(response, "model_dump"):
                try:
                    payload = response.model_dump()
                except Exception:  # pragma: no cover - defensive
                    payload = None
            elif isinstance(response, dict):
                payload = response
            if payload is not None:
                choices = payload.get("choices")
                if payload is not None:
                    self._log.debug(
                        "LLM 响应负载预览：%s",
                        _truncate(_to_json(payload)),
                    )
        else:
            try:
                payload = response.model_dump()
                self._log.debug("LLM 响应负载预览：%s", _truncate(_to_json(payload)))
            except Exception:  # pragma: no cover - defensive
                pass
        if not choices:
            raise ValueError("LLM response missing 'choices'.")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is None:
            raise ValueError("LLM response missing 'message'.")

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Iterable):
            fragments: list[str] = []
            for fragment in content:
                if isinstance(fragment, dict):
                    text = fragment.get("text")
                    if text:
                        fragments.append(str(text))
                elif isinstance(fragment, str):
                    fragments.append(fragment)
            return "".join(fragments).strip()
        raise ValueError("LLM message content has unsupported structure.")


def _truncate(text: str, limit: int = _LOG_PREVIEW_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _to_json(payload: Any) -> str:
    try:
        return json.dumps(payload, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(payload)
