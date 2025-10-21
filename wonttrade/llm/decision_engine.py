"""LLM-based decision engine leveraging pydantic-ai."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import AppConfig, LLMProvider
from ..context import LLMContextBuilder
from ..models import AccountSnapshot, DecisionResult, MarketSnapshot, TargetPosition
from ..telemetry.logger import get_logger

_SYSTEM_PROMPT = (
    "You are the decision engine, operating on Hyperliquid perpetual futures.\n\n"
    "You do not predict; you design trading structures that absorb uncertainty and capture asymmetry.\n\n"
    "At every decision cycle:\n"
    "1. Recall the previous decision and current market/account context. Note whether the prior structure still holds.\n"
    "2. Reflect in Simplified Chinese, briefly stating whether the prior plan remains valid and why. Maintain a calm, structural tone.\n"
    "3. If invalid, rebuild the structure with bounded risk, feedback awareness, and patience. Keep target_size unchanged when the plan remains valid.\n\n"
    "Respond only with JSON matching this schema:\n"
    "{\n"
    '  \"explanation\": \"string (Simplified Chinese, concise structural reasoning)\",\n'
    '  \"invalidation_condition\": \"string (Simplified Chinese, concrete break condition)\",\n'
    '  \"targets\": [\n'
    "    {\n"
    '      \"symbol\": \"string (e.g. BTC-PERP)\",\n'
    '      \"target_size\": \"number (desired net position; positive=long, negative=short)\",\n'
    '      \"stop_loss\": \"number\",\n'
    '      \"take_profit\": \"number\",\n'
    '      \"confidence\": \"number between 0 and 1\",\n'
    '      \"rationale\": \"string (简短的结构逻辑说明)\",\n'
    '      \"margin\": \"optional number representing leverage fraction\"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Describe only the intended final positioning, not the order execution steps.\n"
    "Guiding principles: treat error as information, size positions modestly out of respect for the unknown, leverage time, and maintain structural clarity even in rest."
)

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


class DecisionSchema(BaseModel):
    explanation: str
    invalidation_condition: str
    targets: list[TargetSchema]


@dataclass(slots=True)
class LLMDecisionEngine:
    """Encapsulates prompt generation and response parsing."""

    config: AppConfig
    context_builder: LLMContextBuilder
    _agent: Agent[None, DecisionSchema] = field(init=False, repr=False)
    _log: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        model = self._build_model()
        self._agent = Agent(
            model=model,
            instructions=_SYSTEM_PROMPT,
            output_type=DecisionSchema,
        )
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
        decision_payload, raw_text = self._call_model(prompt)
        explanation = decision_payload.explanation.strip()
        invalidation_condition = decision_payload.invalidation_condition.strip()
        targets = [item.to_domain() for item in decision_payload.targets]
        missing = set(self.config.symbols) - {target.symbol for target in targets}
        if missing:
            raise ValueError(f"LLM response missing symbols: {sorted(missing)}")
        self._log.debug("LLM 原始输出：%s", _truncate(raw_text))
        return DecisionResult(
            explanation=explanation,
            invalidation_condition=invalidation_condition,
            targets=targets,
        )

    def _call_model(self, prompt: str) -> tuple[DecisionSchema, str]:
        retryer = Retrying(
            stop=stop_after_attempt(self.config.llm.retry_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                self._log.debug(
                    "LLM 请求参数：模型=%s，温度=%.3f，请求预览=%s",
                    self.config.llm.model,
                    self.config.llm.temperature,
                    _truncate(prompt),
                )
                settings = self._build_model_settings()
                try:
                    run = self._agent.run_sync(prompt, settings=settings)
                except TypeError:
                    run = self._agent.run_sync(prompt, model_settings=settings)
                payload = run.data
                raw_text = getattr(run, "output_text", "") or ""
                if not isinstance(payload, DecisionSchema):
                    raise ValueError("LLM response failed to parse into DecisionSchema.")
                return payload, raw_text
        raise RuntimeError("LLM retry loop exited without returning output")

    def _build_model(self) -> Any:
        api_key = self.config.llm.api_key or ""
        if self.config.llm.provider is LLMProvider.OPENAI:
            provider = OpenAIProvider(
                base_url=self.config.llm.endpoint,
                api_key=api_key or None,
            )
            model_name = self.config.llm.model
        elif self.config.llm.provider is LLMProvider.AZURE_OPENAI:
            azure = self.config.llm.azure
            if azure is None:
                raise ValueError("Azure OpenAI configuration is missing.")
            provider = AzureProvider(
                azure_endpoint=azure.endpoint,
                api_version=azure.api_version,
                api_key=api_key or None,
            )
            model_name = azure.deployment
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")
        return OpenAIChatModel(
            model_name=model_name,
            provider=provider,
        )

    def _build_model_settings(self) -> ModelSettings:
        """Create model settings compatible with the installed pydantic-ai version."""

        settings = ModelSettings(temperature=self.config.llm.temperature)

        if hasattr(settings, "max_output_tokens"):
            setattr(settings, "max_output_tokens", self.config.llm.max_output_tokens)

        timeout_value = self.config.llm.request_timeout_seconds
        if hasattr(settings, "timeout"):
            setattr(settings, "timeout", timeout_value)
        elif hasattr(settings, "response_timeout"):
            setattr(settings, "response_timeout", timeout_value)

        return settings


def _truncate(text: str, limit: int = _LOG_PREVIEW_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"
