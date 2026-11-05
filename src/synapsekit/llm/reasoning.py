from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from ..observability.tracer import COST_TABLE
from .base import BaseLLM, LLMConfig, _messages_to_prompt
from .effort import (
    EffortClassifier,
    EffortInput,
    EffortLevel,
    normalize_effort,
    resolve_effort,
)
from .utils import extract_reasoning_tokens, extract_text


@dataclass
class ReasoningResponse:
    answer: str
    thinking: str | None
    thinking_tokens: int
    answer_tokens: int
    total_tokens: int
    model: str
    provider: str


@dataclass
class ReasoningUsage:
    thinking_tokens: int
    answer_tokens: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    thinking_cost_usd: float
    answer_cost_usd: float
    total_cost_usd: float
    model: str
    provider: str


@dataclass
class ReasoningStreamChunk:
    text: str
    is_thinking: bool


class BaseReasoningProvider(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        budget_tokens: int | None = None,
        thinking: bool = True,
        provider: str | None = None,
        effort: EffortInput | None = None,
        effort_classifier: EffortClassifier | None = None,
    ) -> None:
        normalized_effort = normalize_effort(effort)
        if budget_tokens is not None and normalized_effort is not None:
            raise ValueError("budget_tokens and effort cannot be used together")
        if normalized_effort is not None and not thinking:
            raise ValueError("effort cannot be used when thinking is disabled")
        self.model = model
        self.api_key = api_key
        self.budget_tokens = budget_tokens
        self.thinking = thinking
        self.provider = provider or "unknown"
        self.effort = normalized_effort
        self.effort_classifier = effort_classifier

    def _effort_for_prompt(self, prompt: str) -> EffortLevel | None:
        return resolve_effort(self.effort, prompt, self.effort_classifier)

    @abstractmethod
    async def generate(self, prompt: str) -> ReasoningResponse:
        raise NotImplementedError

    @abstractmethod
    async def stream(self, prompt: str) -> AsyncGenerator[ReasoningStreamChunk, None]:
        raise NotImplementedError

    def _empty_response(self) -> ReasoningResponse:
        return ReasoningResponse(
            answer="",
            thinking=None,
            thinking_tokens=0,
            answer_tokens=0,
            total_tokens=0,
            model=self.model,
            provider=self.provider,
        )


class ReasoningLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        budget_tokens: int | None = None,
        thinking: bool = True,
        effort: EffortInput | None = None,
        effort_classifier: EffortClassifier | None = None,
    ) -> None:
        normalized_effort = normalize_effort(effort)
        if budget_tokens is not None and normalized_effort is not None:
            raise ValueError("budget_tokens and effort cannot be used together")
        if normalized_effort is not None and not thinking:
            raise ValueError("effort cannot be used when thinking is disabled")
        provider = self._detect_provider(model)
        if normalized_effort is not None and provider not in {"openai", "anthropic", "google"}:
            raise ValueError(
                "ReasoningLLM supports effort routing only for OpenAI, Anthropic, and Google."
            )
        config = LLMConfig(
            model=model,
            api_key=api_key or "",
            provider=provider,
        )
        super().__init__(config)

        self.model = model
        self.provider = provider
        self.budget_tokens = budget_tokens
        self.thinking = thinking
        self.effort = normalized_effort
        self.effort_classifier = effort_classifier
        self._provider = self._build_provider(
            provider=provider,
            model=model,
            api_key=api_key,
            budget_tokens=budget_tokens,
            thinking=thinking,
            effort=normalized_effort,
            effort_classifier=effort_classifier,
        )
        self.last_response: ReasoningResponse | None = None
        self.last_usage: ReasoningUsage | None = None

    @staticmethod
    def _detect_provider(model: str) -> str:
        normalized_model = model.lower()
        if normalized_model.startswith(("o1", "o3")):
            return "openai"
        if "deepseek" in normalized_model:
            return "deepseek"
        if normalized_model.startswith("gemini"):
            return "google"
        if normalized_model.startswith("claude"):
            return "anthropic"
        if normalized_model.startswith("qwq"):
            return "qwen"
        raise ValueError("Unsupported reasoning model")

    def _build_provider(
        self,
        provider: str,
        model: str,
        api_key: str | None,
        budget_tokens: int | None,
        thinking: bool,
        effort: EffortInput | None,
        effort_classifier: EffortClassifier | None,
    ) -> BaseReasoningProvider:
        if provider == "openai":
            from .providers.openai_reasoning import OpenAIReasoning

            return OpenAIReasoning(
                model=model,
                api_key=api_key,
                budget_tokens=budget_tokens,
                thinking=thinking,
                effort=effort,
                effort_classifier=effort_classifier,
            )
        if provider == "anthropic":
            from .providers.anthropic_thinking import AnthropicThinking

            return AnthropicThinking(
                model=model,
                api_key=api_key,
                budget_tokens=budget_tokens,
                thinking=thinking,
                effort=effort,
                effort_classifier=effort_classifier,
            )
        if provider == "google":
            from .providers.google_thinking import GoogleThinking

            return GoogleThinking(
                model=model,
                api_key=api_key,
                budget_tokens=budget_tokens,
                thinking=thinking,
                effort=effort,
                effort_classifier=effort_classifier,
            )
        if provider in {"deepseek", "qwen"}:
            from .providers.deepseek_r1 import DeepSeekR1Reasoning

            return DeepSeekR1Reasoning(
                model=model,
                api_key=api_key,
                budget_tokens=budget_tokens,
                thinking=thinking,
                provider=provider,
            )
        raise ValueError("Unsupported reasoning model")

    def _record_usage(self, response: ReasoningResponse) -> ReasoningUsage:
        output_tokens = response.answer_tokens + response.thinking_tokens
        total_tokens = response.total_tokens or output_tokens
        input_tokens = max(0, total_tokens - output_tokens)

        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

        costs = COST_TABLE.get(self.model, {})
        output_cost = output_tokens * costs.get("output", 0.0)
        thinking_cost = response.thinking_tokens * costs.get("output", 0.0)
        answer_cost = response.answer_tokens * costs.get("output", 0.0)
        total_cost = input_tokens * costs.get("input", 0.0) + output_cost

        usage = ReasoningUsage(
            thinking_tokens=response.thinking_tokens,
            answer_tokens=response.answer_tokens,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_cost_usd=thinking_cost,
            answer_cost_usd=answer_cost,
            total_cost_usd=total_cost,
            model=self.model,
            provider=self.provider,
        )
        self.last_response = response
        self.last_usage = usage
        return usage

    async def agenerate(self, prompt: str) -> ReasoningResponse:
        await self._acquire_rate_limit()
        response = await self._provider.generate(prompt)
        self._record_usage(response)
        return response

    async def generate(self, prompt: str, **kw: Any) -> str:  # type: ignore[override]
        del kw
        response = await self.agenerate(prompt)
        return response.answer

    async def generate_with_messages(self, messages: list[dict], **kw: Any) -> str:
        prompt = _messages_to_prompt(messages)
        return await self.generate(prompt, **kw)

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str, None]:
        del kw
        prompt = _messages_to_prompt(messages)
        async for token in self.stream(prompt):
            yield token

    async def astream(self, prompt: str) -> AsyncGenerator[ReasoningStreamChunk, None]:
        seen_answer = False
        async for chunk in self._provider.stream(prompt):
            if chunk.is_thinking:
                # Enforce reasoning-before-answer ordering.
                # Any thinking emitted after answer is ignored by design.
                if seen_answer:
                    continue
                yield chunk
                continue

            seen_answer = True
            yield chunk

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
        del kw
        async for chunk in self.astream(prompt):
            if not chunk.is_thinking:
                yield chunk.text

    async def _call_with_tools_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.provider == "openai":
            return await self._openai_call_with_tools(messages, tools)
        if self.provider == "anthropic":
            return await self._anthropic_call_with_tools(messages, tools)
        raise NotImplementedError(
            f"ReasoningLLM does not support native function calling for provider '{self.provider}'."
        )

    def _effort_for_prompt(self, prompt: str) -> EffortLevel | None:
        return resolve_effort(self.effort, prompt, self.effort_classifier)

    def _openai_reasoning_effort(self, prompt: str = "") -> str:
        effort = self._effort_for_prompt(prompt)
        if effort is not None:
            return effort.value
        if not self.thinking:
            return "low"
        if self.budget_tokens is None:
            return "medium"
        if self.budget_tokens >= 2048:
            return "high"
        if self.budget_tokens <= 512:
            return "low"
        return "medium"

    def _get_openai_client(self):
        if not hasattr(self, "_openai_client"):
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install synapsekit[openai]"
                ) from None
            self._openai_client = AsyncOpenAI(api_key=self.config.api_key)
        return self._openai_client

    async def _openai_call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        client = self._get_openai_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            reasoning_effort=self._openai_reasoning_effort(_messages_to_prompt(messages)),
            timeout=30.0,
        )
        msg = response.choices[0].message

        usage = getattr(response, "usage", None)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        thinking_tokens = extract_reasoning_tokens(usage)
        answer_tokens = max(0, completion_tokens - thinking_tokens)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = thinking_tokens + answer_tokens

        reasoning_response = ReasoningResponse(
            answer=extract_text(getattr(msg, "content", "")),
            thinking=None,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            total_tokens=total_tokens,
            model=self.model,
            provider=self.provider,
        )
        self._record_usage(reasoning_response)

        if msg.tool_calls:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in msg.tool_calls
                ],
            }
        return {"content": msg.content, "tool_calls": None}

    def _get_anthropic_client(self):
        if not hasattr(self, "_anthropic_client"):
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install synapsekit[anthropic]"
                ) from None
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        return self._anthropic_client

    def _anthropic_thinking_payload(self) -> dict[str, Any] | None:
        if not self.thinking:
            return None
        budget = self.budget_tokens if self.budget_tokens is not None else 1024
        return {"type": "enabled", "budget_tokens": budget}

    async def _anthropic_call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        client = self._get_anthropic_client()
        system = self.config.system_prompt
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            elif m.get("role") == "tool":
                user_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": m["content"],
                            }
                        ],
                    }
                )
            else:
                user_messages.append(m)

        anthropic_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": user_messages,
            "tools": anthropic_tools,
            "max_tokens": self.config.max_tokens,
        }
        prompt = _messages_to_prompt(messages)
        effort = self._effort_for_prompt(prompt)
        thinking_payload = (
            {"type": "adaptive"}
            if effort is not None and self.thinking
            else self._anthropic_thinking_payload()
        )
        if thinking_payload is not None:
            kwargs["thinking"] = thinking_payload
        if effort is not None:
            kwargs["output_config"] = {"effort": effort.value}

        response = await client.messages.create(**kwargs)

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        thinking_tokens = int(getattr(usage, "thinking_tokens", 0) or 0)
        answer_tokens = max(0, output_tokens - thinking_tokens)
        total_tokens = input_tokens + output_tokens

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]
        content_text = text_blocks[0].text if text_blocks else ""

        reasoning_response = ReasoningResponse(
            answer=extract_text(content_text),
            thinking=None,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            total_tokens=total_tokens,
            model=self.model,
            provider=self.provider,
        )
        self._record_usage(reasoning_response)

        if tool_uses:
            return {
                "content": None,
                "tool_calls": [
                    {"id": tu.id, "name": tu.name, "arguments": tu.input} for tu in tool_uses
                ],
            }

        return {"content": content_text, "tool_calls": None}
