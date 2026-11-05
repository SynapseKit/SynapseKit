from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ..effort import EffortClassifier, EffortInput
from ..reasoning import BaseReasoningProvider, ReasoningResponse, ReasoningStreamChunk
from ..utils import extract_text

_DEFAULT_THINKING_BUDGET = 1024


class AnthropicThinking(BaseReasoningProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        budget_tokens: int | None = None,
        thinking: bool = True,
        effort: EffortInput | None = None,
        effort_classifier: EffortClassifier | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            budget_tokens=budget_tokens,
            thinking=thinking,
            provider="anthropic",
            effort=effort,
            effort_classifier=effort_classifier,
        )
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install synapsekit[anthropic]"
                ) from None
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    def _thinking_payload(self) -> dict[str, Any] | None:
        if not self.thinking:
            return None
        if self.effort is not None:
            return {"type": "adaptive"}
        budget = self.budget_tokens if self.budget_tokens is not None else _DEFAULT_THINKING_BUDGET
        return {"type": "enabled", "budget_tokens": budget}

    async def generate(self, prompt: str) -> ReasoningResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "timeout": 30.0,
        }
        effort = self._effort_for_prompt(prompt)
        thinking_payload = self._thinking_payload()
        if thinking_payload is not None:
            kwargs["thinking"] = thinking_payload
        if effort is not None:
            kwargs["output_config"] = {"effort": effort.value}

        try:
            response = await client.messages.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e

        answer, thinking_text = _extract_content_blocks(getattr(response, "content", []))
        if not self.thinking:
            thinking_text = None

        usage = getattr(response, "usage", None)
        # Anthropic's output_tokens covers all output (thinking blocks + text blocks).
        # The SDK does not expose a separate thinking_tokens field, so it is always 0.
        answer_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        thinking_tokens = int(getattr(usage, "thinking_tokens", 0) or 0)
        if not self.thinking:
            thinking_tokens = 0
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = thinking_tokens + answer_tokens

        return ReasoningResponse(
            answer=answer,
            thinking=thinking_text,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            total_tokens=total_tokens,
            model=self.model,
            provider=self.provider,
        )

    async def stream(self, prompt: str) -> AsyncGenerator[ReasoningStreamChunk, None]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "stream": True,
            "timeout": 30.0,
        }
        effort = self._effort_for_prompt(prompt)
        thinking_payload = self._thinking_payload()
        if thinking_payload is not None:
            kwargs["thinking"] = thinking_payload
        if effort is not None:
            kwargs["output_config"] = {"effort": effort.value}

        try:
            stream = await client.messages.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e

        if stream is None:
            return

        try:
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type in {
                    "content_block_delta",
                    "message_delta",
                    "thinking_delta",
                    "text_delta",
                }:
                    delta = getattr(event, "delta", event)
                    thinking_text = _extract_thinking_text(delta)
                    if not self.thinking:
                        thinking_text = ""
                    if thinking_text:
                        yield ReasoningStreamChunk(text=thinking_text, is_thinking=True)
                    answer_text = _extract_answer_text(delta)
                    if answer_text:
                        yield ReasoningStreamChunk(text=answer_text, is_thinking=False)
        except Exception as e:
            raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e


def _extract_content_blocks(content: Any) -> tuple[str, str | None]:
    answer_parts: list[str] = []
    thinking_parts: list[str] = []
    if not isinstance(content, list):
        return "", None

    for block in content:
        block_type = _extract_attr_or_key(block, "type")
        if block_type in {"thinking", "reasoning"}:
            text = extract_text(
                _extract_attr_or_key(block, "thinking") or _extract_attr_or_key(block, "text")
            )
            if text:
                thinking_parts.append(text)
        elif block_type == "text":
            text = extract_text(_extract_attr_or_key(block, "text"))
            if text:
                answer_parts.append(text)

    return "".join(answer_parts), "".join(thinking_parts) or None


def _extract_thinking_text(delta: Any) -> str:
    for key in ("thinking", "thinking_delta", "reasoning", "reasoning_delta"):
        text = _extract_attr_or_key(delta, key)
        extracted = extract_text(text)
        if extracted:
            return extracted
        if isinstance(text, str) and text:
            return text
        if isinstance(text, dict):
            inner = text.get("text") or text.get("thinking")
            if isinstance(inner, str) and inner:
                return inner
    return ""


def _extract_answer_text(delta: Any) -> str:
    for key in ("text", "text_delta"):
        text = _extract_attr_or_key(delta, key)
        extracted = extract_text(text)
        if extracted:
            return extracted
        if isinstance(text, str) and text:
            return text
        if isinstance(text, dict):
            inner = text.get("text")
            if isinstance(inner, str) and inner:
                return inner
    return ""


def _extract_attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
