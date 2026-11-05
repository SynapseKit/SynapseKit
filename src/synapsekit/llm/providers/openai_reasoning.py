from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ..effort import EffortClassifier, EffortInput
from ..reasoning import BaseReasoningProvider, ReasoningResponse, ReasoningStreamChunk
from ..utils import extract_reasoning, extract_reasoning_tokens, extract_text


class OpenAIReasoning(BaseReasoningProvider):
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
            provider="openai",
            effort=effort,
            effort_classifier=effort_classifier,
        )
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install synapsekit[openai]"
                ) from None
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    def _reasoning_effort(self, prompt: str = "") -> str:
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

    async def generate(self, prompt: str) -> ReasoningResponse:
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=self._reasoning_effort(prompt),
                timeout=30.0,
            )
        except Exception as e:
            raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e

        choices = getattr(response, "choices", None) or []
        if not choices:
            return self._empty_response()

        message = getattr(choices[0], "message", None)
        answer = extract_text(getattr(message, "content", ""))
        thinking = extract_reasoning(message)
        if not self.thinking:
            thinking = None

        usage = getattr(response, "usage", None)
        answer_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        thinking_tokens = extract_reasoning_tokens(usage)
        if not self.thinking:
            thinking_tokens = 0
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = thinking_tokens + answer_tokens

        return ReasoningResponse(
            answer=answer,
            thinking=thinking,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            total_tokens=total_tokens,
            model=self.model,
            provider=self.provider,
        )

    async def stream(self, prompt: str) -> AsyncGenerator[ReasoningStreamChunk, None]:
        client = self._get_client()
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=self._reasoning_effort(prompt),
                stream=True,
                timeout=30.0,
            )
        except Exception as e:
            raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e

        if stream is None:
            return

        try:
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                thinking_text = extract_reasoning(delta)
                if not self.thinking:
                    thinking_text = None
                if thinking_text:
                    yield ReasoningStreamChunk(text=thinking_text, is_thinking=True)

                answer_text = extract_text(getattr(delta, "content", None))
                if answer_text:
                    yield ReasoningStreamChunk(text=answer_text, is_thinking=False)
        except Exception as e:
            raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e
