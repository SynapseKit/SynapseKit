from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ..effort import (
    GEMINI_THINKING_BUDGETS,
    EffortClassifier,
    EffortInput,
)
from ..reasoning import BaseReasoningProvider, ReasoningResponse, ReasoningStreamChunk
from ..utils import extract_text

_DEFAULT_THINKING_BUDGET = 1024


class GoogleThinking(BaseReasoningProvider):
    """
    Google Gemini reasoning provider with extended thinking.

    NOTE:
    Gemini thinking detection is heuristic.
    The API does not guarantee stable flags for reasoning vs answer.
    Detection may vary across model versions.
    """

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
            provider="google",
            effort=effort,
            effort_classifier=effort_classifier,
        )
        self._model: Any = None
        self._effort_client: Any = None

    def _get_model(self):
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai required: pip install synapsekit[gemini]"
                ) from None
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(model_name=self.model)
        return self._model

    def _get_effort_client(self):
        if self._effort_client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai>=1.0 required for Gemini effort routing; "
                    "install synapsekit[gemini]"
                ) from None
            self._effort_client = genai.Client(api_key=self.api_key)
        return self._effort_client

    def _effort_generation_config(self, prompt: str) -> Any:
        effort = self._effort_for_prompt(prompt)
        if effort is None:
            return None
        try:
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai>=1.0 required for Gemini effort routing; install synapsekit[gemini]"
            ) from None
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGETS[effort]),
            http_options=types.HttpOptions(timeout=30_000),
        )

    def _generation_config(self) -> dict[str, Any] | None:
        if not self.thinking:
            return None
        budget = self.budget_tokens if self.budget_tokens is not None else _DEFAULT_THINKING_BUDGET
        return {"thinkingConfig": {"thinkingBudget": budget}}

    async def generate(self, prompt: str) -> ReasoningResponse:
        if getattr(self, "effort", None) is None:
            model = self._get_model()
            kwargs: dict[str, Any] = {}
            generation_config = self._generation_config()
            if generation_config is not None:
                kwargs["generation_config"] = generation_config
            kwargs["request_options"] = {"timeout": 30.0}

            try:
                response = await model.generate_content_async(prompt, **kwargs)
            except Exception as e:
                raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e
        else:
            client = self._get_effort_client()
            config = self._effort_generation_config(prompt)
            try:
                response = await client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            except Exception as e:
                raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e

        if response is None:
            return self._empty_response()

        answer, thinking = _extract_response_parts(response)
        if not self.thinking:
            thinking = None

        usage = getattr(response, "usage_metadata", None)
        answer_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        if not self.thinking:
            thinking_tokens = 0
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
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
        if getattr(self, "effort", None) is None:
            model = self._get_model()
            kwargs: dict[str, Any] = {"stream": True}
            generation_config = self._generation_config()
            if generation_config is not None:
                kwargs["generation_config"] = generation_config
            kwargs["request_options"] = {"timeout": 30.0}

            try:
                stream = await model.generate_content_async(prompt, **kwargs)
            except Exception as e:
                raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e
        else:
            client = self._get_effort_client()
            config = self._effort_generation_config(prompt)
            try:
                stream = await client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            except Exception as e:
                raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e

        if stream is None:
            return

        try:
            async for chunk in stream:
                if not hasattr(chunk, "candidates"):
                    text = extract_text(getattr(chunk, "text", ""))
                    if text:
                        yield ReasoningStreamChunk(text=text, is_thinking=False)
                    continue

                thinking_text, answer_text = _extract_chunk_parts(chunk)
                if not self.thinking:
                    thinking_text = ""
                if thinking_text:
                    yield ReasoningStreamChunk(text=thinking_text, is_thinking=True)
                if answer_text:
                    yield ReasoningStreamChunk(text=answer_text, is_thinking=False)
        except Exception as e:
            raise RuntimeError(f"{self.provider} reasoning request failed: {e}") from e


def _extract_response_parts(response: Any) -> tuple[str, str | None]:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        text = extract_text(getattr(response, "text", ""))
        return text, None

    answer_parts: list[str] = []
    thinking_parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        for part in parts:
            part_text = _extract_part_text(part)
            if not part_text:
                continue
            if _is_thinking_part(part):
                thinking_parts.append(part_text)
            else:
                answer_parts.append(part_text)
    return "".join(answer_parts), "".join(thinking_parts) or None


def _extract_chunk_parts(chunk: Any) -> tuple[str, str]:
    if chunk is None:
        return "", ""

    if hasattr(chunk, "candidates") and chunk.candidates:
        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        for candidate in chunk.candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                text = _extract_part_text(part)
                if not text:
                    continue
                if _is_thinking_part(part):
                    thinking_parts.append(text)
                else:
                    answer_parts.append(text)
        return "".join(thinking_parts), "".join(answer_parts)

    text = extract_text(getattr(chunk, "text", ""))
    return "", text


def _extract_part_text(part: Any) -> str:
    return extract_text(part)


def _is_thinking_part(part: Any) -> bool:
    for attr in ("thought", "is_thought", "is_thinking"):
        value = getattr(part, attr, None)
        if isinstance(value, bool):
            return value
    if isinstance(part, dict):
        for key in ("thought", "isThought", "is_thought", "isThinking", "is_thinking"):
            value = part.get(key)
            if isinstance(value, bool):
                return value
    return False
