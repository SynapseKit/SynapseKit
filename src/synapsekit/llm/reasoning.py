"""Unified adapter for extended / chain-of-thought reasoning models."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any, Literal

from .base import BaseLLM, LLMConfig

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)

_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": "o3",
    "anthropic": "claude-opus-4-5-20251001",
    "google": "gemini-2.0-flash-thinking-exp",
    "deepseek": "deepseek-reasoner",
}

Provider = Literal["openai", "anthropic", "google", "deepseek"]


def _strip_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text).strip()


def _extract_reasoning(text: str) -> str:
    m = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


class ReasoningLLM(BaseLLM):
    """Unified adapter for reasoning / chain-of-thought LLMs.

    Delegates to the appropriate provider LLM with extended-thinking params.
    Supported providers: ``openai``, ``anthropic``, ``google``, ``deepseek``.
    """

    def __init__(
        self,
        provider: Provider = "openai",
        model: str | None = None,
        api_key: str | None = None,
        budget_tokens: int = 8000,
        effort: Literal["low", "medium", "high"] = "medium",
    ) -> None:
        if provider not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"provider must be one of {list(_PROVIDER_DEFAULTS)}; got {provider!r}"
            )
        resolved_model = model or _PROVIDER_DEFAULTS[provider]
        config = LLMConfig(
            model=resolved_model,
            api_key=api_key or "",
            provider=provider,
        )
        super().__init__(config)
        self._provider = provider
        self._budget_tokens = budget_tokens
        self._effort = effort
        self._api_key = api_key
        self._delegate: BaseLLM | None = None

    # ------------------------------------------------------------------
    # Internal: build provider delegate lazily
    # ------------------------------------------------------------------

    def _get_delegate(self) -> BaseLLM:
        if self._delegate is not None:
            return self._delegate

        p = self._provider
        model = self.config.model
        key = self._api_key

        if p == "openai":
            self._delegate = _build_openai_reasoning(model, key, self._effort)
        elif p == "anthropic":
            self._delegate = _build_anthropic_reasoning(model, key, self._budget_tokens)
        elif p == "google":
            self._delegate = _build_google_reasoning(model, key)
        elif p == "deepseek":
            self._delegate = _build_deepseek_reasoning(model, key)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported provider: {p!r}")

        return self._delegate

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        delegate = self._get_delegate()
        async for token in delegate.stream(prompt, **kw):
            yield token

    async def generate(self, prompt: str, **kw: Any) -> str:
        raw = await self._get_delegate().generate(prompt, **kw)
        return _strip_thinking(raw)

    async def generate_with_reasoning(self, prompt: str, **kw: Any) -> tuple[str, str]:
        """Return ``(reasoning, answer)`` tuple.

        For providers that return ``<thinking>…</thinking>`` blocks, the
        reasoning is extracted from that block; otherwise the full raw output
        is used as reasoning and the stripped text as the answer.
        """
        raw = await self._get_delegate().generate(prompt, **kw)
        reasoning = _extract_reasoning(raw)
        answer = _strip_thinking(raw)
        if not reasoning:
            # Provider doesn't emit <thinking> blocks — use full text as both
            reasoning = raw
        return reasoning, answer


# ------------------------------------------------------------------
# Provider-specific builder helpers
# ------------------------------------------------------------------


def _build_openai_reasoning(model: str, api_key: str | None, effort: str) -> BaseLLM:
    """OpenAI o-series reasoning models use ``reasoning_effort`` param."""

    import os

    from .base import LLMConfig

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai package required: pip install synapsekit[openai]") from None

    _key = api_key or os.environ.get("OPENAI_API_KEY", "")
    cfg = LLMConfig(model=model, api_key=_key, provider="openai")

    class _OpenAIReasoningLLM(BaseLLM):
        def __init__(self) -> None:
            super().__init__(cfg)
            self._client = AsyncOpenAI(api_key=_key)

        async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=effort,
                stream=False,
            )
            text = response.choices[0].message.content or ""
            self._output_tokens += response.usage.completion_tokens or 0
            self._input_tokens += response.usage.prompt_tokens or 0
            for ch in text:
                yield ch

    return _OpenAIReasoningLLM()


def _build_anthropic_reasoning(model: str, api_key: str | None, budget_tokens: int) -> BaseLLM:
    """Anthropic extended thinking via ``thinking`` param."""

    import os

    from .base import LLMConfig

    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: pip install synapsekit[anthropic]") from None

    _key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    cfg = LLMConfig(model=model, api_key=_key, provider="anthropic", max_tokens=16000)

    class _AnthropicReasoningLLM(BaseLLM):
        def __init__(self) -> None:
            super().__init__(cfg)
            self._client = anthropic.AsyncAnthropic(api_key=_key)

        async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                messages=[{"role": "user", "content": prompt}],
            )
            self._input_tokens += response.usage.input_tokens or 0
            self._output_tokens += response.usage.output_tokens or 0
            # Emit thinking block then text block
            for block in response.content:
                if block.type == "thinking":
                    text = f"<thinking>{block.thinking}</thinking>"
                    for ch in text:
                        yield ch
                elif block.type == "text":
                    for ch in block.text:
                        yield ch

    return _AnthropicReasoningLLM()


def _build_google_reasoning(model: str, api_key: str | None) -> BaseLLM:
    """Google Gemini thinking-exp model."""

    import os

    from .base import LLMConfig

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai required: pip install synapsekit[gemini]") from None

    _key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    cfg = LLMConfig(model=model, api_key=_key, provider="google")
    genai.configure(api_key=_key)
    _gmodel = genai.GenerativeModel(model_name=model)

    class _GoogleReasoningLLM(BaseLLM):
        def __init__(self) -> None:
            super().__init__(cfg)

        async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
            async for chunk in await _gmodel.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": kw.get("max_tokens", self.config.max_tokens),
                },
                stream=True,
            ):
                if chunk.text:
                    self._output_tokens += 1
                    yield chunk.text

    return _GoogleReasoningLLM()


def _build_deepseek_reasoning(model: str, api_key: str | None) -> BaseLLM:
    """DeepSeek reasoner — exposes ``reasoning_content`` in chain-of-thought field."""

    import os

    from .base import LLMConfig

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai package required: pip install synapsekit[openai]") from None

    _key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    cfg = LLMConfig(
        model=model,
        api_key=_key,
        provider="deepseek",
        max_tokens=8192,
    )
    _client = AsyncOpenAI(api_key=_key, base_url="https://api.deepseek.com")

    class _DeepSeekReasoningLLM(BaseLLM):
        def __init__(self) -> None:
            super().__init__(cfg)

        async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
            response = await _client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or ""
            answer = msg.content or ""
            full = f"<thinking>{reasoning}</thinking>{answer}" if reasoning else answer
            if hasattr(response, "usage") and response.usage:
                self._input_tokens += response.usage.prompt_tokens or 0
                self._output_tokens += response.usage.completion_tokens or 0
            for ch in full:
                yield ch

    return _DeepSeekReasoningLLM()
