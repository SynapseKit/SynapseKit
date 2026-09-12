"""``GuardedLLM`` -- wrap any :class:`BaseLLM` so a policy runs on every call.

The wrapper is itself a :class:`BaseLLM`, so it drops in wherever an LLM is
expected. Input guards run on the prompt before the wrapped model sees it (a
redact-mode guard rewrites the prompt); output guards run on the full response
before it is returned (a redact-mode guard rewrites the response). A guard in
``block``/``require_human`` mode raises :class:`GuardrailBlockedError`.

Streaming is buffered: the wrapped model streams internally, but the guarded
response is only yielded once the output guards have inspected (and possibly
rewritten) the whole text -- otherwise redaction could not work. The wrapper
adds no cache or retry layer of its own; the wrapped model keeps its behaviour.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ..llm.base import BaseLLM, LLMConfig
from .policy import GuardrailPolicy
from .types import GuardrailBlockedError


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the content of the last user message (for input guarding)."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _rewrite_last_user(messages: list[dict[str, Any]], new_text: str) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with the last user message's content replaced."""
    out = [dict(m) for m in messages]
    for message in reversed(out):
        if message.get("role") == "user":
            message["content"] = new_text
            break
    return out


class GuardedLLM(BaseLLM):
    """Apply a :class:`GuardrailPolicy` to every call of a wrapped LLM."""

    def __init__(self, llm: BaseLLM, policy: GuardrailPolicy) -> None:
        wrapped_cfg = llm.config
        # No cache/retry layer of our own -- delegate everything to the wrapped
        # model, which keeps its own caching, retries, and rate limiting.
        super().__init__(
            LLMConfig(
                model=f"guarded:{wrapped_cfg.model}",
                api_key="",
                provider=wrapped_cfg.provider,
                max_retries=0,
            )
        )
        self._wrapped = llm
        self._policy = policy

    @property
    def policy(self) -> GuardrailPolicy:
        return self._policy

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        in_report = await self._policy.check_input(prompt)
        if not in_report.allowed:
            raise GuardrailBlockedError(in_report)

        chunks: list[str] = []
        async for token in self._wrapped.stream(in_report.text, **kw):
            chunks.append(token)

        out_report = await self._policy.check_output("".join(chunks))
        if not out_report.allowed:
            raise GuardrailBlockedError(out_report)
        if out_report.text:
            yield out_report.text

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        in_report = await self._policy.check_input(_last_user_text(messages))
        if not in_report.allowed:
            raise GuardrailBlockedError(in_report)
        # Propagate an input-stage redaction into the wrapped call.
        guarded_messages = _rewrite_last_user(messages, in_report.text)

        chunks: list[str] = []
        async for token in self._wrapped.stream_with_messages(guarded_messages, **kw):
            chunks.append(token)

        out_report = await self._policy.check_output("".join(chunks))
        if not out_report.allowed:
            raise GuardrailBlockedError(out_report)
        if out_report.text:
            yield out_report.text

    @property
    def tokens_used(self) -> dict[str, int]:
        return self._wrapped.tokens_used

    async def aclose(self) -> None:
        await self._wrapped.aclose()
