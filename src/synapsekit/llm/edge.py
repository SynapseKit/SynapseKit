"""EdgeRuntime: local-first inference with explicit cloud fallback gates."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

from ..agents.pii_redactor import PIIRedactor, RedactionResult
from .base import BaseLLM, LLMConfig, _messages_to_prompt

EdgeRoute = Literal["local", "cloud"]
FallbackReason = Literal["context_exceeds", "tool_unsupported", "user_opt_in", "local_error"]


class EdgeFallbackBlockedError(RuntimeError):
    """Raised when a request needs cloud fallback but policy does not permit it."""


@dataclass(frozen=True)
class FallbackDecision:
    """Decision object returned by ``FallbackPolicy.evaluate``."""

    allowed: bool
    reason: FallbackReason | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class FallbackPolicy:
    """Declarative routing rules for local-first edge inference.

    Cloud fallback is default-deny. A fallback route is available only when a
    specific gate is enabled and the runtime has a configured cloud model.
    """

    if_context_exceeds: int | None = None
    if_tool_unsupported_locally: bool = False
    if_user_opts_in: bool = False
    require_pii_redaction_before_fallback: bool = True
    fallback_on_local_error: bool = False

    def evaluate(
        self,
        *,
        estimated_context_tokens: int,
        tool_unsupported: bool = False,
        user_opted_in: bool = False,
        local_error: BaseException | None = None,
    ) -> FallbackDecision:
        """Return whether cloud fallback is allowed for the current call."""

        if (
            self.if_context_exceeds is not None
            and estimated_context_tokens > self.if_context_exceeds
        ):
            return FallbackDecision(True, "context_exceeds")

        if tool_unsupported:
            if self.if_tool_unsupported_locally:
                return FallbackDecision(True, "tool_unsupported")
            return FallbackDecision(
                False,
                blocked_reason=(
                    "local model does not support tools and "
                    "if_tool_unsupported_locally is disabled"
                ),
            )

        if user_opted_in:
            if self.if_user_opts_in:
                return FallbackDecision(True, "user_opt_in")
            return FallbackDecision(
                False,
                blocked_reason="user requested cloud fallback but if_user_opts_in is disabled",
            )

        if local_error is not None:
            if self.fallback_on_local_error:
                return FallbackDecision(True, "local_error")
            return FallbackDecision(
                False,
                blocked_reason=f"local model failed and fallback_on_local_error is disabled: {local_error}",
            )

        return FallbackDecision(False, blocked_reason="no fallback condition matched")


@dataclass(frozen=True)
class EdgeRouteMetadata:
    """Metadata describing the most recent edge runtime call."""

    route: EdgeRoute
    fallback_reason: FallbackReason | None = None
    estimated_context_tokens: int = 0
    pii_types_found: tuple[str, ...] = ()


class EdgeRuntime(BaseLLM):
    """Local-first LLM wrapper with policy-controlled cloud fallback.

    ``EdgeRuntime`` is itself a ``BaseLLM`` so it can be passed anywhere a
    regular SynapseKit model is accepted, including agents and RAG pipelines.
    It delegates to the local model by default and only calls the cloud model
    after a ``FallbackPolicy`` gate permits that route.
    """

    def __init__(
        self,
        *,
        local_llm: BaseLLM,
        cloud_llm: BaseLLM | None = None,
        fallback: FallbackPolicy | None = None,
        local_embeddings: Any | None = None,
        pii_redactor: PIIRedactor | None = None,
        token_chars: int = 4,
    ) -> None:
        super().__init__(
            LLMConfig(
                model=f"edge:{local_llm.config.model}",
                api_key="",
                provider="edge",
                system_prompt=local_llm.config.system_prompt,
                temperature=local_llm.config.temperature,
                max_tokens=local_llm.config.max_tokens,
            )
        )
        if token_chars <= 0:
            raise ValueError("token_chars must be positive")

        self.local_llm = local_llm
        self.cloud_llm = cloud_llm
        self.fallback = fallback or FallbackPolicy()
        self.local_embeddings = local_embeddings
        self._pii_redactor = pii_redactor or PIIRedactor()
        self._token_chars = token_chars
        self._last_metadata = EdgeRouteMetadata(route="local")
        self._last_redaction: RedactionResult | None = None

    @property
    def last_route(self) -> EdgeRoute:
        """Route used by the most recent call."""

        return self._last_metadata.route

    @property
    def last_fallback_reason(self) -> FallbackReason | None:
        """Fallback reason for the most recent cloud call, if any."""

        return self._last_metadata.fallback_reason

    @property
    def last_metadata(self) -> EdgeRouteMetadata:
        """Structured metadata for the most recent call."""

        return self._last_metadata

    @property
    def last_redaction(self) -> RedactionResult | None:
        """PII redaction result from the most recent cloud fallback."""

        return self._last_redaction

    def _estimate_tokens(self, text: str) -> int:
        compact = " ".join(text.split())
        if not compact:
            return 0
        return max(1, (len(compact) + self._token_chars - 1) // self._token_chars)

    def _estimate_message_tokens(self, messages: list[dict[str, Any]]) -> int:
        return self._estimate_tokens(_messages_to_prompt(messages))

    def _evaluate_fallback(
        self,
        *,
        estimated_context_tokens: int,
        tool_unsupported: bool = False,
        user_opted_in: bool = False,
        local_error: BaseException | None = None,
    ) -> FallbackDecision:
        decision = self.fallback.evaluate(
            estimated_context_tokens=estimated_context_tokens,
            tool_unsupported=tool_unsupported,
            user_opted_in=user_opted_in,
            local_error=local_error,
        )
        if decision.allowed and self.cloud_llm is None:
            raise EdgeFallbackBlockedError("cloud fallback was allowed but no cloud_llm is configured")
        return decision

    def _select_for_prompt(self, prompt: str, kw: dict[str, Any]) -> tuple[BaseLLM, FallbackDecision]:
        estimated = self._estimate_tokens(prompt)
        user_opted_in = bool(kw.pop("allow_cloud_fallback", False))
        decision = self._evaluate_fallback(
            estimated_context_tokens=estimated,
            user_opted_in=user_opted_in,
        )
        if decision.allowed:
            return self._require_cloud(decision, estimated), decision
        return self.local_llm, FallbackDecision(False)

    def _select_for_messages(
        self,
        messages: list[dict[str, Any]],
        kw: dict[str, Any],
    ) -> tuple[BaseLLM, FallbackDecision]:
        estimated = self._estimate_message_tokens(messages)
        user_opted_in = bool(kw.pop("allow_cloud_fallback", False))
        decision = self._evaluate_fallback(
            estimated_context_tokens=estimated,
            user_opted_in=user_opted_in,
        )
        if decision.allowed:
            return self._require_cloud(decision, estimated), decision
        return self.local_llm, FallbackDecision(False)

    def _require_cloud(self, decision: FallbackDecision, estimated: int) -> BaseLLM:
        if self.cloud_llm is None:
            raise EdgeFallbackBlockedError("cloud fallback requested but cloud_llm is not configured")
        self._last_metadata = EdgeRouteMetadata(
            route="cloud",
            fallback_reason=decision.reason,
            estimated_context_tokens=estimated,
        )
        return self.cloud_llm

    def _record_local(self, estimated: int) -> None:
        self._last_redaction = None
        self._last_metadata = EdgeRouteMetadata(
            route="local",
            fallback_reason=None,
            estimated_context_tokens=estimated,
        )

    def _redact_text_for_cloud(self, text: str) -> str:
        if not self.fallback.require_pii_redaction_before_fallback:
            self._last_redaction = None
            return text
        redaction = self._pii_redactor.redact(text)
        self._last_redaction = redaction
        self._last_metadata = EdgeRouteMetadata(
            route="cloud",
            fallback_reason=self._last_metadata.fallback_reason,
            estimated_context_tokens=self._last_metadata.estimated_context_tokens,
            pii_types_found=tuple(redaction.pii_types_found),
        )
        return redaction.redacted_text

    def _redact_messages_for_cloud(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.fallback.require_pii_redaction_before_fallback:
            self._last_redaction = None
            return messages

        combined_mapping: dict[str, str] = {}
        pii_types: set[str] = set()
        redacted_messages: list[dict[str, Any]] = []

        for message in messages:
            copied = dict(message)
            content = copied.get("content")
            if isinstance(content, str):
                redaction = self._pii_redactor.redact(content)
                copied["content"] = redaction.redacted_text
                combined_mapping.update(redaction.mapping)
                pii_types.update(redaction.pii_types_found)
            redacted_messages.append(copied)

        self._last_redaction = RedactionResult(
            redacted_text=_messages_to_prompt(redacted_messages),
            mapping=combined_mapping,
            pii_types_found=sorted(pii_types),
        )
        self._last_metadata = EdgeRouteMetadata(
            route="cloud",
            fallback_reason=self._last_metadata.fallback_reason,
            estimated_context_tokens=self._last_metadata.estimated_context_tokens,
            pii_types_found=tuple(sorted(pii_types)),
        )
        return redacted_messages

    async def _fallback_after_error(
        self,
        *,
        prompt: str,
        kw: dict[str, Any],
        error: BaseException,
    ) -> str:
        estimated = self._estimate_tokens(prompt)
        decision = self._evaluate_fallback(
            estimated_context_tokens=estimated,
            local_error=error,
        )
        if not decision.allowed:
            raise error
        cloud = self._require_cloud(decision, estimated)
        return await cloud.generate(self._redact_text_for_cloud(prompt), **kw)

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        call_kw = dict(kw)
        estimated = self._estimate_tokens(prompt)
        llm, decision = self._select_for_prompt(prompt, call_kw)
        if decision.allowed:
            async for token in llm.stream(self._redact_text_for_cloud(prompt), **call_kw):
                yield token
            return

        self._record_local(estimated)
        try:
            async for token in self.local_llm.stream(prompt, **call_kw):
                yield token
        except Exception as exc:
            fallback_text = await self._fallback_after_error(prompt=prompt, kw=call_kw, error=exc)
            for token in fallback_text:
                yield token

    async def generate(self, prompt: str, **kw: Any) -> str:  # type: ignore[override]
        call_kw = dict(kw)
        estimated = self._estimate_tokens(prompt)
        llm, decision = self._select_for_prompt(prompt, call_kw)
        if decision.allowed:
            return await llm.generate(self._redact_text_for_cloud(prompt), **call_kw)

        self._record_local(estimated)
        try:
            return await self.local_llm.generate(prompt, **call_kw)
        except Exception as exc:
            return await self._fallback_after_error(prompt=prompt, kw=call_kw, error=exc)

    async def stream_with_messages(
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> AsyncGenerator[str]:
        call_kw = dict(kw)
        estimated = self._estimate_message_tokens(messages)
        llm, decision = self._select_for_messages(messages, call_kw)
        if decision.allowed:
            safe_messages = self._redact_messages_for_cloud(messages)
            async for token in llm.stream_with_messages(safe_messages, **call_kw):
                yield token
            return

        self._record_local(estimated)
        async for token in self.local_llm.stream_with_messages(messages, **call_kw):
            yield token

    async def generate_with_messages(
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> str:
        call_kw = dict(kw)
        estimated = self._estimate_message_tokens(messages)
        llm, decision = self._select_for_messages(messages, call_kw)
        if decision.allowed:
            return await llm.generate_with_messages(
                self._redact_messages_for_cloud(messages),
                **call_kw,
            )

        self._record_local(estimated)
        return await self.local_llm.generate_with_messages(messages, **call_kw)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        estimated = self._estimate_message_tokens(messages)
        try:
            self._record_local(estimated)
            return await self.local_llm.call_with_tools(messages, tools)
        except NotImplementedError as exc:
            decision = self._evaluate_fallback(
                estimated_context_tokens=estimated,
                tool_unsupported=True,
            )
            if not decision.allowed:
                blocked = decision.blocked_reason or "tool fallback denied"
                raise EdgeFallbackBlockedError(blocked) from exc
            cloud = self._require_cloud(decision, estimated)
            return await cloud.call_with_tools(self._redact_messages_for_cloud(messages), tools)
