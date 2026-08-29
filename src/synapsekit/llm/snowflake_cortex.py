"""Snowflake Cortex REST provider."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig

_DEFAULT_TOKEN_TYPE = "PROGRAMMATIC_ACCESS_TOKEN"
_CORTEX_PATH = "/api/v2/cortex/v1/chat/completions"


def _load_httpx() -> Any:
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx required: pip install synapsekit[snowflake-cortex]") from None
    return httpx


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_coerce_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("content", "text", "value"):
            if key in value:
                return _coerce_text(value[key])
    return ""


def _event_from_line(line: str) -> tuple[dict[str, Any] | None, bool]:
    stripped = line.strip()
    if not stripped or stripped.startswith(":") or stripped.startswith("event:"):
        return None, False
    if stripped.startswith("data:"):
        stripped = stripped[5:].strip()
    if stripped == "[DONE]":
        return None, True
    try:
        event = json.loads(stripped)
    except json.JSONDecodeError:
        return None, False
    return event if isinstance(event, dict) else None, False


def _first_count(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _content_from_event(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and (text := _coerce_text(delta.get("content"))):
                return text
            message = choice.get("message")
            if isinstance(message, dict) and (text := _coerce_text(message.get("content"))):
                return text
            for key in ("messages", "content", "text"):
                if text := _coerce_text(choice.get(key)):
                    return text
    for key in ("content", "messages", "response", "text"):
        if text := _coerce_text(event.get(key)):
            return text
    return ""


class SnowflakeCortexLLM(BaseLLM):
    """Snowflake Cortex inference over its native async REST endpoint."""

    def __init__(
        self,
        config: LLMConfig,
        base_url: str | None = None,
        account_identifier: str | None = None,
        authorization_token_type: str | None = None,
        token_type: str | None = None,
    ) -> None:
        super().__init__(config)
        self._base_url = base_url
        self._account_identifier = account_identifier
        self._authorization_token_type = (
            authorization_token_type
            or token_type
            or os.environ.get("SNOWFLAKE_AUTH_TOKEN_TYPE")
            or _DEFAULT_TOKEN_TYPE
        )
        self._client: Any = None

    def _resolved_api_key(self) -> str | None:
        return (
            self.config.api_key
            or os.environ.get("SNOWFLAKE_PAT")
            or os.environ.get("SNOWFLAKE_API_KEY")
        )

    def _resolved_base_url(self) -> str:
        if self._base_url:
            return self._base_url.rstrip("/")
        account_identifier = (
            self._account_identifier
            or os.environ.get("SNOWFLAKE_ACCOUNT_IDENTIFIER")
            or os.environ.get("SNOWFLAKE_ACCOUNT")
        )
        if not account_identifier:
            raise ValueError(
                "account_identifier or base_url is required for SnowflakeCortexLLM. "
                "Pass one to the constructor or set SNOWFLAKE_ACCOUNT_IDENTIFIER."
            )
        return f"https://{account_identifier}.snowflakecomputing.com"

    def _validate_config(self) -> str:
        api_key = self._resolved_api_key()
        if not api_key:
            raise ValueError(
                "api_key is required for SnowflakeCortexLLM. "
                "Pass it in LLMConfig or set SNOWFLAKE_PAT or SNOWFLAKE_API_KEY."
            )
        self._resolved_base_url()
        return api_key

    def _endpoint(self) -> str:
        return f"{self._resolved_base_url()}{_CORTEX_PATH}"

    def _get_client(self) -> Any:
        self._validate_config()
        if self._client is None:
            httpx = _load_httpx()
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout if self.config.timeout is not None else 120.0
            )
        return self._client

    def _record_usage(self, event: dict[str, Any]) -> None:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            return
        input_tokens = _first_count(usage, "prompt_tokens", "input_tokens")
        output_tokens = _first_count(usage, "completion_tokens", "output_tokens")
        if input_tokens is not None:
            self._input_tokens += input_tokens
        if output_tokens is not None:
            self._output_tokens += output_tokens

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        async for token in self.stream_with_messages(messages, **kw):
            yield token

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": kw.get("temperature", self.config.temperature),
            "max_completion_tokens": kw.get("max_tokens", self.config.max_tokens),
        }
        if self.config.top_p is not None or "top_p" in kw:
            payload["top_p"] = kw.get("top_p", self.config.top_p)
        client = self._get_client()
        async with client.stream(
            "POST",
            self._endpoint(),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._validate_config()}",
                "Content-Type": "application/json",
                "X-Snowflake-Authorization-Token-Type": self._authorization_token_type,
            },
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                event, done = _event_from_line(line)
                if done:
                    break
                if event is None:
                    continue
                self._record_usage(event)
                if text := _content_from_event(event):
                    yield text

    async def _call_with_tools_impl(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Call Cortex's OpenAI-compatible endpoint with native tools."""
        tool_payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            tool_payload["top_p"] = self.config.top_p
        response = await self._get_client().post(
            self._endpoint(),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._validate_config()}",
                "Content-Type": "application/json",
                "X-Snowflake-Authorization-Token-Type": self._authorization_token_type,
            },
            json=tool_payload,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return {"content": None, "tool_calls": None}
        self._record_usage(data)
        choices = data.get("choices")
        message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        if not isinstance(message, dict):
            return {"content": None, "tool_calls": None}
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function", {})
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                normalized.append(
                    {
                        "id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": arguments,
                    }
                )
            if normalized:
                return {"content": None, "tool_calls": normalized}
        return {"content": message.get("content"), "tool_calls": None}
