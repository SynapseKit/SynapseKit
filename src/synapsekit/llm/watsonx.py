"""IBM watsonx.ai REST provider with native IAM authentication."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig

_IAM_URL = "https://iam.cloud.ibm.com/identity/token"
_DEFAULT_BASE_URL = "https://us-south.ml.cloud.ibm.com"
_DEFAULT_VERSION = "2024-10-01"
_GRANT_TYPE = "urn:ibm:params:oauth:grant-type:apikey"


def _load_httpx() -> Any:
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx required: pip install synapsekit[watsonx]") from None
    return httpx


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


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
    results = event.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            for key in ("generated_text", "text"):
                if text := _coerce_text(result.get(key)):
                    return text
            delta = result.get("delta")
            if isinstance(delta, dict) and (text := _coerce_text(delta.get("content"))):
                return text

    delta = event.get("delta")
    if isinstance(delta, dict) and (text := _coerce_text(delta.get("content"))):
        return text

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
            for key in ("content", "text"):
                if text := _coerce_text(choice.get(key)):
                    return text

    for key in ("generated_text", "content", "text"):
        if text := _coerce_text(event.get(key)):
            return text
    return ""


class WatsonxLLM(BaseLLM):
    """watsonx.ai chat provider using IAM API-key token exchange."""

    def __init__(
        self,
        config: LLMConfig,
        project_id: str | None = None,
        space_id: str | None = None,
        base_url: str | None = None,
        version: str = _DEFAULT_VERSION,
        iam_url: str | None = None,
    ) -> None:
        super().__init__(config)
        self._project_id = project_id
        self._space_id = space_id
        self._base_url = base_url
        self._version = version
        self._iam_url = iam_url or _IAM_URL
        self._client: Any = None
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    def _resolved_api_key(self) -> str | None:
        return self.config.api_key or _env_first(
            "WATSONX_API_KEY", "WATSONX_APIKEY", "IBM_CLOUD_API_KEY", "IBM_API_KEY"
        )

    def _resolved_scope(self) -> tuple[str | None, str | None]:
        project_id = self._project_id or _env_first("WATSONX_PROJECT_ID", "IBM_PROJECT_ID")
        space_id = self._space_id or _env_first("WATSONX_SPACE_ID", "IBM_SPACE_ID")
        return project_id, space_id

    def _resolved_base_url(self) -> str:
        return (
            self._base_url or _env_first("WATSONX_BASE_URL", "WATSONX_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")

    def _validate_config(self) -> tuple[str, str | None, str | None]:
        project_id, space_id = self._resolved_scope()
        if not project_id and not space_id:
            raise ValueError(
                "project_id or space_id is required for WatsonxLLM. "
                "Pass one to the constructor or set WATSONX_PROJECT_ID/WATSONX_SPACE_ID."
            )
        api_key = self._resolved_api_key()
        if not api_key:
            raise ValueError(
                "api_key is required for WatsonxLLM. Pass it in LLMConfig or set WATSONX_API_KEY."
            )
        return api_key, project_id, space_id

    def _endpoint(self) -> str:
        return f"{self._resolved_base_url()}/ml/v1/text/chat?version={self._version}"

    def _get_client(self) -> Any:
        self._validate_config()
        if self._client is None:
            httpx = _load_httpx()
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout if self.config.timeout is not None else 120.0
            )
        return self._client

    @staticmethod
    def _token_ttl(payload: dict[str, Any]) -> float:
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
            return max(float(expires_in), 0.0)

        expiration = payload.get("expiration")
        if isinstance(expiration, (int, float)) and not isinstance(expiration, bool):
            expiration_value = float(expiration)
            if expiration_value > time.time():
                return max(expiration_value - time.time(), 0.0)
            return max(expiration_value, 0.0)
        return 3600.0

    async def _get_access_token(self) -> str:
        api_key, _, _ = self._validate_config()
        if self._access_token and time.monotonic() < self._token_expires_at:
            return self._access_token

        client = self._get_client()
        response = await client.post(
            self._iam_url,
            data={"grant_type": _GRANT_TYPE, "apikey": api_key},
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        response.raise_for_status()
        payload = response.json()
        access_token = payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("watsonx IAM response did not contain an access_token")
        self._access_token = access_token
        self._token_expires_at = time.monotonic() + self._token_ttl(payload)
        return access_token

    @staticmethod
    def _usage_counts(event: dict[str, Any]) -> tuple[int | None, int | None]:
        usage = event.get("usage")
        if isinstance(usage, dict):
            return (
                _first_count(usage, "prompt_tokens", "input_tokens"),
                _first_count(usage, "completion_tokens", "output_tokens"),
            )

        results = event.get("results")
        if not isinstance(results, list):
            return None, None
        input_total: int | None = None
        output_total: int | None = None
        for result in results:
            if not isinstance(result, dict):
                continue
            input_tokens = _first_count(
                result, "input_token_count", "prompt_tokens", "input_tokens"
            )
            output_tokens = _first_count(
                result, "generated_token_count", "completion_tokens", "output_tokens"
            )
            if input_tokens is not None:
                input_total = max(input_total or 0, input_tokens)
            if output_tokens is not None:
                output_total = max(output_total or 0, output_tokens)
        return input_total, output_total

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
        request_input_tokens = 0
        request_output_tokens = 0
        _, project_id, space_id = self._validate_config()
        payload: dict[str, Any] = {
            "model_id": self.config.model,
            "messages": messages,
            "parameters": {
                "temperature": kw.get("temperature", self.config.temperature),
                "max_new_tokens": kw.get(
                    "max_new_tokens", kw.get("max_tokens", self.config.max_tokens)
                ),
            },
            "stream": True,
        }
        if self.config.top_p is not None or "top_p" in kw:
            payload["parameters"]["top_p"] = kw.get("top_p", self.config.top_p)
        if project_id:
            payload["project_id"] = project_id
        else:
            payload["space_id"] = space_id

        access_token = await self._get_access_token()
        client = self._get_client()
        async with client.stream(
            "POST",
            self._endpoint(),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
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
                input_tokens, output_tokens = self._usage_counts(event)
                if input_tokens is not None:
                    request_input_tokens = max(request_input_tokens, input_tokens)
                if output_tokens is not None:
                    request_output_tokens = max(request_output_tokens, output_tokens)
                if text := _content_from_event(event):
                    yield text
        self._input_tokens += request_input_tokens
        self._output_tokens += request_output_tokens

    async def _call_with_tools_impl(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Call watsonx.ai with the native tools request fields."""
        _, project_id, space_id = self._validate_config()
        payload: dict[str, Any] = {
            "model_id": self.config.model,
            "messages": messages,
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens,
            },
            "tools": tools,
            "tool_choice_option": "auto",
        }
        if self.config.top_p is not None:
            payload["parameters"]["top_p"] = self.config.top_p
        if project_id:
            payload["project_id"] = project_id
        else:
            payload["space_id"] = space_id

        access_token = await self._get_access_token()
        response = await self._get_client().post(
            self._endpoint(),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return {"content": None, "tool_calls": None}
        input_tokens, output_tokens = self._usage_counts(data)
        if input_tokens is not None:
            self._input_tokens += input_tokens
        if output_tokens is not None:
            self._output_tokens += output_tokens

        candidates: list[dict[str, Any]] = []
        choices = data.get("choices")
        if isinstance(choices, list):
            candidates.extend(item for item in choices if isinstance(item, dict))
        results = data.get("results")
        if isinstance(results, list):
            candidates.extend(item for item in results if isinstance(item, dict))

        for candidate in tuple(candidates):
            message = candidate.get("message")
            tools_in_message = None
            if isinstance(message, dict):
                candidates.append(message)
                tools_in_message = message.get("tool_calls") or message.get("function_calls")
            tool_calls = candidate.get("tool_calls") or candidate.get("function_calls")
            if not isinstance(tool_calls, list) and isinstance(tools_in_message, list):
                tool_calls = tools_in_message
            if not isinstance(tool_calls, list):
                continue
            normalized = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function", tool_call)
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                normalized.append(
                    {
                        "id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": arguments,
                    }
                )
            if normalized:
                return {"content": None, "tool_calls": normalized}

        content = ""
        for candidate in candidates:
            content = _coerce_text(candidate.get("content")) or _coerce_text(
                candidate.get("generated_text")
            )
            if content:
                break
        return {"content": content or None, "tool_calls": None}
