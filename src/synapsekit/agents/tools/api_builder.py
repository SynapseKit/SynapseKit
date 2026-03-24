from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, cast
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from ..base import BaseTool, ToolResult


class APIBuilderTool(BaseTool):
    """Build and execute API calls from OpenAPI specs or natural-language intent."""

    name = "api_request_builder"
    description = (
        "Build and execute an API request from an OpenAPI spec or user intent. "
        "Can infer the operation from the provided intent or use explicit path/method inputs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "Natural-language description of the API call to make",
            },
            "openapi_spec": {
                "type": ["string", "object"],
                "description": "Inline OpenAPI/Swagger JSON spec as a string or object",
                "default": "",
            },
            "openapi_url": {
                "type": "string",
                "description": "URL to an OpenAPI/Swagger JSON spec",
                "default": "",
            },
            "operation_id": {
                "type": "string",
                "description": "Explicit OpenAPI operationId to invoke",
                "default": "",
            },
            "path": {
                "type": "string",
                "description": "Explicit API path to invoke (for example /users/{id})",
                "default": "",
            },
            "method": {
                "type": "string",
                "description": "HTTP method to use when path is provided directly",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "default": "GET",
            },
            "server_url": {
                "type": "string",
                "description": "Base server URL override",
                "default": "",
            },
            "path_params": {
                "type": "object",
                "description": "Path parameters used to fill templated route segments",
                "default": {},
            },
            "query_params": {
                "type": "object",
                "description": "Query parameters to append to the request URL",
                "default": {},
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers to send with the request",
                "default": {},
            },
            "body": {
                "type": ["string", "object"],
                "description": "Optional request body as a string or JSON object",
                "default": "",
            },
        },
        "required": ["intent"],
    }

    def __init__(self, llm: Any | None = None, timeout: int = 30) -> None:
        self._llm = llm
        self._timeout = timeout

    async def run(self, intent: str = "", **kwargs: Any) -> ToolResult:
        intent = intent or kwargs.get("input", "")
        if not intent:
            return ToolResult(output="", error="No intent provided for APIBuilderTool.")

        try:
            spec = await self._load_spec(kwargs)
            operation = await self._select_operation(intent, spec, kwargs)
            if operation is None:
                return ToolResult(output="", error="Could not determine an API operation to call.")

            url, method, headers, body, operation_label = self._build_request(
                operation=operation,
                spec=spec,
                kwargs=kwargs,
            )
            response_text = await self._perform_request(url, method, headers, body)
            return ToolResult(
                output=(
                    f"Selected operation: {operation_label}\n"
                    f"Request: {method} {url}\n\n"
                    f"{response_text}"
                )
            )
        except Exception as e:
            return ToolResult(output="", error=f"API builder failed: {e}")

    async def _load_spec(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        spec = kwargs.get("openapi_spec")
        if isinstance(spec, dict):
            return spec
        if isinstance(spec, str) and spec.strip():
            text = spec.strip()
            if text.startswith("{"):
                return cast(dict[str, Any], json.loads(text))

        spec_url = kwargs.get("openapi_url", "")
        if not spec_url:
            return None

        request = Request(spec_url, headers={"Accept": "application/json"})

        def _fetch() -> dict[str, Any]:
            with urlopen(request, timeout=self._timeout) as resp:
                payload = resp.read().decode("utf-8")
            return cast(dict[str, Any], json.loads(payload))

        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _fetch)

    async def _select_operation(
        self,
        intent: str,
        spec: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        operation_id = kwargs.get("operation_id", "")
        if operation_id and spec:
            found = self._find_operation_by_id(spec, operation_id)
            if found is not None:
                return found

        path = kwargs.get("path", "")
        method = (kwargs.get("method", "GET") or "GET").upper()
        if path:
            return {
                "path": path,
                "method": method,
                "operationId": operation_id or f"{method} {path}",
            }

        if not spec:
            return None

        operations = list(self._iter_operations(spec))
        if not operations:
            return None

        if self._llm is not None:
            prompt = self._selection_prompt(intent, operations)
            try:
                result = self._llm.generate(prompt)
                if hasattr(result, "__await__"):
                    result = await result
                chosen = str(result).strip()
                matched = self._find_operation_by_id(spec, chosen)
                if matched is not None:
                    return matched
            except Exception:
                pass

        intent_tokens = self._tokenize(intent)
        best_score = -1
        best_operation: dict[str, Any] | None = None
        for operation in operations:
            score = self._score_operation(intent_tokens, operation)
            if score > best_score:
                best_score = score
                best_operation = operation
        return best_operation

    def _find_operation_by_id(
        self, spec: dict[str, Any], operation_id: str
    ) -> dict[str, Any] | None:
        for operation in self._iter_operations(spec):
            if operation.get("operationId") == operation_id:
                return operation
        return None

    def _iter_operations(self, spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
        paths = spec.get("paths", {}) or {}
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, operation in methods.items():
                if not isinstance(operation, dict):
                    continue
                if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                    continue
                entry = dict(operation)
                entry["path"] = path
                entry["method"] = method.upper()
                yield entry

    def _score_operation(self, intent_tokens: set[str], operation: dict[str, Any]) -> int:
        haystack = " ".join(
            str(operation.get(key, "")) for key in ("operationId", "summary", "description", "path")
        ).lower()
        score = 0
        for token in intent_tokens:
            if token and token in haystack:
                score += 2
        if operation.get("method") == "GET" and any(
            word in intent_tokens for word in {"get", "list", "fetch", "show"}
        ):
            score += 1
        if operation.get("method") in {"POST", "PUT", "PATCH"} and any(
            word in intent_tokens for word in {"create", "add", "update", "modify", "submit"}
        ):
            score += 1
        return score

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if token}

    def _selection_prompt(self, intent: str, operations: list[dict[str, Any]]) -> str:
        lines = [
            "Select the best operationId for this API request.",
            f"Intent: {intent}",
            "Operations:",
        ]
        for operation in operations:
            lines.append(
                f"- operationId={operation.get('operationId', '')} method={operation.get('method', '')} path={operation.get('path', '')} summary={operation.get('summary', '')}"
            )
        lines.append("Return only the operationId.")
        return "\n".join(lines)

    def _build_request(
        self,
        operation: dict[str, Any],
        spec: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> tuple[str, str, dict[str, str], bytes | None, str]:
        method = str(operation.get("method", kwargs.get("method", "GET"))).upper()
        path = str(operation.get("path", kwargs.get("path", "")))
        if not path:
            raise ValueError("No API path available to build the request.")

        base_url = kwargs.get("server_url", "")
        if not base_url and spec:
            servers = spec.get("servers") or []
            if servers and isinstance(servers, list) and isinstance(servers[0], dict):
                base_url = servers[0].get("url", "")
        if not base_url:
            base_url = ""

        path_params = dict(kwargs.get("path_params") or {})
        for key, value in path_params.items():
            path = path.replace(f"{{{key}}}", str(value))

        url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/")) if base_url else path

        query_params = dict(kwargs.get("query_params") or {})
        if query_params:
            url = f"{url}?{urlencode(query_params, doseq=True)}"

        headers = {str(k): str(v) for k, v in dict(kwargs.get("headers") or {}).items()}
        body = kwargs.get("body")
        payload: bytes | None = None
        if method in {"POST", "PUT", "PATCH"} and body not in (None, "", {}, []):
            if isinstance(body, (dict, list)):
                payload = json.dumps(body).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            elif isinstance(body, str):
                text = body.strip()
                if text.startswith("{") or text.startswith("["):
                    payload = text.encode("utf-8")
                    headers.setdefault("Content-Type", "application/json")
                else:
                    payload = text.encode("utf-8")
            else:
                payload = json.dumps(body).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")

        label = operation.get("operationId") or f"{method} {path}"
        return url, method, headers, payload, str(label)

    async def _perform_request(
        self,
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> str:
        request = Request(url, data=body, headers=headers, method=method)

        import asyncio

        def _fetch() -> str:
            with urlopen(request, timeout=self._timeout) as resp:
                status = getattr(resp, "status", 200)
                text = resp.read().decode("utf-8", errors="replace")
            if text.strip():
                try:
                    parsed = json.loads(text)
                    text = json.dumps(parsed, indent=2, sort_keys=True)
                except Exception:
                    pass
            return f"HTTP {status}\n{text}"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _fetch)
