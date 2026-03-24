from __future__ import annotations

import json
import os
from typing import Any

from ..base import BaseTool, ToolResult


class AWSLambdaTool(BaseTool):
    """Invoke AWS Lambda functions."""

    name = "aws_lambda"
    description = (
        "Invoke an AWS Lambda function. "
        "Input: function_name, optional payload, invocation_type, qualifier, and region_name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "function_name": {
                "type": "string",
                "description": "The Lambda function name or ARN",
            },
            "payload": {
                "type": ["object", "string"],
                "description": "Optional JSON payload to send to the function",
                "default": {},
            },
            "invocation_type": {
                "type": "string",
                "description": "Invocation type: RequestResponse, Event, or DryRun",
                "enum": ["RequestResponse", "Event", "DryRun"],
                "default": "RequestResponse",
            },
            "qualifier": {
                "type": "string",
                "description": "Version or alias to invoke",
                "default": "",
            },
            "region_name": {
                "type": "string",
                "description": "AWS region name (defaults to environment or boto3 config)",
                "default": "",
            },
        },
        "required": ["function_name"],
    }

    def __init__(self, region_name: str | None = None) -> None:
        self._region_name = (
            region_name or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )

    async def run(
        self,
        function_name: str = "",
        payload: Any | None = None,
        invocation_type: str = "RequestResponse",
        qualifier: str = "",
        region_name: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        function_name = function_name or kwargs.get("input", "")
        if not function_name:
            return ToolResult(output="", error="No function_name provided for AWS Lambda.")

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 required for AWSLambdaTool: pip install synapsekit[aws-lambda]"
            ) from None

        effective_region = region_name or self._region_name

        try:
            client = (
                boto3.client("lambda", region_name=effective_region)
                if effective_region
                else boto3.client("lambda")
            )
            invoke_kwargs: dict[str, Any] = {
                "FunctionName": function_name,
                "InvocationType": invocation_type,
            }
            if qualifier:
                invoke_kwargs["Qualifier"] = qualifier

            if payload is not None:
                invoke_kwargs["Payload"] = self._serialize_payload(payload)

            import asyncio

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: client.invoke(**invoke_kwargs))
            return ToolResult(output=self._format_response(response))
        except Exception as e:
            return ToolResult(output="", error=f"AWS Lambda invocation failed: {e}")

    def _serialize_payload(self, payload: Any) -> bytes:
        if isinstance(payload, (dict, list)):
            return json.dumps(payload).encode("utf-8")
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return b""
            try:
                return json.dumps(json.loads(text)).encode("utf-8")
            except Exception:
                return text.encode("utf-8")
        return json.dumps(payload).encode("utf-8")

    def _format_response(self, response: dict[str, Any]) -> str:
        parts = []
        status = response.get("StatusCode", "unknown")
        parts.append(f"StatusCode: {status}")

        function_error = response.get("FunctionError")
        if function_error:
            parts.append(f"FunctionError: {function_error}")

        executed_version = response.get("ExecutedVersion")
        if executed_version:
            parts.append(f"ExecutedVersion: {executed_version}")

        payload_stream = response.get("Payload")
        if payload_stream is not None:
            raw = payload_stream.read()
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace")
            else:
                text = str(raw)
            text = text.strip()
            if text:
                try:
                    parsed = json.loads(text)
                    text = json.dumps(parsed, indent=2, sort_keys=True)
                except Exception:
                    pass
            parts.append("Payload:")
            parts.append(text or "(empty)")

        return "\n".join(parts)
