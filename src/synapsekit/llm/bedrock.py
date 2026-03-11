from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from .base import BaseLLM, LLMConfig, _messages_to_prompt


class BedrockLLM(BaseLLM):
    """AWS Bedrock provider (Claude/Titan/Llama) with async streaming via boto3."""

    def __init__(self, config: LLMConfig, region: str = "us-east-1") -> None:
        super().__init__(config)
        self._region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 required: pip install synapsekit[bedrock]") from None
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                **(
                    {"aws_access_key_id": "dummy", "aws_secret_access_key": "dummy"}
                    if self.config.api_key == "env"
                    else {}
                ),
            )
        return self._client

    def _build_body(self, prompt: str, **kw) -> dict:
        model = self.config.model
        temperature = kw.get("temperature", self.config.temperature)
        max_tokens = kw.get("max_tokens", self.config.max_tokens)

        if "claude" in model:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "system": self.config.system_prompt,
            }
        elif "titan" in model:
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                },
            }
        else:
            return {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
            }

    def _extract_chunk(self, chunk: dict, model: str) -> str:
        if "claude" in model:
            delta = chunk.get("delta", {})
            return delta.get("text", "")
        elif "titan" in model:
            return chunk.get("outputText", "")
        else:
            return chunk.get("generation", "")

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        import asyncio

        client = self._get_client()
        body = self._build_body(prompt, **kw)
        model = self.config.model

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.invoke_model_with_response_stream(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
            ),
        )
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            text = self._extract_chunk(chunk, model)
            if text:
                self._output_tokens += 1
                yield text

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        prompt = _messages_to_prompt(messages)
        async for token in self.stream(prompt, **kw):
            yield token
