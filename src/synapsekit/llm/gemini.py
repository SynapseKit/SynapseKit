from __future__ import annotations

from collections.abc import AsyncGenerator

from .base import BaseLLM, LLMConfig, _messages_to_prompt


class GeminiLLM(BaseLLM):
    """Google Gemini provider with async streaming."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai required: pip install synapsekit[gemini]"
                ) from None
            genai.configure(api_key=self.config.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.config.model,
                system_instruction=self.config.system_prompt,
            )
        return self._model

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        model = self._get_model()
        async for chunk in await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": kw.get("temperature", self.config.temperature),
                "max_output_tokens": kw.get("max_tokens", self.config.max_tokens),
            },
            stream=True,
        ):
            if chunk.text:
                self._output_tokens += 1
                yield chunk.text

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        prompt = _messages_to_prompt(messages)
        async for token in self.stream(prompt, **kw):
            yield token
