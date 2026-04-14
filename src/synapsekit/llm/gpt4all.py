from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from queue import Queue
from threading import Thread
from typing import Any

from .base import BaseLLM, LLMConfig


class GPT4AllLLM(BaseLLM):
    """GPT4All local provider using the gpt4all Python bindings."""

    def __init__(
        self,
        config: LLMConfig,
        model_path: str | None = None,
        allow_download: bool = False,
        device: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        super().__init__(config)
        self._model_path = model_path
        self._allow_download = allow_download
        self._device = device
        self._model_kwargs = model_kwargs
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from gpt4all import GPT4All
            except ImportError:
                raise ImportError("gpt4all required: pip install synapsekit[gpt4all]") from None

            kwargs: dict[str, Any] = {
                "model_name": self.config.model,
                "allow_download": self._allow_download,
                **self._model_kwargs,
            }
            if self._model_path is not None:
                kwargs["model_path"] = self._model_path
            if self._device is not None:
                kwargs["device"] = self._device
            self._model = GPT4All(**kwargs)
        return self._model

    @staticmethod
    def _flatten_messages(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for m in messages:
            role = str(m.get("role", "user")).capitalize()
            content = str(m.get("content", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

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
        model = self._get_model()
        prompt = self._flatten_messages(messages)
        temperature = kw.get("temperature", self.config.temperature)
        max_tokens = kw.get("max_tokens", self.config.max_tokens)
        top_p = kw.get("top_p", 0.95)

        queue: Queue[str | None] = Queue()
        errors: list[BaseException] = []

        def _produce() -> None:
            emitted = False

            def _callback(_token_id: int, response: str) -> bool:
                nonlocal emitted
                if response:
                    emitted = True
                    queue.put(response)
                return True

            try:
                result = model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    top_p=top_p,
                    streaming=True,
                    callback=_callback,
                )
                if result:
                    if isinstance(result, str):
                        if not emitted and result:
                            queue.put(result)
                    else:
                        # GPT4All returns an iterator in streaming mode.
                        # Consume it to drive generation; if callback is active,
                        # it already emitted chunks into the queue.
                        for token in result:
                            if not emitted and token:
                                queue.put(str(token))
            except BaseException as exc:  # pragma: no cover - defensive wrapper
                errors.append(exc)
            finally:
                queue.put(None)

        thread = Thread(target=_produce, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        while True:
            token = await loop.run_in_executor(None, queue.get)
            if token is None:
                break
            self._output_tokens += 1
            yield token

        thread.join()
        if errors:
            raise errors[0]
