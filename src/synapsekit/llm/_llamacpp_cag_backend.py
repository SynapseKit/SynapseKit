from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from ..llm.base import BaseLLM
from ..llm.llamacpp import LlamaCppLLM
from ..rag.cag_router import CAGBackend


class LlamaCppCAGBackend(CAGBackend):
    """CAG Backend using llama-cpp-python's save_state/load_state."""

    def supports(self, llm: BaseLLM) -> bool:
        return isinstance(llm, LlamaCppLLM)

    async def build_cache(self, llm: BaseLLM, corpus_text: str) -> Any:
        if not self.supports(llm):
            raise ValueError("LLM is not supported by LlamaCppCAGBackend")

        # Type assert to help static analyzers
        assert isinstance(llm, LlamaCppLLM)

        model = llm._get_model()

        def _build() -> Any:
            model.reset()
            # Tokenize corpus_text
            tokens = model.tokenize(corpus_text.encode("utf-8"), add_bos=True)
            # Evaluate tokens to fill KV cache
            model.eval(tokens)
            # Save the state
            state = model.save_state()
            state_bytes = state.data if hasattr(state, "data") else state
            return {
                "state": state_bytes,
                "corpus_text": corpus_text,
            }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _build)

    async def generate_with_cache(
        self, llm: BaseLLM, cache_handle: Any, query: str
    ) -> AsyncGenerator[str]:
        if not self.supports(llm):
            raise ValueError("LLM is not supported by LlamaCppCAGBackend")

        assert isinstance(llm, LlamaCppLLM)
        model = llm._get_model()

        # Load the KV cache state
        state = cache_handle["state"]
        if isinstance(state, bytes):
            try:
                from llama_cpp import LlamaState

                state = LlamaState(data=state)
            except Exception:
                pass
        model.load_state(state)

        # Reconstruct the full prompt
        corpus_text = cache_handle["corpus_text"]
        full_prompt = corpus_text + query

        # Use create_completion with stream=True. This will reuse the KV cache via prefix matching.
        import contextlib
        import queue
        import threading

        temperature = llm.config.temperature
        max_tokens = llm.config.max_tokens
        top_p = llm._top_p

        # Bounded queue so a slow/absent consumer cannot make the producer thread
        # buffer the whole generation in memory. A stop event lets the producer
        # exit at the next token boundary if the consumer breaks early.
        q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=256)
        stop_event = threading.Event()
        error: list[BaseException] = []

        def _produce() -> None:
            try:
                for chunk in model.create_completion(
                    prompt=full_prompt,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                ):
                    if stop_event.is_set():
                        break
                    # Re-check the stop event while parked on a full queue so an
                    # abandoned generation cannot block this thread forever.
                    while not stop_event.is_set():
                        try:
                            q.put(chunk, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except BaseException as exc:
                error.append(exc)
            finally:
                with contextlib.suppress(queue.Full):
                    q.put_nowait(None)

        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        try:
            while True:
                chunk = await loop.run_in_executor(None, q.get)
                if chunk is None:
                    break
                content = chunk["choices"][0].get("text", "")
                if content:
                    llm._output_tokens += 1
                    yield content
            thread.join()
            if error:
                raise error[0]
        finally:
            # On an early break/aclose, signal the producer and drain the queue
            # so a thread parked on ``put`` can observe the stop and exit.
            stop_event.set()
            with contextlib.suppress(queue.Empty):
                while True:
                    q.get_nowait()
            if thread.is_alive():
                thread.join(timeout=1.0)

    def load_state(self, llm: BaseLLM, state_bytes: bytes) -> None:
        if not self.supports(llm):
            raise ValueError("LLM is not supported by LlamaCppCAGBackend")
        assert isinstance(llm, LlamaCppLLM)
        model = llm._get_model()

        state = state_bytes
        if isinstance(state, bytes):
            try:
                from llama_cpp import LlamaState

                state = LlamaState(data=state)
            except Exception:
                pass
        model.load_state(state)
