"""Real OllamaLLM integration test — actual local inference via testcontainers.

Boots a real Ollama server, pulls a tiny model, and streams a real completion
through OllamaLLM (validating the new ``host`` parameter and the real chat/stream
path). This is the one LLM provider that can run real inference in a container;
it's heavy (image + model pull + CPU inference), so it lives in its own CI job.
Part of #829 (LLM track).
"""

from __future__ import annotations

import inspect
import time

import pytest

pytest.importorskip("ollama")
_container_mod = pytest.importorskip("testcontainers.core.container")

from synapsekit.llm.base import LLMConfig  # noqa: E402
from synapsekit.llm.ollama import OllamaLLM  # noqa: E402

_OLLAMA_IMAGE = "ollama/ollama:latest"
_MODEL = "qwen2.5:0.5b"  # ~400MB — smallest reliable instruct model


@pytest.fixture(scope="module")
def ollama_host():
    import ollama

    container = _container_mod.DockerContainer(_OLLAMA_IMAGE).with_exposed_ports(11434)
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(11434)
        endpoint = f"http://{host}:{port}"
        client = ollama.Client(host=endpoint)
        last_err: Exception | None = None
        for _ in range(60):
            try:
                client.list()
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"ollama did not become ready: {last_err}")
        # Pull the model once for the module (blocking; can take a minute).
        client.pull(_MODEL)
        yield endpoint


@pytest.mark.asyncio
async def test_real_inference_streams_tokens(ollama_host):
    llm = OllamaLLM(
        LLMConfig(model=_MODEL, api_key="", provider="ollama", max_tokens=32),
        host=ollama_host,
    )
    out = "".join([t async for t in llm.stream("Reply with a short greeting.")])
    assert out.strip(), "expected non-empty real inference output"
    assert llm._output_tokens > 0


def test_stream_is_async_generator():
    assert inspect.isasyncgenfunction(OllamaLLM.stream)
