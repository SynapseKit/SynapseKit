"""LLM API Recording & Replay ("LLM VCR") for testing."""

import hashlib
import json
import os
from collections.abc import AsyncGenerator
from contextlib import contextmanager
from typing import Any

import yaml

from synapsekit.llm.base import BaseLLM


def get_all_subclasses(cls: type) -> set[type]:
    """Recursively get all subclasses of a given class."""
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


@contextmanager
def use_cassette(cassette_path: str):
    """
    Context manager to record and replay LLM requests.

    If the cassette file exists, it will replay exact recorded responses
    instead of hitting live LLM APIs. If it doesn't exist, it will allow
    real network calls and record the responses to the cassette file.
    """
    if os.path.exists(cassette_path):
        with open(cassette_path, encoding="utf-8") as f:
            cassette = yaml.safe_load(f) or {}
    else:
        cassette = {}

    classes_to_patch = [BaseLLM, *list(get_all_subclasses(BaseLLM))]
    original_methods = {}

    def get_hash(method_name: str, instance: BaseLLM, *args: Any, **kwargs: Any) -> str:
        data = {
            "method": method_name,
            "provider": getattr(instance.config, "provider", "unknown"),
            "model": getattr(instance.config, "model", "unknown"),
            "system_prompt": getattr(instance.config, "system_prompt", ""),
            "args": args,
            "kwargs": kwargs,
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def make_patched_generate_uncached(orig_method):
        async def patched(self: BaseLLM, prompt: str, **kw: Any) -> str:
            key = get_hash("_generate_uncached", self, prompt, **kw)
            if key in cassette:
                return cassette[key]["response"]
            result = await orig_method(self, prompt, **kw)
            cassette[key] = {"request": {"prompt": prompt, "kwargs": kw}, "response": result}
            return result

        return patched

    def make_patched_generate_with_messages_uncached(orig_method):
        async def patched(self: BaseLLM, messages: list[dict[str, Any]], **kw: Any) -> str:
            key = get_hash("_generate_with_messages_uncached", self, messages, **kw)
            if key in cassette:
                return cassette[key]["response"]
            result = await orig_method(self, messages, **kw)
            cassette[key] = {"request": {"messages": messages, "kwargs": kw}, "response": result}
            return result

        return patched

    def make_patched_call_with_tools_impl(orig_method):
        async def patched(
            self: BaseLLM, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> dict[str, Any]:
            key = get_hash("_call_with_tools_impl", self, messages, tools)
            if key in cassette:
                return cassette[key]["response"]
            result = await orig_method(self, messages, tools)
            cassette[key] = {"request": {"messages": messages, "tools": tools}, "response": result}
            return result

        return patched

    def make_patched_stream(orig_method):
        async def patched(self: BaseLLM, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
            key = get_hash("stream", self, prompt, **kw)
            if key in cassette:
                for token in cassette[key]["response"]:
                    yield token
                return
            tokens = []
            async for token in orig_method(self, prompt, **kw):
                tokens.append(token)
                yield token
            cassette[key] = {"request": {"prompt": prompt, "kwargs": kw}, "response": tokens}

        return patched

    def make_patched_stream_with_messages(orig_method):
        async def patched(
            self: BaseLLM, messages: list[dict[str, Any]], **kw: Any
        ) -> AsyncGenerator[str, None]:
            key = get_hash("stream_with_messages", self, messages, **kw)
            if key in cassette:
                for token in cassette[key]["response"]:
                    yield token
                return
            tokens = []
            async for token in orig_method(self, messages, **kw):
                tokens.append(token)
                yield token
            cassette[key] = {"request": {"messages": messages, "kwargs": kw}, "response": tokens}

        return patched

    patch_map = {
        "_generate_uncached": make_patched_generate_uncached,
        "_generate_with_messages_uncached": make_patched_generate_with_messages_uncached,
        "_call_with_tools_impl": make_patched_call_with_tools_impl,
        "stream": make_patched_stream,
        "stream_with_messages": make_patched_stream_with_messages,
    }

    try:
        for cls in classes_to_patch:
            original_methods[cls] = {}
            for method_name, factory in patch_map.items():
                if method_name in cls.__dict__:
                    orig_method = cls.__dict__[method_name]
                    original_methods[cls][method_name] = orig_method
                    setattr(cls, method_name, factory(orig_method))
        yield
    finally:
        for cls, methods in original_methods.items():
            for method_name, orig_method in methods.items():
                setattr(cls, method_name, orig_method)

        if cassette:
            os.makedirs(os.path.dirname(os.path.abspath(cassette_path)), exist_ok=True)
            with open(cassette_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cassette, f, sort_keys=False)
