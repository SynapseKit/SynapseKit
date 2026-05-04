"""Tokenizer-aware token counting utilities for retrieval packing."""

from __future__ import annotations

from collections.abc import Callable


class TokenCounter:
    """Count tokens using model-aware tokenizers.

    Resolution order:
    1) explicit backend (``tiktoken`` or ``transformers``)
    2) ``auto``: tiktoken first, transformers second

    A custom ``count_fn`` can be passed for deterministic tests.
    """

    def __init__(
        self,
        model: str | None = None,
        backend: str = "auto",
        count_fn: Callable[[str], int] | None = None,
    ) -> None:
        self.model = model
        self.backend = backend

        self._cache: dict[str, int] = {}
        self._cache_order: list[str] = []
        self._cache_maxsize = 8192

        if count_fn is not None:
            self._count_impl = count_fn
            self._backend_used = "custom"
            return

        self._count_impl, self._backend_used = self._resolve_counter(backend)

    @property
    def backend_used(self) -> str:
        return self._backend_used

    def count(self, text: str) -> int:
        if not text:
            return 0
        return self._count_impl(text)

    def count_cached(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        value = self.count(text)
        self._cache[text] = value
        self._cache_order.append(text)

        if len(self._cache_order) > self._cache_maxsize:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

        return value

    def _resolve_counter(self, backend: str) -> tuple[Callable[[str], int], str]:
        if backend not in {"auto", "tiktoken", "transformers"}:
            raise ValueError(
                f"Unknown token counter backend: {backend!r}. "
                "Use 'auto', 'tiktoken', or 'transformers'."
            )

        if backend in {"auto", "tiktoken"}:
            counter = self._try_tiktoken_counter()
            if counter is not None:
                return counter, "tiktoken"
            if backend == "tiktoken":
                raise ImportError(
                    "tiktoken backend requested but unavailable. Install with: pip install tiktoken"
                )

        if backend in {"auto", "transformers"}:
            counter = self._try_transformers_counter()
            if counter is not None:
                return counter, "transformers"
            if backend == "transformers":
                raise ImportError(
                    "transformers backend requested but unavailable. Install with: pip install transformers"
                )

        raise ImportError(
            "No supported tokenizer backend available for token counting. "
            "Install tiktoken or transformers."
        )

    def _try_tiktoken_counter(self) -> Callable[[str], int] | None:
        try:
            import tiktoken
        except ImportError:
            return None

        if self.model:
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            return len(enc.encode(text, disallowed_special=()))

        return _count

    def _try_transformers_counter(self) -> Callable[[str], int] | None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return None

        tokenizer_name = self.model or "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def _count(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        return _count
