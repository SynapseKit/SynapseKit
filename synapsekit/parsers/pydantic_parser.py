from __future__ import annotations

from typing import Any, Type, TypeVar

T = TypeVar("T")


class PydanticParser:
    """Parse LLM JSON output into a Pydantic model."""

    def __init__(self, model: Type[T]) -> None:
        self._model = model

    def parse(self, text: str) -> Any:
        try:
            from pydantic import BaseModel  # noqa: F401
        except ImportError:
            raise ImportError("pydantic required: pip install pydantic")

        from .json_parser import JSONParser

        data = JSONParser().parse(text)
        return self._model(**data)
