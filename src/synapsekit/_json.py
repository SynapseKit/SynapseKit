"""Fast JSON helpers — uses ``orjson`` when available, falls back to stdlib."""

from __future__ import annotations

from typing import Any

try:
    import orjson

    def dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode()

    def dumps_bytes(obj: Any) -> bytes:
        return orjson.dumps(obj)

    def loads(s: str | bytes) -> Any:
        return orjson.loads(s)

except ImportError:
    import json

    def dumps(obj: Any) -> str:  # type: ignore[misc]
        return json.dumps(obj)

    def dumps_bytes(obj: Any) -> bytes:  # type: ignore[misc]
        return json.dumps(obj).encode()

    def loads(s: str | bytes) -> Any:  # type: ignore[misc]
        return json.loads(s)
