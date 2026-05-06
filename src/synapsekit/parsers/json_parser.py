from __future__ import annotations

import re
from typing import Any

from .._json import loads as _json_loads


class JSONParser:
    """Extract and parse JSON from LLM output text."""

    __slots__ = ()

    def parse(self, text: str) -> Any:
        text = text.strip()
        try:
            return _json_loads(text)
        except (ValueError, TypeError):
            pass

        # Try to extract JSON object or array with regex
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return _json_loads(match.group(1))
            except (ValueError, TypeError):
                pass

        raise ValueError(f"Could not parse JSON from: {text[:100]!r}")
