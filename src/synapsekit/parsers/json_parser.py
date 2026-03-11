from __future__ import annotations

import json
import re
from typing import Any


class JSONParser:
    """Extract and parse JSON from LLM output text."""

    def parse(self, text: str) -> Any:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object or array with regex
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from: {text[:100]!r}")
