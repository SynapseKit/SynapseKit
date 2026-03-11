from __future__ import annotations

import re


class ListParser:
    """Parse bullet or numbered list from LLM output into a Python list."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().splitlines()
        result = []
        for line in lines:
            cleaned = re.sub(r"^[\s]*(?:[-*•]|\d+[.):])\s*", "", line).strip()
            if cleaned:
                result.append(cleaned)
        return result
