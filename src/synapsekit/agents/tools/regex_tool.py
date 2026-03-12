from __future__ import annotations

import re
from typing import Any

from ..base import BaseTool, ToolResult


class RegexTool(BaseTool):
    """Apply regex operations: match, search, findall, replace."""

    name = "regex"
    description = (
        "Apply a regular expression to text. "
        "Input: pattern, text, and action (match/search/findall/replace)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression pattern"},
            "text": {"type": "string", "description": "Text to apply the regex to"},
            "action": {
                "type": "string",
                "description": "Action: 'findall' (default), 'match', 'search', 'replace', 'split'",
                "enum": ["findall", "match", "search", "replace", "split"],
                "default": "findall",
            },
            "replacement": {
                "type": "string",
                "description": "Replacement string (for 'replace' action)",
                "default": "",
            },
            "flags": {
                "type": "string",
                "description": "Regex flags: 'i' (ignore case), 'm' (multiline), 's' (dotall)",
                "default": "",
            },
        },
        "required": ["pattern", "text"],
    }

    def _parse_flags(self, flags_str: str) -> int:
        flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
        result = 0
        for ch in flags_str.lower():
            if ch in flag_map:
                result |= flag_map[ch]
        return result

    async def run(
        self,
        pattern: str = "",
        text: str = "",
        action: str = "findall",
        replacement: str = "",
        flags: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        if not pattern:
            return ToolResult(output="", error="No pattern provided.")
        if not text:
            return ToolResult(output="", error="No text provided.")

        try:
            re_flags = self._parse_flags(flags)

            if action == "findall":
                matches = re.findall(pattern, text, re_flags)
                return ToolResult(
                    output="\n".join(str(m) for m in matches) if matches else "(no matches)"
                )

            elif action == "match":
                m = re.match(pattern, text, re_flags)
                if m:
                    return ToolResult(output=f"Match: {m.group(0)}\nGroups: {m.groups()}")
                return ToolResult(output="(no match)")

            elif action == "search":
                m = re.search(pattern, text, re_flags)
                if m:
                    return ToolResult(
                        output=f"Found: {m.group(0)}\nPosition: {m.start()}-{m.end()}\nGroups: {m.groups()}"
                    )
                return ToolResult(output="(no match)")

            elif action == "replace":
                result = re.sub(pattern, replacement, text, flags=re_flags)
                return ToolResult(output=result)

            elif action == "split":
                parts = re.split(pattern, text, flags=re_flags)
                return ToolResult(output="\n".join(parts))

            else:
                return ToolResult(output="", error=f"Unknown action: {action!r}")

        except re.error as e:
            return ToolResult(output="", error=f"Invalid regex: {e}")
        except Exception as e:
            return ToolResult(output="", error=f"Regex error: {e}")
