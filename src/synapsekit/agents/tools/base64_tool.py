from __future__ import annotations

import base64
from typing import Any

from ..base import BaseTool, ToolResult


class Base64Tool(BaseTool):
    """Encode or decode Base64 strings."""

    name = "base64"
    description = (
        "Encode text to Base64 or decode Base64 back to text. "
        "LLMs often struggle with exact Base64 string manipulation due to tokenization, "
        "so this tool ensures exact, character-perfect encoding/decoding. "
        "Input: action ('encode' or 'decode') and the string 'text'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["encode", "decode"],
                "description": "Whether to 'encode' plain text to Base64 or 'decode' Base64 to text.",
            },
            "text": {
                "type": "string",
                "description": "The text to encode or decode.",
            },
        },
        "required": ["action", "text"],
    }

    async def run(self, action: str = "", text: str = "", **kwargs: Any) -> ToolResult:
        """Run the base64 tool."""
        _action = action or kwargs.get("input", "")
        _text = text or kwargs.get("value", "")

        if not _action or not _text:
            return ToolResult(output="", error="Both 'action' and 'text' must be provided.")

        try:
            if _action == "encode":
                encoded_bytes = base64.b64encode(_text.encode("utf-8"))
                return ToolResult(output=encoded_bytes.decode("utf-8"))
            elif _action == "decode":
                decoded_bytes = base64.b64decode(_text.encode("utf-8"), validate=True)
                return ToolResult(output=decoded_bytes.decode("utf-8"))
            else:
                return ToolResult(
                    output="", error=f"Unknown action: {_action!r}. Use 'encode' or 'decode'."
                )
        except Exception as e:
            return ToolResult(output="", error=f"Base64 error: {e}")
