from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import BaseTool, ToolResult


class FileWriteTool(BaseTool):
    """Write content to a local file."""

    name = "file_write"
    description = (
        "Write content to a file on disk. Creates parent directories if needed. "
        "Input: path and content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The file path to write to"},
            "content": {"type": "string", "description": "The content to write"},
            "append": {
                "type": "boolean",
                "description": "Append to file instead of overwriting (default: false)",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir else None

    async def run(
        self,
        path: str = "",
        content: str = "",
        append: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not path:
            return ToolResult(output="", error="No file path provided.")

        try:
            resolved = Path(path).resolve()
            if self._base_dir is not None and not str(resolved).startswith(str(self._base_dir)):
                return ToolResult(output="", error="Access denied: path is outside the allowed directory.")

            resolved.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(resolved, mode, encoding="utf-8") as f:
                f.write(content)

            action = "Appended to" if append else "Written to"
            return ToolResult(output=f"{action} {path} ({len(content)} chars)")
        except PermissionError:
            return ToolResult(output="", error=f"Permission denied: {path!r}")
        except Exception as e:
            return ToolResult(output="", error=f"Error writing file: {e}")
