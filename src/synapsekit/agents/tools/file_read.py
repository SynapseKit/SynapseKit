from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class FileReadTool(BaseTool):
    """Read the contents of a local file."""

    name = "file_read"
    description = "Read the contents of a file from disk. Input: an absolute or relative file path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to read",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding (default: utf-8)",
                "default": "utf-8",
            },
        },
        "required": ["path"],
    }

    async def run(self, path: str = "", encoding: str = "utf-8", **kwargs: Any) -> ToolResult:
        file_path = path or kwargs.get("input", "")
        if not file_path:
            return ToolResult(output="", error="No file path provided.")
        try:
            with open(file_path, encoding=encoding) as f:
                content = f.read()
            return ToolResult(output=content)
        except FileNotFoundError:
            return ToolResult(output="", error=f"File not found: {file_path!r}")
        except PermissionError:
            return ToolResult(output="", error=f"Permission denied: {file_path!r}")
        except Exception as e:
            return ToolResult(output="", error=f"Error reading file: {e}")
