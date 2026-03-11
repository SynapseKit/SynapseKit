from __future__ import annotations

import math
from typing import Any

from ..base import BaseTool, ToolResult

_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "divmod": divmod,
    "int": int,
    "float": float,
    # math module
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "inf": math.inf,
}


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely."""

    name = "calculator"
    description = (
        "Evaluate a mathematical expression. "
        "Input: a math expression string, e.g. '2 + 2 * 3' or 'sqrt(144)'. "
        "Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, pi, e, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate, e.g. '2 ** 10'",
            }
        },
        "required": ["expression"],
    }

    async def run(self, expression: str = "", **kwargs: Any) -> ToolResult:
        expr = expression or kwargs.get("input", "")
        if not expr:
            return ToolResult(output="", error="No expression provided.")
        try:
            result = eval(expr, _SAFE_GLOBALS, {})
            return ToolResult(output=str(result))
        except ZeroDivisionError:
            return ToolResult(output="", error="Division by zero.")
        except Exception as e:
            return ToolResult(output="", error=f"Could not evaluate expression: {e}")
