from __future__ import annotations

from typing import Any

# Sentinel: a node's condition_fn returns END to signal graph termination.
END = "__end__"

# Type alias — graphs pass around plain dicts.
GraphState = dict[str, Any]
