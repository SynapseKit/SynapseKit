"""Private metadata carried alongside provider-compatible chat messages."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_SOURCE_ID_KEYS = frozenset(
    {
        "source_id",
        "source_ids",
        "document_id",
        "document_ids",
        "doc_id",
        "doc_ids",
        "citation_id",
    }
)


class SourceAwareMessages(list[dict[str, Any]]):
    """A normal list of provider messages with private audit-only metadata."""

    __slots__ = (
        "_trusted_citation_ids",
        "_trusted_source_ids",
        "_unresolvable_citation_ids",
    )

    def __init__(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        trusted_source_ids: Iterable[str] = (),
        trusted_citation_ids: Mapping[str, str] | None = None,
        unresolvable_citation_ids: Iterable[str] = (),
    ) -> None:
        super().__init__(messages)
        self._trusted_source_ids = tuple(dict.fromkeys(trusted_source_ids))
        self._trusted_citation_ids = tuple((trusted_citation_ids or {}).items())
        self._unresolvable_citation_ids = tuple(dict.fromkeys(unresolvable_citation_ids))


def source_ids_from_results(results: Any) -> list[str]:
    """Extract explicit source IDs from retrieved result metadata only."""
    found: dict[str, None] = {}

    def reference_values(value: Any) -> Iterable[str]:
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            for item in value:
                yield from reference_values(item)
        elif (
            isinstance(value, str | int | float)
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        ):
            text = str(value)
            if text:
                yield text

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if isinstance(key, str) and key.lower() in _SOURCE_ID_KEYS:
                    for source_id in reference_values(child):
                        found.setdefault(source_id, None)
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, str | bytes | bytearray):
            for child in node:
                visit(child)

    if isinstance(results, Mapping):
        visit(results.get("metadata", {}))
    elif isinstance(results, Sequence) and not isinstance(results, str | bytes | bytearray):
        for result in results:
            if isinstance(result, Mapping):
                visit(result.get("metadata", {}))
    return list(found)
