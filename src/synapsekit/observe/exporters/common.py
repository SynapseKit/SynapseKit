from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime import ObserveSpan


class BufferedSpanExporter:
    def __init__(
        self,
        *,
        service_name: str = "synapsekit",
        kind: str,
        endpoint: str | None = None,
    ) -> None:
        self.service_name = service_name
        self.kind = kind
        self.endpoint = endpoint
        self.spans: list[ObserveSpan] = []

    def export(self, span: "ObserveSpan") -> None:
        self.spans.append(span)

    def clear(self) -> None:
        self.spans.clear()

    def export_dicts(self) -> list[dict]:
        return [span.to_dict() for span in self.spans]
