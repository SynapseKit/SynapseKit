from __future__ import annotations

from .common import BufferedSpanExporter


class ConsoleExporter(BufferedSpanExporter):
    def __init__(self, *, service_name: str = "synapsekit", endpoint: str | None = None) -> None:
        super().__init__(service_name=service_name, kind="console", endpoint=endpoint)
