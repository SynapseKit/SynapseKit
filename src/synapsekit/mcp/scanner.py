"""Heuristic scanning of MCP tool descriptions and input schemas."""

from __future__ import annotations

import asyncio
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from typing import Any

from .pinning import ToolDefinition


@dataclass(frozen=True, slots=True)
class ScanFinding:
    tool_name: str
    fingerprint: str
    severity: str
    rule: str
    message: str


@dataclass(frozen=True, slots=True)
class ScanReport:
    findings: tuple[ScanFinding, ...]


class ToolScanBlockedError(RuntimeError):
    """Registration was blocked because the report contains high findings."""

    def __init__(self, report: ScanReport) -> None:
        self.report = report
        super().__init__("MCP tool registration blocked by high-severity scan findings")


class ToolScanLimitError(ValueError):
    """The input exceeds the scanner's safety limits; no report was produced."""


MAX_TOOLS = 256
MAX_DESCRIPTION_CHARS = 16_384
MAX_TOTAL_DESCRIPTION_CHARS = 262_144
MAX_SCHEMA_NODES = 16_384
MAX_SCHEMA_DEPTH = 64


def _schema(tool: Any) -> Any:
    value = getattr(tool, "inputSchema", None)
    return getattr(tool, "input_schema", None) if value is None else value


def _schema_nodes(value: Any) -> Iterable[tuple[Any, int]]:
    """Walk keys and values iteratively without allocating a width-sized stack."""

    def children(mapping: dict, depth: int) -> Iterable[tuple[Any, int]]:
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise ValueError("MCP scan schema keys must be strings")
            yield key, depth
            yield item, depth

    def elements(items: list, depth: int) -> Iterable[tuple[Any, int]]:
        for item in items:
            yield item, depth

    stack = [iter(((value, 0),))]
    while stack:
        try:
            node, depth = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        yield node, depth
        if isinstance(node, dict):
            stack.append(iter(children(node, depth + 1)))
        elif isinstance(node, list):
            stack.append(iter(elements(node, depth + 1)))


def _bounded_tools(tools: Iterable[ToolDefinition]) -> list[ToolDefinition]:
    """Bound work before snapshots, normalization, and regular expressions."""
    bounded: list[ToolDefinition] = []
    total = 0
    names_total = 0
    nodes = 0
    for tool in tools:
        if len(bounded) >= MAX_TOOLS:
            raise ToolScanLimitError("MCP scan tool count limit exceeded")
        description = getattr(tool, "description", None)
        length = len(description) if description else len(tool.name) + len("MCP tool: ")
        if length > MAX_DESCRIPTION_CHARS:
            raise ToolScanLimitError("MCP scan individual description limit exceeded")
        total += length
        if total > MAX_TOTAL_DESCRIPTION_CHARS:
            raise ToolScanLimitError("MCP scan total text limit exceeded")
        name_length = len(tool.name)
        names_total += name_length
        if name_length > MAX_DESCRIPTION_CHARS or names_total > MAX_TOTAL_DESCRIPTION_CHARS:
            raise ToolScanLimitError("MCP scan tool name limit exceeded")
        for value, depth in _schema_nodes(_schema(tool)):
            nodes += 1
            if nodes > MAX_SCHEMA_NODES or depth > MAX_SCHEMA_DEPTH:
                raise ToolScanLimitError("MCP scan schema structure limit exceeded")
            if isinstance(value, str):
                if len(value) > MAX_DESCRIPTION_CHARS:
                    raise ToolScanLimitError("MCP scan individual schema text limit exceeded")
                total += len(value)
                if total > MAX_TOTAL_DESCRIPTION_CHARS:
                    raise ToolScanLimitError("MCP scan total text limit exceeded")
            elif isinstance(value, int) and value.bit_length() > 4096:
                raise ToolScanLimitError("MCP scan schema number limit exceeded")
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError("MCP scan schema contains a non-finite number")
            elif not isinstance(value, (dict, list, int, float, type(None))):
                raise ValueError("MCP scan schema contains a non-JSON value")
        bounded.append(tool)
    return bounded


_RULES = (
    (
        "instruction_override",
        "high",
        "Possible instruction override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:(?:all|the)\s+)?(?:previous|prior|earlier|system)\s+(?:instructions|rules|prompts)\b",
            re.I,
        ),
    ),
    (
        "concealment",
        "high",
        "Possible instruction concealment",
        re.compile(
            r"\b(?:do not|don't|never)\s+(?:tell|inform|show|reveal|mention)\b.{0,80}\b(?:user|human)\b",
            re.I | re.S,
        ),
    ),
    (
        "role_spoofing",
        "high",
        "Possible role spoofing",
        re.compile(r"(?:\[\s*(?:system|developer)\s*\]|<\s*(?:system|developer)\s*>)", re.I),
    ),
    (
        "remote_execution",
        "high",
        "Possible remote script execution",
        re.compile(r"\b(?:curl|wget)\b[^\n]{0,500}\|\s*(?:ba|z|a)?sh\b", re.I),
    ),
)
_SECRET = re.compile(
    r"(?:\.ssh[/\\]|id_rsa|\.env\b|credentials|api[ _-]?keys?|tokens?|passwords?|secrets?)", re.I
)
_TRANSFER = re.compile(r"\b(?:send|upload|transmit|exfiltrate|post|forward)\b", re.I)
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")


async def scan_tools(tools: Iterable[ToolDefinition]) -> ScanReport:
    """Inspect effective descriptions and schema strings in each snapshot."""
    return await asyncio.to_thread(_scan_tools_sync, tools)


def _scan_tools_sync(tools: Iterable[ToolDefinition]) -> ScanReport:
    findings: list[ScanFinding] = []
    for tool in _bounded_tools(tools):
        matched: set[str] = set()
        texts = (value for value, _ in _schema_nodes(_schema(tool)) if isinstance(value, str))
        for raw in chain((tool.description or f"MCP tool: {tool.name}",), texts):
            text = unicodedata.normalize("NFKC", raw)
            normalized = _INVISIBLE.sub("", text)
            for rule, _, _, pattern in _RULES:
                if pattern.search(normalized):
                    matched.add(rule)
            if _SECRET.search(normalized) and _TRANSFER.search(normalized):
                matched.add("secret_exfiltration")
            if _INVISIBLE.search(text):
                matched.add("hidden_text")
        for rule, severity, message, _ in _RULES:
            if rule in matched:
                findings.append(ScanFinding(tool.name, tool.fingerprint, severity, rule, message))
        if "secret_exfiltration" in matched:
            findings.append(
                ScanFinding(
                    tool.name,
                    tool.fingerprint,
                    "high",
                    "secret_exfiltration",
                    "Possible secret exfiltration",
                )
            )
        if "hidden_text" in matched:
            findings.append(
                ScanFinding(
                    tool.name,
                    tool.fingerprint,
                    "medium",
                    "hidden_text",
                    "Invisible or bidirectional characters in tool text",
                )
            )
    return ScanReport(findings=tuple(findings))
